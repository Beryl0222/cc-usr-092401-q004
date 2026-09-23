"""并发一致性契约：两台扫码终端几乎同时提交时，交接链必须只有一条。

覆盖：

- 线程竞争：同一扫码号的完全相同重试只有一条写入，其余取回首条交接；
  同扫码号但签认/日期/状态/地点/关联区段不同一律冲突；不同扫码抢同一
  生命周期步骤只允许一条成功，保管方不得跳变。
- 失败回滚：字段错误、跳步、冻结拦截都不消费扫码号，之后该码仍可用。
- 损伤竞态：损伤上报与“良好”上报同时抢同一步骤时，冻结必先于任何
  后续保管权转移；交接链永远是生命周期的严格前缀。
- 双方复核：解除须有交出方与接收方两条书面结论；作品下仍有未结损伤时
  冻结不解除。
- HTTP 区分：重放 200（Idempotency-Replayed 头 + replay 字段）、冲突
  409（kind=scan_conflict / conflict）、字段错误 400（kind=invalid_request）。
- 赛后查询：作品视图的 handover_chain 扫码唯一、步骤连续、保管状态一致。
"""

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from domain import (
    ConflictError,
    DomainError,
    HANDOVER_TYPES,
    Incident,
    LoanRegistry,
    ReplayConflict,
    _new_id,
)
from domain_contract import (
    agreement_payload,
    handover_payload,
    make_long_scroll,
)


# ---------------------------------------------------------------------------
# 领域层：直接对 LoanRegistry 做线程竞争
# ---------------------------------------------------------------------------


def new_registry_with_work():
    registry = LoanRegistry()
    work = registry.register_work("并发试卷", "独立作品", "甲馆")
    work_id = work["work"]["work_id"]
    registry.create_agreement(agreement_payload(work_id))
    return registry, work_id


def run_together(callables, workers=None):
    """所有任务在同一道栅栏后同时起跑，最大化临界区竞争。"""
    barrier = threading.Barrier(len(callables))

    def run(fn):
        barrier.wait()
        try:
            return ("ok", fn())
        except Exception as error:  # noqa: BLE001 - 需要把异常类型带回主线程断言
            return ("error", error)

    with ThreadPoolExecutor(max_workers=workers or len(callables)) as pool:
        return list(pool.map(run, callables))


class DomainConcurrencyTest(unittest.TestCase):
    def test_identical_concurrent_submissions_produce_single_handover(self):
        registry, work_id = new_registry_with_work()
        payload = handover_payload(work_id, "出库", "SCAN-RACE", "2026-09-25")

        results = run_together([lambda: registry.record_handover(payload) for _ in range(8)])

        views = [v for kind, v in results if kind == "ok"]
        replays = [v for kind, v in results if kind == "ok" and v.get("replay")]
        self.assertEqual(len(views), 8)  # 全部按 200 语义成功：1 条首交 + 7 条重放
        self.assertEqual(len(replays), 7)
        self.assertEqual(len({v["handover_id"] for v in views}), 1)
        first_id = views[0]["handover_id"]
        self.assertTrue(all(v["replay_of"] == first_id for v in replays))

        # 交接链只有一条，扫码只被占用一次，保管停在“出库 → 运输方”。
        self.assertEqual(len(registry.handovers), 1)
        self.assertEqual(len(registry._scan_codes), 1)
        custody = registry.get_work_view(work_id)["custody"]
        self.assertEqual((custody["status"], custody["custodian_role"]), ("运输中", "运输方"))

    def test_same_scan_different_content_one_wins_rest_conflict(self):
        registry, work_id = new_registry_with_work()

        def payload(i):
            p = handover_payload(work_id, "出库", "SCAN-CLASH", f"2026-10-0{i + 1}")
            p["to_party"]["person"] = f"押运员{i:02d}"  # 签认人各不相同
            return p

        payloads = [payload(i) for i in range(6)]
        results = run_together([lambda p=p: registry.record_handover(p) for p in payloads])

        successes = [v for kind, v in results if kind == "ok"]
        conflicts = [v for kind, v in results
                     if kind == "error" and isinstance(v, ReplayConflict)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(conflicts), 5)
        self.assertEqual(len(registry.handovers), 1)
        self.assertEqual(len(registry._scan_codes), 1)
        # 赢家之外的签认人没有造成保管方跳变。
        winner = registry.handovers[0]
        self.assertEqual(winner.to_party.role, "运输方")
        self.assertEqual(registry.get_work_view(work_id)["custody"]["since_handover"],
                         winner.handover_id)

    def test_distinct_scans_racing_same_step_only_one_commits(self):
        registry, work_id = new_registry_with_work()
        scans = [f"SCAN-STEP-{i}" for i in range(8)]
        results = run_together([
            lambda s=s: registry.record_handover(
                handover_payload(work_id, "出库", s, "2026-09-25"))
            for s in scans
        ])

        self.assertEqual(sum(1 for kind, _ in results if kind == "ok"), 1)
        self.assertEqual(sum(
            1 for kind, v in results
            if kind == "error" and isinstance(v, ConflictError)
        ), 7)
        self.assertEqual(len(registry.handovers), 1)
        # 失败请求没有消耗扫码号：7 个落败码都未被占用。
        winner_scan = registry.handovers[0].scan_code
        for scan in scans:
            if scan != winner_scan:
                self.assertNotIn(scan, registry._scan_codes)
        # 落败码在它真正对应的步骤仍可使用（这里直接用到下一步“到馆”）。
        reused = registry.record_handover(
            handover_payload(work_id, "到馆", scans[0], "2026-09-27",
                             location="乙馆收货区"))
        self.assertEqual(reused["type"], "到馆")

    def test_mixed_step_storm_leaves_strict_lifecycle_prefix(self):
        registry, work_id = new_registry_with_work()
        storm = []
        for round_index in range(2):  # 每步放多个不同扫码、并混入跳步请求
            for step_index, htype in enumerate(HANDOVER_TYPES):
                scan = f"SCAN-MIX-{round_index}-{step_index}"
                day = f"2026-10-{10 + step_index:02d}"
                location = {"出库": "甲馆库房", "到馆": "乙馆收货区", "布展": "乙馆三号厅",
                            "撤展": "乙馆三号厅", "归还": "甲馆库房"}[htype]
                storm.append(lambda htype=htype, scan=scan, day=day, location=location:
                             registry.record_handover(
                                 handover_payload(work_id, htype, scan, day, location=location)))
        run_together(storm)

        # 无论线程如何交织，交接链只能是生命周期的一个严格前缀：
        # 无重复步骤、无跳步、扫码全局唯一。
        view = registry.get_work_view(work_id)
        chain = view["handover_chain"]
        self.assertEqual([c["type"] for c in chain], list(HANDOVER_TYPES[: len(chain)]))
        self.assertEqual(len({c["scan_code"] for c in chain}), len(chain))
        self.assertEqual(view["custody"]["status"], chain[-1]["resulting_status"])


class ReplayFingerprintTest(unittest.TestCase):
    """完全相同的重放取回首条；签认/日期/状态/区段/地点任一不同即冲突。"""

    def setUp(self):
        self.registry = LoanRegistry()
        self.scroll = make_long_scroll(self.registry)
        self.work_id = self.scroll["work"]["work_id"]
        self.segments = [s["segment_id"] for s in self.scroll["segments"]]
        self.registry.create_agreement(agreement_payload(self.work_id))
        self.base = handover_payload(
            self.work_id, "出库", "SCAN-FP", "2026-09-25",
            linked_segments=[self.segments[0]],
        )
        self.first = self.registry.record_handover(self.base)

    def retry_expecting_replay(self, payload):
        view = self.registry.record_handover(payload)
        self.assertTrue(view["replay"])
        self.assertEqual(view["handover_id"], self.first["handover_id"])
        self.assertEqual(len(self.registry.handovers), 1)

    def retry_expecting_conflict(self, payload):
        with self.assertRaises(ReplayConflict):
            self.registry.record_handover(payload)
        self.assertEqual(len(self.registry.handovers), 1)

    def test_identical_retry_returns_first_handover(self):
        self.retry_expecting_replay(dict(self.base))

    def test_different_from_signature_conflicts(self):
        p = json.loads(json.dumps(self.base, ensure_ascii=False))
        p["from_party"]["person"] = "另一位出库员"
        self.retry_expecting_conflict(p)

    def test_different_date_conflicts(self):
        p = dict(self.base)
        p["on_date"] = "2026-09-26"
        self.retry_expecting_conflict(p)

    def test_different_condition_conflicts_without_creating_incident(self):
        p = json.loads(json.dumps(self.base, ensure_ascii=False))
        p["report"] = {
            "condition": "损伤", "damage_note": "重放时谎称损伤",
            "before_hashes": ["a" * 64], "after_hashes": ["b" * 64],
        }
        self.retry_expecting_conflict(p)
        # 冲突请求不能登记风险事件，也不能冻结作品。
        self.assertEqual(self.registry.incidents, [])
        self.assertFalse(self.registry.get_work_view(self.work_id)["frozen"])

    def test_different_linked_segments_conflicts(self):
        p = json.loads(json.dumps(self.base, ensure_ascii=False))
        p["linked_segments"] = [self.segments[1]]
        self.retry_expecting_conflict(p)

    def test_different_location_conflicts(self):
        p = dict(self.base)
        p["at_location"] = "另一处月台"
        self.retry_expecting_conflict(p)

    def test_extra_hash_entries_are_equivalent_for_identity_images(self):
        # 同一张图用明文与用其 sha256 提交，归一化后应视为同一份证据。
        import hashlib
        digest = hashlib.sha256("出库全景照".encode("utf-8")).hexdigest()
        p = json.loads(json.dumps(self.base, ensure_ascii=False))
        p["report"]["image_hashes"] = [digest]
        self.retry_expecting_replay(p)


class FailedRequestRollbackTest(unittest.TestCase):
    def setUp(self):
        self.registry, self.work_id = new_registry_with_work()

    def test_field_error_does_not_consume_scan(self):
        bad = handover_payload(self.work_id, "出库", "SCAN-SAVE", "2026-09-25")
        bad["to_party"]["person"] = ""
        with self.assertRaises(DomainError):
            self.registry.record_handover(bad)
        self.assertNotIn("SCAN-SAVE", self.registry._scan_codes)
        # 同一扫码修正后立即成功。
        ok = self.registry.record_handover(
            handover_payload(self.work_id, "出库", "SCAN-SAVE", "2026-09-25"))
        self.assertEqual(ok["scan_code"], "SCAN-SAVE")
        self.assertEqual(len(self.registry.handovers), 1)

    def test_lifecycle_conflict_does_not_consume_scan(self):
        with self.assertRaises(ConflictError):
            self.registry.record_handover(
                handover_payload(self.work_id, "布展", "SCAN-LATER", "2026-09-30"))
        self.assertNotIn("SCAN-LATER", self.registry._scan_codes)
        # 走完前四步，该码在真正的“布展”步可用。
        for htype, scan, day in (
            ("出库", "SCAN-1", "2026-09-25"),
            ("到馆", "SCAN-2", "2026-09-27"),
        ):
            self.registry.record_handover(
                handover_payload(self.work_id, htype, scan, day))
        done = self.registry.record_handover(
            handover_payload(self.work_id, "布展", "SCAN-LATER", "2026-09-30"))
        self.assertEqual(done["type"], "布展")

    def test_frozen_rejection_does_not_consume_scan(self):
        self.registry.record_handover(
            handover_payload(self.work_id, "出库", "SCAN-1", "2026-09-25"))
        self.registry.record_handover(handover_payload(
            self.work_id, "到馆", "SCAN-2", "2026-09-27",
            report={"condition": "损伤", "damage_note": "新增折痕",
                    "before_hashes": ["a" * 64], "after_hashes": ["b" * 64]},
        ))
        with self.assertRaises(ConflictError):
            self.registry.record_handover(
                handover_payload(self.work_id, "布展", "SCAN-FROZEN", "2026-09-30"))
        self.assertNotIn("SCAN-FROZEN", self.registry._scan_codes)


class DamageRaceTest(unittest.TestCase):
    """损伤上报与同一步骤的“正常”上报同时发生时，冻结必先于保管权转移。"""

    DAMAGE_REPORT = {
        "condition": "损伤", "damage_note": "画心左下角新增折痕",
        "image_hashes": ["到馆检视照"],
        "before_hashes": ["a" * 64], "after_hashes": ["b" * 64],
    }
    GOOD_REPORT = {"condition": "良好", "image_hashes": ["到馆检视照"]}

    def _fresh_work_after_outbound(self):
        registry = LoanRegistry()
        work = registry.register_work(f"竞态卷-{_new_id('race')}", "独立作品", "甲馆")
        work_id = work["work"]["work_id"]
        registry.create_agreement(agreement_payload(work_id))
        registry.record_handover(
            handover_payload(work_id, "出库", f"OUT-{work_id}", "2026-09-25"))
        return registry, work_id

    def test_damage_vs_good_race_never_advances_past_an_open_damage(self):
        # 反复开赛：哪条线程赢取决于调度，两种结局都必须保持结构不变量。
        for iteration in range(30):
            registry, work_id = self._fresh_work_after_outbound()
            damage = handover_payload(
                work_id, "到馆", f"DMG-{iteration}", "2026-09-27",
                location="乙馆收货区", report=dict(self.DAMAGE_REPORT))
            good = handover_payload(
                work_id, "到馆", f"OK-{iteration}", "2026-09-27",
                location="乙馆收货区", report=dict(self.GOOD_REPORT))
            results = run_together(
                [lambda: registry.record_handover(damage),
                 lambda: registry.record_handover(good)])

            successes = [v for kind, v in results if kind == "ok"]
            self.assertEqual(len(successes), 1, f"第 {iteration} 轮应恰好一条成功")
            chain = [h for h in registry.handovers if h.work_id == work_id]
            self.assertEqual(len(chain), 2)  # 出库 + 唯一步骤
            self.assertEqual([h.type for h in chain], ["出库", "到馆"])

            winner_damaged = chain[-1].damaged
            view = registry.get_work_view(work_id)
            if winner_damaged:
                # 损伤赢：必须已经冻结、风险已挂账，后续保管权转移全部被挡。
                self.assertTrue(view["frozen"])
                risks = registry.risk_view(work_id)["open_risks"]
                self.assertEqual(len(risks), 1)
                with self.assertRaises(ConflictError):
                    registry.record_handover(handover_payload(
                        work_id, "布展", f"NEXT-{iteration}", "2026-09-30"))
            else:
                # 良好赢：无风险、无冻结，链可以继续；损伤请求没有留下半成品。
                self.assertFalse(view["frozen"])
                self.assertEqual(registry.risk_view(work_id)["open_risks"], [])
                registry.record_handover(handover_payload(
                    work_id, "布展", f"NEXT-{iteration}", "2026-09-30"))

    def test_three_way_race_damage_good_next_step(self):
        for iteration in range(20):
            registry, work_id = self._fresh_work_after_outbound()
            damage = handover_payload(
                work_id, "到馆", f"3W-D-{iteration}", "2026-09-27",
                report=dict(self.DAMAGE_REPORT))
            good = handover_payload(
                work_id, "到馆", f"3W-G-{iteration}", "2026-09-27",
                report=dict(self.GOOD_REPORT))
            # 第三条企图在同场竞争里直接推进到布展。
            jump = handover_payload(
                work_id, "布展", f"3W-J-{iteration}", "2026-09-30")
            run_together([
                lambda: registry.record_handover(damage),
                lambda: registry.record_handover(good),
                lambda: registry.record_handover(jump),
            ])

            chain = [h for h in registry.handovers if h.work_id == work_id]
            types = [h.type for h in chain]
            self.assertEqual(types, list(HANDOVER_TYPES[:len(types)]))
            damaged_indexes = [i for i, h in enumerate(chain) if h.damaged]
            view = registry.get_work_view(work_id)
            if damaged_indexes:
                # 任何已登记损伤之后都不允许再有交接，且作品必须冻结。
                self.assertEqual(damaged_indexes[-1], len(chain) - 1)
                self.assertTrue(view["frozen"])


class ResolveReviewTest(unittest.TestCase):
    def setUp(self):
        self.registry, self.work_id = new_registry_with_work()
        self.registry.record_handover(
            handover_payload(self.work_id, "出库", "SCAN-1", "2026-09-25"))
        self.registry.record_handover(handover_payload(
            self.work_id, "到馆", "SCAN-2", "2026-09-27",
            report={"condition": "损伤", "damage_note": "边缘轻微磨损",
                    "before_hashes": ["c" * 64], "after_hashes": ["d" * 64]},
        ))
        self.incident_id = self.registry.risk_view(self.work_id)["open_risks"][0]["incident_id"]
        # 到馆交接：交出方=运输方（长风运输），接收方=承借馆（乙馆）。
        self.reviews = [
            {"party": "交出方", "org": "长风运输", "role": "运输方", "person": "押运员",
             "conclusion": "运输全程无碰撞，到馆开箱即见磨损"},
            {"party": "接收方", "org": "乙馆", "role": "承借馆", "person": "库管员",
             "conclusion": "复核确认磨损轻微且稳定，不影响后续展出"},
        ]

    def test_resolve_without_or_with_single_review_rejected_and_stays_frozen(self):
        with self.assertRaises(DomainError):
            self.registry.resolve_incident(self.incident_id, "有结论")
        with self.assertRaises(DomainError):
            self.registry.resolve_incident(self.incident_id, "仅接收方", reviews=self.reviews[1:])
        incident = next(i for i in self.registry.incidents if i.incident_id == self.incident_id)
        self.assertFalse(incident.resolved)
        self.assertTrue(self.registry.get_work_view(self.work_id)["frozen"])

    def test_review_with_wrong_role_rejected(self):
        wrong = json.loads(json.dumps(self.reviews, ensure_ascii=False))
        wrong[0]["role"] = "承借馆"
        with self.assertRaises(DomainError):
            self.registry.resolve_incident(self.incident_id, "结论", reviews=wrong)

    def test_review_without_conclusion_text_rejected(self):
        bad = json.loads(json.dumps(self.reviews, ensure_ascii=False))
        bad[1]["conclusion"] = "   "
        with self.assertRaises(DomainError):
            self.registry.resolve_incident(self.incident_id, "结论", reviews=bad)

    def test_both_reviews_resolve_and_chain_resumes(self):
        result = self.registry.resolve_incident(
            self.incident_id, "双方书面复核通过，恢复交接", reviews=self.reviews)
        self.assertTrue(result["resolved"])
        self.assertFalse(result["frozen"])
        self.assertEqual(result["open_incidents"], 0)
        self.assertEqual({r["party"] for r in result["reviews"]}, {"交出方", "接收方"})
        self.registry.record_handover(
            handover_payload(self.work_id, "布展", "SCAN-3", "2026-09-30"))

    def test_freeze_lifts_only_after_every_open_incident_has_reviews(self):
        # 现实里一次冻结通常只挂一条损伤；这里白框补登第二条未结损伤，
        # 验证解除规则按“全部未结损伤均结案”聚合生效。
        first_handover = next(
            h for h in self.registry.handovers
            if h.work_id == self.work_id and h.type == "到馆")
        second = Incident(
            incident_id=_new_id("incident"), work_id=self.work_id,
            handover_id=first_handover.handover_id, on_date="2026-09-28",
            note="复检发现次生病害", before_hashes=["e" * 64], after_hashes=["f" * 64],
        )
        self.registry.incidents.append(second)

        first_result = self.registry.resolve_incident(
            self.incident_id, "第一条损伤双方复核通过", reviews=self.reviews)
        self.assertTrue(first_result["frozen"])
        self.assertEqual(first_result["open_incidents"], 1)
        self.assertTrue(self.registry.get_work_view(self.work_id)["frozen"])
        with self.assertRaises(ConflictError):
            self.registry.record_handover(
                handover_payload(self.work_id, "布展", "SCAN-BLOCKED", "2026-09-30"))

        second_result = self.registry.resolve_incident(
            second.incident_id, "第二条损伤双方复核通过", reviews=self.reviews)
        self.assertFalse(second_result["frozen"])
        self.assertEqual(second_result["open_incidents"], 0)
        resumed = self.registry.record_handover(
            handover_payload(self.work_id, "布展", "SCAN-3", "2026-09-30"))
        self.assertEqual(resumed["type"], "布展")


# ---------------------------------------------------------------------------
# HTTP 层：真实多线程服务器上的竞争与状态码稳定性
# ---------------------------------------------------------------------------


def http_post(base_url, path, payload):
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return (response.status, json.load(response),
                    response.headers.get("Idempotency-Replayed"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8")), None


def http_get(base_url, path):
    with urlopen(f"{base_url}{path}", timeout=5) as response:
        return response.status, json.load(response)


def register_work_http(base_url, title):
    status, body, _ = http_post(base_url, "/works", {
        "title": title, "kind": "独立作品", "owner_org": "甲馆"})
    assert status == 200
    return body["work"]["work_id"]


def http_handover(work_id, htype, scan, day="2026-09-25", report=None,
                  location="甲馆库房", linked_segments=None):
    pairs = {
        "出库": ("出借馆", "运输方", "甲馆", "长风运输"),
        "到馆": ("运输方", "承借馆", "长风运输", "乙馆"),
        "布展": ("承借馆", "承借馆", "乙馆", "乙馆"),
        "撤展": ("承借馆", "运输方", "乙馆", "长风运输"),
        "归还": ("运输方", "出借馆", "长风运输", "甲馆"),
    }
    fr, tr, forg, torg = pairs[htype]
    return {
        "work_id": work_id, "type": htype, "scan_code": scan,
        "on_date": day, "at_location": location,
        "from_party": {"org": forg, "role": fr, "person": "甲"},
        "to_party": {"org": torg, "role": tr, "person": "乙"},
        "report": report or {"condition": "良好", "image_hashes": ["照"]},
        "linked_segments": linked_segments or [],
    }


class HttpConcurrencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from service import Handler
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_concurrent_identical_posts_one_create_rest_replays(self):
        work_id = register_work_http(self.base_url, "HTTP 并发出库卷")
        payload = http_handover(work_id, "出库", f"NET-RACE-{work_id}")
        results = run_together(
            [lambda: http_post(self.base_url, "/handovers", payload) for _ in range(6)])

        statuses = [r[1][0] for r in results]
        self.assertTrue(all(code == 200 for code in statuses))
        bodies = [r[1][1] for r in results]
        self.assertEqual(len({b["handover_id"] for b in bodies}), 1)
        # 恰好一条首交（无 replay 标记），其余都是重放。
        self.assertEqual(sum(1 for b in bodies if not b.get("replay")), 1)
        replays = [r for r in results if r[1][2] == "true"]
        self.assertEqual(len(replays), 5)
        self.assertTrue(all(b.get("replay_of") == bodies[0]["handover_id"]
                            for b in bodies if b.get("replay")))

        _, work_view = http_get(self.base_url, f"/works/{work_id}")
        chain = work_view["handover_chain"]
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["scan_code"], payload["scan_code"])
        self.assertEqual(work_view["custody"]["custodian_role"], "运输方")

    def test_http_status_kinds_stable_between_replay_conflict_field_error(self):
        work_id = register_work_http(self.base_url, "HTTP 三类状态卷")
        first_payload = http_handover(work_id, "出库", f"KIND-{work_id}")
        code, first, replay_header = http_post(self.base_url, "/handovers", first_payload)
        self.assertEqual((code, replay_header), (200, None))

        # 完全相同重试：200 重放。
        code, body, replay_header = http_post(self.base_url, "/handovers", first_payload)
        self.assertEqual(code, 200)
        self.assertEqual(replay_header, "true")
        self.assertTrue(body["replay"])
        self.assertEqual(body["handover_id"], first["handover_id"])

        # 同扫码不同日期：409 scan_conflict。
        clash = dict(first_payload)
        clash["on_date"] = "2026-09-26"
        code, body, _ = http_post(self.base_url, "/handovers", clash)
        self.assertEqual(code, 409)
        self.assertEqual(body["kind"], "scan_conflict")

        # 跳步：409 conflict（另一类状态冲突）。
        code, body, _ = http_post(
            self.base_url, "/handovers", http_handover(work_id, "布展", f"JUMP-{work_id}"))
        self.assertEqual(code, 409)
        self.assertEqual(body["kind"], "conflict")

        # 字段错误：400 invalid_request；该扫码未被消费。
        bad = http_handover(work_id, "到馆", f"FIELD-{work_id}", day="2026-09-27",
                            location="乙馆收货区")
        bad["to_party"]["person"] = ""
        code, body, _ = http_post(self.base_url, "/handovers", bad)
        self.assertEqual(code, 400)
        self.assertEqual(body["kind"], "invalid_request")
        code, retried, _ = http_post(
            self.base_url, "/handovers",
            http_handover(work_id, "到馆", f"FIELD-{work_id}", day="2026-09-27",
                          location="乙馆收货区"))
        self.assertEqual(code, 200)
        self.assertFalse(retried.get("replay"))

        _, work_view = http_get(self.base_url, f"/works/{work_id}")
        self.assertEqual([c["type"] for c in work_view["handover_chain"]], ["出库", "到馆"])

    def test_http_damage_race_freeze_precedes_next_transfer(self):
        for iteration in range(6):
            work_id = register_work_http(self.base_url, f"HTTP 损伤竞态卷 {iteration}")
            http_post(self.base_url, "/handovers",
                      http_handover(work_id, "出库", f"DO-{iteration}-{work_id}"))
            damage = http_handover(
                work_id, "到馆", f"DD-{iteration}", day="2026-09-27",
                location="乙馆收货区",
                report={"condition": "损伤", "damage_note": "新增折痕",
                        "before_hashes": ["a" * 64], "after_hashes": ["b" * 64]})
            good = http_handover(
                work_id, "到馆", f"DG-{iteration}", day="2026-09-27",
                location="乙馆收货区")
            run_together([
                lambda: http_post(self.base_url, "/handovers", damage),
                lambda: http_post(self.base_url, "/handovers", good),
            ])

            _, risk = http_get(self.base_url, f"/works/{work_id}/risk")
            _, work_view = http_get(self.base_url, f"/works/{work_id}")
            chain = work_view["handover_chain"]
            self.assertEqual([c["type"] for c in chain], ["出库", "到馆"])
            if risk["frozen"]:
                self.assertEqual(len(risk["open_risks"]), 1)
                code, body, _ = http_post(
                    self.base_url, "/handovers",
                    http_handover(work_id, "布展", f"DN-{iteration}", day="2026-09-30"))
                self.assertEqual(code, 409)
                self.assertEqual(body["kind"], "conflict")
            else:
                self.assertEqual(risk["open_risks"], [])
                code, _, _ = http_post(
                    self.base_url, "/handovers",
                    http_handover(work_id, "布展", f"DN-{iteration}", day="2026-09-30"))
                self.assertEqual(code, 200)

    def test_http_resolve_requires_both_reviews_and_single_terminal_flow(self):
        work_id = register_work_http(self.base_url, "HTTP 复核解除卷")
        http_post(self.base_url, "/handovers",
                  http_handover(work_id, "出库", f"R1-{work_id}"))
        _, damaged, _ = http_post(self.base_url, "/handovers", http_handover(
            work_id, "到馆", f"R2-{work_id}", day="2026-09-27",
            location="乙馆收货区",
            report={"condition": "损伤", "damage_note": "折痕",
                    "before_hashes": ["a" * 64], "after_hashes": ["b" * 64]}))
        incident_id = damaged["incident_id"]

        resolve = {"resolution_note": "双方复核通过", "reviews": [
            {"party": "交出方", "role": "运输方", "person": "押运员", "conclusion": "无碰撞"},
            {"party": "接收方", "role": "承借馆", "person": "库管员", "conclusion": "可展出"},
        ]}
        code, body, _ = http_post(
            self.base_url, f"/incidents/{incident_id}/resolve",
            {"resolution_note": "有结论"})
        self.assertEqual(code, 400)
        self.assertEqual(body["kind"], "invalid_request")
        code, _, _ = http_post(
            self.base_url, f"/incidents/{incident_id}/resolve",
            {**resolve, "reviews": resolve["reviews"][:1]})
        self.assertEqual(code, 400)
        code, result, _ = http_post(
            self.base_url, f"/incidents/{incident_id}/resolve", resolve)
        self.assertEqual(code, 200)
        self.assertFalse(result["frozen"])

        # 单终端完整流程继续可用；每步之后用完全相同负载重试，必须取回同一条。
        chain = [
            ("布展", f"R3-{work_id}", "2026-09-30", "乙馆三号厅"),
            ("撤展", f"R4-{work_id}", "2027-01-05", "乙馆三号厅"),
            ("归还", f"R5-{work_id}", "2027-01-07", "甲馆库房"),
        ]
        for htype, scan, day, location in chain:
            payload = http_handover(work_id, htype, scan, day=day, location=location)
            code, created, _ = http_post(self.base_url, "/handovers", payload)
            self.assertEqual(code, 200)
            code, retried, header = http_post(self.base_url, "/handovers", payload)
            self.assertEqual(code, 200)
            self.assertEqual(header, "true")
            self.assertEqual(retried["handover_id"], created["handover_id"])

        _, work_view = http_get(self.base_url, f"/works/{work_id}")
        self.assertEqual([c["type"] for c in work_view["handover_chain"]],
                         ["出库", "到馆", "布展", "撤展", "归还"])
        self.assertEqual(work_view["custody"]["status"], "已归还")


if __name__ == "__main__":
    unittest.main()
