"""并发一致性契约：两台扫码终端几乎同时提交时，交接链仍只有一条。

覆盖：
- 同作品同扫码号、内容完全一致的并发重试：只成立一次，其余幂等返回首次交接；
- 同扫码号但签认/日期/状态/关联区段不同：一方成功，其余 409 冲突，不产生第二条链；
- 校验失败（字段错误/跳步/冻结）不消耗扫码号，回滚后该码仍可使用；
- 损伤上报与下一步交接竞态：冻结必先于保管权转移，布展不可能插入；
- 解除风险须出借馆与承借馆双方书面复核，且所有未结损伤结清后才解冻；
- HTTP 层稳定区分重放（200 + replayed）、冲突（409）与字段错误（400）；
- 竞争结束后通过查询视图核对交接链唯一、保管方唯一、风险事件数量正确。
"""

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from domain import (
    ConflictError,
    DomainError,
    Incident,
    LoanRegistry,
    ReplayedHandover,
)
from domain_contract import agreement_payload, handover_payload
from service import Handler


def new_work_with_agreement(registry: LoanRegistry, title: str = "并发测试卷") -> str:
    work = registry.register_work(title, "独立作品", "甲馆")
    work_id = work["work"]["work_id"]
    registry.create_agreement(agreement_payload(work_id))
    return work_id


def run_concurrently(callables, *, rounds: int = 1):
    """让所有任务在同一屏障前同时起跑；返回每轮每个任务的 (结果, 异常) 列表。"""
    all_rounds = []
    for _ in range(rounds):
        n = len(callables)
        barrier = threading.Barrier(n)
        outcomes = [None] * n

        def worker(index, fn):
            barrier.wait()
            try:
                outcomes[index] = (fn(), None)
            except BaseException as error:  # 捕获后由主线程断言，避免线程内静默
                outcomes[index] = (None, error)

        threads = [
            threading.Thread(target=worker, args=(i, fn))
            for i, fn in enumerate(callables)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(outcome is not None for outcome in outcomes), "存在并发任务未结束"
        all_rounds.append(outcomes)
    return all_rounds


class DomainConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.registry = LoanRegistry()

    def test_identical_concurrent_submissions_create_exactly_one_handover(self):
        for _ in range(20):
            registry = LoanRegistry()
            work_id = new_work_with_agreement(registry)
            payload = handover_payload(work_id, "出库", "SCAN-SAME", "2026-09-25")
            terminals = 8

            def submit():
                return registry.record_handover(json.loads(json.dumps(payload)))

            rounds = run_concurrently([submit] * terminals)
            outcomes = rounds[0]
            successes = [value for value, error in outcomes if error is None]
            replays = [error for _value, error in outcomes if isinstance(error, ReplayedHandover)]

            # 一台终端首次办理成功，其余全部识别为同一请求的重放。
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(replays), terminals - 1)

            first_id = successes[0]["handover_id"]
            self.assertTrue(all(error.view["handover_id"] == first_id for error in replays))

            chain = [h for h in registry.handovers if h.work_id == work_id]
            self.assertEqual(len(chain), 1)
            self.assertEqual(chain[0].handover_id, first_id)
            custody = registry.get_work_view(work_id)["custody"]
            self.assertEqual(custody["status"], "运输中")
            self.assertEqual(custody["since_handover"], first_id)

    def test_same_scan_with_different_fields_one_wins_rest_conflict(self):
        for mutation, label in [
            (lambda p: p.update(on_date="2026-09-26"), "交接日期"),
            (lambda p: p["to_party"].update(person="另一位接收员"), "接收方签认"),
            (lambda p: p["from_party"].update(person="另一位出库员"), "交出方签认"),
        ]:
            with self.subTest(变异维度=label):
                registry = LoanRegistry()
                work_id = new_work_with_agreement(registry)
                base = handover_payload(work_id, "出库", "SCAN-DIFF", "2026-09-25")
                mutated = json.loads(json.dumps(base))
                mutation(mutated)

                outcomes = run_concurrently(
                    [
                        lambda: registry.record_handover(json.loads(json.dumps(base))),
                        lambda: registry.record_handover(json.loads(json.dumps(mutated))),
                    ]
                )[0]
                conflicts = [
                    error for _value, error in outcomes if isinstance(error, ConflictError)
                ]
                successes = [value for value, error in outcomes if error is None]
                self.assertEqual(len(successes), 1)
                self.assertEqual(len(conflicts), 1)
                self.assertEqual(conflicts[0].code, "scan_conflict")
                self.assertIn(label, str(conflicts[0]))
                # 冲突落败方没有产生第二条交接，也没有改动保管方。
                self.assertEqual(
                    [h for h in registry.handovers if h.work_id == work_id],
                    [registry.handovers[-1]],
                )

    def test_same_scan_different_condition_or_segments_conflicts(self):
        # 登记一件带区段的作品，用于核对“关联区段不同也必须冲突”。
        view = self.registry.register_work(
            "带区段卷", "长卷", "甲馆",
            segments=[
                {"segment_id": "SEG-A", "label": "引首", "start_cm": 0, "end_cm": 80},
                {"segment_id": "SEG-B", "label": "画心", "start_cm": 80, "end_cm": 360},
            ],
        )
        work_id = view["work"]["work_id"]
        self.registry.create_agreement(agreement_payload(work_id))

        first_payload = handover_payload(
            work_id, "出库", "SCAN-COND", "2026-09-25",
            linked_segments=["SEG-A"],
            report={"condition": "良好", "image_hashes": ["出库全景照"]},
        )
        first_view = self.registry.record_handover(first_payload)

        # 与首次完全一致的请求仍是重放。
        with self.assertRaises(ReplayedHandover):
            self.registry.record_handover(handover_payload(
                work_id, "出库", "SCAN-COND", "2026-09-25", linked_segments=["SEG-A"],
                report={"condition": "良好", "image_hashes": ["出库全景照"]},
            ))

        # 同扫码号但状态结论不同 → 冲突，不能借“重试”把良好改写成损伤。
        with self.assertRaises(ConflictError) as condition_error:
            self.registry.record_handover(handover_payload(
                work_id, "出库", "SCAN-COND", "2026-09-25", linked_segments=["SEG-A"],
                report={"condition": "损伤", "damage_note": "重试里捏造的损伤", "image_hashes": ["x"]},
            ))
        self.assertEqual(condition_error.exception.code, "scan_conflict")
        self.assertIn("状态报告", str(condition_error.exception))

        # 同扫码号但关联区段不同 → 同样冲突。
        with self.assertRaises(ConflictError) as segment_error:
            self.registry.record_handover(handover_payload(
                work_id, "出库", "SCAN-COND", "2026-09-25", linked_segments=["SEG-B"],
                report={"condition": "良好", "image_hashes": ["出库全景照"]},
            ))
        self.assertEqual(segment_error.exception.code, "scan_conflict")
        self.assertIn("关联区段", str(segment_error.exception))

        # 首次交接的状态仍是“良好”，关联区段仍是 SEG-A，没有被冲突请求污染。
        stored = self.registry.handovers[-1]
        self.assertEqual(stored.handover_id, first_view["handover_id"])
        self.assertEqual(
            self.registry._handover_view(stored)["condition"]["condition"], "良好")
        self.assertEqual(stored.linked_segments, ["SEG-A"])

    def test_failed_validation_does_not_consume_scan_under_contention(self):
        registry = self.registry
        work_id = new_work_with_agreement(registry)
        registry.record_handover(
            handover_payload(work_id, "出库", "SCAN-1", "2026-09-25"))

        # 多个终端同时用各自的新扫码跳步办理“布展/归还/撤展”，全部失败，扫码均不得被占用。
        bad_specs = [("布展", "SCAN-BAD-1"), ("归还", "SCAN-BAD-2"), ("撤展", "SCAN-BAD-3")]
        jobs = [
            (lambda htype=htype, scan=scan: registry.record_handover(
                handover_payload(work_id, htype, scan, "2026-09-26")))
            for htype, scan in bad_specs
        ]
        outcomes = run_concurrently(jobs)[0]
        self.assertEqual(sum(1 for _v, e in outcomes if isinstance(e, ConflictError)), 3)
        self.assertEqual(
            [scan for scan in ("SCAN-BAD-1", "SCAN-BAD-2", "SCAN-BAD-3")
             if (work_id, scan) in registry._scan_index],
            [],
        )

        # 失败回滚后生命周期仍停在出库之后；这些码稍后在正确步骤上仍可用。
        registry.record_handover(
            handover_payload(work_id, "到馆", "SCAN-BAD-1", "2026-09-27"))
        registry.record_handover(
            handover_payload(work_id, "布展", "SCAN-2", "2026-09-30"))
        registry.record_handover(
            handover_payload(work_id, "撤展", "SCAN-BAD-2", "2027-01-05"))
        done = registry.record_handover(
            handover_payload(work_id, "归还", "SCAN-BAD-3", "2027-01-07"))
        self.assertEqual(done["type"], "归还")
        self.assertEqual(
            [h.type for h in registry.handovers if h.work_id == work_id],
            ["出库", "到馆", "布展", "撤展", "归还"],
        )

    def test_damage_and_next_handover_race_freeze_always_precedes_transfer(self):
        for _ in range(20):
            registry = LoanRegistry()
            work_id = new_work_with_agreement(registry)
            registry.record_handover(
                handover_payload(work_id, "出库", "SCAN-1", "2026-09-25"))

            damage_report = {
                "condition": "损伤",
                "damage_note": "画心左下角新增折痕",
                "image_hashes": ["到馆检视照"],
                "before_hashes": ["a" * 64],
                "after_hashes": ["b" * 64],
            }

            def report_damage():
                return registry.record_handover(
                    handover_payload(work_id, "到馆", "SCAN-2", "2026-09-27",
                                     report=dict(damage_report)))

            def install_exhibit():
                return registry.record_handover(
                    handover_payload(work_id, "布展", "SCAN-3", "2026-09-30"))

            outcomes = run_concurrently([report_damage, install_exhibit])[0]
            damage_views = [v for v, e in outcomes if e is None and v.get("frozen")]
            install_errors = [
                e for _v, e in outcomes
                if isinstance(e, ConflictError) and e.code in ("work_frozen", "conflict")
            ]

            # 无论锁先判给谁：损伤到馆必然成立并冻结，布展必然失败。
            self.assertEqual(len(damage_views), 1)
            self.assertEqual(len(install_errors), 1)

            chain = [h for h in registry.handovers if h.work_id == work_id]
            self.assertEqual([h.type for h in chain], ["出库", "到馆"])
            self.assertTrue(registry.get_work_view(work_id)["frozen"])
            risk = registry.risk_view(work_id)
            self.assertEqual(len(risk["open_risks"]), 1)
            # 保管权停在运输方→承借馆的到馆点，绝没有跳到“展出中”。
            self.assertEqual(risk["custody"]["status"], "待布展")

            # 冻结后再办布展仍被拒；同一损伤交接的完全相同重试则按重放返回。
            with self.assertRaises(ConflictError):
                registry.record_handover(
                    handover_payload(work_id, "布展", "SCAN-4", "2026-10-01"))
            with self.assertRaises(ReplayedHandover) as replay:
                registry.record_handover(
                    handover_payload(work_id, "到馆", "SCAN-2", "2026-09-27",
                                     report=dict(damage_report)))
            self.assertEqual(replay.exception.view["handover_id"], chain[-1].handover_id)

    def test_resolve_requires_both_party_reviews(self):
        registry = self.registry
        work_id = new_work_with_agreement(registry)
        registry.record_handover(
            handover_payload(work_id, "出库", "SCAN-1", "2026-09-25"))
        registry.record_handover(handover_payload(
            work_id, "到馆", "SCAN-2", "2026-09-27",
            report={"condition": "损伤", "damage_note": "折痕",
                    "before_hashes": ["a" * 64], "after_hashes": ["b" * 64]},
        ))
        incident_id = registry.risk_view(work_id)["open_risks"][0]["incident_id"]

        lender = {"org": "甲馆", "person": "修复师甲", "note": "出借馆复核可继续", "on_date": "2026-09-28"}
        borrower = {"org": "乙馆", "person": "馆员乙", "note": "承借馆复核可继续", "on_date": "2026-09-28"}

        # 缺结论 / 缺任一方 / 结论缺签认人都属于字段错误。
        with self.assertRaises(DomainError):
            registry.resolve_incident(incident_id, {"resolution_note": "已修复"})
        with self.assertRaises(DomainError):
            registry.resolve_incident(
                incident_id, {"resolution_note": "已修复", "lender_review": lender})
        with self.assertRaises(DomainError):
            registry.resolve_incident(incident_id, {
                "resolution_note": "已修复",
                "lender_review": dict(lender, person=""),
                "borrower_review": borrower,
            })
        self.assertTrue(registry.get_work_view(work_id)["frozen"])

        result = registry.resolve_incident(incident_id, {
            "resolution_note": "双方书面复核，确认可继续",
            "lender_review": lender,
            "borrower_review": borrower,
        })
        self.assertFalse(result["frozen"])
        self.assertEqual(result["remaining_open_incidents"], [])
        self.assertFalse(registry.get_work_view(work_id)["frozen"])

        # 已解除的事件不能重复复核。
        with self.assertRaises(ConflictError) as error:
            registry.resolve_incident(incident_id, {
                "resolution_note": "再次确认",
                "lender_review": lender, "borrower_review": borrower,
            })
        self.assertEqual(error.exception.code, "incident_already_resolved")

    def test_freeze_lifts_only_after_all_open_incidents_have_dual_reviews(self):
        registry = self.registry
        work_id = new_work_with_agreement(registry)
        registry.record_handover(
            handover_payload(work_id, "出库", "SCAN-1", "2026-09-25"))
        registry.record_handover(handover_payload(
            work_id, "到馆", "SCAN-2", "2026-09-27",
            report={"condition": "损伤", "damage_note": "损伤一",
                    "before_hashes": ["a" * 64], "after_hashes": ["b" * 64]},
        ))
        first_id = registry.risk_view(work_id)["open_risks"][0]["incident_id"]

        # 白箱构造：同一作品挂着第二起未结损伤（例如巡检另报），
        # 只解除其中一起时冻结必须继续有效。
        second = Incident(
            incident_id="incident-extra",
            work_id=work_id,
            handover_id=registry.handovers[-1].handover_id,
            on_date="2026-09-28",
            note="损伤二",
            before_hashes=["c" * 64],
            after_hashes=["d" * 64],
        )
        registry.incidents.append(second)

        lender = {"org": "甲馆", "person": "修复师甲", "note": "同意", "on_date": "2026-09-29"}
        borrower = {"org": "乙馆", "person": "馆员乙", "note": "同意", "on_date": "2026-09-29"}
        partial = registry.resolve_incident(first_id, {
            "resolution_note": "第一起双方复核完成",
            "lender_review": lender, "borrower_review": borrower,
        })
        self.assertTrue(partial["frozen"])
        self.assertEqual(partial["remaining_open_incidents"], [second.incident_id])
        self.assertTrue(registry.get_work_view(work_id)["frozen"])

        with self.assertRaises(ConflictError):
            registry.record_handover(
                handover_payload(work_id, "布展", "SCAN-3", "2026-09-30"))

        final = registry.resolve_incident(second.incident_id, {
            "resolution_note": "第二起双方复核完成",
            "lender_review": lender, "borrower_review": borrower,
        })
        self.assertFalse(final["frozen"])
        self.assertFalse(registry.get_work_view(work_id)["frozen"])
        registry.record_handover(
            handover_payload(work_id, "布展", "SCAN-3", "2026-09-30"))


def http_post(base_url, path, payload):
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8")), error.headers
    with response:
        return response.status, json.load(response), response.headers


class HttpConcurrencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _register_work(self, title):
        status, body, _ = http_post(self.base_url, "/works", {
            "title": title, "kind": "独立作品", "owner_org": "甲馆",
        })
        self.assertEqual(status, 200)
        return body["work"]["work_id"]

    def test_http_identical_concurrent_posts_return_first_handover_once(self):
        work_id = self._register_work("HTTP 并发重放卷")
        payload = {
            "work_id": work_id, "type": "出库", "scan_code": "HTTP-SAME",
            "on_date": "2026-09-25", "at_location": "甲馆库房",
            "from_party": {"org": "甲馆", "role": "出借馆", "person": "甲"},
            "to_party": {"org": "长风运输", "role": "运输方", "person": "乙"},
            "report": {"condition": "良好", "image_hashes": ["ok"]},
        }
        barrier = threading.Barrier(2)
        responses = [None, None]

        def post(index):
            barrier.wait()
            responses[index] = http_post(
                self.base_url, "/handovers", json.loads(json.dumps(payload)))

        threads = [threading.Thread(target=post, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertTrue(all(r[0] == 200 for r in responses))
        ids = {r[1]["handover_id"] for r in responses}
        self.assertEqual(len(ids), 1)
        replayed_flags = sorted(bool(r[1].get("replayed")) for r in responses)
        self.assertEqual(replayed_flags, [False, True])
        replay_headers = [r[2].get("Idempotency-Replayed") for r in responses]
        self.assertIn("true", replay_headers)

        with urlopen(f"{self.base_url}/works/{work_id}", timeout=5) as response:
            view = json.load(response)
        self.assertEqual(view["custody"]["status"], "运输中")
        self.assertFalse(view["frozen"])

        # 网络恢复后的再次重试依旧返回同一条首次交接。
        status, body, headers = http_post(
            self.base_url, "/handovers", json.loads(json.dumps(payload)))
        self.assertEqual(status, 200)
        self.assertTrue(body["replayed"])
        self.assertEqual(body["handover_id"], next(iter(ids)))
        self.assertEqual(headers.get("Idempotency-Replayed"), "true")

    def test_http_same_scan_different_date_is_conflict(self):
        work_id = self._register_work("HTTP 并发冲突卷")
        base = {
            "work_id": work_id, "type": "出库", "scan_code": "HTTP-DIFF",
            "on_date": "2026-09-25",
            "from_party": {"org": "甲馆", "role": "出借馆", "person": "甲"},
            "to_party": {"org": "长风运输", "role": "运输方", "person": "乙"},
            "report": {"condition": "良好", "image_hashes": ["ok"]},
        }
        changed = json.loads(json.dumps(base))
        changed["on_date"] = "2026-09-26"
        barrier = threading.Barrier(2)
        responses = [None, None]

        def post(index, payload):
            barrier.wait()
            responses[index] = http_post(self.base_url, "/handovers", payload)

        threads = [
            threading.Thread(target=post, args=(0, base)),
            threading.Thread(target=post, args=(1, changed)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        statuses = sorted(r[0] for r in responses)
        self.assertEqual(statuses, [200, 409])
        error_body = next(r[1] for r in responses if r[0] == 409)
        self.assertEqual(error_body["code"], "scan_conflict")
        self.assertIn("交接日期", error_body["error"])

    def test_http_field_errors_are_400_with_stable_code(self):
        work_id = self._register_work("HTTP 字段错误卷")
        bad_role = {
            "work_id": work_id, "type": "出库", "scan_code": "HTTP-BAD",
            "on_date": "2026-09-25",
            "from_party": {"org": "甲馆", "role": "出借馆", "person": "甲"},
            "to_party": {"org": "长风运输", "role": "承借馆", "person": "乙"},
            "report": {"condition": "良好", "image_hashes": ["ok"]},
        }
        status, body, _ = http_post(self.base_url, "/handovers", bad_role)
        self.assertEqual(status, 400)
        self.assertEqual(body["code"], "invalid_request")

        bad_date = dict(bad_role)
        bad_date["scan_code"] = "HTTP-BAD2"
        bad_date["to_party"] = {"org": "长风运输", "role": "运输方", "person": "乙"}
        bad_date["on_date"] = "not-a-date"
        status, body, _ = http_post(self.base_url, "/handovers", bad_date)
        self.assertEqual(status, 400)
        self.assertEqual(body["code"], "invalid_request")

        # 字段错误不消费扫码号：修正后仍可用同一扫码首次办理。
        fixed = dict(bad_role)
        fixed["to_party"] = {"org": "长风运输", "role": "运输方", "person": "乙"}
        status, body, _ = http_post(self.base_url, "/handovers", fixed)
        self.assertEqual(status, 200)
        self.assertEqual(body["scan_code"], "HTTP-BAD")

    def test_http_damage_race_leaves_single_chain_and_freeze(self):
        work_id = self._register_work("HTTP 损伤竞态卷")
        http_post(self.base_url, "/handovers", {
            "work_id": work_id, "type": "出库", "scan_code": "HTTP-R1",
            "on_date": "2026-09-25",
            "from_party": {"org": "甲馆", "role": "出借馆", "person": "甲"},
            "to_party": {"org": "长风运输", "role": "运输方", "person": "乙"},
            "report": {"condition": "良好", "image_hashes": ["ok"]},
        })
        damage = {
            "work_id": work_id, "type": "到馆", "scan_code": "HTTP-R2",
            "on_date": "2026-09-27",
            "from_party": {"org": "长风运输", "role": "运输方", "person": "乙"},
            "to_party": {"org": "乙馆", "role": "承借馆", "person": "丙"},
            "report": {"condition": "损伤", "damage_note": "新增折痕",
                       "before_hashes": ["a" * 64], "after_hashes": ["b" * 64]},
        }
        install = {
            "work_id": work_id, "type": "布展", "scan_code": "HTTP-R3",
            "on_date": "2026-09-30",
            "from_party": {"org": "乙馆", "role": "承借馆", "person": "丙"},
            "to_party": {"org": "乙馆", "role": "承借馆", "person": "丁"},
            "report": {"condition": "良好", "image_hashes": ["ok"]},
        }
        barrier = threading.Barrier(2)
        responses = [None, None]

        def post(index, payload):
            barrier.wait()
            responses[index] = http_post(self.base_url, "/handovers", payload)

        threads = [
            threading.Thread(target=post, args=(0, damage)),
            threading.Thread(target=post, args=(1, install)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        statuses = sorted(r[0] for r in responses)
        self.assertEqual(statuses, [200, 409])
        damage_response = next(r for r in responses if r[0] == 200)[1]
        self.assertTrue(damage_response["frozen"])

        with urlopen(f"{self.base_url}/works/{work_id}/risk", timeout=5) as response:
            risk = json.load(response)
        self.assertTrue(risk["frozen"])
        self.assertEqual(len(risk["open_risks"]), 1)
        self.assertEqual(risk["custody"]["status"], "待布展")

        # 双方复核解除后链路才能继续。
        incident_id = risk["open_risks"][0]["incident_id"]
        review = {"org": "馆方", "person": "复核人", "note": "确认可继续", "on_date": "2026-09-28"}
        status, body, _ = http_post(self.base_url, f"/incidents/{incident_id}/resolve", {
            "resolution_note": "双方复核完成",
            "lender_review": review,
            "borrower_review": dict(review, org="乙馆", person="馆员乙"),
        })
        self.assertEqual(status, 200)
        self.assertFalse(body["frozen"])
        status, _, _ = http_post(self.base_url, "/handovers", install)
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
