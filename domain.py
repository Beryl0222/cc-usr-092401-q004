"""馆际作品借展的领域核心。

只依赖标准库，集中表达四类业务规则：

1. 复合作品结构：实体作品 → 组成区段 → 作者贡献（含合作顺序、题跋）。
2. 借展协议：约束展期、展厅、照度、运输、保险与数字传播用途。
3. 状态交接：出库/到馆/布展/撤展/归还由交接双方签认；重复扫码幂等；
   发现损伤立即冻结后续动作并保全前后图像哈希。
4. 策展展签：发布日期确认后锁定证据快照，后来的学术更正只产生新版，
   不改变旧版展签；任意版本都可从展签定位到实体、贡献区段、当前保管
   责任、授权范围与未解除风险。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from functools import wraps
from typing import Any, Callable, Iterator, Optional

# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------

ROLES = ("出借馆", "承借馆", "运输方", "策展人")

WORK_KINDS = ("独立作品", "合作画", "历史图录", "长卷")

CONTRIBUTION_KINDS = ("作画", "题跋", "书引首", "鉴藏印", "题签")

# 交接类型固定，顺序即借展生命周期
HANDOVER_TYPES = ("出库", "到馆", "布展", "撤展", "归还")

# 每次交接后实体所处的保管状态
STATUS_AFTER = {
    "出库": "运输中",
    "到馆": "待布展",
    "布展": "展出中",
    "撤展": "待归还",
    "归还": "已归还",
}

# 各交接类型的法定交出方 / 接收方角色
TRANSFER_PAIRS = {
    "出库": ("出借馆", "运输方"),
    "到馆": ("运输方", "承借馆"),
    "布展": ("承借馆", "承借馆"),
    "撤展": ("承借馆", "运输方"),
    "归还": ("运输方", "出借馆"),
}

LABEL_STATUS = ("草拟", "已发布", "已更正")

# 指纹各维度在冲突提示中的中文名（仅用于错误信息）。
_FINGERPRINT_FIELD_LABELS = {
    "type": "生命周期位置",
    "from_party": "交出方签认",
    "to_party": "接收方签认",
    "on_date": "交接日期",
    "at_location": "交接地点",
    "linked_segments": "关联区段",
    "report": "状态报告",
}


class DomainError(ValueError):
    """请求违反领域规则（作为 4xx 返回给调用方）。

    ``code`` 是稳定的机器可读错误码，HTTP 层据此区分字段错误、冲突与重放。
    """

    code = "invalid_request"

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConflictError(DomainError):
    """状态冲突（重复交接、已冻结、已锁定等），语义上是 409。"""

    code = "conflict"


class ReplayedHandover(ConflictError):
    """同一作品、同一扫码号且内容完全一致的重试：不产生新记录，返回首次交接。"""

    code = "replayed_handover"

    def __init__(self, handover: "Handover", view: dict[str, Any]) -> None:
        super().__init__(
            f"扫码 {handover.scan_code} 与交接 {handover.handover_id} 内容一致，按重试返回首次交接",
        )
        self.handover = handover
        self.view = view


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


def _new_id(prefix: str) -> str:
    _new_id.counter[prefix] = _new_id.counter.get(prefix, 0) + 1
    return f"{prefix}-{_new_id.counter[prefix]:04d}"


_new_id.counter = {}  # type: ignore[attr-defined]


@dataclass
class Work:
    """作品实体：一件可以被独立出借、运输、投保的物理对象。"""

    title: str
    kind: str
    owner_org: str  # 当前权属机构
    work_id: str = field(default_factory=lambda: _new_id("work"))
    segments: list["Segment"] = field(default_factory=list)
    contributions: list["Contribution"] = field(default_factory=list)

    def require_segment(self, segment_id: str) -> "Segment":
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise DomainError(f"区段 {segment_id} 不属于作品 {self.work_id}")

    def to_ref(self) -> dict[str, str]:
        return {"work_id": self.work_id, "title": self.title, "kind": self.kind}


@dataclass
class Segment:
    """组成区段：长卷/合作画上可独立辨认的物理局部。"""

    label: str
    start_cm: float = 0.0
    end_cm: float = 0.0
    segment_id: str = field(default_factory=lambda: _new_id("seg"))
    note: str = ""


@dataclass
class Contribution:
    """作者贡献：谁、以何种方式、按什么合作顺序、落在哪个区段。"""

    author: str
    kind: str
    order: int
    segment_id: Optional[str] = None
    contribution_id: str = field(default_factory=lambda: _new_id("ctrb"))


@dataclass
class Agreement:
    """借展协议：约束条件随协议保存，授权范围由这里派生。"""

    agreement_id: str
    work_id: str
    lender_org: str
    borrower_org: str
    start_on: str  # 展期起（YYYY-MM-DD）
    end_on: str  # 展期止
    gallery: str
    max_lux: int
    transport: dict[str, Any]
    insurance: dict[str, Any]
    digital_rights: dict[str, Any]
    version: int = 1
    supersedes: Optional[str] = None

    def authorization_scope(self) -> dict[str, Any]:
        """从协议条款派生当前授权范围，供展签与风险视图引用。"""
        return {
            "agreement_id": self.agreement_id,
            "version": self.version,
            "exhibition_period": {"start": self.start_on, "end": self.end_on},
            "gallery": self.gallery,
            "max_lux": self.max_lux,
            "transport": self.transport,
            "insurance": self.insurance,
            "digital_rights": self.digital_rights,
        }


@dataclass
class Signature:
    org: str
    role: str
    person: str


@dataclass
class ConditionReport:
    """状态报告：交接时由双方签认，发现损伤时带损伤说明与前后图像哈希。"""

    condition: str  # 良好 / 损伤
    image_hashes: list[str]
    damage_note: str = ""
    before_hashes: list[str] = field(default_factory=list)
    after_hashes: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Handover:
    handover_id: str
    work_id: str
    type: str
    scan_code: str
    from_party: Signature
    to_party: Signature
    report: ConditionReport
    on_date: str
    at_location: str
    linked_segments: list[str] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def damaged(self) -> bool:
        return self.report.condition == "损伤" or bool(self.report.damage_note)


@dataclass
class Incident:
    """损伤事件：冻结后所有未完成交接，图像哈希作为证据保全。

    解除冻结须同时具备出借馆与承借馆的书面复核结论。
    """

    incident_id: str
    work_id: str
    handover_id: str
    on_date: str
    note: str
    before_hashes: list[str]
    after_hashes: list[str]
    resolved: bool = False
    resolution_note: str = ""
    review_lender: Optional["ReviewConclusion"] = None
    review_borrower: Optional["ReviewConclusion"] = None


@dataclass
class ReviewConclusion:
    """某一方对损伤事件的书面复核结论。"""

    side: str  # lender / borrower
    org: str
    person: str
    note: str
    on_date: str


@dataclass
class LabelVersion:
    """展签的一个不可变版本。发布即锁定证据快照。"""

    version: int
    status: str
    narrative: str
    citations: list[dict[str, str]]
    evidence: dict[str, Any]
    published_on: Optional[str]
    frozen: bool


# ---------------------------------------------------------------------------
# 领域服务
# ---------------------------------------------------------------------------


class LoanRegistry:
    """保存全部借展记录并强制业务规则。

    对外使用命令式方法（register_work / record_handover / ...），
    查询通过 get_work_view / label_version / risk_view 等只读视图。
    """

    def __init__(self) -> None:
        self.works: dict[str, Work] = {}
        self.agreements: dict[str, Agreement] = {}
        self._agreement_history: dict[str, list[str]] = {}  # work_id → 协议ID（含历史版本）
        self.handovers: list[Handover] = []
        # (work_id, scan_code) → 已成立的交接；查重与写入共用同一把锁。
        self._scan_index: dict[tuple[str, str], Handover] = {}
        self.incidents: list[Incident] = []
        self._frozen_works: set[str] = set()
        self.labels: dict[str, list[LabelVersion]] = {}
        # 同一进程内所有写操作串行化，保证“查重 → 校验 → 写入 → 冻结/转移保管权”原子。
        self._lock = threading.RLock()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    @staticmethod
    def _synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
        """让整个方法在注册表锁内执行；RLock 可重入，方法间互相调用安全。"""

        @wraps(method)
        def wrapper(self: "LoanRegistry", *args: Any, **kwargs: Any) -> Any:
            with self._locked():
                return method(self, *args, **kwargs)

        return wrapper

    # -- 作品结构 ----------------------------------------------------------

    @_synchronized
    def register_work(
        self,
        title: str,
        kind: str,
        owner_org: str,
        segments: Optional[list[dict[str, Any]]] = None,
        contributions: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if not title or not title.strip():
            raise DomainError("作品名称不能为空")
        if kind not in WORK_KINDS:
            raise DomainError(f"作品类型须为 {WORK_KINDS} 之一")
        if not owner_org or not owner_org.strip():
            raise DomainError("必须登记当前权属机构")

        work = Work(title=title.strip(), kind=kind, owner_org=owner_org.strip())

        segment_ids: list[str] = []
        for raw in segments or []:
            segment_id = raw.get("segment_id") or _new_id("seg")
            if segment_id in segment_ids:
                raise DomainError(f"区段编号 {segment_id} 重复")
            segment = Segment(
                label=raw["label"],
                start_cm=float(raw.get("start_cm", 0.0)),
                end_cm=float(raw.get("end_cm", 0.0)),
                segment_id=segment_id,
                note=raw.get("note", ""),
            )
            if segment.end_cm < segment.start_cm:
                raise DomainError(f"区段 {segment.label} 的终点不能早于起点")
            work.segments.append(segment)
            segment_ids.append(segment.segment_id)

        orders: set[int] = set()
        for raw in contributions or []:
            order = int(raw["order"])
            if order <= 0:
                raise DomainError("合作顺序须从 1 开始")
            if order in orders:
                raise DomainError(f"合作顺序 {order} 重复")
            orders.add(order)
            contribution_kind = raw.get("kind", "作画")
            if contribution_kind not in CONTRIBUTION_KINDS:
                raise DomainError(f"贡献类型须为 {CONTRIBUTION_KINDS} 之一")
            segment_id = raw.get("segment_id")
            if segment_id is not None and segment_id not in segment_ids:
                raise DomainError(f"贡献指向不存在的区段 {segment_id}")
            work.contributions.append(
                Contribution(
                    author=raw["author"],
                    kind=contribution_kind,
                    order=order,
                    segment_id=segment_id,
                )
            )

        self.works[work.work_id] = work
        return self.get_work_view(work.work_id)

    # -- 协议 --------------------------------------------------------------

    @_synchronized
    def create_agreement(self, payload: dict[str, Any]) -> dict[str, Any]:
        work = self._work(payload["work_id"])
        agreement_id = payload.get("agreement_id") or f"AGR-{work.work_id}-v1"
        self._validate_agreement_id(agreement_id)
        agreement = self._build_agreement(agreement_id, work.work_id, payload, version=1)
        self.agreements[agreement_id] = agreement
        self._agreement_history.setdefault(work.work_id, []).append(agreement_id)
        return self._agreement_view(agreement)

    @_synchronized
    def reschedule_agreement(
        self, current_agreement_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        """跨馆改期：协议条款变更生成新版本，旧版保留，已发布展签不受影响。"""
        old = self.agreements.get(current_agreement_id)
        if old is None:
            raise DomainError(f"协议 {current_agreement_id} 不存在")
        work = self._work(old.work_id)
        if work.work_id in self._frozen_works:
            raise ConflictError(f"作品 {work.work_id} 已因损伤冻结，须先解除风险")

        merged: dict[str, Any] = {
            "lender_org": old.lender_org,
            "borrower_org": old.borrower_org,
            "start_on": old.start_on,
            "end_on": old.end_on,
            "gallery": old.gallery,
            "max_lux": old.max_lux,
            "transport": old.transport,
            "insurance": old.insurance,
            "digital_rights": old.digital_rights,
        }
        merged.update(changes)

        new_id = changes.get("agreement_id") or self._next_agreement_id(work.work_id, old.version + 1)
        self._validate_agreement_id(new_id)
        agreement = self._build_agreement(new_id, work.work_id, merged, version=old.version + 1)
        agreement.supersedes = current_agreement_id
        self.agreements[new_id] = agreement
        self._agreement_history.setdefault(work.work_id, []).append(new_id)
        return self._agreement_view(new_id)

    def _build_agreement(
        self, agreement_id: str, work_id: str, payload: dict[str, Any], version: int
    ) -> Agreement:
        start_on = self._date(payload["start_on"], "展期开始")
        end_on = self._date(payload["end_on"], "展期结束")
        if end_on < start_on:
            raise DomainError("展期结束日不能早于开始日")
        max_lux = int(payload["max_lux"])
        if max_lux <= 0:
            raise DomainError("照度上限须为正数（勒克斯）")
        agreement = Agreement(
            agreement_id=agreement_id,
            work_id=work_id,
            lender_org=payload["lender_org"],
            borrower_org=payload["borrower_org"],
            start_on=start_on.isoformat(),
            end_on=end_on.isoformat(),
            gallery=payload["gallery"],
            max_lux=max_lux,
            transport=dict(payload.get("transport") or {}),
            insurance=dict(payload.get("insurance") or {}),
            digital_rights=dict(payload.get("digital_rights") or {}),
            version=version,
        )
        return agreement

    def _validate_agreement_id(self, agreement_id: str) -> None:
        if agreement_id in self.agreements:
            raise ConflictError(f"协议编号 {agreement_id} 已存在")

    @staticmethod
    def _next_agreement_id(work_id: str, version: int) -> str:
        return f"AGR-{work_id}-v{version}"

    # -- 状态交接 ----------------------------------------------------------

    def record_handover(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._locked():
            return self._record_handover_locked(payload)

    def _record_handover_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        work = self._work(payload["work_id"])
        handover_type = payload["type"]
        if handover_type not in HANDOVER_TYPES:
            raise DomainError(f"交接类型须为 {HANDOVER_TYPES} 之一")

        scan_code = str(payload["scan_code"])
        if not scan_code.strip():
            raise DomainError("扫码标识不能为空")

        scan_key = (work.work_id, scan_code)
        previous = self._scan_index.get(scan_key)

        # 新扫码先做冻结状态校验（与单终端流程的状态码顺序一致）；
        # 已见过的扫码跳过状态校验，进入下面的幂等判定——
        # 这样作品已冻结/生命周期已前进时，完全相同的重试仍能返回首次交接。
        if previous is None and work.work_id in self._frozen_works:
            raise ConflictError(
                f"作品 {work.work_id} 已因损伤冻结，后续交接全部中止",
                code="work_frozen",
            )

        # 解析/校验请求字段（400 在此抛出），失败一律不触碰扫码索引与交接链。
        expected_from, expected_to = TRANSFER_PAIRS[handover_type]
        from_party = self._signature(payload["from_party"], expected_from)
        to_party = self._signature(payload["to_party"], expected_to)
        on_date = self._date(payload["on_date"], "交接日期").isoformat()
        at_location = str(payload.get("at_location", ""))
        linked_segments = list(payload.get("linked_segments") or [])
        for segment_id in linked_segments:
            work.require_segment(segment_id)
        report = self._condition_report(payload.get("report") or {})

        parts = self._fingerprint_parts(
            work.work_id, handover_type, from_party, to_party, report,
            on_date, at_location, linked_segments,
        )
        fingerprint = self._canonical_fingerprint(parts)

        # 幂等层：同一（作品，扫码号）见过的请求直接按首次结果处理。
        if previous is not None:
            if previous.fingerprint == fingerprint:
                raise ReplayedHandover(previous, self._handover_result_view(previous))
            differing = self._differing_fields(parts, self._stored_parts(previous))
            labels = [_FINGERPRINT_FIELD_LABELS.get(key, key) for key in differing]
            raise ConflictError(
                f"扫码 {scan_code} 已在交接 {previous.handover_id} 使用，重复扫码在"
                f"{'、'.join(labels)}上不一致，不能产生第二次交接",
                code="scan_conflict",
            )

        # 全新扫码才做生命周期校验（409），跳步同样不消耗扫码标识。
        self._assert_lifecycle(work.work_id, handover_type)

        handover = Handover(
            handover_id=_new_id("handover"),
            work_id=work.work_id,
            type=handover_type,
            scan_code=scan_code,
            from_party=from_party,
            to_party=to_party,
            report=report,
            on_date=on_date,
            at_location=at_location,
            linked_segments=linked_segments,
            fingerprint=fingerprint,
        )

        # 校验全部通过后才提交：扫码占用、交接入链、（损伤时）冻结与风险登记，
        # 在同一把锁内一次完成，并发的下一步交接不可能插到冻结之前。
        incident: Optional[Incident] = None
        self._scan_index[scan_key] = handover
        self.handovers.append(handover)
        if handover.damaged:
            self._frozen_works.add(work.work_id)
            incident = Incident(
                incident_id=_new_id("incident"),
                work_id=work.work_id,
                handover_id=handover.handover_id,
                on_date=handover.on_date,
                note=report.damage_note,
                before_hashes=list(report.before_hashes),
                after_hashes=list(report.after_hashes),
            )
            self.incidents.append(incident)
        return self._handover_result_view(handover, incident)

    def resolve_incident(self, incident_id: str, payload: dict[str, Any] | str) -> dict[str, Any]:
        """损伤经双方书面复核后解除；该作品仍有其他未结损伤时继续冻结。"""
        with self._locked():
            return self._resolve_incident_locked(incident_id, payload)

    def _resolve_incident_locked(self, incident_id: str, payload: dict[str, Any] | str) -> dict[str, Any]:
        incident = next((i for i in self.incidents if i.incident_id == incident_id), None)
        if incident is None:
            raise DomainError(f"损伤事件 {incident_id} 不存在")
        if incident.resolved:
            raise ConflictError(
                f"损伤事件 {incident_id} 已解除，不能重复复核",
                code="incident_already_resolved",
            )

        data = payload if isinstance(payload, dict) else {"resolution_note": payload}
        resolution_note = str(data.get("resolution_note", "") or "").strip()
        if not resolution_note:
            raise DomainError("解除损伤须填写处理与复核结论")
        lender = self._review_conclusion(data.get("lender_review"), "出借馆", "lender_review")
        borrower = self._review_conclusion(data.get("borrower_review"), "承借馆", "borrower_review")

        incident.resolved = True
        incident.resolution_note = resolution_note
        incident.review_lender = lender
        incident.review_borrower = borrower

        # 仅当该作品所有未结损伤都已具备双方复核结论（即全部结清）时才解除冻结。
        still_open = [
            i for i in self.incidents
            if i.work_id == incident.work_id and not i.resolved
        ]
        if not still_open:
            self._frozen_works.discard(incident.work_id)
        return {
            "incident_id": incident.incident_id,
            "work_id": incident.work_id,
            "resolved": True,
            "resolution_note": incident.resolution_note,
            "reviews": {
                "lender": self._review_view(lender),
                "borrower": self._review_view(borrower),
            },
            "remaining_open_incidents": [i.incident_id for i in still_open],
            "frozen": bool(still_open) or incident.work_id in self._frozen_works,
        }

    def _review_conclusion(self, raw: Any, side_label: str, field_name: str) -> "ReviewConclusion":
        if not isinstance(raw, dict):
            raise DomainError(f"解除损伤须附{side_label}（{field_name}）的书面复核结论")
        person = str(raw.get("person", "") or "").strip()
        note = str(raw.get("note", "") or "").strip()
        if not person:
            raise DomainError(f"{side_label}复核须签认复核人")
        if not note:
            raise DomainError(f"{side_label}复核须填写书面结论")
        on_date = self._date(raw.get("on_date"), f"{side_label}复核日期").isoformat()
        return ReviewConclusion(
            side=side_label,
            org=str(raw.get("org", "") or "").strip(),
            person=person,
            note=note,
            on_date=on_date,
        )

    @staticmethod
    def _review_view(review: "ReviewConclusion") -> dict[str, Any]:
        return {
            "side": review.side,
            "org": review.org,
            "person": review.person,
            "note": review.note,
            "on_date": review.on_date,
        }

    @staticmethod
    def _fingerprint_parts(
        work_id: str,
        handover_type: str,
        from_party: "Signature",
        to_party: "Signature",
        report: "ConditionReport",
        on_date: str,
        at_location: str,
        linked_segments: list[str],
    ) -> dict[str, Any]:
        """决定“同一次交接”的内容维度：生命周期位置、双方签认、日期、状态、区段。"""
        return {
            "work_id": work_id,
            "type": handover_type,
            "from_party": {"org": from_party.org, "role": from_party.role, "person": from_party.person},
            "to_party": {"org": to_party.org, "role": to_party.role, "person": to_party.person},
            "on_date": on_date,
            "at_location": at_location,
            "linked_segments": sorted(linked_segments),
            "report": {
                "condition": report.condition,
                "damage_note": report.damage_note,
                "image_hashes": list(report.image_hashes),
                "before_hashes": list(report.before_hashes),
                "after_hashes": list(report.after_hashes),
            },
        }

    def _stored_parts(self, handover: "Handover") -> dict[str, Any]:
        return self._fingerprint_parts(
            handover.work_id, handover.type, handover.from_party, handover.to_party,
            handover.report, handover.on_date, handover.at_location, handover.linked_segments,
        )

    @staticmethod
    def _canonical_fingerprint(parts: dict[str, Any]) -> str:
        canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _differing_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
        return [key for key in left if left.get(key) != right.get(key)]

    def _assert_lifecycle(self, work_id: str, handover_type: str) -> None:
        completed = [h.type for h in self.handovers if h.work_id == work_id]
        expected = HANDOVER_TYPES[len(completed)]
        if handover_type != expected:
            raise ConflictError(
                f"作品 {work_id} 下一次交接应为 {expected}，不能直接办理 {handover_type}"
            )

    @staticmethod
    def _signature(raw: dict[str, Any], expected_role: str) -> Signature:
        if not raw or not raw.get("person"):
            raise DomainError("交接双方都须指定签认人")
        role = raw.get("role", expected_role)
        if role != expected_role:
            raise DomainError(f"该交接位置须由 {expected_role} 签认，收到的是 {role}")
        return Signature(org=raw.get("org", ""), role=role, person=raw["person"])

    @staticmethod
    def _condition_report(raw: dict[str, Any]) -> ConditionReport:
        condition = raw.get("condition", "良好")
        if condition not in ("良好", "损伤"):
            raise DomainError("状态结论须为 良好 或 损伤")
        hashes = [LoanRegistry._image_hash(h) for h in raw.get("image_hashes", [])]
        damage_note = str(raw.get("damage_note", "") or "").strip()
        if condition == "损伤" and not damage_note:
            raise DomainError("损伤报告必须填写损伤说明")
        return ConditionReport(
            condition=condition,
            image_hashes=hashes,
            damage_note=damage_note,
            before_hashes=[LoanRegistry._image_hash(h) for h in raw.get("before_hashes", [])],
            after_hashes=[LoanRegistry._image_hash(h) for h in raw.get("after_hashes", [])],
            note=str(raw.get("note", "") or ""),
        )

    @staticmethod
    def _image_hash(value: str) -> str:
        """登记图像证据哈希；已是 64 位十六进制（sha256）时原样保全，否则计算。"""
        value = str(value)
        if re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value.lower()
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    # -- 策展展签 ----------------------------------------------------------

    @_synchronized
    def create_label(self, work_id: str, narrative: str, citations: list[dict[str, str]]) -> dict[str, Any]:
        work = self._work(work_id)
        if work_id in self.labels:
            raise ConflictError(f"作品 {work_id} 的展签已存在，应使用更正接口")
        if not narrative or not narrative.strip():
            raise DomainError("展签叙事不能为空")
        version = LabelVersion(
            version=1,
            status="草拟",
            narrative=narrative.strip(),
            citations=self._normalize_citations(citations),
            evidence={},
            published_on=None,
            frozen=False,
        )
        self.labels[work_id] = [version]
        return self._label_view(work_id, version)

    @_synchronized
    def publish_label(self, work_id: str, published_on: str) -> dict[str, Any]:
        """发布日期确认：锁定证据快照。快照只含当时的数据，之后不再变化。"""
        versions = self.labels.get(work_id)
        if not versions:
            raise DomainError(f"作品 {work_id} 尚无展签")
        current = versions[-1]
        if current.frozen:
            raise ConflictError(f"展签 v{current.version} 已发布并锁定，不能重复发布")
        day = self._date(published_on, "发布日期")
        current.status = "已发布"
        current.published_on = day.isoformat()
        current.frozen = True
        current.evidence = self._evidence_snapshot(work_id)
        return self._label_view(work_id, current)

    @_synchronized
    def correct_label(self, work_id: str, narrative: str, citations: list[dict[str, str]]) -> dict[str, Any]:
        """学术更正：另起新版本，旧版展签与其证据快照原样保留。"""
        versions = self.labels.get(work_id)
        if not versions:
            raise DomainError(f"作品 {work_id} 尚无展签")
        current = versions[-1]
        if not current.frozen:
            raise ConflictError("只能更正已发布的展签；草拟版可直接修改")
        new_version = LabelVersion(
            version=current.version + 1,
            status="草拟",
            narrative=narrative.strip(),
            citations=self._normalize_citations(citations),
            evidence={},
            published_on=None,
            frozen=False,
        )
        versions.append(new_version)
        return self._label_view(work_id, new_version)

    def _evidence_snapshot(self, work_id: str) -> dict[str, Any]:
        """发布时点的证据快照：作品、区段、贡献、协议、交接与未解除风险。"""
        work = self.works[work_id]
        agreement = self._current_agreement(work_id)
        handovers = [self._handover_view(h) for h in self.handovers if h.work_id == work_id]
        open_incidents = [
            {
                "incident_id": i.incident_id,
                "handover_id": i.handover_id,
                "on_date": i.on_date,
                "note": i.note,
                "before_hashes": i.before_hashes,
                "after_hashes": i.after_hashes,
            }
            for i in self.incidents
            if i.work_id == work_id and not i.resolved
        ]
        return {
            "snapshot_on": date.today().isoformat(),
            "work": {
                "work_id": work.work_id,
                "title": work.title,
                "kind": work.kind,
                "owner_org": work.owner_org,
            },
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "label": s.label,
                    "start_cm": s.start_cm,
                    "end_cm": s.end_cm,
                }
                for s in work.segments
            ],
            "contributions": [
                {
                    "contribution_id": c.contribution_id,
                    "author": c.author,
                    "kind": c.kind,
                    "order": c.order,
                    "segment_id": c.segment_id,
                }
                for c in sorted(work.contributions, key=lambda c: c.order)
            ],
            "agreement_version": agreement.authorization_scope() if agreement else None,
            "handovers": handovers,
            "custody": self._custody(work_id),
            "open_risks": open_incidents,
        }

    @staticmethod
    def _normalize_citations(citations: list[dict[str, str]]) -> list[dict[str, str]]:
        result = []
        for raw in citations or []:
            if not raw.get("ref"):
                raise DomainError("引用必须指向作品关系（work_id 或 segment_id）")
            result.append({"ref": raw["ref"], "note": raw.get("note", "")})
        return result

    # -- 查询视图 ----------------------------------------------------------

    @_synchronized
    def get_work_view(self, work_id: str) -> dict[str, Any]:
        work = self._work(work_id)
        agreement = self._current_agreement(work_id)
        return {
            "work": {
                "work_id": work.work_id,
                "title": work.title,
                "kind": work.kind,
                "owner_org": work.owner_org,
            },
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "label": s.label,
                    "start_cm": s.start_cm,
                    "end_cm": s.end_cm,
                    "note": s.note,
                }
                for s in work.segments
            ],
            "contributions": [
                {
                    "contribution_id": c.contribution_id,
                    "author": c.author,
                    "kind": c.kind,
                    "order": c.order,
                    "segment_id": c.segment_id,
                }
                for c in sorted(work.contributions, key=lambda c: c.order)
            ],
            "current_agreement": agreement.agreement_id if agreement else None,
            "agreement_versions": list(self._agreement_history.get(work_id, [])),
            "custody": self._custody(work_id),
            "frozen": work_id in self._frozen_works,
        }

    @_synchronized
    def risk_view(self, work_id: str) -> dict[str, Any]:
        """从展签/策展侧回答：实体在哪、谁保管、授权到哪、风险是否解除。"""
        work = self._work(work_id)
        agreement = self._current_agreement(work_id)
        open_incidents = [i for i in self.incidents if i.work_id == work_id and not i.resolved]
        return {
            "work": work.to_ref(),
            "custody": self._custody(work_id),
            "authorization": agreement.authorization_scope() if agreement else None,
            "open_risks": [
                {
                    "incident_id": i.incident_id,
                    "on_date": i.on_date,
                    "note": i.note,
                    "before_hashes": i.before_hashes,
                    "after_hashes": i.after_hashes,
                }
                for i in open_incidents
            ],
            "frozen": work_id in self._frozen_works,
        }

    @_synchronized
    def locate_segment(self, work_id: str, segment_id: str) -> dict[str, Any]:
        """局部状态争议入口：从区段定位实体、贡献、当前保管与风险。"""
        work = self._work(work_id)
        segment = work.require_segment(segment_id)
        linked = [
            {
                "contribution_id": c.contribution_id,
                "author": c.author,
                "kind": c.kind,
                "order": c.order,
            }
            for c in sorted(work.contributions, key=lambda c: c.order)
            if c.segment_id == segment_id
        ]
        segment_handovers = [
            self._handover_view(h)
            for h in self.handovers
            if h.work_id == work_id and (not h.linked_segments or segment_id in h.linked_segments)
        ]
        segment_incidents = [
            {
                "incident_id": i.incident_id,
                "on_date": i.on_date,
                "note": i.note,
                "resolved": i.resolved,
            }
            for i in self.incidents
            if i.work_id == work_id
            and any(
                h.linked_segments and segment_id in h.linked_segments
                for h in self.handovers
                if h.handover_id == i.handover_id
            )
        ]
        return {
            "work": work.to_ref(),
            "segment": {
                "segment_id": segment.segment_id,
                "label": segment.label,
                "start_cm": segment.start_cm,
                "end_cm": segment.end_cm,
            },
            "contributions": linked,
            "custody": self._custody(work_id),
            "segment_handovers": segment_handovers,
            "segment_incidents": segment_incidents,
            "frozen": work_id in self._frozen_works,
        }

    @_synchronized
    def label_version(self, work_id: str, version: Optional[int] = None) -> dict[str, Any]:
        versions = self.labels.get(self._work(work_id).work_id)
        if not versions:
            raise DomainError(f"作品 {work_id} 尚无展签")
        target = versions[-1] if version is None else next(
            (v for v in versions if v.version == version), None
        )
        if target is None:
            raise DomainError(f"展签 v{version} 不存在")
        return self._label_view(work_id, target)

    def _custody(self, work_id: str) -> dict[str, Any]:
        completed = [h for h in self.handovers if h.work_id == work_id]
        if not completed:
            work = self.works[work_id]
            return {"status": "在库", "custodian_role": "出借馆", "custodian_org": work.owner_org}
        last = completed[-1]
        return {
            "status": STATUS_AFTER[last.type],
            "custodian_role": last.to_party.role,
            "custodian_org": last.to_party.org,
            "since_handover": last.handover_id,
            "since": last.on_date,
        }

    def _current_agreement(self, work_id: str) -> Optional[Agreement]:
        history = self._agreement_history.get(work_id)
        if not history:
            return None
        return self.agreements[history[-1]]

    def _work(self, work_id: str) -> Work:
        work = self.works.get(work_id)
        if work is None:
            raise DomainError(f"作品 {work_id} 不存在")
        return work

    @staticmethod
    def _date(value: str, field_name: str) -> date:
        try:
            return date.fromisoformat(str(value))
        except (TypeError, ValueError):
            raise DomainError(f"{field_name} 须为 YYYY-MM-DD 日期")

    # -- 序列化 ------------------------------------------------------------

    def _agreement_view(self, ref: str | Agreement) -> dict[str, Any]:
        agreement = ref if isinstance(ref, Agreement) else self.agreements[ref]
        return {
            "agreement_id": agreement.agreement_id,
            "work_id": agreement.work_id,
            "version": agreement.version,
            "supersedes": agreement.supersedes,
            "lender_org": agreement.lender_org,
            "borrower_org": agreement.borrower_org,
            "start_on": agreement.start_on,
            "end_on": agreement.end_on,
            "gallery": agreement.gallery,
            "max_lux": agreement.max_lux,
            "transport": agreement.transport,
            "insurance": agreement.insurance,
            "digital_rights": agreement.digital_rights,
        }

    def _handover_result_view(
        self, handover: Handover, incident: Optional[Incident] = None
    ) -> dict[str, Any]:
        """一次交接成立（或按重试回放）后返回给调用方的视图。"""
        if incident is None and handover.damaged:
            incident = next(
                (i for i in self.incidents if i.handover_id == handover.handover_id), None
            )
        return self._handover_view(
            handover,
            frozen=handover.damaged,
            incident_id=incident.incident_id if incident else None,
        )

    def _handover_view(self, handover: Handover, frozen: bool = False, incident_id: Optional[str] = None) -> dict[str, Any]:
        view = {
            "handover_id": handover.handover_id,
            "work_id": handover.work_id,
            "type": handover.type,
            "scan_code": handover.scan_code,
            "on_date": handover.on_date,
            "at_location": handover.at_location,
            "linked_segments": handover.linked_segments,
            "from_party": {"org": handover.from_party.org, "role": handover.from_party.role, "person": handover.from_party.person},
            "to_party": {"org": handover.to_party.org, "role": handover.to_party.role, "person": handover.to_party.person},
            "signed_by_both": bool(handover.from_party.person and handover.to_party.person),
            "resulting_status": STATUS_AFTER[handover.type],
            "condition": {
                "condition": handover.report.condition,
                "damage_note": handover.report.damage_note,
                "image_hashes": handover.report.image_hashes,
                "before_hashes": handover.report.before_hashes,
                "after_hashes": handover.report.after_hashes,
            },
        }
        if frozen:
            view["frozen"] = True
            view["frozen_reason"] = "发现损伤，后续动作冻结"
        if incident_id:
            view["incident_id"] = incident_id
        return view

    def _label_view(self, work_id: str, version: LabelVersion) -> dict[str, Any]:
        return {
            "work_id": work_id,
            "version": version.version,
            "status": version.status,
            "frozen": version.frozen,
            "published_on": version.published_on,
            "narrative": version.narrative,
            "citations": version.citations,
            "evidence_snapshot": version.evidence,
        }


# ---------------------------------------------------------------------------
# 应用外观：供 HTTP 层调用
# ---------------------------------------------------------------------------


@dataclass
class Route:
    method: str
    pattern: str
    handler: Callable[[LoanRegistry, dict[str, Any], dict[str, str]], dict[str, Any]]


def build_routes() -> list[Route]:
    return [
        Route("POST", r"^/works$", lambda reg, body, _: reg.register_work(
            body["title"], body["kind"], body["owner_org"],
            body.get("segments"), body.get("contributions"),
        )),
        Route("GET", r"^/works/(?P<id>[^/]+)$", lambda reg, _b, p: reg.get_work_view(p["id"])),
        Route("GET", r"^/works/(?P<id>[^/]+)/risk$", lambda reg, _b, p: reg.risk_view(p["id"])),
        Route("GET", r"^/works/(?P<id>[^/]+)/segments/(?P<sid>[^/]+)$",
              lambda reg, _b, p: reg.locate_segment(p["id"], p["sid"])),
        Route("POST", r"^/agreements$", lambda reg, body, _: reg.create_agreement(body)),
        Route("POST", r"^/agreements/(?P<id>[^/]+)/reschedule$",
              lambda reg, body, p: reg.reschedule_agreement(p["id"], body)),
        Route("POST", r"^/handovers$", lambda reg, body, _: reg.record_handover(body)),
        Route("POST", r"^/incidents/(?P<id>[^/]+)/resolve$",
              lambda reg, body, p: reg.resolve_incident(p["id"], body)),
        Route("POST", r"^/works/(?P<id>[^/]+)/labels$",
              lambda reg, body, p: reg.create_label(p["id"], body["narrative"], body.get("citations", []))),
        Route("POST", r"^/works/(?P<id>[^/]+)/labels/publish$",
              lambda reg, body, p: reg.publish_label(p["id"], body["published_on"])),
        Route("POST", r"^/works/(?P<id>[^/]+)/labels/correct$",
              lambda reg, body, p: reg.correct_label(p["id"], body["narrative"], body.get("citations", []))),
        Route("GET", r"^/works/(?P<id>[^/]+)/labels$",
              lambda reg, _b, p: reg.label_version(p["id"], None)),
        Route("GET", r"^/works/(?P<id>[^/]+)/labels/(?P<v>[0-9]+)$",
              lambda reg, _b, p: reg.label_version(p["id"], int(p["v"]))),
    ]
