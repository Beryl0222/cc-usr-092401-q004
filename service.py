"""馆际作品借展的运行入口。

提供健康检查与领域 JSON 接口；领域规则全部在 domain.py，
本模块只负责 HTTP 编解码与状态码映射。
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from domain import (
    ConflictError,
    DomainError,
    LoanRegistry,
    ReplayConflict,
    Route,
    build_routes,
)

SERVICE_ID = "museum-loan"
SERVICE_NAME = "馆际作品借展"


def health_payload():
    """返回稳定的服务身份信息。"""
    return {"status": "ok", "service": SERVICE_ID, "name": SERVICE_NAME}


class ApiState:
    """进程内共享的领域记录（单实例部署足够；多实例需换持久层）。"""

    def __init__(self):
        self.registry = LoanRegistry()
        self.routes = build_routes()


STATE = ApiState()


class Handler(BaseHTTPRequestHandler):
    """健康检查与借展领域接口，供本地联调和运维巡检使用。"""

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method):
        if method == "GET" and self.path == "/health":
            self._write_json(200, health_payload())
            return

        body = {}
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                self._write_json(400, {"error": "请求体须为 UTF-8 JSON", "kind": "invalid_request"})
                return
            if not isinstance(body, dict):
                self._write_json(400, {"error": "请求体须为 JSON 对象", "kind": "invalid_request"})
                return

        for route in STATE.routes:  # type: Route
            if route.method != method:
                continue
            match = re.match(route.pattern + r"$", self.path)
            if not match:
                continue
            try:
                payload = route.handler(STATE.registry, body, match.groupdict())
            except ReplayConflict as error:
                # 同扫码号但签认/日期/状态/关联区段不同：真正的冲突。
                self._write_error(409, error, "scan_conflict")
            except ConflictError as error:
                # 生命周期跳步、已冻结、已锁定等状态冲突。
                self._write_error(409, error, "conflict")
            except DomainError as error:
                # 字段不合法、缺签认、角色不符等请求错误。
                self._write_error(400, error, "invalid_request")
            else:
                # 完全相同的重试返回首次交接，视图带 replay 标记；
                # 用响应头让调用方无需解析体即可识别重放。
                headers = {"Idempotency-Replayed": "true"} if payload.get("replay") else None
                self._write_json(200, payload, headers)
            return

        self.send_error(404)

    def _write_error(self, status, error, kind):
        """错误响应带稳定 kind，供调用方区分重放/冲突/字段错误。"""
        self._write_json(status, {"error": str(error), "kind": kind})

    def _write_json(self, status, payload, extra_headers=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return


def main():
    parser = argparse.ArgumentParser(description=SERVICE_NAME)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        assert health_payload()["service"] == SERVICE_ID
        LoanRegistry().register_work(
            title="自检样例", kind="独立作品", owner_org="自检馆"
        )
        print("基础检查通过")
        return
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
