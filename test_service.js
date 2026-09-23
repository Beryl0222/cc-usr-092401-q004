"use strict";

const { spawnSync } = require("node:child_process");

// 运行全部 *_contract.py：服务契约、HTTP 契约、领域契约与并发一致性契约。
const result = spawnSync(
  "python3",
  ["-m", "unittest", "-v", "service_contract", "api_contract", "domain_contract", "concurrency_contract"],
  { stdio: "inherit" }
);
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
