<div align="center">

# 🤖 agent

**用 Rust 打造的轻量级 Agent 框架 —— 一次编写，既可作为库嵌入，也可作为多租户 HTTP 服务独立部署。**

[![Rust](https://img.shields.io/badge/Rust-2024-000000?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-130%2B%20passing-brightgreen)](#-开发)
[![Clippy](https://img.shields.io/badge/clippy-clean-success?logo=rust)](#-开发)

</div>

---

## 📖 简介

`agent` 把"大模型 + 工具调用循环"封装为可复用的 Rust 库：内置多轮对话、多租户隔离、结构化输出、MCP 工具生态与向量检索，并提供一个可直接部署的 REST 服务二进制。

## ✨ 主要特性

| 分类 | 能力 |
|---|---|
| 🔁 **对话循环** | 纯文本 / 流式 / 结构化输出（JSON Schema 自动推断）三种模式的工具调用循环 |
| 🧠 **`Agent` 运行时** | 完整事件记录（`ExecutionContext`）+ 无状态多轮续接（`run_continuing`）+ 流式输出（`run_stream`），历史按 token 预算自动裁剪 |
| 🧰 **工具生态** | 内置 `calculator`、`web_search`（Tavily），并通过 `mcp.json` 接入任意 MCP Server |
| 🏢 **多租户** | `Provider` 封装每租户凭据 + 并发限流，互不干扰 |
| 🌐 **HTTP 服务** | `/v1/agent/run` 支持会话续接（`sessionId`）、幂等重试（`Idempotency-Key`）与结构化输出（`responseSchema`） |
| 📚 **向量检索** | 文本分块 / embedding / 余弦相似度检索，适配 RAG 场景 |
| 📊 **基准评测** | 内置 GAIA 数据集评测，量化模型 + 工具组合效果 |
| ✅ **工程质量** | 无 `.unwrap()` 生产路径、零硬编码密钥、130+ 单测、clippy 全绿 |

## 🚀 快速开始

```bash
# 1. 克隆并配置
git clone <repo-url> && cd agent
cp mcp.example.json mcp.json          # 可选：接入 MCP Server
cp tenants.example.json tenants.json  # HTTP 服务需要，填入真实 apiKey/baseUrl

export OPENAI_API_KEY=sk-...
export LLM_MODEL=deepseek-v4-flash    # 默认值，可省略

# 2. 跑一个示例
cargo run --example tool_call_complete

# 3. 启动 HTTP 服务
cargo run --bin server  # 默认监听 0.0.0.0:8080
```

**调用 HTTP 服务（支持多轮续接）：**

```bash
curl -X POST localhost:8080/v1/agent/run \
  -H "Authorization: Bearer <tenants.json 中配置的 token>" \
  -H "Content-Type: application/json" \
  -d '{"input": "5875 乘以 467 是多少", "tools": ["calculator"], "sessionId": "chat-1"}'
```

**调用 HTTP 服务（结构化输出，不支持与 `sessionId` 同时使用）：**

```bash
curl -X POST localhost:8080/v1/agent/run \
  -H "Authorization: Bearer <tenants.json 中配置的 token>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "5875 乘以 467 是多少",
    "tools": ["calculator"],
    "responseSchema": {
      "name": "MultiplicationResult",
      "schema": {"type": "object", "properties": {"product": {"type": "number"}}, "required": ["product"]}
    }
  }'
```

## ⚙️ 配置说明

所有配置均通过环境变量读取（统一入口 [`src/config.rs`](src/config.rs)），常用项如下：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODEL` | `deepseek-v4-flash` | 默认模型 |
| `LLM_MAX_CONCURRENCY` | `3` | 单租户并发上限 |
| `LLM_MAX_TOOL_ROUNDS` | `10` | 单次调用工具轮次预算 |
| `LLM_MAX_RETRIES` | `3` | 单次模型请求失败后的重试次数（指数退避，`0` 关闭重试） |
| `LLM_MAX_HISTORY_TOKENS` | `6000` | 多轮历史 token 软上限 |
| `MCP_CONFIG_PATH` | `mcp.json` | MCP Server 配置路径 |
| `AGENT_HTTP_ADDR` | `0.0.0.0:8080` | HTTP 服务监听地址 |
| `AGENT_TENANTS_PATH` | `tenants.json` | 租户配置路径 |
| `AGENT_SESSION_TTL_SECS` | `1800` | 会话空闲过期时间 |
| `AGENT_IDEMPOTENCY_TTL_SECS` | `86400` | 幂等 key 缓存有效期 |
| `TAVILY_API_KEY` | - | `web_search` 工具密钥 |
| `RUST_LOG` | `info` | 日志级别 |

> 完整列表（含 `HF_TOKEN`、`EMBED_*` 等）见 `src/config.rs`。

## 📁 项目结构

```
src/
├── agent/          Agent 运行时：ExecutionContext / Event / 多轮历史裁剪
├── api/            HTTP 服务：路由、鉴权、SessionStore、IdempotencyStore
├── llm/            Provider、tool_loop、stream、structured、complete
├── tools/          Tool trait 与内置工具（calculator / web_search / mcp）
├── callback/       工具调用前后的回调实现（人工审批 / 搜索结果压缩）
├── gaia/           GAIA 基准数据集与评测
├── knowledge_base/ 文本分块、embedding、向量检索
└── config.rs       环境变量统一读取入口
examples/           15 个可运行示例
```

## 🧪 开发

```bash
cargo check --all-targets   # 编译检查
cargo test                # 130+ 单测 + doctest
cargo clippy --all-targets  # lint（零告警）
cargo fmt                   # 格式化
```

## 🗺️ Roadmap

<details>
<summary>点击展开后续规划</summary>

- **会话与记忆**：`SessionStore` 的 Redis/DB 实现、`/v1/sessions` 资源化、长期记忆分层、租户+终端用户两级隔离
- **协议与可扩展性**：`Tool` trait 与 `async-openai` 解耦、类型化错误（`thiserror`）
- **架构边界**：`gaia` 拆为独立 crate
- **运维**：`Dockerfile` 与部署文档、sweep 任务补充 metrics

</details>

## 🤝 贡献指南

欢迎提交 Issue / PR：

1. Fork 并新建分支
2. 提交前确保 `cargo test && cargo clippy --all-targets && cargo fmt --check` 全部通过
3. 提交 PR 并描述改动动机

## 📄 License

[MIT](LICENSE) © 2026
