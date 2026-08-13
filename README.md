<div align="center">

# 🤖 agent

**用 Rust 打造的轻量级 Agent 框架 —— 一次编写，可作为库嵌入、多租户 HTTP 服务独立部署，也可作为交互式 CLI 使用。**

[![Rust](https://img.shields.io/badge/Rust-2024-000000?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-200%2B%20passing-brightgreen)](#-开发)
[![Clippy](https://img.shields.io/badge/clippy-clean-success?logo=rust)](#-开发)

</div>

---

## 📖 简介

`agent` 把"大模型 + 工具调用循环"封装为可复用的 Rust 库：内置多轮对话、多租户隔离、结构化输出、MCP 工具生态与向量检索，并提供一个可直接部署的 REST 服务二进制，以及一个开箱即用的交互式 CLI。

## ✨ 主要特性

| 分类 | 能力 |
|---|---|
| 🔁 **对话循环** | 纯文本 / 流式 / 结构化输出（JSON Schema 自动推断）三种模式的工具调用循环 |
| 💬 **交互式 CLI** | `cargo run --bin cli`：多轮聊天 + 永不过期的会话持久化（`--list`/`--rm` 管理）+ 自动接入 `mcp.json` 的 MCP 工具 + `--no-stream` 切换非流式 + 高危工具确认（`ApprovalCallback`）+ `web_search` 结果自动压缩（`SearchCompressorCallback`）+ `--workspace` 钉定目录并沙箱化内置文件工具（`WorkspaceGuardCallback`）+ 默认 vi 键位的行编辑器（`--no-vi-mode` 切回 Emacs 键位） |
| 🧠 **`Agent` 运行时** | 完整事件记录（`ExecutionContext`）+ 无状态多轮续接（`run_continuing`）+ 流式输出（`run_stream`），历史按 token 预算自动裁剪 |
| 💾 **会话持久化** | `SessionStore` trait 与 HTTP 层解耦，内置内存（`MemorySessionStore`，HTTP 服务用）与落盘（`FileSessionStore`，按 `(scope, sessionId)` 存 JSON 文件，CLI 用）两种实现 |
| 🧰 **工具生态** | 内置 `calculator`、`web_search`（Tavily），并通过 `mcp.json` 接入任意 MCP Server（stdio / Streamable HTTP 两种传输） |
| 🏢 **多租户** | `Provider` 封装每租户凭据 + 并发限流，互不干扰 |
| 🌐 **HTTP 服务** | `/v1/agent/run` 支持会话续接（`sessionId`）、幂等重试（`Idempotency-Key`）与结构化输出（`responseSchema`） |
| 📚 **向量检索** | 文本分块 / embedding / 余弦相似度检索，适配 RAG 场景 |
| 📊 **基准评测** | 内置 GAIA 数据集评测，量化模型 + 工具组合效果 |
| ✅ **工程质量** | 无 `.unwrap()` 生产路径、零硬编码密钥、200+ 单测、clippy 全绿 |

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

# 3. 交互式聊天（无需 tenants.json，直接用 OPENAI_API_KEY）
cargo run --bin cli

# 4. 启动 HTTP 服务
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

## 💬 CLI 使用方式

```bash
cargo run --bin cli                        # 在 `default` 会话中聊天
cargo run --bin cli -- --session work      # 独立的、命名的会话
cargo run --bin cli -- --workspace ~/projects/foo  # 把这次运行钉在（并沙箱到）指定目录
cargo run --bin cli -- --tools calculator  # 只启用指定内置工具（不加载 MCP）
cargo run --bin cli -- --fresh             # 启动前先清空该会话的历史
cargo run --bin cli -- --list              # 列出所有已保存的会话
cargo run --bin cli -- --rm work           # 删除名为 `work` 的会话
cargo run --bin cli -- --no-stream         # 一次性输出完整回复，而非逐 token 流式
cargo run --bin cli -- --dangerous-tools delete_file,demo__write_file  # 自定义需确认的高危工具
cargo run --bin cli -- --no-approval       # 关闭高危操作确认，直接执行所有工具
cargo run --bin cli -- --no-search-compression  # 关闭 web_search 结果压缩，看模型原始收到的内容
cargo run --bin cli -- --no-sandbox        # 允许文件类工具访问工作区之外的路径
cargo run --bin cli -- --no-vi-mode        # 关闭 vi 键位，改用 Emacs 键位编辑输入行
```

聊天中可用命令：`/reset` 清空当前会话历史；`exit` / `quit` / `:q`（或 Ctrl-D）退出。

`You>` 提示符的行编辑由 [`reedline`](https://github.com/nushell/reedline)（`nushell` 同款行编辑器）提供，默认使用 **vi 键位**：直接打字即为插入模式，`Esc` 进入 normal 模式后可用 `hjkl`/`w`/`b`/`0`/`$`/`dd` 等移动或编辑，`k`/`j` 翻历史，`i`/`a` 回到插入模式——等价于 `bash` 的 `set -o vi` / `zsh` 的 `bindkey -v`。终端光标形状会跟随模式变化（插入模式为竖线，normal 模式为块状，类似 Vim 本身的默认约定），不用看输入内容也能分辨当前处于哪个模式。用 `--no-vi-mode` 切换回 `reedline` 的 Emacs 键位（方向键翻历史、`Ctrl-A`/`Ctrl-E` 等，标准 `bash`/`readline` 默认行为；Emacs 模式没有 insert/normal 之分，因此不切换光标形状）。`Ctrl-C` 只取消当前正在输入的这一行（回到空提示符），不会退出聊天；`Ctrl-D` 仍是退出聊天的方式。

`--list` / `--rm <session>` 是一次性的会话管理命令：打印结果后立即退出，不会进入聊天、也不会初始化 LLM Provider（因此无需配置 `OPENAI_API_KEY` 即可使用）。`--list` 按最近活跃时间倒序列出每个会话的 id、事件数与相对时间（如 `3h ago`）；`--rm` 删除指定会话，删除一个不存在的会话不算错误，只会提示未找到。

会话历史按 `--session` 的名字落盘到 [`agent::session::FileSessionStore`](src/agent/session.rs)（默认目录 `.agent/sessions`，可用 `AGENT_CLI_SESSION_DIR` 覆盖），**进程退出后再次运行同一个 `--session` 仍能续接对话，且永不过期**（用 `FileSessionStore::new_persistent` 构造，与 HTTP 服务的 TTL 会话不同）——多久以前的对话都能接着聊，只有 `--fresh` / `/reset` / `--rm` 会清空。CLI 没有多租户概念，鉴权配置（`tenants.json`）与之无关，只需要 `OPENAI_API_KEY` 等模型凭据。

**`--workspace <dir>`** 把这次运行钉在一个具体目录（默认：启动 `cli` 时所在的目录，因此不传这个参数时行为与以前完全一致），并通过真正的 `std::env::set_current_dir` 切换过去——此后进程里任何相对路径（模型传给文件工具的参数、`mcp.json` 的默认查找路径、`.agent/sessions` 的默认位置）都以它为基准解析。这也意味着不同 `--workspace` 默认拥有各自独立的会话与 MCP 配置（除非用绝对路径的 `AGENT_CLI_SESSION_DIR` / `MCP_CONFIG_PATH` 覆盖）。在此之上，[`WorkspaceGuardCallback`](src/callback/path_guard.rs) 把这个目录变成内置文件类工具（`delete_file`/`read_file`/`list_files`/`unzip_file`）的**硬边界**：模型传入的路径参数一旦解析后落在工作区之外（绝对路径、`../` 逃逸，甚至指向工作区外的符号链接），会在真正执行前直接被拒绝——甚至不会触发确认弹窗。这解决的正是"CLI 运行时没有限定到具体目录，危险操作可能波及工作区之外"的风险。默认开启（沙箱状态显示在启动横幅里），`--no-sandbox` 可关闭这层限制（工作目录本身仍会被钉住，只是不再拦截越权路径）；MCP 工具的参数不在保护范围内（其 schema 运行时才发现，无法预先校验），启用 MCP Server 前请确保信任它。

工具集：默认内置工具（`calculator`、`web_search`、文件系统工具等）之外，若 `mcp.json`（`MCP_CONFIG_PATH`，默认路径 `mcp.json`）存在，会自动连接其中每个已启用的 MCP Server 并把发现的工具一并注册（见 [`ToolRegistry::with_mcp`](src/tools/registry.rs)）；退出聊天时会优雅关闭这些连接。`--tools` 会切换到显式的内置工具子集，此时不加载 MCP（MCP 工具的名字要连接后才知道，无法提前按名选择）。

`delete_file` 默认被视为高危操作：调用前会在终端打印参数并等待 `y`/`n` 确认（[`ApprovalCallback`](src/callback/approval.rs)），拒绝时模型会收到一条错误结果并继续对话而不会中断整轮运行。用 `--dangerous-tools` 指定另一份需确认的工具名单（逗号分隔，可以是内置工具或 `<server>__<tool>` 形式的 MCP 工具），或用 `--no-approval` 完全关闭确认、放行所有工具调用。

`web_search` 的结果默认会被压缩：过长的网页原文会先分块，再按本轮查询做向量检索，只保留最相关的片段写入会话历史（[`SearchCompressorCallback`](src/callback/search_compressor.rs)），避免长文本占满后续每一轮的上下文；压缩失败（如向量检索所需的 embedding 服务不可用）会静默回退为不压缩，不影响本轮对话。用 `--no-search-compression` 关闭，保留原始结果（便于调试模型实际看到的内容)。

## 📦 作为库使用

尚未发布到 crates.io，以 Git 依赖的方式引入：

```toml
# Cargo.toml
[dependencies]
agent = { git = "<repo-url>" }
```

**最小可运行示例**（单轮问答 + 工具调用）：

```rust
use std::sync::Arc;
use agent::{Agent, config, llm::provider::Provider, telemetry, tools::ToolRegistry};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?; // 加载 .env 并初始化日志，务必在读取任何 config::* 之前调用

  let toolbox = Arc::new(ToolRegistry::builtin()?); // 内置 calculator / web_search / mcp
  let agent = Agent::new(
    Provider::shared().clone(),   // 读取 OPENAI_API_KEY 等环境变量
    config::model(),              // 默认 deepseek-v4-flash，见 LLM_MODEL
    Some("You are a helpful assistant."),
    toolbox,
  );

  let result = agent.run("5875 乘以 467 是多少？").await?;
  println!("{}", result.output);
  Ok(())
}
```

**多轮对话**：`Agent` 本身无状态（纯函数：历史事件 + 新输入 → 结果），多轮续接需要调用方自己保存 `result.context.events` 并在下一轮传回 `run_continuing`。内置 `agent::session::{SessionStore, MemorySessionStore, FileSessionStore}` 可直接复用（HTTP 服务用 `MemorySessionStore`，需要跨进程保留会话——例如 CLI——可换成落盘的 `FileSessionStore`）：

```rust
use agent::session::{FileSessionStore, SessionStore};

// `new_persistent`: sessions never expire, matching what a CLI-style, cross-process
// conversation needs. Use `FileSessionStore::new(dir, ttl)` instead for an HTTP-style
// deployment that should evict idle sessions after a fixed TTL.
let store = FileSessionStore::new_persistent("./sessions");

// 第一轮
let history = store.history("local", "chat-1").await; // 空历史
let result = agent.run_continuing(history, "帮我记一下：明天下午 3 点开会").await?;
store.save("local", "chat-1", result.context.events.clone()).await;

// 第二轮，进程重启后依然可续接
let history = store.history("local", "chat-1").await;
let result = agent.run_continuing(history, "我刚才让你记的是什么时间？").await?;
```

**流式输出**：用 `agent.run_stream(input)` 替代 `run`，得到 `impl Stream<Item = AgentStreamEvent>`，逐 token 转发，见 [`examples/agent_stream.rs`](examples/agent_stream.rs)。

**结构化输出**（反序列化为 Rust 类型）：

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct MultiplicationResult { product: f64 }

let result = agent.run_structured::<MultiplicationResult>("5875 乘以 467 是多少？").await?;
println!("{}", result.output.product);
```

更多用法（多租户、MCP 工具、审批回调、向量检索等）见 [`examples/`](examples) 目录下的 18 个可运行示例，每个文件顶部注释都写明了 `cargo run --example <name>` 的运行方式。

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
| `AGENT_SESSION_TTL_SECS` | `1800` | HTTP 会话空闲过期时间（CLI 会话永不过期，不受此项影响） |
| `AGENT_CLI_SESSION_DIR` | `.agent/sessions` | CLI 会话落盘目录 |
| `AGENT_IDEMPOTENCY_TTL_SECS` | `86400` | 幂等 key 缓存有效期 |
| `TAVILY_API_KEY` | - | `web_search` 工具密钥 |
| `RUST_LOG` | `info` | 日志级别 |

> 完整列表（含 `HF_TOKEN`、`EMBED_*` 等）见 `src/config.rs`。

## 📁 项目结构

```
src/
├── agent/          Agent 运行时：ExecutionContext / Event / 多轮历史裁剪 / SessionStore（Memory / File 两种实现）
├── api/            HTTP 服务：路由、鉴权、IdempotencyStore（SessionStore 已下沉至 agent/，此处仅 re-export）
├── llm/            Provider、tool_loop、stream、structured、complete
├── tools/          Tool trait 与内置工具（calculator / web_search / mcp）
├── callback/       工具调用前后的回调实现（人工审批 / 搜索结果压缩）
├── gaia/           GAIA 基准数据集与评测
├── knowledge_base/ 文本分块、embedding、向量检索
├── bin/            二进制入口：cli（交互式聊天）、server（HTTP 服务）、gaia（基准评测）
└── config.rs       环境变量统一读取入口
examples/           18 个可运行示例（`shared/` 为示例间共用的 demo MCP server，非独立示例）
```

## 🧪 开发

```bash
cargo check --all-targets   # 编译检查
cargo test                # 200+ 单测 + doctest
cargo clippy --all-targets  # lint（零告警）
cargo fmt                   # 格式化
```

## 🗺️ Roadmap

<details>
<summary>点击展开后续规划</summary>

- **会话与记忆**：`SessionStore` 的 Redis/DB 实现（已有 `Memory`/`File` 两种单机实现）、`/v1/sessions` 资源化、长期记忆分层、租户+终端用户两级隔离
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
