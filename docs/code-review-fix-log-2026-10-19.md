# 代码审核结果与修复进度

- 日期：2026-10-19
- 审核对象：工作区与暂存区中所有未提交的改动
- 状态：**全部修复完成，可安全提交**

## 审核范围

本次改动共 39 个文件、约 +4348 / -134 行，主题集中在：

1. **运行取消机制**：`CancellationToken` 贯穿模型请求、流式输出与工具执行，中途可取消并以携带 transcript 的 `Cancelled` 错误返回。
2. **工具超时与并发调度**：`Tool::timeout` / `Tool::concurrency_safe` 两个新能力，`execute_with_timeout` 统一超时，`schedule::plan` 用 gate 编排同轮工具的并发与串行。
3. **新增工具**：`write_file`、`edit_file`、`grep`、`run_command`（shell）。
4. **安全加固**：`path_guard` 扩展受保护路径（`.env*`、`.git`、`.agent`、`mcp.json`），shell 不享受"常驻允许"，`builtin()` 与 `builtin_all()` 拆分默认工具集。
5. **上下文压缩增强**：除结果外，新增对 `write_file.content`、`edit_file.old_string/new_string` 的参数裁剪。

## 审核结论

改动整体质量很高：文档注释详尽、测试覆盖充分（含取消竞态、进程组回收、符号链接、幂等性）、安全边界（进程组清理、原子写、凭据剥离、受保护路径）经得起推敲。**未发现阻塞提交的缺陷。**

## 问题清单与修复状态

### 已修复

| # | 严重程度 | 位置 | 问题描述 | 修复方式 |
|---|---------|------|---------|---------|
| 1a | 中 | `src/agent/runtime/mod.rs` | `let slots = &Semaphore::new(...)` 依赖 Rust 临时值生命周期延长，脆弱且可读性差 | 改为具名绑定 `semaphore` + `&semaphore`，并在注释说明借用范围 |
| 1b | 中 | `src/llm/tool_loop.rs` | 同上 | 同上 |
| 2 | 中 | `src/tools/file_write/execute.rs` | `write_atomic` 先以默认 umask 权限创建临时文件，之后才 `set_permissions`，对 `0o600` 敏感文件存在短暂同机可读窗口 | 新增 `write_temp`，用 `OpenOptions` 在创建时即设置目标权限位（Unix）；`create_new` 防覆盖 |
| 3 | 低 | `src/tools/registry.rs` | `builtin()` 保留 `delete_file`/`unzip_file` 两个破坏性工具，但文档未显式说明其破坏性 | 文档补充说明：二者仅因向后兼容保留，无法加闸的部署应用 `select` |
| 4 | 低 | `src/callback/context_optimizer/compaction.rs` | `trimmed_calls` 不校验 ToolResult 配对，孤立 ToolCall 也会被裁剪，语义未说明 | 补充注释：未配对的写/编辑调用与已完成调用同样"已发出"，裁剪无害 |
| 5 | 低 | `src/tools/grep/execute.rs` | `glob_to_regex` 逐字符 `to_string()` + `regex::escape` 产生临时分配 | 改为逐字符判断 regex 元字符，手动加 `\` 前缀 |

### 无需处理（已核实为正确设计）

以下在审查中重点核查、确认无需改动：

- **符号链接逃逸**：`path_guard` 的 `resolve_best_effort` 对"已存在的最长前缀"做 `canonicalize`（解析 symlink），指向 root 外的 symlink 会被解析到真实路径并拒绝。文档已声明两个已知限制（未创建后缀里的 symlink、TOCTOU），单用户单机场景可接受。
- **同步工具超时语义**：`spawn_blocking` 中的同步工具超时后仍在后台跑完，无法真正中断；代码与文档均诚实声明，且错误信息按 `concurrency_safe` 区分"已放弃"与"可能仍在运行"。
- **`Capped` 头尾截断**：分块不变性、多字节切割、省略字节计数经核对均正确。
- **`schedule::plan` 屏障语义**：用 `watch` channel 的 drop 作为完成信号，任何结束路径都能释放后继，`turn()` 幂等且 cancel-safe。

## 验证结果

| 检查项 | 结果 |
|-------|------|
| `cargo check --all-targets` | 通过 |
| `cargo clippy --all-targets` | 无新增警告（仅存示例中 `rmcp::ServerInfo` 的既有 deprecated 警告） |
| `cargo test --lib` | 555 通过 / 0 失败 |
| `cargo test --test cancellation` | 2 通过 / 0 失败 |
| `cargo test --lib tools::file_write` | 9 通过 / 0 失败 |
| `cargo test --lib tools::grep` | 13 通过 / 0 失败 |
