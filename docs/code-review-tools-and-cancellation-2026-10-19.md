# 代码审查：写入/执行类工具与取消/超时（2026-10-19）

> 范围：暂存区变更（30 个文件），即 `docs/agent-gap-analysis.md` 的第 1、2 项。
> 方式：先做静态审查，随后全部修复。状态：✅ 已修复并有测试；🟡 已缓解，仍有说明中的限制；📝 仅记录，未改动。
> 验证（与 `.github/workflows/ci.yml` 一致的检查）：
> - `cargo fmt --all -- --check`：通过。
> - `cargo clippy --all-targets --all-features -- -D warnings`（原生）与 `cargo clippy -p web-ui -p shared --target wasm32-unknown-unknown --all-targets -- -D warnings`：通过。
> - `cargo test --lib --tests` 连续运行 3 次，结果一致：555 个库测试 + 30 + 2 + 9 个集成测试全部通过（用于排查时序类测试是否抖动）；`cargo test --doc`：通过。
> - `cargo doc --no-deps`（`-D warnings`）：有 5 个错误，集中在 `src/bin/cli/web.rs` 的失效 intra-doc 链接。我在导出的未改动 HEAD 树上跑了同一条命令，同样是 5 个，所以是**既有问题**，不是本次引入，未处理。
> - Windows 目标（`x86_64-pc-windows-msvc`）交叉检查：**未能完成**，本机缺少 Windows C 头文件，`zstd-sys` 编译失败，与本次改动无关。因此 L2 仍标为 🟡。
> - `cspell`（不在 CI 内）：把本次新增的技术词补进了 `cspell.json`（共 18 个）。逐个核对过，其余报出的词在 HEAD 里就已存在，不是本次引入。
> - 过程中发现并修正了两个自己的错误：一次替换误把测试函数的首行并到了 `{` 之后（已改回并由 `cargo fmt --check` 确认）；`is_secret_env_name` 的新测试里有一条断言与函数的实际规则不符，已按真实规则（只看名字结尾）改正。

## 1. 总览

| 级别 | 数量 | ✅ | 🟡 | 📝 |
|---|---|---|---|---|
| 🛑 严重 | 2 | 2 | 0 | 0 |
| ⚠️ 重要 | 9 | 9 | 0 | 0 |
| 💡 建议 | 11 | 9 | 1 | 1 |
| 其他观察 | 3 | 3 | 0 | 0 |

## 2. 追踪表

### 🛑 严重

| 编号 | 问题 | 状态 | 修复说明 |
|---|---|---|---|
| S1 | `builtin()` 默认含 `run_command` / `write_file` / `edit_file`，CLI 以外的入口没有沙箱和审批 | ✅ | `ToolRegistry::builtin()` 收窄为不含这三个工具（仍含 `grep`）；新增 `builtin_all()` 给 CLI；新增 `with_mcp_over(base, path)` 以便在 `builtin_all()` 上叠加 MCP。8 个示例和 `tool_loop` 自动获得收窄后的集合。测试：`builtin_leaves_out_the_write_and_execute_tools`、`builtin_all_adds_exactly_the_write_and_execute_tools`。 |
| S2 | 同一轮并发执行有副作用的工具，同文件两个 `edit_file` 会静默丢更新 | ✅ | `Tool` 新增 `concurrency_safe()`，**默认 `false`**；只读内置工具（`calculator`、`web_search`、`read_file`、`list_files`、`read_image`、`grep`）以及带 `readOnlyHint` 的 MCP 工具声明为 `true`。新增 `src/tools/schedule.rs`：不安全的调用是前后双向屏障，按调用顺序执行。只有"工具执行"这一步排队，审批提示仍然同时弹出。`Agent::execute_tool_calls` 与 `tool_loop` 两条路径都接入。屏障靠 `watch` 通道的发送端 drop 释放，因此被拒绝、被挂起、被取消的调用都不会卡住后面的调用。测试见 `schedule.rs` 与 `runtime/tests.rs`（含"被拒绝的调用不阻塞后续"、"取消时释放排队调用"）。 |

### ⚠️ 重要

| 编号 | 问题 | 状态 | 修复说明 |
|---|---|---|---|
| M1 | 输出全部读入内存后才截断 | ✅ | 改为边读边写入有界缓冲 `Capped`（头 + 滚动尾，stdout 32KiB / stderr 16KiB），内存与输出量无关。测试：`runaway_output_is_bounded_and_keeps_both_ends`（2000 万字节）、分块一致性。 |
| M2 | 取消 / drop 时后台子进程残留 | ✅ | 新增 `ProcessGroup` 守卫，任何退出路径（正常结束、超时、future 被 drop）都对整个进程组发 `SIGKILL`；`pgid <= 1` 被拒绝，避免 `kill(-1)`。测试：`dropping_the_call_kills_the_whole_process_group`、`background_processes_are_killed_when_the_command_finishes`。 |
| M3 | 后台进程占住管道，必然超时且丢输出 | ✅ | 命令退出后清理进程组，再给管道 2 秒排空；超时的错误信息附带超时前已捕获的输出。测试：`a_background_process_does_not_hold_the_call_hostage`、`a_timeout_still_reports_the_output_so_far`。 |
| M4 | 密钥黑名单硬编码、不完整 | ✅ | 名单移到 `config::SECRET_ENV_VARS`，与读取这些变量的常量同源；另按名字后缀（`_API_KEY`、`_TOKEN`、`_SECRET`、`_PASSWORD` 等）兜底，规则只看名字**结尾**，故意偏保守（以 `_PASSWORD` 结尾的目录变量也会被清掉）。文档明确说明这**不是安全边界**；`.env` 文件本身由 M7 的路径守卫保护。测试：`secret_env_names_are_recognised_by_the_names_that_carry_credentials`。 |
| M5 | `grep` 不跳过隐藏文件，`.env` 内容会回传给模型 | ✅ | 跳过所有以 `.` 开头的条目，与 `list_files` 一致；显式把隐藏路径作为 `path` 时仍可搜索。路径守卫另外拒绝读 `.env*`。 |
| M6 | `grep` 超时或取消后线程继续扫完 | ✅ | 遍历中检查取消标志，`run` 被 drop 时通过 `CancelOnDrop` 置位。测试：`a_cancelled_search_stops_walking`。 |
| M7 | 写类工具可改写 `mcp.json`、`.agent/`、`.env`、`.git/` | ✅ | `WorkspaceGuardCallback` 区分读写：`.env` / `.env.*` 读写都拒绝；`.git`、`.agent`、`mcp.json` 拒绝写入（读仍允许）。按路径组件、大小写不敏感匹配，只看工作区根以下的部分，名字相近的文件（`environment.md`）不受影响。`unzip_file` 的 `extract_to` 视为写，`zip_path` 视为读。 |
| M8 | 对 `run_command` 选 `always` 等于放开整个 shell | ✅ | `DualApprovalCallback::without_standing_approval(tool)`：「总是允许」只放行当次、不记忆；「总是拒绝」仍然记忆。CLI 对 `run_command` 启用。 |
| M9 | 同步工具超时提示"已放弃"，模型会重试 | ✅ | `execute_with_timeout` 对非 `concurrency_safe` 的工具改用"可能仍在运行或已部分生效，先检查状态再重试"的措辞；只读工具保持原措辞。 |

### 💡 建议

| 编号 | 问题 | 状态 | 修复说明 |
|---|---|---|---|
| L1 | async 里同步 spawn 外部 `kill` | ✅ | 改为直接调用 `libc::kill(-pgid, SIGKILL)`（仅 Unix 依赖 `libc`），不再依赖 PATH，只发一次（`AtomicBool`）。 |
| L2 | Windows 下 `cmd /C` 引号问题 | 🟡 | 改用 `raw_arg` 传命令。**本机是 macOS，Windows 分支未编译验证**。 |
| L3 | 工具自带上限覆盖 `with_tool_timeout`，文档没写清 | ✅ | `execute_with_timeout` 文档写明：工具自带上限双向覆盖调用方设置（可更长也可更短）。 |
| L4 | 先截断再写入，崩溃会留下半截文件 | ✅ | `write_file` / `edit_file` 改为 `write_atomic`：写同目录临时文件再 `rename`；保留已有文件权限；符号链接写穿到目标而不替换链接；失败时清理临时文件。 |
| L5 | 大文件被静默跳过 | ✅ | 结果里追加 `[skipped N file(s) over 2 MiB ...]`。 |
| L6 | 每个文件打开两次、每行分配 `String` | ✅ | 一次打开，用 `fill_buf` 窥视判断二进制，复用行缓冲 `read_until`。 |
| L7 | 新测试抢走了别的测试的文档注释 | ✅ | 注释归位。 |
| L8 | `describe_grep` 丢了 `include` / `case_insensitive` | ✅ | 提示里带上这两个选项，并要求"用同样的参数重跑"。 |
| L9 | gaia 工具集的说法不准确 | ✅ | 注释和文档改为准确描述：固定集合含 `unzip_file`，去掉了 `delete_file`；并说明理由。 |
| L10 | `Conversation` 新增 `pub` 字段是破坏性变更 | 📝 | 仓库内没有任何地方以结构体字面量构造它，所以不影响现有代码。作为库发布时，应在版本说明里标注。**未改动**：改成私有字段会连带改动所有读取方，收益不成比例。 |
| L11 | 测试：固定延时、手写列表过滤、临时文件不清理 | ✅ | `tests/cancellation.rs` 改为事件触发（等请求/工具真的开始）；`shell` 的进程组测试也改为等后台进程写出 pid 后再 abort，不再依赖固定的 500ms；路径守卫回归测试改为"每个内置工具必须被分类"（守卫或路径无关二选一），新工具漏登记会失败；新增 `tools::testing::TempDir`（drop 时清理），shell / grep / file_write / file_edit 的测试改用它。 |

### 其他观察

| 编号 | 问题 | 状态 | 修复说明 |
|---|---|---|---|
| O1 | `--dangerous-tools` 显式指定会静默丢掉新增工具的审批 | ✅ | 启动时对"已启用却不在名单里的默认高危工具"打印警告；README 说明它是替换而非追加。 |
| O2 | `write_file` 的 `content` 留在历史里 | ✅ | 确认是真问题（`Compaction` 只改结果）。现在对不再"新"的调用，`write_file.content` 与 `edit_file.old_string/new_string` 超过 256 字符时替换为"[N chars, M lines, omitted from history]"；键保留，调用形状不变；幂等；未注册的工具不动。 |
| O3 | 摘要请求不受取消控制 | ✅ | before-LLM 钩子改为与取消令牌竞争，取消后停止后续钩子。 |

## 3. 已知限制（修复后仍然存在）

- **Windows 路径未验证**：`raw_arg` 与进程组相关分支只在 Unix 上编译和测试过。
- **同步工具超时**：`read_file`、`delete_file`、`unzip_file` 等已在阻塞线程池里运行的同步工具无法被中断，只能让调用方停止等待。现在超时提示会提醒先检查状态，`grep` 是例外（会检查取消标志）。
- **屏障是按位置而非按路径**：两个不安全的调用即使操作不同文件也不会重叠。这是有意的保守选择，调度器无法判断任意两个工具调用是否冲突。
- **`run_command` 的命令内容无法校验**：路径守卫只能看 `workdir`；`.env*` 保护只覆盖文件工具，shell 里仍可 `cat .env`。真正的把关是人工审批。
- **MCP 的 `readOnlyHint` 取自服务器自述**：只用来放宽并发，不影响审批和路径守卫；写错只会让两个调用重叠，不会绕过任何防护。

## 4. 新增 / 改动的公开接口

| 接口 | 说明 |
|---|---|
| `ToolRegistry::builtin_all()` | 完整内置集合（含写入与执行） |
| `ToolRegistry::with_mcp_over(base, path)` | 在给定注册表上叠加 MCP |
| `Tool::concurrency_safe()` | 默认 `false`；只读工具声明 `true` |
| `simple_tool!` 的 `concurrency_safe = <bool>` | 宏的新可选参数 |
| `DualApprovalCallback::without_standing_approval(tool)` | 某工具的"总是允许"只放行当次 |
| `config::SECRET_ENV_VARS` / `config::is_secret_env_name` | 密钥环境变量名单与按名字判断 |
| `tools::schedule::{plan, CallGate}` | 一轮内调用的排队规则 |
| **行为变更** | `builtin()` 不再含 `write_file` / `edit_file` / `run_command`；`grep` 不再搜索隐藏文件；`write_file` / `edit_file` 改为原子写 |

## 5. 修复清单

```markdown
### 🛑 Critical
- [x] S1 builtin() 收窄，新增 builtin_all()
- [x] S2 副作用工具按调用顺序执行

### ⚠️ Major
- [x] M1 输出限量读取
- [x] M2 drop/取消时终止进程组
- [x] M3 后台进程与超时不丢输出
- [x] M4 密钥过滤复用 config
- [x] M5 grep 跳过隐藏文件
- [x] M6 grep 可取消
- [x] M7 受保护路径
- [x] M8 run_command 禁用 always
- [x] M9 同步工具超时措辞

### 💡 Minor
- [x] L1 L3 L4 L5 L6 L7 L8 L9 L11
- [~] L2 已改，Windows 未验证
- [-] L10 仅记录
- [x] O1 O2 O3
```
