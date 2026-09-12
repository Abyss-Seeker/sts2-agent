# sts2-agent — 用 LLM 玩《杀戮尖塔 2》

一个独立于 RL 训练代码的控制台：把本项目底层的游戏桥接（TCP NDJSON 协议，
克隆自 `sts2_env/bridge/`）接到任意 **OpenAI 兼容 LLM API** 上，让没有任何
游戏知识的 LLM 通过 Web UI 实际游玩《杀戮尖塔 2》。

## 架构

```
游戏 (bridge_mod, TCP :9002)
        │  状态 JSON ↓ / 动作 JSON ↑
bridge_client.py        # 底层操控（克隆自 sts2_env/bridge，零依赖）
game_state.py           # 状态 → LLM 可读文本 + 运行级记忆 (RunMemory)
knowledge.py            # 卡牌/权力/意图知识库（可选复用 sts2_env 卡牌注册表）
prompts.py              # 完整规则书 + 输出契约 + 提示词模板（可在 UI 覆盖）
context_manager.py      # 上下文管理：滚动历史 + 字符预算裁剪
llm_client.py           # OpenAI 兼容 /chat/completions（纯标准库）
agent.py                # 决策循环：状态→提示→LLM→校验→重试→执行
server.py + static/     # 本地 Web UI（配置 API/模板、观测决策日志）
```

## 使用步骤

游戏内录屏展示现支持独立 Mod：见 [native_overlay/README.md](native_overlay/README.md)。
角色讲解可独立选择简体中文 / English，不必修改游戏语言。

1. **启动游戏**（需已安装 `bridge_mod`，见仓库根目录 README），确认
   `[BridgeServer]` 监听 `127.0.0.1:9002`。
2. **启动控制台**（在仓库根目录的 `.venv` 中执行，也可用任何 Python ≥3.11）：

   ```powershell
   cd sts2-agent
   python server.py --open
   ```

3. 浏览器打开 `http://127.0.0.1:8770/`：
   - 填入 LLM API 的 **Base URL / API Key / 模型**（OpenAI、DeepSeek、
     Moonshot、Qwen、Ollama、vLLM 等任何 OpenAI 兼容端点均可）；
   - 按需编辑**提示词模板**（默认内置完整规则书，占位符
     `{{RULEBOOK}}`/`{{CONTRACT}}`/`{{RUN_MEMORY}}`/`{{STATE}}`）；
   - 调整**上下文管理**参数；
   - 点击 **▶ 启动 Agent**。
4. 游戏内开一局（Agent 会自动接管地图/战斗/商店/事件/奖励等所有决策），
   在 UI 的"决策日志"中实时观察 LLM 的思考与出牌。

## 决策模式（beta）

### `single_action`（基线 / 默认）
当前最小可用行为：每个状态一次 LLM 调用，每次只执行一个动作。

### `action_chunk`（beta，仅战斗）
一次模型推理可以提交一组（ActionChunk）**已经由当前可见信息完全确定**的动作：
模型用"计划作用域引用"（`h0`/`e0`...）规划，harness 在每个动作执行前针对
**最新权威状态**重新解析引用并校验合法性，再逐个发送；桥接协议保持
一状态一动作握手不变，改变的只是 LLM 调用频率。

> 核心不变量：**每个动作后都会观测游戏状态，但只在认知边界调用 LLM。**
> 一次模型推理可以安全地驱动多个逐个被桥接确认的 STS2 动作；
> harness 负责校验与中断计划，绝不代替模型做策略决策。

计划会在以下**认知边界**被中断并向模型重新询问（全部是协议/信息层规则，
不含任何策略判断）：新回合、手牌出现新卡（抽牌/生成）、目标死亡/丢失、
计划引用无法解析、计划动作非法、药水不可用、动作被游戏拒绝（状态无变化）、
模型显式 `checkpoint_after:true`、屏幕切换。

### benchmark 纯度（`failure_policy`）
- `demo_resilient`（默认）：允许确定性兜底动作继续演示，并清楚记录
  `FALLBACK` 事件；评测有效性指标仍会标记该次兜底。
- `benchmark_strict`：LLM 失败（API 错误/超时/全部解析失败）时
  **不做任何非 LLM 策略兜底**——标记 benchmark 失效并停止 Agent。

### 指标
严格区分四个维度（`/api/status`），且 **single_action 与 action_chunk
使用完全相同的确认语义**（SENT → 下一权威状态 → CONFIRMED/REJECTED）：
- **推理请求**：`llm_request_count`（每一次**真实 HTTP 推理尝试**——由
  `LLMClient.on_http_attempt` 在每次 urlopen 前触发，内部
  `max_retries` 重试逐次计数；含超时/HTTP 错误/thinking-only/坏 JSON）、
  `llm_success_count`、`llm_failed_request_count`、
  `logical_inspection_count`（认知边界数，一次 inspect 可能对应多次
  请求）；
- **游戏动作**：`game_action_sent_count`（已发送）、
  `game_action_confirmed_count`（被下一权威状态确认；`game_action_count`
  为其兼容别名）、`game_action_rejected_count`（可见状态无变化被拒）、
  `game_action_unconfirmable_count`（无法可靠判定——保守地不计
  confirmed）；
- **计划**：完成/中断分开统计——计划是否完成与检查点原因是两个独立维度
  （最后一击把战斗带进 reward_screen 时 reason=SCREEN_CHANGED 但
  plan_completed=true）；`executed_vs_planned_ratio` 只统计 **confirmed**
  的计划动作；
- **成本/延迟**：`prompt_tokens`、`completion_tokens`、
  `reasoning_tokens`、`prompt_cache_hit_tokens`/`prompt_cache_miss_tokens`
  与 `cache_hit_ratio`（官方 DeepSeek usage）、首 token 延迟
  `first_reasoning_token_ms_p50` / `first_content_token_ms_p50`（流式下
  每次调用只记一次）。

核心 KPI：`actions_per_llm_call > 1`（分母 = 每一次真实 HTTP 尝试，含
失败与内部重试），且不增加非法动作率、无隐藏信息。

### DeepSeek 能力（可选）
`reasoning_max_chars` 默认 6000，0 关闭护栏。对支持原生 thinking 的端点，
护栏会启用流式；收到思考增量时检查字符预算及本次时间预算的 60%。
达到预算会关闭原流，用已有分析和完整状态关闭 thinking 请求最终 JSON，
两次请求共用原截止时间。它不是服务端 token 上限，不保证质量完全不变；
无数据到达时仍由 HTTP 截止时间保护。中转需支持原生字段并选 deepseek profile。
`max_tokens` 是思考与答案共用额度，单纯调小可能反而导致无答案。

`thinking_enabled` / `reasoning_effort`（low|high|max）通过
`provider_profile` 控制是否下发：`auto`（默认）**仅当主机名是官方
api.deepseek.com 时**启用，绝不凭模型名猜测（中转上跑
`deepseek-*` 模型仍是普通 OpenAI 兼容端点）；`generic` 永远不发；
`deepseek` 显式强制开启。`stream_mode` 默认 `off`（部分中转的流式重组会
损坏内容），`on`/`auto` 时也只执行**完整**回复，绝不解析流式半截 JSON。

### Delta 观测策略（正确性优先）
`delta_observations` 默认 **关闭**。即使打开，凡涉及新牌/引用失效的检查点
（HAND_CHANGED / NEXT_ACTION_ILLEGAL / CARD_GONE / TARGET_GONE /
POTION_INVALID）以及任何新增手牌信息都直接发送完整状态。其他检查点也会在
delta 后附完整当前状态，避免历史关闭或裁剪后丢失卡面、状态与合法动作信息。

### 信息对等（human parity，保持不变）
模型可见当前公开状态，以及真人反复游玩可以学到的静态知识（包括敌人行为
模式、条件与概率）。不提供 seed、RNG、抽牌堆顺序、未来随机结果、'?' 房间
隐藏类型或隐藏事件结果。静态条件推演不等于读取本局隐藏状态。

## 无游戏基础 LLM 的适配

- **规则书**：系统提示内置一份从零讲起的规则书（能量/格挡/伤害公式/意图/
  地图/构筑/药水/商店），假设 LLM 完全不了解游戏。
- **状态格式化**：桥接的原始 JSON 被转成带索引的结构化文本——手牌标注费用/
  目标/是否可打（`*`），敌人标注意图与预估伤害，权力附白话释义。
- **卡牌知识库**：`knowledge.py` 按需查询模拟器卡牌注册表（费用/伤害/格挡/
  类型），LLM 无需背卡表；`sts2_env` 不可用时自动降级。
- **输出契约**：LLM 只需返回一个严格 JSON（`play`/`end_turn`/`potion`/
  `choose`/`skip`），非法回复会被捕获并带着错误反馈重试（最多 3 次），
  之后回退到安全动作；动作执行前还会做越界/能量/目标校验。
- **安全模式**：连接后自动发送 `set_fallback=false`，游戏端不会随机替打，
  所有结果都是 LLM 自己的决策。

## 上下文管理

LLM API 是无状态的，长局远超窗口，因此每步决策的消息构成为：

1. 固定系统提示（规则书 + 输出契约）；
2. **运行记忆**（`RunMemory`：楼层/HP/金币/遗物/近期事件，每步重新生成，
   永不裁剪）；
3. **滚动历史**：最近 N 个（局面→决策→结果）轮次，按 `max_context_chars`
   预算从最旧开始丢弃；必要的卡面与选项信息不会为了字符预算被整行删除；
4. 当前局面作为最终 user 消息。

## 日志

每个决策都会写入 `sts2-agent/logs/run_*.jsonl`（可在 UI 中关掉），便于
复盘与提示词迭代。配置持久化在 `sts2-agent/config.json`（含 API Key，请勿
提交到 git）。

## 测试（无游戏也能跑）

```powershell
python smoke_test.py            # 纯逻辑冒烟：JSON 提取/动作校验/格式化
python e2e_test.py              # 假桥接 + MockLLM 的完整单动作决策回路
python tests\test_core_runtime.py   # ActionChunk 核心运行时（解析/执行器/检查点）
python tests\test_beta_e2e.py       # action_chunk 验收：一次调用多动作/抽牌检查点/拒绝/严格失败
```
## Enemy Behavior Knowledge

游戏版本、数据同步状态和升级检查清单见 [版本维护记录](docs/GAME_VERSION_MAINTENANCE.md)。游戏升级、重新提取数据或部署桥接后，应同步更新该记录。

`enemy_behavior_knowledge` defaults to `true`. The control panel checkbox
"敌人行为知识（默认开）" saves the same setting. Set it to `false` to stop
adding this reference to model requests; visible intents and powers remain.
Existing configurations inherit the enabled default without editing secrets.

Both single-action and ActionChunk requests, including delta checkpoints and
in-combat selections, receive reference information for living enemy types.
It contains static move sequences, conditional/random branches, cooldowns,
buff/debuff amounts and relevant power implementations. The reference is
included separately from disposable decision history so disabling history
does not remove enemy knowledge. Disabling the feature stops new injection;
it does not erase facts the model previously mentioned in decision history.

The bundled `docs/enemy_behaviors.json` is generated from 121 local decompiled
monster classes using the SDK's Roslyn C# parser. It preserves gameplay method
bodies rather than relying on the older, inaccurate MONSTERS_REFERENCE.md.
It includes source SHA-256 hashes for reproducibility. The model receives
static definitions, never live AI fields or future random rolls. Conditional
probabilities must be normalized over eligible moves; unknown history stays
uncertain. Source-build definitions may differ from the installed game, whose
visible values always take priority. Unrecognized enemies are explicitly
marked as lacking a reference.

Regenerate after updating the decompiled source (.NET 9 SDK required):

```powershell
dotnet run --project tools/enemy_reference -- ../decompiled docs/enemy_behaviors.json
```

Ceremonial Beast has an explicit explanation of the Plow HP threshold,
Strength removal and stun, phase-two cycle, and Ringing's card-play restriction.
Flyconid's explanation distinguishes cooldowns from weights (its opening is
50% Frail Spores / 50% Smash). Other effects retain their static definitions
and conditions rather than inventing a predicted move for the live instance.

## Prompt Refinement (Schema 4)

Default prompts distinguish learned rules from hidden run-specific facts,
evaluate resources across the turn and run, and mark historical proposals as
unconfirmed until execution is observed. Chunk instructions use one consistent
JSON contract. A missing reasoning summary does not invalidate a legal action;
empty-response retries do not suggest ending the turn or constrain internal
reasoning to one sentence.

Custom `user_template` now applies in both request paths. `{{STATE}}` is rendered
once (or appended if omitted); `{{RUN_MEMORY}}` is supported. Recognized old
default system templates migrate automatically; customized templates survive.
Formatting failures stop model requests instead of exposing raw bridge JSON.

Observations include potion usage/effects, merchant entrance context, shop card
details, all visible intent types, and authoritative target-preview indices.
Full-belt potion rewards remain visible but unavailable until a slot is freed.
These bridge-side additions require the rebuilt `STS2BridgeMod.dll` and a game
restart; Python prompt changes require restarting the agent server. Build with
`dotnet build bridge_update/STS2BridgeMod.csproj`. The output is under
`bridge_update/.godot/mono/temp/bin/Debug/`.

Enemy reference generation removes cosmetic calls and unused private helpers,
preserves gameplay conditions and amounts, and shares duplicate power definitions
within a request. No live move identifier is passed to the model.
Automated checks validate formatting, contracts and information boundaries;
they do not establish an improvement in win rate without gameplay comparisons.
