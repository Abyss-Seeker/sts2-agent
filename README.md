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
   预算从最旧开始丢弃，单局面文本超限自动截断；
4. 当前局面作为最终 user 消息。

## 日志

每个决策都会写入 `sts2-agent/logs/run_*.jsonl`（可在 UI 中关掉），便于
复盘与提示词迭代。配置持久化在 `sts2-agent/config.json`（含 API Key，请勿
提交到 git）。

## 测试（无游戏也能跑）

```powershell
python smoke_test.py
```

纯逻辑冒烟测试：JSON 提取、动作校验、上下文预算裁剪、战斗/地图状态格式化，
全部不依赖游戏进程。
