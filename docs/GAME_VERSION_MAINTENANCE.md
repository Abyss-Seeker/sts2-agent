# STS2 版本与数据维护记录

本文件记录本机游戏、Agent 知识和桥接的同步状态。每次游戏升级、重新提取数据或部署桥接后更新本文件；未检查的项目应保留为“待核实”，不能仅修改版本号就宣称兼容。

## 当前基线

核实日期：2026-09-12（Asia/Shanghai）。这是本机安装版本，不代表官方最新版本。

- 游戏版本：`v0.111.0`。
- 游戏提交：`41cef1ea`。
- 发布信息时间：`2026-08-13T17:39:18-07:00`。
- Steam App ID：`2868840`；Build ID：`24724944`；订阅分支：`public-beta`。
- `release_info.json` 内的 branch：`v0.111.0`；main_assembly_hash：`222455745`（游戏自己的字段，不是 SHA-256）。
- 安装目录：`E:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2`。
- 版本证据：安装目录的 `release_info.json`、Steam 的 `appmanifest_2868840.acf`。
- 当前 `data_sts2_windows_x86_64/sts2.dll` SHA-256：`0861BFA1DF347538D932F22D580E75420F08082792EB914E53B4882764ACDBE9`。

## 数据与部署同步状态

- `docs/enemy_behaviors.json`：schema 1，121 种敌人，记录 184 个来源文件哈希；本次核对全部匹配 `../decompiled` 中的对应源码。
- 敌人知识 JSON SHA-256：`c5aadb5fdbdab6f137d006f398d684d7867c891c1ac6ec066aac283f08645c88`。
- **待核实：反编译源码与当前安装 DLL 的对应关系。** 来源文件哈希匹配只证明知识库与这些源码一致；当前记录没有建立反编译输入 DLL 的版本证据链。
- **待部署核实：游戏 mods 内的桥接 DLL 与本地最新构建不同。** 已安装 SHA-256：`62D0B897C1FF449B85B85409B88DAA9D49426A3ED1DA535B58450D37DB9F3F13`；本地构建 SHA-256：`639E35FFE1479421F13FFE9F528CF4291531AA0920FC9D7FA2EB274FC898FF2E`。不能把源码已修改、构建已通过当作游戏已加载新版。
- 当前验证记录：Python 196 项测试及 4 项子测试通过；桥接编译通过。尚未通过新版本实战验证完整兼容性或胜率。
- 目前没有一键更新流程，也没有版本变化后自动停用手写解释的机制。

## 版本变化后的检查清单

版本号、Build ID 或 DLL 哈希任一变化，都需要重新评估。即使版本号相同，DLL 哈希变化也不能跳过检查。

### 每次更新都检查

- [ ] 记录新版本、分支、Build ID、DLL SHA-256 和检查日期，保留旧基线及差异摘要。
- [ ] 从本次安装 DLL 重新反编译到独立目录，记录输入 DLL 哈希和反编译工具版本；不要把新旧源码混在一起。
- [ ] 对比怪物、卡牌、药水、遗物、状态效果及公共战斗机制的变更范围。
- [ ] 重新生成敌人参考；检查新增、删除、改名和未识别的敌人 ID，并核对来源文件哈希。
- [ ] 检查 `enemy_knowledge.py` 的通用解释与手写 `NOTES`，以及 `knowledge.py` 的静态兜底和 `prompts.py` 规则书。
- [ ] 用新游戏程序集编译 `bridge_update/STS2BridgeMod.csproj`；运行相关测试，再完成下述游戏内验证。
- [ ] 部署后核对桥接 DLL 哈希并重启游戏与 Agent 服务，确保已加载新 DLL、模块和缓存中的知识。
- [ ] 开启控制台“显示完整发送消息”，抽查模型实际收到的状态、效果、敌人参考及重试内容。
- [ ] 更新本文件的同步状态、验证结果和未解决事项；只有检查完成的部分才标为已同步。

### 1. 数值变化

检查伤害、次数、能量、力量增长、状态层数、阈值、难度差异及概率权重。当前可见数值优先，但这不会自动修复静态未来行为中的旧数值。

涉及文件：敌人参考、`enemy_knowledge.py`、`knowledge.py`、`game_state.py`、桥接的卡牌和药水序列化。确认目标预览和 intent 伤害是否已包含修正，避免提示词引导模型重复计算。

### 2. 机制重做

检查行动状态机、阶段切换、触发时机、目标范围、冷却与重复限制、状态效果生命周期，以及公共随机分支 API 的语义。仅重新提取方法正文或编译成功不足以证明解释正确。

当前特别需要人工复核的手写解释：

- `CEREMONIAL_BEAST`：Plow 的 HP 阈值与受伤触发、清除 Strength、眩晕及阶段切换、后续行动循环、Strength 增长，以及 Ringing 的施加范围和出牌限制。
- `FLYCONID`：开场概率、后续可选集合、冷却与 CannotRepeat，以及 Vulnerable/Frail 的效果。
- `enemy_knowledge.py` 的 `POLICY`：概率归一化、`AddBranch` 参数含义、难度阈值和通用效果解释。

上述解释目前是手写常量，不会随 JSON 生成自动更新。若相关实现已变化但解释尚未验证，应修正或暂时移除相应解释；若整个参考来源不确定，可暂时关闭 `enemy_behavior_knowledge`。源码哈希自动失效机制尚未实现。

### 3. 新增或删除内容

检查生成器是否发现新怪，以及新怪引用的 power、辅助方法和公共机制是否完整提取。覆盖数量增加不等于机制完整。对缺少参考的 ID 保持明确“未知”，不编造概率。

新卡牌、药水、遗物、状态、目标类型和界面同样需要检查：是否能格式化真实效果、是否需要更新静态兜底、动作校验、索引映射或桥接处理器。

### 游戏内最小验证

- [ ] 普通战斗与多 intent 敌人：伤害、强化、debuff 和目标预览正确。
- [ ] 受影响的敌人：至少验证改动涉及的阶段或触发条件；未覆盖分支明确记录。
- [ ] 抽牌、选牌、回合切换：ActionChunk 检查点、当前引用和执行确认正常。
- [ ] 满药水栏奖励及商店：效果可见，腾出槽位后可领取或购买。
- [ ] 上下文没有 seed、RNG、隐藏当前 AI 状态或实际未来随机结果；允许静态模式和基于公开历史的概率推演。

## 现有维护命令

在仓库根目录运行。生成命令的输入必须是本次已核实版本的反编译目录；下面的 `../decompiled` 仅是当前约定路径。

```powershell
Get-Content 'E:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2/release_info.json'
Get-FileHash 'E:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2/data_sts2_windows_x86_64/sts2.dll' -Algorithm SHA256
dotnet run --project tools/enemy_reference -- ../decompiled docs/enemy_behaviors.json
dotnet build bridge_update/STS2BridgeMod.csproj
..\.venv\Scripts\python.exe -c "import sys,pytest; sys.argv=['pytest']; raise SystemExit(pytest.main(['tests','-q','-p','no:cacheprovider']))"
```

生成器需要 .NET 9 SDK；桥接构建还需要可用的 Godot SDK/NuGet 缓存及项目指定的游戏程序集目录。这些命令不会自动完成反编译、部署或游戏内验证。

## 维护历史

- 2026-09-12：建立 v0.111.0 / Build 24724944 基线；核实敌人参考与 184 个本地源码文件一致；记录源码到安装 DLL 的来源关系待核实、桥接部署不同步，以及手写机制解释需要人工复核。
