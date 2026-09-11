# 事件悬浮信息与药水动作更新

这里保存本次桥接源码变更，原桥接工程位于 `../../bridge_mod`。
构建工程引用其余原有源码，避免复制整个工程；输出仍为 STS2BridgeMod.dll。
本次只改 C#，使用已有桥接 PCK。

- 事件选项传递原生 HoverTips，包含卡牌、遗物、附魔等类型和可见描述。
- 每个决策状态包含完整药水栏、can_use/can_discard/queued；检查当前安装版本的
  CanUseOrRemovePotions、战斗阶段、选卡界面和 PassesCustomUsabilityCheck。
- potion / discard_potion 在合法界面调用游戏原生动作队列，之后重新发状态，
  保留当前房间选择。非战斗目标使用本玩家；污浊药水使用原生无生物目标。
- 商店入口和关闭货架后增加模型决策机会：污浊药水只有货架关闭、商人可见时
  才合法。假商人原来直接点击继续，现先让模型决策。
- 商店购买列表与事件/休息站描述在药水操作后重新构建。

核对依据：当前安装 sts2.dll 中 NPotionPopup、FoulPotion、UsePotionAction；
本地 decompiled 目录的接口版本部分不同，因此最终以当前 DLL 编译和检查为准。

验证：Python 全量 tests、Godot 原生气泡资源预览、两个 C# 工程编译。
未自动开启或消耗真实对局。
