// RlRewardsScreenHandler.cs -- bridge-driven reward screen handler.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public class RlRewardsScreenHandler : IScreenHandler, IHandler
{
    private const int RewardClickSettleDelayMs = 500;
    private const int ProceedCloseTimeoutSeconds = 10;
    private const int ActDisplayIndexOffset = 1;
    private const string ScreenLogName = "NRewardsScreen";
    private const string UnknownRewardLabel = "unknown";
    private const string GoldRewardLabelPrefix = "gold:";
    private static TimeSpan AgentTimeout => TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public Type ScreenType => typeof(NRewardsScreen);
    public TimeSpan Timeout => TimeSpan.FromSeconds(BridgeServer.HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.EnterScreen(ScreenLogName);
        NRewardsScreen screen = AutoSlayer.GetCurrentScreen<NRewardsScreen>();
        HashSet<NRewardButton> attemptedButtons = new();

        while (true)
        {
            ct.ThrowIfCancellationRequested();
            List<NRewardButton> rewardButtons = AvailableRewardButtons(screen, attemptedButtons).ToList();
            NProceedButton? proceedButton = UiHelper.FindFirst<NProceedButton>(screen);
            if (rewardButtons.Count == 0)
            {
                if (proceedButton != null)
                {
                    await ClickProceed(screen, proceedButton, ct);
                }
                AutoSlayLog.ExitScreen(ScreenLogName);
                return;
            }

            RewardScreenChoice choice = await ChooseRewardScreenOption(rewardButtons, proceedButton, random, ct);
            if (choice.ShouldProceed)
            {
                if (proceedButton != null)
                {
                    await ClickProceed(screen, proceedButton, ct);
                }
                AutoSlayLog.ExitScreen(ScreenLogName);
                return;
            }

            NRewardButton rewardButton = choice.Button!;
            attemptedButtons.Add(rewardButton);
            AutoSlayLog.Action("Clicking reward button: " + RewardLabel(rewardButton));
            await UiHelper.Click(rewardButton);
            await Task.Delay(RewardClickSettleDelayMs, ct);

            IOverlayScreen overlayScreen = NOverlayStack.Instance?.Peek();
            if (overlayScreen != null && overlayScreen != screen)
            {
                AutoSlayLog.Action("Child screen opened, returning to drain loop");
                AutoSlayLog.ExitScreen(ScreenLogName);
                return;
            }
        }
    }

    private static IEnumerable<NRewardButton> AvailableRewardButtons(
        NRewardsScreen screen,
        HashSet<NRewardButton> attemptedButtons)
    {
        bool hasPotionSlots = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState())?.HasOpenPotionSlots ?? false;
        return UiHelper.FindAll<NRewardButton>(screen)
            .Where(button => button.IsEnabled)
            .Where(button => !attemptedButtons.Contains(button))
            .Where(button => BridgeServer.Instance.IsClientConnected || !(button.Reward is PotionReward) || hasPotionSlots);
    }

    private static async Task<RewardScreenChoice> ChooseRewardScreenOption(
        List<NRewardButton> rewardButtons,
        NProceedButton? proceedButton,
        Rng random,
        CancellationToken ct)
    {
        if (!BridgeServer.Instance.IsClientConnected)
        {
            return RewardScreenChoice.Pick(random.NextItem(rewardButtons));
        }

        try
        {
            string Snapshot()
            {
                List<Dictionary<string, object>> options = rewardButtons
                    .Select((button, index) => RewardOption(button, index))
                    .ToList();
                if (proceedButton != null && proceedButton.IsEnabled)
                {
                    options.Add(new Dictionary<string, object>
                    {
                        ["index"] = options.Count,
                        ["id"] = NonCombatBridgeProtocol.ProceedAction,
                        ["action"] = NonCombatBridgeProtocol.ProceedAction,
                        ["label"] = NonCombatBridgeProtocol.ProceedAction,
                        ["enabled"] = true,
                    });
                }

                RunState runState = RunManager.Instance.DebugOnlyGetState();
                return JsonSerializer.Serialize(new Dictionary<string, object>
                {
                    ["type"] = NonCombatBridgeProtocol.RewardScreenState,
                    ["options"] = options,
                    ["floor"] = runState.TotalFloor,
                    ["act"] = runState.CurrentActIndex + ActDisplayIndexOffset,
                });
            }
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                Snapshot,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return RewardScreenChoice.Pick(random.NextItem(rewardButtons));
            }

            int chosenIndex = ReadChoiceIndex(responseJson);
            if (chosenIndex >= 0 && chosenIndex < rewardButtons.Count)
            {
                return RewardScreenChoice.Pick(rewardButtons[chosenIndex]);
            }
            if (chosenIndex == rewardButtons.Count && proceedButton != null && proceedButton.IsEnabled)
            {
                return RewardScreenChoice.Proceed();
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlRewards] Agent error: " + ex.Message);
        }

        return RewardScreenChoice.Pick(random.NextItem(rewardButtons));
    }

    private static Dictionary<string, object> RewardOption(NRewardButton button, int index)
    {
        var option = new Dictionary<string, object>
        {
            ["index"] = index,
            ["id"] = RewardLabel(button),
            ["action"] = NonCombatBridgeProtocol.PickRewardAction,
            ["label"] = RewardLabel(button),
            ["description"] = RewardDescription(button),
            ["enabled"] = button.IsEnabled,
        };
        if (button.Reward is PotionReward potionReward && potionReward.Potion is { } potion)
        {
            bool hasSlot = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState())?.HasOpenPotionSlots ?? false;
            option["id"] = potion.Id.Entry;
            option["label"] = CardSerialization.CleanText(potion.Title.GetFormattedText());
            option["effect"] = CardSerialization.CleanText(potion.DynamicDescription.GetFormattedText());
            option["hover_info"] = CardSerialization.SerializeVisibleHoverTips(potion.HoverTips);
            option["requires_empty_potion_slot"] = !hasSlot;
            option["enabled"] = button.IsEnabled && hasSlot;
        }
        return option;
    }

    private static string RewardLabel(NRewardButton button)
    {
        Reward? reward = button.Reward;
        if (reward == null)
        {
            return UnknownRewardLabel;
        }
        if (reward is GoldReward goldReward)
        {
            return $"{GoldRewardLabelPrefix}{goldReward.Amount}";
        }
        return reward.GetType().Name;
    }

    private static string RewardDescription(NRewardButton button)
    {
        try
        {
            return button.Reward?.Description.GetFormattedText() ?? "";
        }
        catch
        {
            return "";
        }
    }

    private static int ReadChoiceIndex(string responseJson)
    {
        using JsonDocument doc = JsonDocument.Parse(responseJson);
        JsonElement root = doc.RootElement;
        string action = root.GetProperty("action").GetString() ?? "";
        if (action == NonCombatBridgeProtocol.ChooseAction &&
            root.TryGetProperty("index", out JsonElement indexProp))
        {
            return indexProp.GetInt32();
        }
        return -1;
    }

    private static async Task ClickProceed(NRewardsScreen screen, NProceedButton proceedButton, CancellationToken ct)
    {
        AutoSlayLog.Action("Clicking proceed");
        await UiHelper.Click(proceedButton);
        await WaitHelper.Until(
            () => !GodotObject.IsInstanceValid(screen)
                || NOverlayStack.Instance?.Peek() != screen
                || (NMapScreen.Instance?.IsOpen ?? false),
            ct,
            TimeSpan.FromSeconds(ProceedCloseTimeoutSeconds),
            "Rewards screen did not close or map did not open after clicking proceed");
    }

    private readonly record struct RewardScreenChoice(bool ShouldProceed, NRewardButton? Button)
    {
        public static RewardScreenChoice Proceed() => new(true, null);

        public static RewardScreenChoice Pick(NRewardButton button) => new(false, button);
    }
}
