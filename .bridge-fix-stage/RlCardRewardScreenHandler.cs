// RlCardRewardScreenHandler.cs -- RL-agent-driven card reward screen handler.
//
// This handles the NCardRewardSelectionScreen overlay. When the card selector
// (RlCardSelector) is active, this screen may not appear because CardSelectCmd
// bypasses it. But if it does appear (e.g. due to some code path not using
// Selector), this handler sends the options to Python and clicks the chosen card.
//
// Falls back to random selection if Python is disconnected or times out.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace STS2BridgeMod;

public class RlCardRewardScreenHandler : IScreenHandler, IHandler
{
    private const int InitialSettleDelayMs = 400;
    private const int CloseTimeoutSeconds = 10;
    private static TimeSpan AgentTimeout => TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public Type ScreenType => typeof(NCardRewardSelectionScreen);
    public TimeSpan Timeout => TimeSpan.FromSeconds(BridgeServer.HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[RlCardReward] Card reward screen appeared");
        NCardRewardSelectionScreen screen =
            (NCardRewardSelectionScreen)NOverlayStack.Instance.Peek();
        await Task.Delay(InitialSettleDelayMs, ct);

        List<NCardHolder> holders = UiHelper.FindAll<NCardHolder>(screen);
        if (holders.Count == 0)
        {
            Logger.Log("[RlCardReward] No card holders found");
            return;
        }

        // Build state message
        var cards = new List<Dictionary<string, object>>();
        for (int i = 0; i < holders.Count; i++)
        {
            var card = holders[i].CardModel;
            if (card != null)
            {
                // Full human-visible card face incl. upgrade preview
                // (the reward UI shows candidates with hover previews).
                var cardData = CardSerialization.SerializeCardFull(card, includeUpgradePreview: true);
                cardData["index"] = i;
                cards.Add(cardData);
            }
            else
            {
                cards.Add(new Dictionary<string, object> { ["index"] = i });
            }
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = NonCombatBridgeProtocol.CardRewardState,
            ["cards"] = cards,
            ["can_skip"] = FindSkipButton(screen) != null,
        };

        NCardHolder chosenHolder = null;
        string? actionRequestId = null;

        if (BridgeServer.Instance.IsClientConnected)
        {
            try
            {
                string stateJson = JsonSerializer.Serialize(stateMsg);
                string responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                    stateJson,
                    AgentTimeout, ct);

                if (responseJson != null)
                {
                    using var doc = JsonDocument.Parse(responseJson);
                    var root = doc.RootElement;
                    actionRequestId = root.TryGetProperty("request_id", out var rid) ? rid.GetString() : null;
                    string action = root.GetProperty("action").GetString() ?? "";

                    if (action == NonCombatBridgeProtocol.SkipAction)
                    {
                        Logger.Log("[RlCardReward] Agent chose to skip");
                        await SkipCardRewardAsync(screen, actionRequestId, ct);
                        return;
                    }

                    if (action == NonCombatBridgeProtocol.ChooseAction &&
                        root.TryGetProperty("index", out var idxProp))
                    {
                        int idx = idxProp.GetInt32();
                        if (idx >= holders.Count)
                        {
                            Logger.Log("[RlCardReward] Agent chose to skip via out-of-range choose");
                            await SkipCardRewardAsync(screen, actionRequestId, ct);
                            return;
                        }
                        if (idx >= 0 && idx < holders.Count)
                        {
                            chosenHolder = holders[idx];
                            Logger.Log($"[RlCardReward] Agent chose card at index {idx}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[RlCardReward] Agent error: {ex.Message}");
            }
        }

        // Fallback to random
        if (chosenHolder == null)
        {
            if (!BridgeServer.AllowRandomFallback)
            {
                // Human-parity safe mode: never take a card for the agent.
                // Abort the run (caught by RlAutoSlayer) so the reward is
                // left untouched for a clean evaluation handoff.
                throw new OperationCanceledException(
                    "Safe mode: no agent decision for card reward");
            }
            Logger.Log("[RlCardReward] Falling back to random selection");
            chosenHolder = random.NextItem(holders);
        }

        chosenHolder.EmitSignal(NCardHolder.SignalName.Pressed, chosenHolder);
        await WaitHelper.Until(
            () => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(),
            ct, TimeSpan.FromSeconds(CloseTimeoutSeconds),
            "Card reward screen did not close after selection");
        BridgeServer.Instance.RecordActionResult(actionRequestId, true, "card_reward_selected");
        Logger.Log("[RlCardReward] Card reward screen handled");
    }

    // The UI creates buttons in _extraOptions order. Match the stable option ID,
    // never localized button text or a guessed RewardScreen singleton.
    private static NCardRewardAlternativeButton? FindSkipButton(NCardRewardSelectionScreen screen)
    {
        var options = typeof(NCardRewardSelectionScreen)
            .GetField("_extraOptions", BindingFlags.Instance | BindingFlags.NonPublic)
            ?.GetValue(screen) as IReadOnlyList<CardRewardAlternative>;
        var container = typeof(NCardRewardSelectionScreen)
            .GetField("_rewardAlternativesContainer", BindingFlags.Instance | BindingFlags.NonPublic)
            ?.GetValue(screen) as Control;
        if (options == null || container == null)
            return null;
        var buttons = container.GetChildren().OfType<NCardRewardAlternativeButton>().ToList();
        if (buttons.Count != options.Count)
            return null;
        for (int i = 0; i < options.Count; i++)
            if (string.Equals(options[i].OptionId, "Skip", StringComparison.OrdinalIgnoreCase)
                && buttons[i].IsVisibleInTree())
                return buttons[i];
        return null;
    }

    private static async Task SkipCardRewardAsync(
        NCardRewardSelectionScreen screen, string? requestId, CancellationToken ct)
    {
        var button = FindSkipButton(screen);
        if (button == null)
        {
            BridgeServer.Instance.RecordActionResult(requestId, false, "card_reward_skip_unavailable");
            Logger.Log("[RlCardReward] Skip button unavailable; action rejected");
            return;
        }
        try
        {
            button.EmitSignal(NClickableControl.SignalName.Released, button);
            await WaitHelper.Until(
                () => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(),
                ct, TimeSpan.FromSeconds(CloseTimeoutSeconds),
                "Card reward screen did not close after skip");
            BridgeServer.Instance.RecordActionResult(requestId, true, "card_reward_skipped");
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            BridgeServer.Instance.RecordActionResult(requestId, false, "card_reward_skip_failed");
            Logger.Log($"[RlCardReward] Skip failed: {ex.Message}");
        }
    }
}
