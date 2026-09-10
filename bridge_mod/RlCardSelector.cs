// RlCardSelector.cs -- RL-agent-driven card selector.
//
// Implements ICardSelector (MegaCrit.Sts2.Core.TestSupport) to intercept
// all card selection prompts in the game (card rewards, deck upgrades,
// deck transforms, deck enchants, hand selections, etc.).
//
// When CardSelectCmd has a Selector set, it bypasses the UI screens and
// calls these methods directly. This is the same mechanism AutoSlay uses
// with AutoSlayCardSelector, but we replace random with RL agent decisions.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace STS2BridgeMod;

public class RlCardSelector : ICardSelector
{
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    /// <summary>
    /// Called for deck upgrade, deck transform, deck enchant, hand selection,
    /// and various other card selection prompts.
    /// </summary>
    public async Task<IEnumerable<CardModel>> GetSelectedCards(
        IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        // This prompt is frequently opened from INSIDE another decision --
        // e.g. a potion used during combat, which is exactly the case that
        // used to preempt the combat handler's pending request and abort the
        // run. Mark it as nested so the outer handler can recognise that its
        // own wait is being preempted (and wait instead of failing).
        BridgeServer.EnterNestedDecision();
        try
        {
            return await GetSelectedCardsCore(options, minSelect, maxSelect);
        }
        finally
        {
            BridgeServer.ExitNestedDecision();
        }
    }

    private async Task<IEnumerable<CardModel>> GetSelectedCardsCore(
        IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        List<CardModel> cardList = options.ToList();
        if (cardList.Count == 0)
            return Array.Empty<CardModel>();

        // If there's only one option and we must select at least one, auto-select
        if (cardList.Count <= minSelect)
            return cardList;

        // Build the state message
        var cards = new List<Dictionary<string, object>>();
        for (int i = 0; i < cardList.Count; i++)
        {
            CardModel card = cardList[i];
            var cardData = CardSerialization.SerializeCardFull(card, includeUpgradePreview: true);
            cardData["index"] = i;
            cards.Add(cardData);
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = NonCombatBridgeProtocol.CardSelectState,
            ["cards"] = cards,
            ["min_select"] = minSelect,
            ["max_select"] = maxSelect,
        };

        // Selections are very often opened from INSIDE combat ("exhaust a
        // card", "upgrade a card in your hand", "choose a card to discard",
        // a potion that grants a card). A human deciding there still sees the
        // board, so attach it: without energy / enemies / hand the agent has
        // to choose which card to burn completely blind.
        try
        {
            Dictionary<string, object>? combatContext =
                RlCombatHandler.BuildCombatContext();
            if (combatContext != null)
                stateMsg["combat_context"] = combatContext;
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCardSelector] Combat context unavailable: {ex.Message}");
        }

        // Try to get decision from Python agent
        if (BridgeServer.Instance.IsClientConnected)
        {
            string stateJson = JsonSerializer.Serialize(stateMsg);
            while (true)
            {
                string responseJson = null;
                try
                {
                    using var cts = new CancellationTokenSource(AgentTimeout);
                    responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                        stateJson,
                        AgentTimeout, cts.Token);

                    if (responseJson != null)
                    {
                        return ParseCardSelectResponse(responseJson, cardList, minSelect, maxSelect);
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log($"[RlCardSelector] Agent error: {ex.Message}");
                }

                if (!BridgeServer.AllowRandomFallback)
                {
                    // Human-parity safe mode: never decide for the agent.
                    // Keep waiting while a client is connected; abort the
                    // run (caught by RlAutoSlayer) if the client goes away.
                    if (!BridgeServer.Instance.IsClientConnected)
                        throw new OperationCanceledException(
                            "Agent disconnected during safe-mode card selection");
                    Logger.Log("[RlCardSelector] Safe mode: still waiting for agent selection...");
                    continue;
                }
                break;
            }
        }
        else if (!BridgeServer.AllowRandomFallback)
        {
            throw new OperationCanceledException(
                "Safe mode: no agent connected for card selection");
        }

        // Fallback: select the first legal cards.
        Logger.Log("[RlCardSelector] Falling back to deterministic selection");
        return FallbackSelect(cardList, minSelect, maxSelect);
    }

    /// <summary>
    /// Called specifically for card reward screens (post-combat card picks).
    /// </summary>
    public CardRewardSelection GetSelectedCardReward(
        IReadOnlyList<CardCreationResult> options,
        IReadOnlyList<CardRewardAlternative> alternatives)
    {
        BridgeServer.EnterNestedDecision();
        try
        {
            return GetSelectedCardRewardCore(options, alternatives);
        }
        finally
        {
            BridgeServer.ExitNestedDecision();
        }
    }

    private CardRewardSelection GetSelectedCardRewardCore(
        IReadOnlyList<CardCreationResult> options,
        IReadOnlyList<CardRewardAlternative> alternatives)
    {
        if (options.Count == 0)
            return default;

        // Build the state message
        var cards = new List<Dictionary<string, object>>();
        for (int i = 0; i < options.Count; i++)
        {
            CardModel card = options[i].Card;
            var cardData = CardSerialization.SerializeCardFull(card, includeUpgradePreview: true);
            cardData["index"] = i;
            cards.Add(cardData);
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = NonCombatBridgeProtocol.CardRewardState,
            ["cards"] = cards,
            ["can_skip"] = true,
        };

        // This method is synchronous, so we use a blocking wait
        if (BridgeServer.Instance.IsClientConnected)
        {
            string stateJson = JsonSerializer.Serialize(stateMsg);
            while (true)
            {
                string responseJson = null;
                try
                {
                    using var cts = new CancellationTokenSource(AgentTimeout);
                    responseJson = BridgeServer.Instance.SendStateAndWaitForActionAsync(
                        stateJson,
                        AgentTimeout, cts.Token).GetAwaiter().GetResult();
                }
                catch (Exception ex)
                {
                    Logger.Log($"[RlCardSelector] Agent error for card reward: {ex.Message}");
                }

                if (responseJson != null)
                {
                    using var doc = JsonDocument.Parse(responseJson);
                    var root = doc.RootElement;
                    string action = root.GetProperty("action").GetString() ?? "";

                    if (action == NonCombatBridgeProtocol.SkipAction)
                    {
                        Logger.Log("[RlCardSelector] Agent chose to skip card reward");
                        return default;
                    }

                    if (action == NonCombatBridgeProtocol.ChooseAction &&
                        root.TryGetProperty("index", out var idxProp))
                    {
                        int idx = idxProp.GetInt32();
                        if (idx >= options.Count)
                        {
                            Logger.Log("[RlCardSelector] Agent chose to skip card reward via out-of-range choose");
                            return default;
                        }
                        if (idx >= 0 && idx < options.Count)
                        {
                            Logger.Log(
                                $"[RlCardSelector] Agent chose card reward: {options[idx].Card.Id.Entry}");
                            return new CardRewardSelection { card = options[idx].Card };
                        }
                    }
                }

                if (!BridgeServer.AllowRandomFallback)
                {
                    // Human-parity safe mode: never take a card for the agent.
                    if (!BridgeServer.Instance.IsClientConnected)
                        throw new OperationCanceledException(
                            "Agent disconnected during safe-mode card reward");
                    Logger.Log("[RlCardSelector] Safe mode: still waiting for card reward decision...");
                    continue;
                }
                break;
            }
        }
        else if (!BridgeServer.AllowRandomFallback)
        {
            throw new OperationCanceledException(
                "Safe mode: no agent connected for card reward decision");
        }

        // Fallback: pick the first card
        Logger.Log("[RlCardSelector] Falling back: picking first card reward");
        return new CardRewardSelection { card = options[0].Card };
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private IEnumerable<CardModel> ParseCardSelectResponse(
        string json, List<CardModel> cardList, int minSelect, int maxSelect)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            string action = root.GetProperty("action").GetString() ?? "";

            if (action == NonCombatBridgeProtocol.SkipAction)
            {
                if (minSelect <= 0)
                {
                    Logger.Log("[RlCardSelector] Agent chose to skip");
                    return Array.Empty<CardModel>();
                }
                // Can't skip if min is > 0, fall through
            }

            if (action == NonCombatBridgeProtocol.ChooseAction)
            {
                // Single index
                if (root.TryGetProperty("index", out var idxProp))
                {
                    int idx = idxProp.GetInt32();
                    if (idx >= cardList.Count && minSelect <= 0)
                    {
                        Logger.Log("[RlCardSelector] Agent chose to skip via out-of-range single index");
                        return Array.Empty<CardModel>();
                    }
                    if (idx >= 0 && idx < cardList.Count)
                    {
                        if (minSelect > 1)
                        {
                            Logger.Log("[RlCardSelector] Single index response rejected because min_select > 1");
                        }
                        else
                        {
                            Logger.Log($"[RlCardSelector] Agent chose card: {cardList[idx].Id.Entry}");
                            return new[] { cardList[idx] };
                        }
                    }
                }

                // Multiple indexes
                if (root.TryGetProperty("indexes", out var idxsProp) &&
                    idxsProp.ValueKind == JsonValueKind.Array)
                {
                    var selected = new List<CardModel>();
                    foreach (var elem in idxsProp.EnumerateArray())
                    {
                        int idx = elem.GetInt32();
                        if (idx >= 0 && idx < cardList.Count)
                        {
                            selected.Add(cardList[idx]);
                        }
                    }
                    if (selected.Count == 0 && minSelect <= 0)
                    {
                        Logger.Log("[RlCardSelector] Agent chose to skip via empty indexes array");
                        return Array.Empty<CardModel>();
                    }
                    if (selected.Count >= minSelect && selected.Count <= maxSelect)
                    {
                        Logger.Log(
                            $"[RlCardSelector] Agent chose {selected.Count} cards");
                        return selected;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCardSelector] Error parsing response: {ex.Message}");
        }

        Logger.Log("[RlCardSelector] Invalid response, falling back to deterministic selection");
        return FallbackSelect(cardList, minSelect, maxSelect);
    }

    private static IEnumerable<CardModel> FallbackSelect(
        List<CardModel> cards, int minSelect, int maxSelect)
    {
        int count = Math.Min(maxSelect, cards.Count);
        if (count < minSelect)
            count = Math.Min(minSelect, cards.Count);
        return cards.Take(count);
    }
}
