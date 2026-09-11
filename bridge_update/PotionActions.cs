using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

internal static class PotionActions
{
    // Mirror NPotionPopup's actual enabled-button checks, including custom
    // rules (e.g. FoulPotion in merchant / FakeMerchant), not potion-name rules.
    private static readonly Dictionary<PotionModel, GameAction> Pending = new();
    public static bool InSelection => (NPlayerHand.Instance?.IsInCardSelection ?? false)
        || NOverlayStack.Instance?.Peek() is ICardSelector;

    public static bool CanDiscard(PotionModel potion) => !potion.IsQueued
        && !(Pending.TryGetValue(potion, out var action) && !action.CompletionTask.IsCompleted)
        && !potion.Owner.Creature.IsDead && potion.Owner.CanUseOrRemovePotions;

    public static bool CanUse(PotionModel potion)
    {
        if (!CanDiscard(potion) || !potion.PassesCustomUsabilityCheck) return false;
        if (potion.Usage == PotionUsage.AnyTime) return true;
        return potion.Usage == PotionUsage.CombatOnly && CombatManager.Instance.IsInProgress
            && potion.Owner.Creature.CombatState?.CurrentSide == potion.Owner.Creature.Side
            && !InSelection && !CombatManager.Instance.PlayerActionsDisabled;
    }

    public static List<Dictionary<string, object>> Serialize(Player player)
    {
        foreach (var key in Pending.Where(p => p.Value.CompletionTask.IsCompleted).Select(p => p.Key).ToArray())
            Pending.Remove(key);
        return player.PotionSlots.Select((potion, slot) => potion == null
            ? new Dictionary<string, object> { ["slot"] = slot, ["empty"] = true, ["can_use"] = false, ["can_discard"] = false }
            : new Dictionary<string, object>
            {
                ["slot"] = slot, ["id"] = potion.Id.Entry,
                ["name"] = CardSerialization.CleanText(potion.Title.GetFormattedText()),
                ["effect"] = CardSerialization.CleanText(potion.DynamicDescription.GetFormattedText()),
                ["hover_info"] = CardSerialization.SerializeVisibleHoverTips(potion.HoverTips),
                ["usage"] = potion.Usage.ToString(), ["can_use"] = CanUse(potion),
                ["can_discard"] = CanDiscard(potion), ["target"] = potion.TargetType.ToString(),
                ["requires_target"] = potion.TargetType.ToString() == "AnyEnemy",
                ["queued"] = potion.IsQueued || Pending.ContainsKey(potion),
            }).ToList();
    }

    public static async Task<bool> Execute(JsonElement command, CancellationToken ct)
    {
        var player = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        if (player == null || !command.TryGetProperty("slot", out var raw) || !raw.TryGetInt32(out int slot)
            || slot < 0 || slot >= player.PotionSlots.Count) return false;
        var potion = player.PotionSlots[slot];
        if (potion == null) return false;
        bool discard = command.GetProperty("action").GetString() == "discard_potion";
        if (discard ? !CanDiscard(potion) : !CanUse(potion)) return false;
        // Enemy-targeted combat use remains in RlCombatHandler, which resolves
        // current living enemy indices and owns its nested selection lifecycle.
        var target = potion.TargetType.ToString() == "TargetedNoCreature" ? null : player.Creature;
        GameAction action = discard ? new DiscardPotionGameAction(player, (uint)slot, CombatManager.Instance.IsInProgress)
            : new UsePotionAction(potion, target, CombatManager.Instance.IsInProgress);
        bool cancelled = false;
        action.BeforeCancelled += _ => cancelled = true;
        Pending[potion] = action;
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);
        // The native UI allows queueing AnyTime potions during a selection.
        // Let that selection finish rather than deadlocking its pending action.
        if (InSelection) return true;
        await action.CompletionTask.WaitAsync(TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds + 30), ct);
        if (action.Exception != null) throw new InvalidOperationException("Potion action failed", action.Exception);
        return !cancelled;
    }
}
