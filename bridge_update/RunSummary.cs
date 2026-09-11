// RunSummary.cs -- human-visible run/player snapshot attached to every
// non-combat state payload so the agent can see everything a human player
// can see (HP, gold, character, relics, potions, full deck) on every screen.
//
// All tooltip text comes from the game's own localization pipeline
// (Title/DynamicDescription), i.e. exactly what the player sees on hover.

using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

internal static class RunSummary
{
    /// <summary>
    /// Build the FLAT player summary dictionary from the current run state.
    /// Returns null when no run/player is available. The bridge injects this
    /// object directly as the state's "player" field (never nested), so the
    /// Python formatter reads hp/gold/relics/deck at one level.
    /// Mirrors the player fields serialized by RlCombatHandler.
    /// </summary>
    public static Dictionary<string, object>? BuildPlayerSummary()
    {
        try
        {
            RunState runState = RunManager.Instance.DebugOnlyGetState();
            if (runState == null)
                return null;

            var player = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
            if (player == null)
                return null;

            var playerObj = new Dictionary<string, object>
            {
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold,
                ["ascension"] = runState.AscensionLevel,
            };

            // Character identity: visible on every UI screen for a human.
            try
            {
                playerObj["character_id"] = player.Character.Id.Entry;
                playerObj["character_name"] = CardSerialization.CleanText(
                    player.Character.Title.GetFormattedText());
            }
            catch { }

            // Relics: display name, player-visible tooltip, UI counter and
            // spent state. No hidden internal counters.
            var relics = new List<Dictionary<string, object>>();
            foreach (var relic in player.Relics)
            {
                var relicObj = new Dictionary<string, object>
                {
                    ["id"] = relic.Id.Entry,
                };
                try
                {
                    var name = CardSerialization.CleanText(relic.Title.GetFormattedText());
                    var desc = CardSerialization.CleanText(relic.DynamicDescription.GetFormattedText());
                    if (!string.IsNullOrEmpty(name))
                        relicObj["name"] = name;
                    if (!string.IsNullOrEmpty(desc))
                        relicObj["description"] = desc;
                }
                catch { }
                if (relic.StackCount != 0)
                    relicObj["counter"] = relic.StackCount;
                if (relic.IsUsedUp)
                    relicObj["used_up"] = true;
                relics.Add(relicObj);
            }
            playerObj["relics"] = relics;

            // Potion belt: EVERY slot (empty slots included) plus capacity.
            playerObj["potions"] = PotionActions.Serialize(player);
            playerObj["potion_slot_capacity"] = player.PotionSlots.Count;

            // Full deck composition (unordered, grouped by player-visible
            // identity: id + upgraded + enchantment + affliction).
            playerObj["deck"] = PileComposition(player.Deck.Cards);
            playerObj["deck_count"] = player.Deck.Cards.Count;

            return playerObj;
        }
        catch (Exception ex)
        {
            Logger.Log("[RunSummary] Error building summary: " + ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Current act identity + revealed boss(es), or null. Injected at the
    /// STATE top level (not inside the player object).
    /// </summary>
    public static Dictionary<string, object>? BuildActInfo()
    {
        try
        {
            RunState runState = RunManager.Instance.DebugOnlyGetState();
            return runState == null ? null : CardSerialization.SerializeActInfo(runState);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Stable top-level progress fields for every real state.</summary>
    public static Dictionary<string, object>? BuildProgressInfo()
    {
        try
        {
            RunState runState = RunManager.Instance.DebugOnlyGetState();
            if (runState == null)
                return null;
            return new Dictionary<string, object>
            {
                ["floor"] = runState.TotalFloor,
                ["act"] = runState.CurrentActIndex + 1,
            };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Unordered pile composition: grouped by (card id, upgraded,
    /// enchantment, affliction) and sorted so that internal pile order can
    /// never leak through serialization. Same shape as
    /// RlCombatHandler.SerializePileComposition.
    /// </summary>
    private static List<Dictionary<string, object>> PileComposition(
        IEnumerable<MegaCrit.Sts2.Core.Models.CardModel> cards)
    {
        var grouped = new Dictionary<(string Id, bool Up, string Ench, string Aff), int>();
        if (cards != null)
        {
            foreach (var card in cards)
            {
                var key = (
                    card.Id.Entry,
                    card.IsUpgraded,
                    card.Enchantment?.Id.Entry ?? "",
                    card.Affliction?.Id.Entry ?? "");
                grouped[key] = grouped.GetValueOrDefault(key) + 1;
            }
        }

        var result = new List<Dictionary<string, object>>();
        foreach (var key in grouped.Keys.OrderBy(k => k.Id).ThenBy(k => k.Up))
        {
            var entry = new Dictionary<string, object>
            {
                ["id"] = key.Id,
                ["upgraded"] = key.Up,
                ["count"] = grouped[key],
            };
            if (!string.IsNullOrEmpty(key.Ench))
                entry["enchantment"] = key.Ench;
            if (!string.IsNullOrEmpty(key.Aff))
                entry["affliction"] = key.Aff;
            result.Add(entry);
        }
        return result;
    }
}
