// RlNativeSelectionCoordinator.cs -- headful native-selection driver (§14).
//
// When the game opens a NATIVE selection UI (hand-select mode during
// combat, or a card-selection overlay screen), this coordinator:
//   1. observes the REAL screen (visible card holders only),
//   2. serializes the human-visible options to Python via the bridge,
//   3. waits for the agent's choice,
//   4. CLICKS the real UI holders (and the confirm button for
//      multi-select) so the game's own CardsSelected() / hand-select
//      task resolves NATURALLY.
//
// It never returns CardModels directly, mutates the selection model, or
// completes the game's task manually (§14/§23). The RlCardSelector
// bypass must NOT be installed in headful mode (§27).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public static class RlNativeSelectionCoordinator
{
    private static readonly TimeSpan AgentTimeout =
        TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    // §20: bounded polling, never a 0ms busy loop.
    private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(200);

    private static readonly SemaphoreSlim Gate = new(1, 1);
    private static NPlayerHand? _handSeenInSelectMode;
    private static IOverlayScreen? _overlaySeen;
    private static int _generation;

    /// <summary>
    /// §28: TOPMOST player-actionable modal owns the current decision.
    /// While a native selection is active the underlying combat/room
    /// handlers must SUSPEND production of their own states -- otherwise a
    /// stale combat_action preempts the selection request in the bridge's
    /// single pending slot (this is exactly the Headbutt desync).
    /// </summary>
    public static bool IsSelectionActive => _overlaySeen != null
        || _handSeenInSelectMode != null;

    /// <summary>Monotonic selection id (diagnostics, never LLM-visible).</summary>
    public static int Generation => _generation;

    /// <summary>
    /// §20/§17: the coordinator owns every card-selection screen the agent
    /// must decide on. Detection is TYPE-NAME driven so unknown/new
    /// current-build selection screens (Headbutt discard-pile, potion,
    /// deck upgrade/remove/transform/enchant, simple pick, multi-select)
    /// are covered generically -- never a per-card special case.
    /// </summary>
    public static bool OwnsOverlay(IOverlayScreen screen)
    {
        if (screen is NChooseACardSelectionScreen
            or NSimpleCardSelectScreen
            or NDeckCardSelectScreen)
        {
            return true;
        }
        string name = screen.GetType().Name;
        return name.Contains("CardSelect", StringComparison.OrdinalIgnoreCase)
               || name.Contains("CardSelection", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// §28: explicit ownership query used by RlAutoSlayer's drain path and the
    /// watcher. Equivalent to <see cref="OwnsOverlay"/>; kept as a named alias
    /// for the documented API surface so both paths converge on one owner.
    /// </summary>
    public static bool CanHandle(IOverlayScreen screen)
        => OwnsOverlay(screen);

    /// <summary>
    /// §12: combat/room handlers call this before emitting their own state.
    /// Returns true when the caller must RE-OBSERVE (a selection was served
    /// while it waited) instead of using its stale snapshot.
    /// </summary>
    public static async Task<bool> WaitForSelectionClearAsync(
        string caller, CancellationToken ct)
    {
        if (!IsSelectionActive)
            return false;
        Logger.Log($"[NativeSel] {caller} suspending: native selection owns"
                   + " the current decision");
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(
            BridgeServer.AgentTimeoutSeconds + 30);
        while (IsSelectionActive && DateTime.UtcNow < deadline)
        {
            await Task.Delay(PollInterval, ct);
        }
        bool cleared = !IsSelectionActive;
        Logger.Log($"[NativeSel] {caller} resume"
                   + (cleared ? " (selection cleared)" : " (STILL ACTIVE)"));
        return true;
    }

    /// <summary>
    /// Background watcher (one per active run). Bounded 200ms polling.
    /// </summary>
    public static async Task RunLoopAsync(CancellationToken ct)
    {
        Logger.Log("[NativeSel] Coordinator watcher started (headful)");
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await WatchOnceAsync(ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Logger.Log($"[NativeSel] watcher error: {ex.Message}");
            }
            await Task.Delay(PollInterval, ct);
        }
        Logger.Log("[NativeSel] Coordinator watcher stopped");
    }

    private static async Task WatchOnceAsync(CancellationToken ct)
    {
        // Path 1 (P0, combat): the hand is in a selection mode
        // (Headbutt discard, hand upgrade, potion-triggered hand pick...).
        var hand = NPlayerHand.Instance;
        if (hand != null && hand.CurrentMode != NPlayerHand.Mode.Play)
        {
            if (_handSeenInSelectMode == null)
            {
                await Gate.WaitAsync(ct);
                try
                {
                    if (_handSeenInSelectMode == null)
                    {
                        _handSeenInSelectMode = hand;
                        await HandleHandSelectionAsync(hand, ct);
                        _handSeenInSelectMode = null;
                    }
                }
                finally
                {
                    Gate.Release();
                }
            }
            return;
        }
        _handSeenInSelectMode = null;

        // Path 2: card-selection overlay screens (choose-a-card, simple
        // grid / potion, deck selection). All are driven through their
        // NATIVE holders -- never through ICardSelector (§15).
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay == null || _overlaySeen == overlay)
            return;

        bool claimed =
            overlay is NChooseACardSelectionScreen
                ? await ClaimAsync(overlay,
                    () => HandleChooseACardAsync(
                        (NChooseACardSelectionScreen)overlay, ct), ct)
                : overlay is NSimpleCardSelectScreen
                ? await ClaimAsync(overlay,
                    () => HandleGridSelectionAsync(
                        (Control)(object)overlay, "simple_card_select", ct), ct)
                : overlay is NDeckCardSelectScreen
                ? await ClaimAsync(overlay,
                    () => HandleGridSelectionAsync(
                        (Control)(object)overlay, "deck_card_select", ct), ct)
                : false;
        if (claimed)
            Logger.Log("[NativeSel] screen handled natively");
    }

    /// <summary>
    /// §18: one bridge decision request per screen instance -- the screen is
    /// marked seen before the request and released only after it is handled,
    /// so the 200ms poll can never re-send the same card_select.
    /// </summary>
    private static async Task<bool> ClaimAsync(
        IOverlayScreen overlay, Func<Task> handler, CancellationToken ct)
    {
        _overlaySeen = overlay;
        int gen = ++_generation;
        Logger.Log($"[NativeSel] selection observed id=S{gen}"
                   + $" screen={overlay.GetType().Name}");
        await Gate.WaitAsync(ct);
        try
        {
            await handler();
            // §21: confirm the screen actually closed before relinquishing.
            await WaitForCloseAsync(overlay, gen, ct);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Logger.Log($"[NativeSel] handler error: {ex.Message}");
        }
        finally
        {
            Gate.Release();
            _overlaySeen = null;
        }
        return true;
    }

    private static async Task WaitForCloseAsync(
        IOverlayScreen overlay, int gen, CancellationToken ct)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (DateTime.UtcNow < deadline)
        {
            bool stillOpen = ReferenceEquals(
                NOverlayStack.Instance?.Peek(), overlay);
            if (!stillOpen)
            {
                Logger.Log($"[NativeSel] selection closed id=S{gen}");
                return;
            }
            await Task.Delay(PollInterval, ct);
        }
        Logger.Log($"[NativeSel] WARNING: selection id=S{gen} did NOT close"
                   + " within the bounded window -- diagnostic, no blind"
                   + " re-click");
    }

    /// <summary>
    /// §15: native selection screens ignore input for ~350ms after they open.
    /// Block until at least <paramref name="milliseconds"/> have elapsed so the
    /// first real click is not silently dropped by the game.
    /// </summary>
    private static async Task WaitAtLeastMsAsync(int milliseconds, CancellationToken ct)
    {
        await Task.Delay(milliseconds, ct);
    }

    // ------------------------------------------------------------------
    // Path 3: generic grid selection screens (potion / simple select /
    // deck upgrade-remove-transform-enchant). Supports MULTI-SELECT:
    // click every chosen native holder, then click the REAL confirm
    // button -- never a backend submit (§22).
    // ------------------------------------------------------------------

    private static async Task HandleGridSelectionAsync(
        Control screen, string kind, CancellationToken ct)
    {
        // §15: brief guard so the first click on a freshly-opened simple grid
        // (Headbutt discard, potion pick, deck select) is not dropped.
        await WaitAtLeastMsAsync(300, ct);
        var holders = UiHelper.FindAll<NCardHolder>(screen)
            .Where(h => h.IsVisibleInTree()).ToList();
        if (holders.Count == 0)
        {
            Logger.Log($"[NativeSel] {kind}: no visible holders yet");
            return;
        }

        var options = new List<Dictionary<string, object>>();
        for (int i = 0; i < holders.Count; i++)
        {
            var model = holders[i].CardNode?.Model;
            var data = model != null
                ? CardSerialization.SerializeCardFull(
                    model, includeUpgradePreview: true)
                : new Dictionary<string, object>();
            data["index"] = i;
            options.Add(data);
        }

        bool canConfirm = FindConfirmButton(screen) != null;
        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = "card_select",
            ["selection_kind"] = kind,
            ["min_select"] = 1,
            ["max_select"] = canConfirm ? holders.Count : 1,
            ["requires_confirm"] = canConfirm,
            ["options"] = options,
        };
        Logger.Log($"[NativeSel] sending {kind} ({holders.Count} options,"
                   + $" confirm={canConfirm})");
        string? response = await BridgeServer.Instance
            .SendStateAndWaitForActionAsync(
                JsonSerializer.Serialize(stateMsg), AgentTimeout, ct);

        var picks = ParseChoices(response);
        if (picks.Count == 0)
            picks.Add(0);
        foreach (int pick in picks)
        {
            if (pick < 0 || pick >= holders.Count)
                continue;
            Logger.Log($"[NativeSel] clicking {kind} index {pick}");
            await ClickHolderAsync(holders[pick]);
            await Task.Delay(120, ct);
        }

        NButton? confirm = FindConfirmButton(screen);
        if (confirm != null)
        {
            Logger.Log("[NativeSel] clicking native Confirm button");
            await UiHelper.Click(confirm);
        }
    }

    private static NButton? FindConfirmButton(Control screen)
    {
        // NSimpleCardSelectScreen keeps its confirm as _confirmButton; fall
        // back to any visible NButton whose name mentions confirm.
        NButton? reflect = screen.GetType()
            .GetField("_confirmButton",
                System.Reflection.BindingFlags.NonPublic
                | System.Reflection.BindingFlags.Instance)
            ?.GetValue(screen) as NButton;
        if (reflect != null && reflect.IsVisibleInTree())
            return reflect;
        return UiHelper.FindAll<NButton>(screen).FirstOrDefault(
            b => b.IsVisibleInTree()
                 && (b.Name?.ToString() ?? "")
                     .Contains("onfirm", StringComparison.OrdinalIgnoreCase));
    }

    private static List<int> ParseChoices(string? response)
    {
        var picks = new List<int>();
        if (string.IsNullOrWhiteSpace(response))
            return picks;
        try
        {
            using var doc = JsonDocument.Parse(response);
            var root = doc.RootElement;
            if (root.TryGetProperty("indexes", out var arr)
                && arr.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in arr.EnumerateArray())
                {
                    if (item.TryGetInt32(out int i))
                        picks.Add(i);
                }
                if (picks.Count > 0)
                    return picks;
            }
            if (root.TryGetProperty("index", out var idx)
                && idx.TryGetInt32(out int single))
                picks.Add(single);
        }
        catch (Exception ex)
        {
            Logger.Log($"[NativeSel] response parse failed: {ex.Message}");
        }
        return picks;
    }

    // ------------------------------------------------------------------
    // Path 1: combat hand selection (Headbutt discard, hand upgrade, ...)
    // ------------------------------------------------------------------

    private static async Task HandleHandSelectionAsync(
        NPlayerHand hand, CancellationToken ct)
    {
        Logger.Log("[NativeSel] hand selection mode detected");
        var holders = hand.ActiveHolders.ToList();
        var cards = holders
            .Where(h => h.CardNode?.Model != null)
            .Select(h => (holder: h, model: h.CardNode!.Model))
            .ToList();
        if (cards.Count == 0)
        {
            Logger.Log("[NativeSel] no visible hand cards; waiting for UI");
            return;
        }

        var options = new List<Dictionary<string, object>>();
        for (int i = 0; i < cards.Count; i++)
        {
            var data = CardSerialization.SerializeCardFull(
                cards[i].model, includeUpgradePreview: true);
            data["index"] = i;
            options.Add(data);
        }

        var combat = BuildCombatContext();
        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = "card_select",
            ["selection_kind"] = "hand_select",
            ["min_select"] = 1,
            ["max_select"] = 1,
            ["options"] = options,
            ["combat_context"] = combat,
        };

        Logger.Log($"[NativeSel] sending card_select ({cards.Count} options)");
        string? response = await BridgeServer.Instance
            .SendStateAndWaitForActionAsync(
                JsonSerializer.Serialize(stateMsg), AgentTimeout, ct);

        int index = ParseChoice(response);
        if (index < 0 || index >= cards.Count)
        {
            // No valid answer: click the first visible card so the game's
            // own hand-select resolves (a human must click something too).
            Logger.Log("[NativeSel] no valid choice; clicking first card");
            index = 0;
        }

        var holder = cards[index].holder;
        Logger.Log($"[NativeSel] clicking hand card index {index}");
        await ClickHolderAsync(holder);
        // Single-select hand modes resolve as soon as one card is clicked.
    }

    // ------------------------------------------------------------------
    // Path 2: NChooseACardSelectionScreen (generated-card / discover UI)
    // ------------------------------------------------------------------

    private static async Task HandleChooseACardAsync(
        NChooseACardSelectionScreen screen, CancellationToken ct)
    {
        Logger.Log("[NativeSel] choose-a-card overlay detected");
        // §15: respect the ~350ms native input guard before clicking.
        await WaitAtLeastMsAsync(400, ct);
        // The screen itself implements ICardSelector; its visible holders
        // are the native click targets. We drive the screen through the
        // bridge WITHOUT invoking its selector logic.
        var holders = UiHelper.FindAll<NCardHolder>(screen)
            .Where(h => h.IsVisibleInTree()).ToList();
        if (holders.Count == 0)
        {
            Logger.Log("[NativeSel] no visible card holders yet");
            return;
        }

        var options = new List<Dictionary<string, object>>();
        for (int i = 0; i < holders.Count; i++)
        {
            var model = holders[i].CardNode?.Model;
            var data = model != null
                ? CardSerialization.SerializeCardFull(
                    model, includeUpgradePreview: true)
                : new Dictionary<string, object>();
            data["index"] = i;
            options.Add(data);
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = "card_select",
            ["selection_kind"] = "choose_a_card",
            ["min_select"] = 1,
            ["max_select"] = 1,
            ["options"] = options,
        };
        Logger.Log($"[NativeSel] sending choose_a_card ({holders.Count} options)");
        string? response = await BridgeServer.Instance
            .SendStateAndWaitForActionAsync(
                JsonSerializer.Serialize(stateMsg), AgentTimeout, ct);

        int index = ParseChoice(response);
        if (index < 0 || index >= holders.Count)
            index = 0;
        Logger.Log($"[NativeSel] clicking overlay card index {index}");
        await ClickHolderAsync(holders[index]);
    }

    /// <summary>
    /// §23: click the REAL UI control. NCardHolder routes clicks through
    /// its private _hitbox (an NClickableControl), so we drive THAT node's
    /// ForceClick instead of emitting signals directly.
    /// </summary>
    private static async Task ClickHolderAsync(NCardHolder holder)
    {
        var hitbox = holder.GetType()
            .GetField("_hitbox",
                System.Reflection.BindingFlags.NonPublic
                | System.Reflection.BindingFlags.Instance)
            ?.GetValue(holder) as NClickableControl;
        if (hitbox != null)
        {
            await UiHelper.Click(hitbox);
            return;
        }
        // Fallback: the holder's own Pressed signal (same subscriber the
        // game uses when a human clicks).
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
        await Task.Delay(100);
    }

    private static int ParseChoice(string? response)
    {
        if (string.IsNullOrWhiteSpace(response))
            return -1;
        try
        {
            using var doc = JsonDocument.Parse(response);
            if (doc.RootElement.TryGetProperty("index", out var idx)
                && idx.TryGetInt32(out int i))
                return i;
            if (doc.RootElement.TryGetProperty("indexes", out var arr)
                && arr.ValueKind == JsonValueKind.Array
                && arr.GetArrayLength() > 0
                && arr[0].TryGetInt32(out int first))
                return first;
        }
        catch (Exception ex)
        {
            Logger.Log($"[NativeSel] response parse failed: {ex.Message}");
        }
        return -1;
    }

    /// <summary>
    /// §22: only what a human can still see/inspect on the combat screen.
    /// </summary>
    private static Dictionary<string, object> BuildCombatContext()
    {
        var combat = new Dictionary<string, object>();
        try
        {
            var runState = RunManager.Instance.DebugOnlyGetState();
            combat["in_combat"] = true;
            combat["act_floor"] = runState.ActFloor;
            var player = LocalContext.GetMe(runState);
            combat["energy"] = player?.PlayerCombatState?.Energy;
            combat["hand"] = PileType.Hand.GetPile(player).Cards
                .Select(c => CardSerialization.SerializeCardFull(
                    c, includeUpgradePreview: false))
                .ToList();
        }
        catch (Exception ex)
        {
            combat["serialize_error"] = ex.Message;
        }
        return combat;
    }
}
