// RlCombatHandler.cs -- RL-agent-driven combat handler.
//
// Replaces AutoSlay's CombatRoomHandler. Instead of applying god-mode buffs
// and playing random cards, this handler:
//   1. Waits for combat to start and the play phase
//   2. Serializes the combat state to JSON
//   3. Sends state to the Python RL agent via BridgeServer
//   4. Waits for the agent's response (play card or end turn)
//   5. Executes the action using CardCmd.AutoPlay or PlayerCmd.EndTurn
//   6. Loops until combat ends
//
// If the Python agent is not connected or times out, falls back to random play.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public class RlCombatHandler : IRoomHandler, IHandler
{
    private static TimeSpan AgentTimeout => TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);
    private const int MaxRlHandSlots = 10;
    // A potion / card can open a nested card selection, which preempts this
    // handler's pending request. Cap how often one turn may be re-queried
    // for that reason so a pathological loop can never spin forever.
    private const int MaxPreemptedWaits = 8;

    public RoomType[] HandledTypes => new RoomType[]
    {
        RoomType.Monster, RoomType.Elite, RoomType.Boss
    };

    public TimeSpan Timeout => TimeSpan.FromMinutes(10);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[RlCombat] Waiting for combat to start");
        // AutoSlayConfig.watchdogTimeout is only 30s, while a single agent
        // decision may legitimately take up to AgentTimeoutSeconds (90s+ for
        // slow reasoning models). WaitHelper.Until() calls Watchdog.Check() on
        // EVERY poll, so refresh the watchdog from the condition delegate --
        // it is evaluated BEFORE each Check -- otherwise a long but perfectly
        // healthy turn gets killed as "stuck".
        //
        // TRANSITION INVARIANT (resume point): a saved run can resume with
        // CurrentRoom = a combat room whose combat ALREADY finished
        // (NCombatRoom created with mode=FinishedCombat, rewards still on
        // screen). Waiting for "combat to start" there can never succeed and
        // used to burn all recovery attempts until the run aborted with
        // "Combat not started" -> run_complete(terminated). If the combat is
        // not running but the rewards screen / map is visible, the room is
        // already finished: skip combat handling entirely and let the run
        // loop continue with rewards / drain / map navigation.
        await WaitHelper.Until(
            delegate {
                RlAutoSlayer.CurrentWatchdog?.Reset("Waiting for combat to start");
                if (CombatManager.Instance.IsInProgress)
                    return true;
                if (NOverlayStack.Instance?.Peek() is NRewardsScreen
                    || (NMapScreen.Instance?.IsOpen ?? false))
                {
                    return true;
                }
                // §31/§32: route from the CURRENT runtime state, not from the
                // room type. A resumed Elite may already be FinishedCombat
                // (room.IsPreFinished / every enemy dead) while rewards and
                // map have not opened yet -- waiting for "combat to start"
                // there can never succeed and burned all recovery attempts.
                if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom
                        is CombatRoom combatRoom)
                {
                    if (combatRoom.IsPreFinished)
                        return true;
                    var enemies = combatRoom.CombatState.Enemies.ToList();
                    if (enemies.Count > 0 && enemies.All(e => !e.IsAlive))
                        return true;
                }
                return false;
            },
            ct, AutoSlayConfig.nodeWaitTimeout, "Combat not started");

        if (!CombatManager.Instance.IsInProgress)
        {
            Logger.Log(
                "[RlCombat] Combat already finished at this resume point"
                + " (rewards/map visible); skipping combat handling so the"
                + " run loop continues with rewards and map navigation");
            return;
        }

        Logger.Log("[RlCombat] Combat started");
        Player player = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());

        int turnCount = 0;
        while (CombatManager.Instance.IsInProgress && turnCount < 200)
        {
            ct.ThrowIfCancellationRequested();
            turnCount++;

            // Wait for play phase
            // THIS is where runs used to die: the watchdog had not been reset
            // since the turn began -- several LLM round-trips (and tens of
            // seconds) earlier -- so the very first Check() inside Until()
            // threw AutoSlayTimeoutException and terminated the run
            // immediately after end_turn, with no error on the agent side.
            await WaitHelper.Until(
                delegate {
                    RlAutoSlayer.CurrentWatchdog?.Reset(
                        $"Combat turn {turnCount}: waiting for play phase");
                    return player.PlayerCombatState?.Phase == PlayerTurnPhase.Play
                           || !CombatManager.Instance.IsInProgress;
                },
                ct, TimeSpan.FromSeconds(30), "Play phase not started");

            if (!CombatManager.Instance.IsInProgress)
                break;

            RlAutoSlayer.CurrentWatchdog?.Reset($"Combat turn {turnCount}");
            Logger.Log($"[RlCombat] Turn {turnCount}: awaiting agent decision");

            int cardsPlayed = 0;
            int preemptedWaits = 0;
            bool turnEnded = false;

            while (!turnEnded && cardsPlayed < 50 && player.PlayerCombatState?.Phase == PlayerTurnPhase.Play)
            {
                ct.ThrowIfCancellationRequested();

                if (cardsPlayed > 0 && cardsPlayed % 10 == 0)
                {
                    RlAutoSlayer.CurrentWatchdog?.Reset(
                        $"Combat turn {turnCount}, played {cardsPlayed} cards");
                }

                // §12/§28: a native modal selection (Headbutt discard
                // pick, potion pick, ...) owns the current decision while
                // the originating command is still awaiting CardsSelected().
                // Suspend combat-state production until it clears, then
                // RE-OBSERVE -- never emit a stale combat_action.
                if (await RlNativeSelectionCoordinator
                        .WaitForSelectionClearAsync("RlCombat", ct))
                {
                    if (!CombatManager.Instance.IsInProgress)
                        break;
                    player = LocalContext.GetMe(
                        RunManager.Instance.DebugOnlyGetState());
                    Logger.Log("[RlCombat] re-observed combat after"
                               + " native selection resolved");
                }

                // Serialize combat state
                string stateJson = SerializeCombatState(player);

                // Send to Python and wait for response
                string responseJson = null;
                bool clientConnected = BridgeServer.Instance.IsClientConnected;
                Logger.Log($"[RlCombat] Client connected: {clientConnected}, sending state ({stateJson.Length} bytes)");
                if (clientConnected)
                {
                    try
                    {
                        Logger.Log("[RlCombat] State sent, waiting for agent response...");
                        responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                            () => SerializeCombatState(player),
                            AgentTimeout, ct);
                        Logger.Log($"[RlCombat] Agent response: {responseJson ?? "null"}");
                    }
                    catch (Exception ex)
                    {
                        Logger.Log($"[RlCombat] Agent communication error: {ex.Message}");
                    }
                }

                // A nested card selection (opened by the potion / card we just
                // used) preempts this wait: it takes over the single pending
                // request slot and cancels ours, after which the agent's reply
                // to THIS request is dropped as stale because its request_id
                // no longer matches. That is NOT an agent timeout -- waiting
                // for the nested prompt and re-querying with a freshly
                // serialized (already settled) state is the correct response.
                if (responseJson == null && BridgeServer.HasNestedDecision
                    && preemptedWaits < MaxPreemptedWaits)
                {
                    preemptedWaits++;
                    Logger.Log(
                        $"[RlCombat] Combat request preempted by a nested card"
                        + $" selection ({preemptedWaits}/{MaxPreemptedWaits});"
                        + " waiting for it to settle before re-querying.");
                    await WaitForNoNestedDecisionAsync("preempted combat wait", ct);
                    continue;
                }

                // The agent just answered -- that IS progress. Refresh the
                // watchdog so any waiting below starts from now instead of
                // from the start of a turn that may have already spanned
                // several multi-second LLM round-trips.
                RlAutoSlayer.CurrentWatchdog?.Reset(
                    $"Combat turn {turnCount}: agent responded");

                // Parse and execute the response, or fall back to random
                if (responseJson != null)
                {
                    AgentActionOutcome outcome = await ExecuteAgentAction(
                        responseJson, player, random, ct);
                    turnEnded = outcome.TurnEnded;
                    RecordAgentActionOutcome(responseJson, outcome);
                }
                else if (!BridgeServer.AllowRandomFallback)
                {
                    // Human-parity safe mode: never decide for the agent.
                    // Terminate the episode instead of playing random cards,
                    // so evaluation outcomes contain only agent decisions.
                    Logger.Log("[RlCombat] Agent did not respond and random fallback "
                        + "is disabled (human-parity safe mode). Aborting episode.");
                    throw new OperationCanceledException(
                        "Agent response timeout with random fallback disabled");
                }
                else
                {
                    Logger.Log("[RlCombat] No agent response, falling back to random");
                    turnEnded = await PlayRandomFallback(player, random, ct);
                }

                if (!turnEnded)
                    cardsPlayed++;

                await Task.Delay(100, ct);
            }

            // If we ran out of cards to play without ending turn, end it
            if (player.PlayerCombatState?.Phase == PlayerTurnPhase.Play && CombatManager.Instance.IsInProgress && !turnEnded)
            {
                PlayerCmd.EndTurn(player, canBackOut: false);
            }
        }

        await WaitHelper.Until(
            delegate {
                RlAutoSlayer.CurrentWatchdog?.Reset("Waiting for combat to end");
                return !CombatManager.Instance.IsInProgress;
            },
            ct, TimeSpan.FromSeconds(30), "Combat did not end");
        Logger.Log("[RlCombat] Combat finished");
    }

    /// <summary>
    /// AUTHORITATIVE game-side outcome of one agent gameplay command.
    /// ``Accepted`` is the ONLY acceptance signal Python may use (a successful
    /// TCP send proves nothing). ``Reason`` is diagnostic text.
    /// </summary>
    private readonly record struct AgentActionOutcome(
        bool TurnEnded, bool Accepted, string Reason);

    /// <summary>
    /// Execute an action from the Python agent response JSON and report
    /// whether the game-side handler ACCEPTED it.
    /// </summary>
    private async Task<AgentActionOutcome> ExecuteAgentAction(
        string json, Player player, Rng random, CancellationToken ct)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            string action = root.GetProperty("action").GetString() ?? "";

            switch (action.ToLowerInvariant())
            {
                case "play":
                {
                    int cardIndex = root.GetProperty("card_index").GetInt32();
                    int targetIndex = root.TryGetProperty("target_index", out var ti)
                        ? ti.GetInt32() : -1;

                    if (cardIndex >= MaxRlHandSlots)
                    {
                        int potionSlot = cardIndex - MaxRlHandSlots;
                        Logger.Log($"[RlCombat] Using potion slot {potionSlot} -> target_index {targetIndex}");
                        bool potionOk = await UsePotionAndWaitAsync(
                            player, potionSlot, targetIndex, ct);
                        return new AgentActionOutcome(
                            false, potionOk,
                            potionOk ? $"used potion slot {potionSlot}"
                                     : $"potion slot {potionSlot} not usable");
                    }

                    CardPile hand = PileType.Hand.GetPile(player);
                    if (cardIndex < 0 || cardIndex >= hand.Cards.Count)
                    {
                        Logger.Log($"[RlCombat] Invalid card_index {cardIndex}, hand size {hand.Cards.Count}");
                        return new AgentActionOutcome(
                            false, false, $"invalid card_index {cardIndex}");
                    }

                    CardModel card = hand.Cards[cardIndex];

                    UnplayableReason reason;
                    AbstractModel preventer;
                    if (!card.CanPlay(out reason, out preventer))
                    {
                        Logger.Log($"[RlCombat] Card {card.Id.Entry} not playable: {reason}");
                        return new AgentActionOutcome(
                            false, false, $"card {card.Id.Entry} not playable: {reason}");
                    }

                    Creature target = ResolveTarget(card, targetIndex);
                    if (card.TargetType == TargetType.AnyEnemy && target == null)
                    {
                        Logger.Log($"[RlCombat] Invalid target_index {targetIndex} for {card.Id.Entry}");
                        return new AgentActionOutcome(
                            false, false,
                            $"invalid target_index {targetIndex} for {card.Id.Entry}");
                    }
                    Logger.Log($"[RlCombat] Playing card: {card.Id.Entry} -> target_index {targetIndex}");

                    await PlayCardAndWaitAsync(player, card, target, ct);
                    return new AgentActionOutcome(
                        false, true, $"played card {card.Id.Entry}");
                }

                case "end_turn":
                {
                    Logger.Log("[RlCombat] Agent chose to end turn");
                    PlayerCmd.EndTurn(player, canBackOut: false);
                    return new AgentActionOutcome(true, true, "ended turn");
                }

                case "potion":
                {
                    int slot = root.GetProperty("slot").GetInt32();
                    int targetIndex = root.TryGetProperty("target_index", out var ti)
                        ? ti.GetInt32() : -1;
                    Logger.Log($"[RlCombat] Using potion slot {slot} -> target_index {targetIndex}");
                    bool potionOk = await UsePotionAndWaitAsync(
                        player, slot, targetIndex, ct);
                    return new AgentActionOutcome(
                        false, potionOk,
                        potionOk ? $"used potion slot {slot}"
                                 : $"potion slot {slot} not usable");
                }

                default:
                    Logger.Log($"[RlCombat] Unknown action: {action}");
                    return new AgentActionOutcome(
                        false, false, $"unknown action {action}");
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCombat] Error executing agent action: {ex.Message}");
            return new AgentActionOutcome(
                false, false, $"error: {ex.Message}");
        }
    }

    /// <summary>
    /// Publish the game-side outcome of the command that answered the current
    /// state, correlated by that state's request_id, so BridgeServer can
    /// attach it to the NEXT authoritative state. Never inferred from the
    /// Python socket write.
    /// </summary>
    private static void RecordAgentActionOutcome(
        string responseJson, AgentActionOutcome outcome)
    {
        string requestId = "";
        try
        {
            using var doc = JsonDocument.Parse(responseJson);
            if (doc.RootElement.TryGetProperty("request_id", out var rid))
                requestId = rid.GetString() ?? "";
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCombat] Could not read request_id for action"
                       + $" outcome: {ex.Message}");
        }
        BridgeServer.Instance.RecordActionResult(
            requestId, outcome.Accepted, outcome.Reason);
    }

    /// <summary>
    /// Fallback: play a random playable card, then end turn.
    /// Returns true (turn ended).
    /// </summary>
    private async Task<bool> PlayRandomFallback(
        Player player, Rng random, CancellationToken ct)
    {
        CardPile hand = PileType.Hand.GetPile(player);
        UnplayableReason reason;
        AbstractModel preventer;
        List<CardModel> playable = hand.Cards
            .Where(c => c.CanPlay(out reason, out preventer))
            .ToList();

        if (playable.Count > 0)
        {
            CardModel card = random.NextItem(playable);
            Creature target = GetRandomTarget(card, random);
            Logger.Log($"[RlCombat] Random fallback: playing {card.Id.Entry}");
            await PlayCardAndWaitAsync(player, card, target, ct);
            return false;
        }
        else
        {
            Logger.Log("[RlCombat] Random fallback: no playable cards, ending turn");
            PlayerCmd.EndTurn(player, canBackOut: false);
            return true;
        }
    }

    /// <summary>
    /// Resolve a target creature from the target_index.
    /// </summary>
    private Creature? ResolveTarget(CardModel card, int targetIndex)
    {
        if (card.TargetType != TargetType.AnyEnemy)
            return null;

        ICombatState combatState = card.CombatState;
        if (combatState == null)
            return null;

        List<Creature> allEnemies = combatState.Enemies.ToList();
        if (allEnemies.Count == 0)
            return null;

        if (targetIndex >= 0)
        {
            if (targetIndex >= allEnemies.Count)
                return null;
            Creature indexedEnemy = allEnemies[targetIndex];
            return indexedEnemy.IsHittable ? indexedEnemy : null;
        }

        return combatState.HittableEnemies.FirstOrDefault();
    }

    private static Creature? ResolvePotionTarget(Player player, PotionModel? potion, int targetIndex)
    {
        if (potion == null)
            return null;

        string targetType = "Self";
        try
        {
            targetType = potion.TargetType.ToString();
        }
        catch
        {
            return player.Creature;
        }

        if (targetType == "AnyEnemy")
        {
            ICombatState? combatState = player.Creature?.CombatState;
            if (combatState == null)
                return null;
            List<Creature> allEnemies = combatState.Enemies.ToList();
            if (targetIndex >= 0)
            {
                if (targetIndex >= allEnemies.Count)
                    return null;
                Creature indexedEnemy = allEnemies[targetIndex];
                return indexedEnemy.IsHittable ? indexedEnemy : null;
            }
            return combatState.HittableEnemies.FirstOrDefault();
        }

        if (targetType == "Self" || targetType == "AnyPlayer")
            return player.Creature;

        return null;
    }

    private static Creature? GetRandomTarget(CardModel card, Rng random)
    {
        if (card.TargetType != TargetType.AnyEnemy)
            return null;
        ICombatState combatState = card.CombatState;
        if (combatState == null)
            return null;
        List<Creature> hittable = combatState.HittableEnemies.ToList();
        if (hittable.Count == 0)
            return null;
        return random.NextItem(hittable);
    }

    private static async Task PlayCardAndWaitAsync(
        Player player, CardModel card, Creature? target, CancellationToken ct)
    {
        // §31: a play may open a NATIVE selection (Headbutt discard, relic
        // choice, ...) that RlNativeSelectionCoordinator must resolve WHILE this
        // action is still awaiting its full GameAction completion. We wait on
        // CompletionTask -- NOT energy/hand changes -- so the combat loop never
        // advances before the originating card finishes. That stale advance was
        // the root cause of the Headbutt combat_action / end_turn loops. The
        // selection screen is driven by the coordinator concurrently; we only
        // await here.
        string? previousSource = ActiveActionSource;
        ActiveActionSource = $"played card {card.Id.Entry}";
        try
        {
            var playAction = new PlayCardAction(card, target);

            Logger.Log(
                $"[RlCombat] Enqueue PlayCardAction {card.Id.Entry}; " +
                $"waiting for full GameAction completion");

            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(playAction);

            // Prefer the shared agent-decision timeout plus settle margin over a
            // hard-coded constant, so slow reasoning models still fit.
            TimeSpan actionTimeout =
                TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds)
                + TimeSpan.FromSeconds(30);

            try
            {
                await playAction.CompletionTask.WaitAsync(actionTimeout, ct);
            }
            catch (TimeoutException)
            {
                Logger.Log(
                    $"[RlCombat] PlayCardAction timeout: " +
                    $"card={card.Id.Entry} state={playAction.State}");
                throw new TimeoutException(
                    $"PlayCardAction did not complete: {card.Id.Entry}; " +
                    $"state={playAction.State}");
            }

            if (playAction.Exception != null)
            {
                throw new InvalidOperationException(
                    $"PlayCardAction failed: {card.Id.Entry}",
                    playAction.Exception);
            }

            Logger.Log(
                $"[RlCombat] PlayCardAction completed: {card.Id.Entry}; " +
                $"state={playAction.State}");
        }
        finally
        {
            ActiveActionSource = previousSource;
        }
    }

    /// <summary>
    /// Use a potion and wait for its full GameAction completion.
    /// Returns TRUE only when the game-side handler accepted and started it
    /// (false for an unusable/absent slot or an invalid target).
    /// </summary>
    private static async Task<bool> UsePotionAndWaitAsync(
        Player player, int slot, int targetIndex, CancellationToken ct)
    {
        if (slot < 0)
            return false;

        dynamic potion = null;
        try
        {
            potion = player.GetPotionAtSlotIndex(slot);
        }
        catch
        {
            return false;
        }

        if (potion == null || !PotionActions.CanUse((PotionModel)potion))
            return false;

        Creature? target = ResolvePotionTarget(player, potion, targetIndex);
        if (potion.TargetType.ToString() == "AnyEnemy" && target == null)
            return false;

        string? previousSource = ActiveActionSource;
        ActiveActionSource = $"used potion in slot {slot}";
        try
        {
            var usePotionAction = new UsePotionAction(
                potion,
                target,
                CombatManager.Instance.IsInProgress);

            Logger.Log(
                $"[RlCombat] Enqueue UsePotionAction slot={slot}; " +
                $"waiting for full GameAction completion");

            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(usePotionAction);

            // §31: a potion is NOT resolved when its belt slot empties -- several
            // potions open a player-choice UI of their own. Wait on CompletionTask
            // so the granted effect (and any nested selection) finishes before we
            // continue. Prefer the shared agent-decision timeout + settle margin.
            TimeSpan actionTimeout =
                TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds)
                + TimeSpan.FromSeconds(30);

            try
            {
                await usePotionAction.CompletionTask.WaitAsync(actionTimeout, ct);
            }
            catch (TimeoutException)
            {
                Logger.Log(
                    $"[RlCombat] UsePotionAction timeout: " +
                    $"slot={slot} state={usePotionAction.State}");
                throw new TimeoutException(
                    $"UsePotionAction did not complete for slot {slot}; " +
                    $"state={usePotionAction.State}");
            }

            if (usePotionAction.Exception != null)
            {
                throw new InvalidOperationException(
                    $"UsePotionAction failed for slot {slot}",
                    usePotionAction.Exception);
            }

            Logger.Log(
                $"[RlCombat] UsePotionAction completed: slot={slot}; " +
                $"state={usePotionAction.State}");
        }
        finally
        {
            ActiveActionSource = previousSource;
        }
        return true;
    }

    /// <summary>
    /// Wait until no nested agent decision is pending.
    ///
    /// A nested decision is an agent request raised from inside another one
    /// (a card selection opened by a potion or by a card effect). Until it is
    /// answered, the game state is half-settled and must not be serialized
    /// for the next combat decision.
    ///
    /// Deliberately never throws on timeout: a stuck selection must degrade
    /// into the normal retry path instead of aborting the whole run.
    /// </summary>
    private static async Task<bool> WaitForNoNestedDecisionAsync(
        string source, CancellationToken ct)
    {
        if (!BridgeServer.HasNestedDecision)
            return true;

        Logger.Log(
            $"[RlCombat] {source} opened a nested card selection; waiting for"
            + " the agent to answer it before continuing.");
        RlAutoSlayer.CurrentWatchdog?.Reset(
            $"Waiting for nested card selection ({source})");

        int maxMs = BridgeServer.HandlerTimeoutSeconds * 1000;
        int waitedMs = 0;
        while (BridgeServer.HasNestedDecision && waitedMs < maxMs)
        {
            // Keep refreshing: this wait can last a full agent round-trip
            // (tens of seconds), far past the 30s watchdog timeout, and the
            // next WaitHelper call would otherwise trip on it.
            RlAutoSlayer.CurrentWatchdog?.Reset(
                $"Waiting for nested card selection ({source})");
            await Task.Delay(50, ct);
            waitedMs += 50;
        }

        if (BridgeServer.HasNestedDecision)
        {
            Logger.Log(
                $"[RlCombat] WARNING: nested card selection still pending after"
                + $" {maxMs}ms ({source}); continuing anyway.");
            return false;
        }
        return true;
    }

    // ----------------------------------------------------------------
    // State serialization
    // ----------------------------------------------------------------

    private string SerializeCombatState(Player player)
    {
        try
        {
            var cm = CombatManager.Instance;
            CombatState combatState = cm.DebugOnlyGetState();
            Creature playerCreature = player.Creature;
            PlayerCombatState pcs = player.PlayerCombatState;

            Logger.Log($"[RlCombat] Serialize: cm={cm != null}, cs={combatState != null}, creature={playerCreature != null}, pcs={pcs != null}");
            if (playerCreature != null)
                Logger.Log($"[RlCombat] Player: HP={playerCreature.CurrentHp}/{playerCreature.MaxHp} Block={playerCreature.Block}");
            if (pcs != null)
                Logger.Log($"[RlCombat] Energy={pcs.Energy}/{pcs.MaxEnergy} Hand={pcs.Hand.Cards.Count} Draw={pcs.DrawPile.Cards.Count}");
            if (combatState != null)
                Logger.Log($"[RlCombat] Enemies={combatState.Enemies.Count()} Round={combatState.RoundNumber}");

            // Player info
            var playerObj = new Dictionary<string, object>
            {
                ["hp"] = playerCreature.CurrentHp,
                ["max_hp"] = playerCreature.MaxHp,
                ["block"] = playerCreature.Block,
                ["energy"] = pcs?.Energy ?? 0,
                ["max_energy"] = pcs?.MaxEnergy ?? 3,
            };

            // Character identity (visible on every UI screen for a human).
            try
            {
                playerObj["character_id"] = player.Character.Id.Entry;
                playerObj["character_name"] = CardSerialization.CleanText(
                    player.Character.Title.GetFormattedText());
            }
            catch { }

            // Character-specific visible state (Stars / Orbs / Osty / ...).
            // Only explicitly serialized fields are exposed.
            var visibleCharacterState = CardSerialization.BuildVisibleCharacterState(player);
            if (visibleCharacterState != null)
                playerObj["visible_character_state"] = visibleCharacterState;

            // Player powers: only UI-visible powers, with the game's own
            // runtime tooltip (PowerModel.HoverTips pipeline).
            var powers = CardSerialization.SerializePowers(playerCreature);
            if (powers.Count > 0)
                playerObj["powers"] = powers;

            // Relics: visible belt -- identity + UI counter + spent state.
            // Hidden internal proc counters are deliberately not serialized.
            var relics = new List<Dictionary<string, object>>();
            foreach (RelicModel relic in player.Relics)
            {
                var relicObj = new Dictionary<string, object>
                {
                    ["id"] = relic.Id.Entry,
                };
                if (relic.StackCount != 0)
                    relicObj["counter"] = relic.StackCount;
                if (relic.IsUsedUp)
                    relicObj["used_up"] = true;
                relics.Add(relicObj);
            }
            if (relics.Count > 0)
                playerObj["relics"] = relics;

            // Gold: top-bar run metadata.
            playerObj["gold"] = player.Gold;

            // Hand cards
            var handCards = new List<Dictionary<string, object>>();
            if (pcs != null)
            {
                foreach (CardModel card in pcs.Hand.Cards)
                {
                    handCards.Add(SerializeCard(card));
                }
            }

            // Enemies
            var enemies = new List<Dictionary<string, object>>();
            if (combatState != null)
            {
                foreach (Creature enemy in combatState.Enemies)
                {
                    enemies.Add(SerializeEnemy(enemy));
                }
            }

            // Run state info
            RunState runState = RunManager.Instance.DebugOnlyGetState();

            // Current master deck: the human can open the Deck viewer during
            // combat, so the CURRENT runtime deck is serialized on every
            // combat request (unordered composition). Python no longer has
            // to rely on a cached snapshot.
            List<Dictionary<string, object>> currentDeck = SerializePileComposition(
                player.Deck.Cards);

            List<Dictionary<string, object>> potions = SerializePotions(player);
            var state = new Dictionary<string, object>
            {
                ["type"] = "combat_action",
                ["player"] = playerObj,
                ["hand"] = handCards,
                ["enemies"] = enemies,
                ["potions"] = potions,
                ["potion_slot_capacity"] = player.MaxPotionCount,
                ["available_actions"] = GetAvailableActions(potions),
                // Piles: visible count + UNORDERED composition (grouped by
                // (card id, upgraded), canonically sorted). Internal pile
                // order is never serialized -- the pile viewer only shows
                // unordered content.
                ["draw_pile_count"] = pcs?.DrawPile.Cards.Count ?? 0,
                ["draw_pile"] = SerializePileComposition(pcs?.DrawPile.Cards),
                ["discard_pile_count"] = pcs?.DiscardPile.Cards.Count ?? 0,
                ["discard_pile"] = SerializePileComposition(pcs?.DiscardPile.Cards),
                ["exhaust_pile_count"] = pcs?.ExhaustPile.Cards.Count ?? 0,
                ["exhaust_pile"] = SerializePileComposition(pcs?.ExhaustPile.Cards),
                ["round"] = combatState?.RoundNumber ?? 0,
                ["floor"] = runState?.TotalFloor ?? 0,
                ["act"] = (runState?.CurrentActIndex ?? 0) + 1,
                ["ascension"] = runState?.AscensionLevel ?? 0,
                // Multiplayer boundary: the agent officially supports
                // singleplayer only (see Python side warning).
                ["player_count"] = combatState?.Players.Count ?? 1,
            };

            // Act identity + currently revealed boss(es) (map-screen info).
            if (runState != null)
            {
                var actInfo = CardSerialization.SerializeActInfo(runState);
                if (actInfo != null)
                    state["act_info"] = actInfo;
            }

            // Current runtime master deck (combat state carries it directly).
            if (currentDeck.Count > 0)
            {
                state["deck"] = currentDeck;
                state["deck_count"] = player.Deck.Cards.Count;
            }

            return JsonSerializer.Serialize(state);
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCombat] Error serializing combat state: {ex.Message}");
            return "{\"type\":\"combat_action\",\"error\":\"serialization_failed\"}";
        }
    }

    /// <summary>
    /// Unordered pile composition: grouped by (card id, upgraded,
    /// enchantment, affliction) and sorted so that the internal draw order
    /// can never leak through serialization. Two cards the player could tell
    /// apart in the pile viewer are never merged into one group.
    /// </summary>
    private static List<Dictionary<string, object>> SerializePileComposition(
        IEnumerable<CardModel> cards)
    {
        var grouped = new Dictionary<(string Id, bool Up, string Ench, string Aff), int>();
        if (cards != null)
        {
            foreach (CardModel card in cards)
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

    private static Dictionary<string, object> SerializeCard(CardModel card)
    {
        // Full human-visible card face: current rendered text with dynamic
        // values resolved, current vs base energy/star costs, keywords,
        // enchantment/affliction, upgrade level and unplayable reasons.
        // Upgrade preview is not shown for hand cards in the UI, so it is
        // not included here (reward/select screens do include it).
        try
        {
            return CardSerialization.SerializeCardFull(card, includeUpgradePreview: false);
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCombat] SerializeCard fallback for {card.Id.Entry}: {ex.Message}");
            UnplayableReason reason;
            AbstractModel preventer;
            return new Dictionary<string, object>
            {
                ["id"] = card.Id.Entry,
                ["cost"] = card.EnergyCost.Canonical,
                ["type"] = card.Type.ToString(),
                ["target"] = card.TargetType.ToString(),
                ["playable"] = card.CanPlay(out reason, out preventer),
                ["upgraded"] = card.IsUpgraded,
            };
        }
    }

    private static Dictionary<string, object> SerializeEnemy(Creature enemy)
    {
        var data = new Dictionary<string, object>
        {
            ["id"] = enemy.IsMonster ? enemy.Monster!.Id.Entry : "UNKNOWN",
            ["hp"] = enemy.CurrentHp,
            ["max_hp"] = enemy.MaxHp,
            ["block"] = enemy.Block,
            ["is_alive"] = enemy.IsAlive,
        };

        // Powers: only UI-visible powers, with runtime tooltips.
        var powers = CardSerialization.SerializePowers(enemy);
        if (powers.Count > 0)
            data["powers"] = powers;

        // Intent
        if (enemy.IsMonster && enemy.Monster != null)
        {
            try
            {
                var nextMove = enemy.Monster.NextMove;
                if (nextMove?.Intents != null && nextMove.Intents.Count > 0)
                {
                    AbstractIntent firstIntent = nextMove.Intents[0];
                    data["intent"] = firstIntent.IntentType.ToString();
                    data["intent_move_id"] = nextMove.Id;

                    if (firstIntent is AttackIntent attackIntent)
                    {
                        ICombatState cs = enemy.CombatState;
                        if (cs != null)
                        {
                            try
                            {
                                data["intent_damage"] = attackIntent.GetSingleDamage(
                                    cs.PlayerCreatures, enemy);
                                data["intent_hits"] = attackIntent.Repeats > 0
                                    ? attackIntent.Repeats : 1;
                            }
                            catch { }
                        }
                    }
                }
            }
            catch
            {
                data["intent"] = "UNKNOWN";
            }
        }

        return data;
    }

    /// <summary>
    /// What caused the agent decision currently being resolved, e.g.
    /// "played card burning_pact". Set ONLY while a play / potion is being
    /// resolved, so a selection opened by it can name its trigger. It is null
    /// at all other times: a selection opened by an enemy move must never be
    /// mislabelled with the last card we happened to play.
    /// </summary>
    internal static string? ActiveActionSource;

    /// <summary>
    /// Board snapshot for agent decisions requested from INSIDE combat.
    ///
    /// Many cards (and potions) open a selection while combat runs: "exhaust
    /// a card", "upgrade a card in your hand", "choose a card to discard",
    /// "choose a card the enemy offers you". A human deciding there still
    /// sees energy, the enemies and the rest of their hand -- without this
    /// the agent would have to pick which card to burn completely blind.
    /// Returns null outside combat.
    /// </summary>
    public static Dictionary<string, object>? BuildCombatContext()
    {
        try
        {
            if (!CombatManager.Instance.IsInProgress)
                return null;

            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            if (runState == null)
                return null;

            Player player = LocalContext.GetMe(runState);
            PlayerCombatState? pcs = player.PlayerCombatState;
            if (pcs == null)
                return null;

            var ctx = new Dictionary<string, object>
            {
                ["in_combat"] = true,
                ["round"] = CombatManager.Instance.DebugOnlyGetState()?.RoundNumber ?? 0,
                ["energy"] = pcs.Energy,
                ["max_energy"] = pcs.MaxEnergy,
                ["block"] = player.Creature.Block,
            };

            if (!string.IsNullOrEmpty(ActiveActionSource))
                ctx["trigger"] = ActiveActionSource;

            var enemies = new List<Dictionary<string, object>>();
            CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
            if (combatState != null)
            {
                foreach (Creature enemy in combatState.Enemies)
                    enemies.Add(SerializeEnemy(enemy));
            }
            ctx["enemies"] = enemies;

            var hand = new List<Dictionary<string, object>>();
            foreach (CardModel card in pcs.Hand.Cards)
                hand.Add(SerializeCard(card));
            ctx["hand"] = hand;

            return ctx;
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCombat] BuildCombatContext failed: {ex.Message}");
            return null;
        }
    }

    private static List<Dictionary<string, object>> SerializePotions(Player player) => PotionActions.Serialize(player);

    private static List<string> GetAvailableActions(IEnumerable<Dictionary<string, object>> potions)
    {
        var actions = new List<string> { "PLAY", "END_TURN" };
        if (potions.Any(p => p.TryGetValue("can_use", out object? canUse) && canUse is bool b && b))
            actions.Add("POTION");
        if (potions.Any(p => p.TryGetValue("can_discard", out var discard) && discard is true)) actions.Add("DISCARD_POTION");
        return actions;
    }
}
