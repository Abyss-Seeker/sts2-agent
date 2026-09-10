// RlAutoSlayer.cs -- RL-agent-driven AutoSlayer.
//
// This is a modified version of the game's AutoSlayer that replaces
// random decision handlers with RL agent handlers communicating via TCP.
// The overall game flow (main menu, room loop, screen draining, map navigation)
// is preserved from the original AutoSlayer.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Handlers.Rooms;
using MegaCrit.Sts2.Core.AutoSlay.Handlers.Screens;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Timeline;
using MegaCrit.Sts2.Core.Timeline.Epochs;

namespace STS2BridgeMod;

/// <summary>
/// RL-agent-driven AutoSlayer. Mirrors the structure of the game's built-in
/// AutoSlayer but replaces random decision handlers with RL agent handlers
/// that communicate with Python via BridgeServer TCP.
///
/// Combat, map navigation, card rewards, events, shops, rest sites, treasure,
/// and boss relic choices are bridge-driven. Most other screen handlers still
/// use AutoSlay helpers.
/// </summary>
public class RlAutoSlayer
{
    private const string MainMenuPath = "/root/Game/RootSceneContainer/MainMenu";
    private const string RestSiteProceedButtonPath =
        "/root/Game/RootSceneContainer/Run/RoomContainer/RestSiteRoom/ProceedButton";
    private const string EventRoomPath =
        "/root/Game/RootSceneContainer/Run/RoomContainer/EventRoom";
    private const string AbandonRunOptionsButtonPath =
        "/root/Game/RootSceneContainer/Run/GlobalUi/TopBar/RightAlignedStuff/Options";
    private const string AbandonRunButtonPath =
        "/root/Game/RootSceneContainer/Run/GlobalUi/CapstoneScreenContainer/OptionsScreen/AbandonRunButton";
    private const string AbandonRunProceedButtonPath =
        "/root/Game/RootSceneContainer/Run/GlobalUi/OverlayScreensContainer/GameOverScreen/UI/ProceedButton";
    private const string AbandonRunMenuButtonPath = "MainMenuTextButtons/AbandonRunButton";
    private const string ContinueRunButtonPath = "MainMenuTextButtons/ContinueButton";
    private const string AbandonPopupPrimaryYesButtonPath = "VerticalPopup/YesButton";
    private const string AbandonPopupFallbackYesButtonPath = "YesButton";
    private const string SingleplayerButtonPath = "MainMenuTextButtons/SingleplayerButton";
    private const string CharacterSelectScreenPath = "Submenus/CharacterSelectScreen";
    private const string StandardRunButtonPath = "Submenus/SingleplayerSubmenu/StandardButton";
    private const string CharacterButtonContainerPath = "CharSelectButtons/ButtonContainer";
    private const string CharacterConfirmButtonPath =
        "Submenus/CharacterSelectScreen/ConfirmButton";
    public const string PreferredCharacterId = "Ironclad";
    private const int FinalRunFloor = 49;
    private const int RunTimeoutMinutes = 60;
    private const int RunStateTimeoutSeconds = 60;
    private const int RoomAssignmentTimeoutSeconds = 60;
    private const int NonCombatSettleDelayMs = 500;
    private const int BossTransitionTimeoutSeconds = 10;
    private const int ActTransitionTimeoutSeconds = 5;
    private const int OverlayCloseRetryLimit = 3;
    private const int OverlayDrainSettleDelayMs = 100;
    private const int EventProceedTimeoutSeconds = 5;
    private const int RewardsScreenTimeoutSeconds = 10;
    private const int MainMenuTimeoutSeconds = 30;
    private const int AbandonPopupTimeoutSeconds = 5;
    private const int AbandonRunSettleDelayMs = 1000;
    private const int MenuClickSettleDelayMs = 500;
    // Fault tolerance: how many BACK-TO-BACK room failures actually end the
    // run, and how long to let the game settle before carrying on.
    private const int MaxConsecutiveFailures = 5;
    private const int RecoverySettleDelayMs = 1500;
    private const int CharacterSelectDelayMs = 100;
    private readonly Dictionary<RoomType, IRoomHandler> _roomHandlers;
    private readonly Dictionary<Type, IScreenHandler> _screenHandlers;
    private readonly RlMapHandler _mapHandler;

    private CancellationTokenSource? _cts;
    private Task? _runTask;
    private Task? _watcherTask;
    private Rng? _random;
    private Watchdog? _watchdog;
    private IDisposable? _cardSelectorScope;
    private bool _completionSignalSent;

    public static bool IsActive { get; private set; }

    /// <summary>
    /// §B1: the live slayer instance, so the bridge can start an EXPLICIT
    /// run (seed + character) instead of relying on AutoSlay's implicit
    /// "start whatever at the main menu" behaviour.
    /// </summary>
    public static RlAutoSlayer? Active { get; private set; }

    /// <summary>Seed currently requested by the controller (B1 ack).</summary>
    public static string RequestedSeed { get; private set; } = "";

    /// <summary>
    /// Resume an existing run save at startup instead of abandoning it and
    /// starting over. Default true: an aborted run should cost at most the
    /// current room, never the whole run.
    /// </summary>
    public static bool ResumeSavedRun = true;

    /// <summary>
    /// Public watchdog for use by handlers. Since we can't set
    /// AutoSlayer.CurrentWatchdog (private setter), we expose our own.
    /// </summary>
    public static Watchdog? CurrentWatchdog { get; private set; }

    public RlAutoSlayer()
    {
        // Use our RL combat handler for all combat room types
        var combatHandler = new RlCombatHandler();
        _roomHandlers = new Dictionary<RoomType, IRoomHandler>
        {
            [RoomType.Monster] = combatHandler,
            [RoomType.Elite] = combatHandler,
            [RoomType.Boss] = combatHandler,
            [RoomType.Event] = new RlEventRoomHandler(),
            [RoomType.Shop] = new RlShopRoomHandler(),
            [RoomType.Treasure] = new RlTreasureRoomHandler(),
            [RoomType.RestSite] = new RlRestSiteRoomHandler(),
        };

        _mapHandler = new RlMapHandler();

        _screenHandlers = new Dictionary<Type, IScreenHandler>
        {
            [typeof(NRewardsScreen)] = new RlRewardsScreenHandler(),
            [typeof(NCardRewardSelectionScreen)] = new RlCardRewardScreenHandler(),
            [typeof(NDeckUpgradeSelectScreen)] = new DeckUpgradeScreenHandler(),
            [typeof(NDeckTransformSelectScreen)] = new DeckTransformScreenHandler(),
            [typeof(NDeckEnchantSelectScreen)] = new DeckEnchantScreenHandler(),
            [typeof(NDeckCardSelectScreen)] = new DeckCardSelectScreenHandler(),
            [typeof(NSimpleCardSelectScreen)] = new SimpleCardSelectScreenHandler(),
            [typeof(NChooseACardSelectionScreen)] = new ChooseACardScreenHandler(),
            [typeof(NChooseABundleSelectionScreen)] = new RlCardBundleScreenHandler(),
            [typeof(NChooseARelicSelection)] = new RlChooseARelicScreenHandler(),
            [typeof(NGameOverScreen)] = new RlGameOverScreenHandler(),
            [typeof(NCrystalSphereScreen)] = new RlCrystalSphereScreenHandler(),
        };
    }

    public void Start(string seed, string? logFile = null)
    {
        // All startup paths run on the game thread and share one owner.
        if (IsActive)
        {
            Logger.Log("[RlAutoSlayer] Start ignored: controller already active");
            return;
        }
        if (logFile != null)
        {
            AutoSlayLog.OpenLogFile(logFile);
        }
        Active = this;
        RequestedSeed = seed;
        _completionSignalSent = false;
        _lastAbortException = null;
        RlNativeSelectionCoordinator.Reset();
        IsActive = true;
        SetAutoSlayerActive(true);
        _cts = new CancellationTokenSource();
        // Keep one watcher alive for the run. It is dormant in headless
        // mode, which allows set_headful to take effect without restarting
        // the automation lifecycle.
        _watcherTask = RlNativeSelectionCoordinator.RunLoopAsync(_cts.Token);
        _runTask = RunAsync(seed, _cts.Token);
        TaskHelper.RunSafely(_runTask);
    }

    public void Stop()
    {
        _cts?.Cancel();
    }

    /// <summary>
    /// B1 acknowledgement: requested vs ACTUAL seed. The runner may only
    /// claim seed_applied_to_game when seed_match is true.
    /// </summary>
    public static string BuildStartRunAck(
        string seed, string character, int difficulty, bool success,
        string error)
    {
        string actualSeed = "";
        int floor = -1;
        try
        {
            actualSeed = NGame.Instance?.DebugSeedOverride ?? "";
            var state = RunManager.Instance.DebugOnlyGetState();
            if (state != null)
                floor = state.TotalFloor;
        }
        catch { }

        bool match = success
            && string.Equals(actualSeed, seed, StringComparison.Ordinal);
        var fields = new Dictionary<string, object>
        {
            ["type"] = "start_run_ack",
            ["success"] = success,
            ["requested_seed"] = seed,
            ["actual_seed"] = actualSeed,
            ["seed_match"] = match,
            ["character"] = character,
            ["difficulty"] = difficulty,
            ["floor"] = floor,
            ["error"] = error,
        };
        return JsonSerializer.Serialize(fields);
    }

    /// <summary>
    /// B1: start an EXPLICIT run for the benchmark controller.
    ///
    /// Stops the current run loop and abandons the live run (best effort),
    /// then restarts the slayer with the REQUESTED seed so the same seed can
    /// be replayed under both decision modes. This never restarts STS2: the
    /// game process and the bridge connection stay up (§11/B1.5).
    ///
    /// Character is the game's standard singleplayer selection
    /// (PreferredCharacterId = Ironclad) and difficulty is the standard
    /// run (ascension 0) -- see docs/run_start_seed_audit.md.
    /// </summary>
    public async Task<bool> StartRunAsync(string seed, CancellationToken ct)
    {
        RequestedSeed = seed;
        Logger.Log($"[RlAutoSlayer] start_run requested: seed={seed}");
        Stop();
        if (_runTask != null)
            await _runTask;
        await Task.Delay(AbandonRunSettleDelayMs, ct);
        try
        {
            await AbandonRunAsync(ct);
        }
        catch (Exception ex)
        {
            // Abandoning is best effort: if the game is on the main menu
            // there is nothing to abandon.
            Logger.Log($"[RlAutoSlayer] start_run: abandon skipped"
                       + $" ({ex.Message})");
        }

        // Never let the abandon-save resume resurrect the run we just
        // abandoned, we want a FRESH run with the requested seed.
        bool resumePref = ResumeSavedRun;
        ResumeSavedRun = false;
        try
        {
            Start(seed);
            await WaitHelper.Until(
                delegate {
                    if (NGame.Instance == null)
                        return false;
                    if (!string.Equals(NGame.Instance.DebugSeedOverride, seed,
                            StringComparison.Ordinal))
                        return false;
                    return RunManager.Instance.DebugOnlyGetState() != null;
                },
                ct, TimeSpan.FromSeconds(RunStateTimeoutSeconds),
                "run did not start with the requested seed");
            Logger.Log($"[RlAutoSlayer] start_run accepted: seed={seed}");
            return true;
        }
        finally
        {
            _cts?.Cancel();
            if (_watcherTask != null)
            {
                try { await _watcherTask; }
                catch (OperationCanceledException) { }
            }
            _cts?.Dispose();
            _cts = null;
            RlNativeSelectionCoordinator.Reset();
            ResumeSavedRun = resumePref;
        }
    }

    // ------------------------------------------------------------------
    // §29-§32: bridge automation is NOT NonInteractiveMode.
    //
    // HeadfulMode (default true) keeps the game fully interactive while
    // the agent plays: BGM, ambience, SFX, normal animation waits. When
    // false (headless/CI), the legacy behaviour is preserved and the game
    // suppresses music/visuals exactly like AutoSlay does.
    // BridgeServer exposes set_headful / set_fast_mode so the Python UI
    // can flip these at runtime.
    // ------------------------------------------------------------------
    public static bool HeadfulMode { get; set; } = true;
    public static bool FastModeEnabled { get; set; } = true;

    public static void ApplyHeadfulMode(bool enabled)
    {
        HeadfulMode = enabled;
        var active = Active;
        if (active != null && IsActive)
        {
            if (enabled)
            {
                active._cardSelectorScope?.Dispose();
                active._cardSelectorScope = null;
            }
            else if (active._cardSelectorScope == null)
            {
                active._cardSelectorScope = CardSelectCmd.UseSelector(
                    new RlCardSelector());
            }
        }
        SetAutoSlayerActive(IsActive);
    }

    /// <summary>
    /// Restart the controller loop after a recoverable termination. The game
    /// save remains authoritative; PlayMainMenuAsync will choose Continue.
    /// </summary>
    public static bool EnsureAutomationActive()
    {
        if (IsActive)
            return true;
        var slayer = Active ?? new RlAutoSlayer();
        string seed = string.IsNullOrWhiteSpace(RequestedSeed)
            ? SeedHelper.GetRandomSeed()
            : RequestedSeed;
        ResumeSavedRun = true;
        slayer.Start(seed);
        return IsActive;
    }

    public static void ApplyFastMode(bool enabled)
    {
        FastModeEnabled = enabled;
        try
        {
            SaveManager.Instance.PrefsSave.FastMode = enabled
                ? FastModeType.Fast
                : FastModeType.Normal;
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlAutoSlayer] FastMode preference deferred: {ex.Message}");
        }
    }

    /// <summary>
    /// Set AutoSlayer.IsActive via NonInteractiveMode.AutoSlayerCheck.
    /// The original AutoSlayer static constructor wires this up, but since
    // we're running our own slayer, we set it directly to report our state.
    /// In headful mode the check reports FALSE so the game keeps music,
    /// ambience, SFX and normal waits; automation state lives in
    /// RlAutoSlayer.IsActive only.
    /// </summary>
    private static void SetAutoSlayerActive(bool active)
    {
        if (HeadfulMode)
        {
            // Headful: the game stays interactive (music/ambience/SFX/
            // animation waits all run). Bridge automation state lives in
            // RlAutoSlayer.IsActive -- NonInteractiveMode is NOT the way
            // we express "the agent is playing".
            NonInteractiveMode.AutoSlayerCheck = () => false;
            Logger.Log(
                $"[RlAutoSlayer] Headful mode: NonInteractiveMode.IsActive"
                + $"=false (automation active={active}); BGM/SFX/waits on");
        }
        else
        {
            NonInteractiveMode.AutoSlayerCheck = () => active;
            Logger.Log(
                $"[RlAutoSlayer] Headless mode: NonInteractiveMode follows"
                + $" automation (active={active})");
        }
    }

    private async Task RunAsync(string seed, CancellationToken ct)
    {
        Logger.Log($"[RlAutoSlayer] Run started with seed: {seed}");
        try
        {
            await WaitHelper.WithTimeout(
                (CancellationToken token) => PlayRunAsync(seed, token),
                TimeSpan.FromMinutes(RunTimeoutMinutes),
                ct);
            Logger.Log($"[RlAutoSlayer] Run completed with seed: {seed}");
        }
        catch (Exception ex)
        {
            _lastAbortException = ex;
            Logger.Log($"[RlAutoSlayer] Run failed:\n{ex}");
        }
        finally
        {
            IsActive = false;
            SetAutoSlayerActive(false);
            CurrentWatchdog = null;
            SetSharedCurrentWatchdog(null);
            _watchdog = null;
            _cardSelectorScope?.Dispose();
            _cardSelectorScope = null;
            AutoSlayLog.CloseLogFile();

            // Notify Python that the run is over. The debug fields are
            // developer diagnostics ONLY -- the agent's terminal-state
            // formatter never renders them, so they cannot leak into the
            // LLM prompt.
            if (!_completionSignalSent)
            {
                BridgeServer.Instance.SendState(RunCompleteState(
                    NonCombatBridgeProtocol.TerminatedResult));
                _completionSignalSent = true;
            }
        }
    }

    private async Task PlayRunAsync(string seed, CancellationToken ct)
    {
        await WaitHelper.Until(() => NGame.Instance != null, ct,
            AutoSlayConfig.gameInitTimeout, "Game instance not initialized");

        NGame.Instance.DebugSeedOverride = seed;
        // §35: FastMode is an INDEPENDENT setting (it accelerates native
        // animations but must never gate audio/UI/selection screens).
        SaveManager.Instance.PrefsSave.FastMode = FastModeEnabled
            ? FastModeType.Fast
            : FastModeType.Normal;
        SaveManager.Instance.SetFtuesEnabled(enabled: false);

        // Unlock all epochs
        SaveManager.Instance.ObtainEpochOverride(
            EpochModel.GetId<Silent1Epoch>(), EpochState.Revealed);
        SaveManager.Instance.ObtainEpochOverride(
            EpochModel.GetId<Regent1Epoch>(), EpochState.Revealed);
        SaveManager.Instance.ObtainEpochOverride(
            EpochModel.GetId<Defect1Epoch>(), EpochState.Revealed);
        SaveManager.Instance.ObtainEpochOverride(
            EpochModel.GetId<Necrobinder1Epoch>(), EpochState.Revealed);

        _random = new Rng((uint)StringHelper.GetDeterministicHashCode(seed));

        // §27: in headful mode CardSelectCmd.Selector MUST stay null --
        // native selection UIs are driven by RlNativeSelectionCoordinator
        // instead (visible UI + real clicks). Headless keeps the legacy
        // selector bypass.
        if (!HeadfulMode)
        {
            // Install our RL card selector for deck upgrade/transform/card selection screens
            _cardSelectorScope = CardSelectCmd.UseSelector(new RlCardSelector());
            Logger.Log("[RlAutoSlayer] Headless: RlCardSelector installed (legacy bypass)");
        }
        else
        {
            Logger.Log("[RlAutoSlayer] Headful: CardSelectCmd.Selector left null"
                + " (native UI driven by coordinator)");
        }

        // §16/§17: in headful mode the native selection coordinator must own
        // every card-selection screen; no strategic selector bypass may be
        // installed. Fail loudly if the invariant is violated.
        AssertNativeUiInvariant();

        _watchdog = new Watchdog();
        CurrentWatchdog = _watchdog;
        SetSharedCurrentWatchdog(_watchdog);
        _watchdog.Reset("Playing main menu");

        var root = ((SceneTree)Engine.GetMainLoop()).Root;
        if (root.GetNodeOrNull("/root/Game/RootSceneContainer/Run") != null
            && RunManager.Instance.DebugOnlyGetState() != null)
            Logger.Log("[RlAutoSlayer] Resuming controller in existing run scene");
        else
            await PlayMainMenuAsync(ct);

        await WaitHelper.Until(
            () => RunManager.Instance.DebugOnlyGetState() != null, ct,
            TimeSpan.FromSeconds(RunStateTimeoutSeconds), "Run state not initialized");

        RunState runState = RunManager.Instance.DebugOnlyGetState();
        Logger.Log($"[RlAutoSlayer] RunState available. Floor: {runState.TotalFloor}");

        await WaitHelper.Until(
            () => {
                var room = runState.CurrentRoom;
                if (room != null)
                    Logger.Log($"[RlAutoSlayer] Waiting for room... type={room.RoomType}");
                return room != null && room.RoomType != RoomType.Unassigned;
            },
            ct, TimeSpan.FromSeconds(RoomAssignmentTimeoutSeconds), "Room type not assigned");

        // Main game loop
        //
        // FAULT TOLERANCE: one room blowing up must never throw away the whole
        // run. In practice nearly every abort is harmless -- a screen that did
        // not close inside its timeout, a room that had already advanced, a
        // watchdog timeout fired while a slow LLM was still thinking. Each
        // iteration is guarded: on failure we settle, drain leftover screens
        // and carry on from whatever the game shows NOW. Only back-to-back
        // failures up to MaxConsecutiveFailures, or a real cancellation, end
        // the run.
        int consecutiveFailures = 0;
        while (runState.TotalFloor < FinalRunFloor)
        {
            ct.ThrowIfCancellationRequested();

            AbstractRoom? currentRoom = runState.CurrentRoom;
            if (currentRoom == null)
            {
                Logger.Log("[RlAutoSlayer] No current room yet; waiting for the run to settle.");
                await Task.Delay(NonCombatSettleDelayMs, ct);
                continue;
            }
            RoomType roomType = currentRoom.RoomType;

            try
            {
                bool finished = await RunRoomCycleAsync(runState, roomType, ct);
                consecutiveFailures = 0;
                if (finished)
                    return;
            }
            catch (OperationCanceledException)
            {
                // A real cancel (agent stopped / user abort) ends the run; it
                // must not be swallowed by the recovery path.
                throw;
            }
            catch (Exception ex)
            {
                consecutiveFailures++;
                Logger.Log(
                    $"[RlAutoSlayer] Recoverable error handling {roomType} (failure"
                    + $" {consecutiveFailures}/{MaxConsecutiveFailures}): {ex.Message}");
                if (consecutiveFailures >= MaxConsecutiveFailures)
                {
                    Logger.Log("[RlAutoSlayer] Too many consecutive failures; aborting the run.");
                    throw;
                }
                await RecoverAsync(roomType, ct);
            }
        }

        Logger.Log("[RlAutoSlayer] Run completed (max floor reached). Abandoning");
        await AbandonRunAsync(ct);
    }

    /// <summary>
    /// One full room cycle: handle the room, wait for / claim rewards, drain
    /// leftover screens, run act transitions and navigate the map.
    /// Returns true when the RUN is over (victory), false to keep looping.
    /// </summary>
    private async Task<bool> RunRoomCycleAsync(
        RunState runState, RoomType roomType, CancellationToken ct)
    {
        _watchdog?.Reset(
            $"Entering {roomType} room (Act {runState.CurrentActIndex + 1}, Floor {runState.ActFloor})");
        Logger.Log(
            $"[RlAutoSlayer] Entering {roomType} (Act {runState.CurrentActIndex + 1}, Floor {runState.ActFloor})");

        await HandleRoomAsync(roomType, ct);

        // After combat rooms, wait for rewards screen
        if (roomType == RoomType.Monster || roomType == RoomType.Elite ||
            roomType == RoomType.Boss)
        {
            await WaitForRewardsScreenAsync(ct);
        }
        else
        {
            await Task.Delay(NonCombatSettleDelayMs, ct);
        }

        await DrainOverlayScreensAsync(ct);

        if (roomType == RoomType.RestSite)
        {
            await ClickRestSiteProceedIfNeeded(ct);
        }
        if (roomType == RoomType.Event)
        {
            await ClickEventProceedIfNeeded(ct);
        }

        // Boss room: handle act transition
        if (roomType == RoomType.Boss)
        {
            _watchdog?.Reset("Waiting for act transition after boss");
            RoomType postBossRoomType = RoomType.Boss;
            await WaitHelper.Until(delegate
            {
                AbstractRoom? bossRoom = runState.CurrentRoom;
                if (bossRoom == null) return false;
                postBossRoomType = bossRoom.RoomType;
                return postBossRoomType != RoomType.Boss;
            }, ct, TimeSpan.FromSeconds(BossTransitionTimeoutSeconds),
                "Act transition did not start after boss");

            Logger.Log($"[RlAutoSlayer] Post-boss transition: room type is now {postBossRoomType}");

            if (postBossRoomType == RoomType.Event &&
                runState.CurrentActIndex >= runState.Acts.Count - 1)
            {
                _watchdog?.Reset(
                    $"Entering {postBossRoomType} room (Act {runState.CurrentActIndex + 1}, Floor {runState.ActFloor})");
                await HandleRoomAsync(postBossRoomType, ct);
                await Task.Delay(NonCombatSettleDelayMs, ct);
                await DrainOverlayScreensAsync(ct);
                _watchdog?.Reset("Waiting for main menu after victory");
                await WaitForMainMenuAsync(ct);
                Logger.Log("[RlAutoSlayer] Victory! Run completed and returned to main menu");

                // Notify Python of victory
                BridgeServer.Instance.SendState(RunCompleteState(
                    NonCombatBridgeProtocol.VictoryResult));
                _completionSignalSent = true;
                return true;
            }

            await WaitHelper.Until(
                () => runState.VisitedMapCoords.Count == 0, ct,
                TimeSpan.FromSeconds(ActTransitionTimeoutSeconds),
                "Act transition did not complete (VisitedMapCoords not cleared)");
        }

        _watchdog?.Reset("Navigating map");
        LogTransitionState("MAP_HANDLER_ENTER");
        await _mapHandler.HandleAsync(_random, ct);
        return false;
    }

    /// <summary>
    /// Bring the game back to a state the run loop can continue from.
    ///
    /// Nearly every abort observed in practice is harmless: a screen that did
    /// not close inside its timeout, a room that had already advanced, or a
    /// watchdog timeout fired while a slow LLM was still thinking. None of
    /// them justify discarding the whole run. So: let the game settle, give
    /// the watchdog a fresh lease, drain whatever screen is left over, then
    /// let the loop continue from the CURRENT state -- which may already be
    /// the next room, the rewards screen, or the map.
    /// </summary>
    private async Task RecoverAsync(RoomType roomType, CancellationToken ct)
    {
        Logger.Log($"[RlAutoSlayer] Recovering after a {roomType} failure; the run continues.");
        _watchdog?.Reset($"Recovering after {roomType} failure");

        try
        {
            await Task.Delay(RecoverySettleDelayMs, ct);
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlAutoSlayer] Recovery settle interrupted: {ex.Message}");
        }

        try
        {
            await DrainOverlayScreensAsync(ct);
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlAutoSlayer] Recovery drain failed (continuing anyway): {ex.Message}");
        }

        // TRANSITION BARRIER: draining the leftover screens (rewards / card
        // reward / proceed) does NOT advance the room. If the map is now
        // open we MUST navigate it here -- otherwise the run loop re-enters
        // the SAME (already finished) room, burns all recovery attempts and
        // aborts the run with "Combat not started" (observed on Floor 7
        // Elite resume: card_reward confirmed -> 46s -> terminated).
        if (NMapScreen.Instance?.IsOpen ?? false)
        {
            Logger.Log("[RlAutoSlayer] Recovery: map is open; navigating so the agent chooses the next room");
            _watchdog?.Reset("Recovery map navigation");
            await _mapHandler.HandleAsync(_random, ct);
        }

        _watchdog?.Reset($"Recovered, continuing after {roomType} failure");
    }

    private async Task HandleRoomAsync(RoomType roomType, CancellationToken ct)
    {
        if (!_roomHandlers.TryGetValue(roomType, out IRoomHandler handler))
        {
            Logger.Log($"[RlAutoSlayer] No handler for room type: {roomType}");
            return;
        }
        await WaitHelper.WithTimeout(
            (CancellationToken token) => handler.HandleAsync(_random, token),
            handler.Timeout, ct);
    }

    /// <summary>
    /// §16: formal headful invariant -- when the game stays interactive
    /// (headful), no ICardSelector bypass may be installed, otherwise the
    /// native selection UI is silently resolved by strategy and the run is not
    /// benchmark-valid. Throws if CardSelectCmd.Selector is non-null while
    /// HeadfulMode is true.
    /// </summary>
    private static void AssertNativeUiInvariant()
    {
        if (!HeadfulMode)
            return;

        if (CardSelectCmd.Selector != null)
        {
            throw new InvalidOperationException(
                "NATIVE_UI_BYPASS: unexpected CardSelectCmd.Selector="
                + CardSelectCmd.Selector.GetType().FullName);
        }
    }

    private async Task DrainOverlayScreensAsync(CancellationToken ct)
    {
        if (NOverlayStack.Instance == null)
        {
            await WaitHelper.Until(() => NOverlayStack.Instance != null, ct,
                AutoSlayConfig.nodeWaitTimeout, "Overlay stack not initialized");
        }

        HashSet<IOverlayScreen> handledScreens = new HashSet<IOverlayScreen>();
        int consecutiveFailures = 0;

        while (true)
        {
            NOverlayStack? instance = NOverlayStack.Instance;
            if (instance == null || instance.ScreenCount <= 0)
                break;

            ct.ThrowIfCancellationRequested();

            IOverlayScreen currentOverlay = NOverlayStack.Instance.Peek();
            if (currentOverlay == null)
                break;

            // §18: a native selection screen the coordinator owns must be driven
            // by RlNativeSelectionCoordinator (visible UI + real clicks), never
            // by the legacy AutoSlay handler. The coordinator's watcher handles
            // it concurrently; we suspend the AutoSlayer handler here so the two
            // never race on the same modal.
            if (HeadfulMode && RlNativeSelectionCoordinator.CanHandle(currentOverlay))
            {
                Logger.Log(
                    $"[RlAutoSlayer] Delegating native selection " +
                    $"{currentOverlay.GetType().Name} to coordinator");
                await RlNativeSelectionCoordinator.WaitForSelectionClearAsync(
                    "DrainOverlay", ct);
                consecutiveFailures = 0;
                await Task.Delay(OverlayDrainSettleDelayMs, ct);
                continue;
            }

            if (handledScreens.Contains(currentOverlay))
            {
                consecutiveFailures++;
                if (consecutiveFailures >= OverlayCloseRetryLimit)
                {
                    Logger.Log(
                        $"[RlAutoSlayer] Infinite loop: screen {currentOverlay.GetType().Name} not closing after {OverlayCloseRetryLimit} attempts");
                    throw new InvalidOperationException(
                        "Screen " + currentOverlay.GetType().Name + " not closing after being handled");
                }
            }
            else
            {
                handledScreens.Add(currentOverlay);
                consecutiveFailures = 0;
            }

            Node node = (Node)currentOverlay;
            Type type = node.GetType();

            if (!_screenHandlers.TryGetValue(type, out IScreenHandler handler))
            {
                // Fail loudly instead of silently leaving the overlay open:
                // "drain returned" must never be mistaken for "the room is
                // ready to navigate". The recovery loop logs the exception
                // and the run continues, so this is diagnostics, not a
                // hard abort.
                throw new InvalidOperationException(
                    $"UNHANDLED_OVERLAY_DURING_TRANSITION: {type.FullName}");
            }

            _watchdog?.Reset("Handling screen: " + type.Name);
            Logger.Log($"[RlAutoSlayer] Handling screen: {type.Name}");
            await WaitHelper.WithTimeout(
                (CancellationToken token) => handler.HandleAsync(_random, token),
                handler.Timeout, ct);

            if (currentOverlay is NRewardsScreen &&
                (NMapScreen.Instance?.IsOpen ?? false))
            {
                break;
            }

            await Task.Delay(OverlayDrainSettleDelayMs, ct);
        }
    }

    private async Task ClickRestSiteProceedIfNeeded(CancellationToken ct)
    {
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        NProceedButton nodeOrNull = root.GetNodeOrNull<NProceedButton>(
            RestSiteProceedButtonPath);
        if (nodeOrNull != null && nodeOrNull.IsEnabled)
        {
            Logger.Log("[RlAutoSlayer] Clicking rest site proceed button");
            await UiHelper.Click(nodeOrNull);
        }
    }

    private async Task ClickEventProceedIfNeeded(CancellationToken ct)
    {
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        Node eventRoom = root.GetNodeOrNull(EventRoomPath);
        if (eventRoom == null)
            return;

        NEventOptionButton proceedOption = null;
        await WaitHelper.Until(delegate
        {
            NMapScreen? instance = NMapScreen.Instance;
            if (instance != null && instance.IsOpen) return true;

            List<NEventOptionButton> list = (from o in UiHelper.FindAll<NEventOptionButton>(eventRoom)
                where !o.Option.IsLocked && o.Option.IsProceed
                select o).ToList();
            if (list.Count > 0)
            {
                proceedOption = list[0];
                return true;
            }
            return false;
        }, ct, TimeSpan.FromSeconds(EventProceedTimeoutSeconds),
            "Event proceed option or map did not appear");

        if (proceedOption != null)
        {
            Logger.Log("[RlAutoSlayer] Clicking event proceed option");
            await UiHelper.Click(proceedOption);
        }
    }

    private async Task WaitForRewardsScreenAsync(CancellationToken ct)
    {
        Logger.Log("[RlAutoSlayer] Waiting for rewards screen");
        // Runs right after a combat that may have taken minutes of agent
        // thinking: refresh the watchdog from the condition delegate so the
        // 30s AutoSlayConfig.watchdogTimeout cannot fire here.
        await WaitHelper.Until(
            delegate {
                CurrentWatchdog?.Reset("Waiting for rewards screen after combat");
                return NOverlayStack.Instance?.Peek() is NRewardsScreen ||
                       (NMapScreen.Instance?.IsOpen ?? false);
            },
            ct, TimeSpan.FromSeconds(RewardsScreenTimeoutSeconds),
            "Rewards screen did not appear after combat");
    }

    private async Task WaitForMainMenuAsync(CancellationToken ct)
    {
        Logger.Log("[RlAutoSlayer] Waiting for main menu");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        await WaitHelper.Until(
            () => root.GetNodeOrNull<Control>(MainMenuPath)?.IsVisibleInTree() ?? false,
            ct, TimeSpan.FromSeconds(MainMenuTimeoutSeconds),
            "Main menu did not appear after game over");
    }

    private async Task PlayMainMenuAsync(CancellationToken ct)
    {
        Logger.Log("[RlAutoSlayer] Playing main menu");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        Control mainMenu = await WaitHelper.ForNode<Control>(
            root, MainMenuPath, ct, TimeSpan.FromSeconds(MainMenuTimeoutSeconds));

        // RESUME: continue an existing run save when there is one. This is
        // what makes an aborted run cheap -- restarting the game (or just the
        // agent) picks the run back up from its last save instead of
        // abandoning it and starting all over.
        if (await TryResumeSavedRunAsync(mainMenu, ct))
        {
            return;
        }

        // Abandon existing run if present (best effort)
        try
        {
            NButton abandonBtn = mainMenu.GetNodeOrNull<NButton>(
                AbandonRunMenuButtonPath);
            if (abandonBtn != null && abandonBtn.Visible)
            {
                Logger.Log("[RlAutoSlayer] Abandoning existing run");
                await UiHelper.Click(abandonBtn);
                await Task.Delay(MenuClickSettleDelayMs, ct);
                // Try to find and click Yes on the confirmation popup
                try
                {
                    await WaitHelper.Until(
                        () => NModalContainer.Instance?.OpenModal != null, ct,
                        TimeSpan.FromSeconds(AbandonPopupTimeoutSeconds), "Abandon popup");
                    Node popup = (Node)NModalContainer.Instance.OpenModal;
                    NButton yesBtn = popup.GetNodeOrNull<NButton>(AbandonPopupPrimaryYesButtonPath)
                        ?? popup.GetNodeOrNull<NButton>(AbandonPopupFallbackYesButtonPath);
                    if (yesBtn != null)
                    {
                        await UiHelper.Click(yesBtn);
                        await Task.Delay(MenuClickSettleDelayMs, ct);
                    }
                }
                catch
                {
                    Logger.Log("[RlAutoSlayer] Popup not found, trying to continue anyway");
                }
                await Task.Delay(AbandonRunSettleDelayMs, ct);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlAutoSlayer] Could not abandon run: {ex.Message}, continuing...");
        }

        // Click singleplayer
        NButton spButton = mainMenu.GetNode<NButton>(
            SingleplayerButtonPath);
        Logger.Log("[RlAutoSlayer] Clicking singleplayer");
        await UiHelper.Click(spButton);

        // Navigate to character select
        Control charSelectScreen = mainMenu.GetNodeOrNull<Control>(
            CharacterSelectScreenPath);
        NButton standardButton = mainMenu.GetNodeOrNull<NButton>(
            StandardRunButtonPath);
        await WaitHelper.Until(delegate
        {
            charSelectScreen = mainMenu.GetNodeOrNull<Control>(
                CharacterSelectScreenPath);
            standardButton = mainMenu.GetNodeOrNull<NButton>(
                StandardRunButtonPath);
            bool csVisible = charSelectScreen?.Visible ?? false;
            bool sbVisible = standardButton?.Visible ?? false;
            return csVisible || sbVisible;
        }, ct, AutoSlayConfig.nodeWaitTimeout,
            "Neither CharacterSelectScreen nor SingleplayerSubmenu became visible");

        if (standardButton?.Visible ?? false)
        {
            Control csCtrl = charSelectScreen;
            if (csCtrl == null || !csCtrl.Visible)
            {
                Logger.Log("[RlAutoSlayer] Clicking standard run");
                await UiHelper.Click(standardButton);
                await WaitHelper.Until(
                    () => mainMenu.GetNodeOrNull<Control>(
                        CharacterSelectScreenPath)?.Visible ?? false,
                    ct, AutoSlayConfig.nodeWaitTimeout,
                    "CharacterSelectScreen did not become visible");
                charSelectScreen = mainMenu.GetNode<Control>(
                    CharacterSelectScreenPath);
            }
        }

        // Select Ironclad (first character) — our agent was trained on Ironclad
        Node buttonContainer = charSelectScreen.GetNode(
            CharacterButtonContainerPath);
        List<NCharacterSelectButton> buttons =
            UiHelper.FindAll<NCharacterSelectButton>(buttonContainer);
        foreach (NCharacterSelectButton btn in buttons)
        {
            btn.UnlockIfPossible();
        }
        List<NCharacterSelectButton> available =
            buttons.Where(b => !b.IsLocked).ToList();

        // Pick Ironclad (first available) instead of random
        NCharacterSelectButton selectedChar = available.FirstOrDefault(
            b => b.Character.Id.Entry.Contains(PreferredCharacterId,
                StringComparison.OrdinalIgnoreCase))
            ?? available.First();
        Logger.Log($"[RlAutoSlayer] Selecting character: {selectedChar.Character.Id}");
        selectedChar.Select();
        await Task.Delay(CharacterSelectDelayMs, ct);

        NButton confirmBtn = await WaitHelper.ForNode<NButton>(
            mainMenu, CharacterConfirmButtonPath, ct);
        Logger.Log("[RlAutoSlayer] Confirming character");
        await UiHelper.Click(confirmBtn);
    }

    /// <summary>
    /// Resume an existing run save when one exists.
    ///
    /// This is what makes an aborted run recoverable: restarting the game (or
    /// just restarting the agent) picks the run back up from its last save
    /// instead of abandoning it and starting from scratch. Enabled by default
    /// (<see cref="ResumeSavedRun"/>).
    /// </summary>
    private static async Task<bool> TryResumeSavedRunAsync(
        Control mainMenu, CancellationToken ct)
    {
        if (!ResumeSavedRun)
            return false;

        try
        {
            NButton? continueButton =
                mainMenu.GetNodeOrNull<NButton>(ContinueRunButtonPath);
            if (continueButton == null || !continueButton.Visible
                || !continueButton.IsEnabled)
            {
                return false;
            }

            AutoSlayLog.Action("Saved run found -- CONTINUING it");
            await UiHelper.Click(continueButton);
            await WaitHelper.Until(delegate {
                RunState? rs = RunManager.Instance.DebugOnlyGetState();
                return rs != null && rs.CurrentRoom != null;
            }, ct, TimeSpan.FromSeconds(RunStateTimeoutSeconds),
                "Continued run did not load");
            AutoSlayLog.Action("Saved run loaded, resuming");
            return true;
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn(
                $"[RlAutoSlayer] Resume failed, starting a new run instead: {ex.Message}");
            return false;
        }
    }

    private async Task AbandonRunAsync(CancellationToken ct)
    {
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        await Task.Delay(AbandonRunSettleDelayMs, ct);
        await UiHelper.Click(await WaitHelper.ForNode<NButton>(
            root,
            AbandonRunOptionsButtonPath,
            ct));
        await UiHelper.Click(await WaitHelper.ForNode<NButton>(
            root,
            AbandonRunButtonPath,
            ct));
        await UiHelper.Click(await WaitHelper.ForNode<NButton>(
            root,
            AbandonRunProceedButtonPath,
            ct));
    }

    private static Exception? _lastAbortException;

    private static string RunCompleteState(string result)
    {
        var fields = new Dictionary<string, object>
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.RunCompleteState,
            [NonCombatBridgeProtocol.ResultField] = result,
        };
        if (_lastAbortException != null)
        {
            fields["debug_reason"] = _lastAbortException.Message;
            fields["debug_exception"] = _lastAbortException.GetType().Name;
        }
        return JsonSerializer.Serialize(fields);
    }

    /// <summary>
    /// Developer-only transition breadcrumb (GPT-reviewed remediation):
    /// logs the state-machine context around a screen/room transition.
    /// NEVER serialized to the agent -- Logger output only.
    /// </summary>
    internal static void LogTransitionState(string phase)
    {
        try
        {
            var runState = RunManager.Instance?.DebugOnlyGetState();
            var overlay = NOverlayStack.Instance?.Peek();
            Logger.Log(
                "[Transition] " + phase
                + $" floor={runState?.TotalFloor}"
                + $" actFloor={runState?.ActFloor}"
                + $" room={runState?.CurrentRoom?.RoomType}"
                + $" visited={runState?.VisitedMapCoords?.Count}"
                + $" overlayCount={NOverlayStack.Instance?.ScreenCount}"
                + $" overlay={overlay?.GetType().Name ?? "null"}"
                + $" mapOpen={NMapScreen.Instance?.IsOpen}");
        }
        catch (Exception ex)
        {
            Logger.Log($"[Transition] {phase} (logging failed: {ex.Message})");
        }
    }

    private static void SetSharedCurrentWatchdog(Watchdog? watchdog)
    {
        try
        {
            PropertyInfo? property = typeof(AutoSlayer).GetProperty(
                "CurrentWatchdog",
                BindingFlags.Public | BindingFlags.Static);
            property?.SetValue(null, watchdog);
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlAutoSlayer] Could not mirror watchdog: {ex.Message}");
        }
    }
}

/// <summary>
/// RL-aware GameOverScreenHandler. Same as the original but also notifies
/// the Python agent about game over.
/// </summary>
public class RlGameOverScreenHandler : IScreenHandler, IHandler
{
    private const int HandlerTimeoutMinutes = 2;
    private const int ContinueButtonTimeoutSeconds = 30;
    private const int SummaryAnimationTimeoutSeconds = 90;
    private const int WatchdogRefreshCycles = 20;

    public Type ScreenType => typeof(NGameOverScreen);
    public TimeSpan Timeout => TimeSpan.FromMinutes(HandlerTimeoutMinutes);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[RlGameOver] Game over screen appeared");
        NGameOverScreen screen =
            (NGameOverScreen)NOverlayStack.Instance.Peek();

        // Notify Python that the game is over
        BridgeServer.Instance.SendState(JsonSerializer.Serialize(
            new Dictionary<string, object>
            {
                [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.GameOverState,
                [NonCombatBridgeProtocol.MessageField] = NonCombatBridgeProtocol.GameOverMessage,
            }));

        NGameOverContinueButton continueButton =
            UiHelper.FindFirst<NGameOverContinueButton>(screen);
        if (continueButton == null)
        {
            Logger.Log("[RlGameOver] Continue button not found");
            return;
        }

        await WaitHelper.Until(() => continueButton.IsEnabled, ct,
            TimeSpan.FromSeconds(ContinueButtonTimeoutSeconds),
            "Continue button did not become enabled");
        await UiHelper.Click(continueButton);

        NReturnToMainMenuButton mainMenuButton = null;
        int waitCycles = 0;
        await WaitHelper.Until(delegate
        {
            if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
                return true;
            mainMenuButton = UiHelper.FindFirst<NReturnToMainMenuButton>(screen);
            waitCycles++;
            if (waitCycles % WatchdogRefreshCycles == 0)
            {
                RlAutoSlayer.CurrentWatchdog?.Reset("Waiting for game over summary animation");
            }
            return mainMenuButton != null && mainMenuButton.Visible && mainMenuButton.IsEnabled;
        }, ct, TimeSpan.FromSeconds(SummaryAnimationTimeoutSeconds),
            "Main menu button did not become enabled");

        if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
            return;

        await UiHelper.Click(mainMenuButton);
        await WaitHelper.Until(
            () => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(),
            ct, TimeSpan.FromSeconds(ContinueButtonTimeoutSeconds),
            "Game over screen did not close");
    }
}
