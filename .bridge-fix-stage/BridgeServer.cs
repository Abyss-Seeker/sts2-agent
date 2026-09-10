// BridgeServer.cs -- TCP server for RL agent communication.
//
// Protocol: newline-delimited JSON over TCP (one JSON object per line).
//   Game -> Agent:  state messages (combat_action, map_select, card_reward, etc.)
//   Agent -> Game:  action messages (play, end_turn, choose, skip)
//
// Threading model:
//   - TCP accept/read loop runs on a background thread (Task.Run)
//   - Game handlers call SendState() and WaitForActionAsync() from the game thread
//   - WaitForActionAsync() blocks the calling async context until a response arrives
//
// The server accepts exactly one client at a time. If the client disconnects,
// it goes back to listening for a new connection.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;

namespace STS2BridgeMod;

public class BridgeServer
{
    public static readonly BridgeServer Instance = new();

    /// <summary>
    /// Human-parity safe mode switch. When false, combat handlers abort the
    /// episode on agent timeout instead of playing a random card (which would
    /// make evaluation outcomes partly non-agent decisions). Default true for
    /// interactive/diagnostic use; training/evaluation clients should send
    /// {"action":"set_fallback","enabled":false} right after connecting.
    /// </summary>
    public static bool AllowRandomFallback = false;

    /// <summary>
    /// Per-decision agent wait used by every screen/room handler. The Python
    /// client can retune it at runtime via
    /// {"action":"set_agent_timeout","seconds":N} (clamped to 10..300) so slow
    /// reasoning models still fit inside the game's decision window.
    /// </summary>
    public static int AgentTimeoutSeconds = 90;

    /// <summary>Handler-level watchdog: per-screen total, agent wait + margin.</summary>
    public static int HandlerTimeoutSeconds => AgentTimeoutSeconds + 15;

    // ----------------------------------------------------------------
    // Nested decisions
    // ----------------------------------------------------------------
    //
    // One decision can be requested from INSIDE another: using a potion (or
    // playing a card) during combat may open a card-selection prompt, and
    // that prompt asks the agent through the very same bridge.
    //
    // The bridge holds exactly ONE pending request slot, so the nested call
    // preempts the outer one:
    //   1. SendStateAndWaitForActionAsync() cancels the outer pending wait;
    //   2. the agent's reply -- correlated with the OUTER request_id -- is
    //      then dropped as "stale";
    //   3. the outer handler sees a null response and, in safe mode, aborts
    //      the whole run as an "agent timeout".
    // That is exactly what happened when a potion opened a card selection
    // during combat: the agent's end_turn was dropped, the turn never ended,
    // and the run was terminated.
    //
    // Tracking the nesting depth lets the outer handler recognise this case:
    // wait for the nested prompt to finish, then re-send a freshly
    // serialized (already settled) state instead of failing the run.
    private static int _nestedDecisionDepth;

    /// <summary>
    /// True while an agent decision is being requested from inside another
    /// (outer) decision -- e.g. a card selection opened by a potion used
    /// during combat.
    /// </summary>
    public static bool HasNestedDecision => Volatile.Read(ref _nestedDecisionDepth) > 0;

    public static void EnterNestedDecision() => Interlocked.Increment(ref _nestedDecisionDepth);

    public static void ExitNestedDecision() => Interlocked.Decrement(ref _nestedDecisionDepth);

    private TcpListener? _listener;
    private TcpClient? _client;
    private NetworkStream? _stream;
    private readonly object _lock = new();
    private bool _running;
    private CancellationTokenSource? _cts;

    private readonly byte[] _readBuffer = new byte[8192];
    private string _readRemainder = "";

    // Pending action response mechanism: when a handler sends state and waits
    // for a response, it sets _pendingAction. The read loop sets the result
    // when a complete line arrives.
    private TaskCompletionSource<string>? _pendingAction;
    private string? _pendingRequestId;
    private readonly object _pendingLock = new();
    private long _requestCounter;

    // Authoritative outcome of ONE gameplay command, waiting to be attached
    // to the very NEXT real game state as "previous_action_result".
    // request_id correlates it with the exact state/action handshake.
    private string? _pendingActionResultJson;

    /// <summary>
    /// Whether a Python client is currently connected.
    /// </summary>
    public bool IsClientConnected
    {
        get
        {
            lock (_lock)
            {
                return _client?.Connected == true;
            }
        }
    }

    private BridgeServer() { }

    /// <summary>
    /// Start listening for client connections on the given port.
    /// </summary>
    public void Start(int port)
    {
        if (_running) return;
        _running = true;
        _cts = new CancellationTokenSource();

        _listener = new TcpListener(IPAddress.Loopback, port);
        _listener.Start();
        Logger.Log($"[BridgeServer] Listening on 127.0.0.1:{port}");

        Task.Run(() => AcceptLoopAsync(_cts.Token));
    }

    /// <summary>
    /// Stop the server and disconnect any client.
    /// </summary>
    public void Stop()
    {
        _running = false;
        _cts?.Cancel();
        CancelPendingAction("Server stopped");
        DisconnectClient();
        _listener?.Stop();
        Logger.Log("[BridgeServer] Server stopped.");
    }

    /// <summary>
    /// Send a state JSON message to the connected client.
    /// Thread-safe; can be called from any thread.
    /// </summary>
    public void SendState(string stateJson)
    {
        SendStateInternal(stateJson);
    }

    private bool SendStateInternal(string stateJson)
    {
        lock (_lock)
        {
            if (_stream == null || _client?.Connected != true)
                return false;

            try
            {
                // Do not consume previous_action_result when nobody can
                // receive it. It remains queued for the next real state.
                stateJson = EnrichWithRunSummary(stateJson);
                stateJson = AttachPendingActionResult(stateJson);
                byte[] data = Encoding.UTF8.GetBytes(stateJson + "\n");
                _stream.Write(data, 0, data.Length);
                _stream.Flush();
                return true;
            }
            catch (Exception ex)
            {
                Logger.Log($"[BridgeServer] Error sending state: {ex.Message}");
                DisconnectClient();
                return false;
            }
        }
    }

    /// <summary>
    /// Record the AUTHORITATIVE game-side outcome of one gameplay command so
    /// it can be attached to the very next real game state.
    ///
    /// SENT != ACCEPTED: a successful TCP send from Python proves only that
    /// bytes left the client. Only the game-side handler can say whether the
    /// command was accepted, so this is the sole source of truth.
    ///
    /// <paramref name="requestId"/> is the request_id of the state whose
    /// action was answered; Python correlates the outcome with the exact
    /// in-flight command and refuses to let a stale ACK classify a newer one.
    /// </summary>
    public void RecordActionResult(string? requestId, bool accepted, string? reason = null)
    {
        var payload = new Dictionary<string, object?>
        {
            ["request_id"] = requestId ?? "",
            ["accepted"] = accepted,
        };
        if (!string.IsNullOrEmpty(reason))
            payload["reason"] = reason;

        lock (_lock)
        {
            _pendingActionResultJson = JsonSerializer.Serialize(payload);
        }
    }

    /// <summary>
    /// Attach the recorded previous_action_result (if any) to the next REAL
    /// game state, exactly once. Ack/control messages (pong/ok/error) are
    /// never carriers, so a pending result can never be lost to a control
    /// reply. Controller metadata only -- it is correlated by request_id and
    /// is not part of the agent's human-visible observation.
    /// </summary>
    private string AttachPendingActionResult(string stateJson)
    {
        try
        {
            if (JsonNode.Parse(stateJson) is not JsonObject node)
                return stateJson;
            string type = node["type"] is JsonValue v
                ? v.GetValue<string>() : "";
            if (type is "pong" or "ok" or "error")
                return stateJson;

            string? result;
            lock (_lock)
            {
                result = _pendingActionResultJson;
                _pendingActionResultJson = null;
            }
            if (result == null)
                return stateJson;

            node["previous_action_result"] = JsonNode.Parse(result);
            return node.ToJsonString();
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] AttachActionResult failed: {ex.Message}");
            return stateJson;
        }
    }

    /// <summary>
    /// Human-parity: attach the run/player snapshot (HP, gold, relics,
    /// potions, full deck) to every state message that does not already
    /// carry one, so the agent sees on every screen what a human player
    /// sees. Ack messages (pong/ok) are left untouched.
    /// </summary>
    private static string EnrichWithRunSummary(string stateJson)
    {
        try
        {
            var node = JsonNode.Parse(stateJson) as JsonObject;
            if (node == null || node["type"] == null)
                return stateJson;

            string type = node["type"]!.GetValue<string>();
            if (type == "pong" || type == "ok" || type == "error")
                return stateJson;

            // FLAT injection: the player summary becomes the state's
            // "player" object directly (hp/gold/relics/deck at one level),
            // and the act info goes to the state top level. Nesting the
            // summary used to hide the player fields from the agent.
            var playerSummary = RunSummary.BuildPlayerSummary();
            if (node["player"] == null && playerSummary != null)
                node["player"] = JsonSerializer.SerializeToNode(playerSummary);
            var actInfo = RunSummary.BuildActInfo();
            if (node["act_info"] == null && actInfo != null)
                node["act_info"] = JsonSerializer.SerializeToNode(actInfo);
            var progress = RunSummary.BuildProgressInfo();
            if (progress != null)
            {
                if (node["floor"] == null)
                    node["floor"] = JsonValue.Create((int)progress["floor"]);
                if (node["act"] == null)
                    node["act"] = JsonValue.Create((int)progress["act"]);
            }
            return node.ToJsonString();
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] RunSummary enrich failed: {ex.Message}");
            return stateJson;
        }
    }

    public async Task<string?> SendStateAndWaitForActionAsync(
        string stateJson, TimeSpan timeout, CancellationToken ct)
    {
        string requestId = Interlocked.Increment(ref _requestCounter).ToString();
        TaskCompletionSource<string> tcs;
        lock (_pendingLock)
        {
            if (_pendingAction != null && !_pendingAction.Task.IsCompleted)
            {
                Logger.Log(
                    "[BridgeServer] ERROR: overlapping pending agent " +
                    "decisions detected");
            }
            _pendingAction?.TrySetCanceled();
            tcs = new TaskCompletionSource<string>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            _pendingAction = tcs;
            _pendingRequestId = requestId;
        }

        try
        {
            string payload = AttachRequestId(stateJson, requestId);
            if (!SendStateInternal(payload))
            {
                return null;
            }
            return await WaitForPendingActionAsync(tcs, timeout, ct);
        }
        finally
        {
            // An agent round-trip can legitimately take far longer than
            // AutoSlayConfig.watchdogTimeout (30s), and handlers wait via
            // WaitHelper.Until(), which calls Watchdog.Check() on EVERY poll.
            // Refreshing here means "the agent just answered" always counts as
            // progress: without it, any Until() entered right after a slow
            // decision aborts the run as a bogus watchdog timeout. This is the
            // central guard -- it also covers handlers that were not
            // individually updated.
            RlAutoSlayer.CurrentWatchdog?.Reset("Agent decision received");

            lock (_pendingLock)
            {
                if (_pendingAction == tcs)
                {
                    _pendingAction = null;
                    _pendingRequestId = null;
                }
            }
        }
    }

    /// <summary>
    /// Wait for the next action message from the Python client.
    /// This is the primary mechanism for handlers to receive agent decisions.
    ///
    /// Returns the raw JSON string of the action message, or null on timeout.
    /// </summary>
    public async Task<string?> WaitForActionAsync(
        TimeSpan timeout, CancellationToken ct)
    {
        TaskCompletionSource<string> tcs = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            lock (_pendingLock)
            {
                _pendingAction?.TrySetCanceled();
                _pendingAction = tcs;
                _pendingRequestId = null;
            }
            return await WaitForPendingActionAsync(tcs, timeout, ct);
        }
        finally
        {
            lock (_pendingLock)
            {
                if (_pendingAction == tcs)
                {
                    _pendingAction = null;
                    _pendingRequestId = null;
                }
            }
        }
    }

    private static async Task<string?> WaitForPendingActionAsync(
        TaskCompletionSource<string> tcs, TimeSpan timeout, CancellationToken ct)
    {
        try
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
                ct, timeoutCts.Token);

            using var reg = linkedCts.Token.Register(() =>
            {
                tcs.TrySetCanceled();
            });

            return await tcs.Task;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] WaitForAction error: {ex.Message}");
            return null;
        }
    }

    // ----------------------------------------------------------------
    // Background thread methods
    // ----------------------------------------------------------------

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        while (_running && !ct.IsCancellationRequested)
        {
            try
            {
                Logger.Log("[BridgeServer] Waiting for client connection...");
                var client = await _listener!.AcceptTcpClientAsync(ct);
                Logger.Log(
                    $"[BridgeServer] Client connected from {client.Client.RemoteEndPoint}");

                lock (_lock)
                {
                    client.SendTimeout = 5000;
                    client.ReceiveTimeout = 5000;
                    _client = client;
                    _stream = client.GetStream();
                    _stream.WriteTimeout = 5000;
                    _stream.ReadTimeout = 5000;
                    _readRemainder = "";
                }

                await HandleClientAsync(ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Logger.Log($"[BridgeServer] Accept error: {ex.Message}");
                await Task.Delay(1000, ct);
            }
        }
    }

    private async Task HandleClientAsync(CancellationToken ct)
    {
        try
        {
            while (_running && !ct.IsCancellationRequested)
            {
                NetworkStream? stream;
                lock (_lock)
                {
                    stream = _stream;
                }
                if (stream == null) break;

                int bytesRead = await stream.ReadAsync(
                    _readBuffer, 0, _readBuffer.Length, ct);
                if (bytesRead == 0)
                {
                    Logger.Log("[BridgeServer] Client disconnected (read 0 bytes).");
                    break;
                }

                _readRemainder += Encoding.UTF8.GetString(_readBuffer, 0, bytesRead);

                while (_readRemainder.Contains('\n'))
                {
                    int idx = _readRemainder.IndexOf('\n');
                    string line = _readRemainder[..idx].Trim();
                    _readRemainder = _readRemainder[(idx + 1)..];

                    if (string.IsNullOrEmpty(line))
                        continue;

                    await ProcessIncomingMessage(line);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] Client read error: {ex.Message}");
        }
        finally
        {
            CancelPendingAction("Client disconnected");
            DisconnectClient();
        }
    }

    /// <summary>
    /// B1: start a run with an explicit seed and acknowledge it.
    /// Ack: {"type":"start_run_ack","success":bool,"requested_seed":...,
    /// "actual_seed":...,"character":...,"difficulty":...,"run_id":...,
    /// "error":...}
    /// The seed is benchmark-controller metadata -- it is NEVER part of the
    /// agent's observation (human parity).
    /// </summary>
    private async Task HandleStartRunAsync(JsonElement root)
    {
        string seed = root.TryGetProperty("seed", out var seedProp)
            ? seedProp.GetString() ?? ""
            : "";
        string character = root.TryGetProperty("character", out var charProp)
            ? charProp.GetString() ?? RlAutoSlayer.PreferredCharacterId
            : RlAutoSlayer.PreferredCharacterId;
        int difficulty = 0;
        if (root.TryGetProperty("difficulty", out var diffProp)
            && diffProp.TryGetInt32(out int d))
        {
            difficulty = d;
        }

        try
        {
            var slayer = RlAutoSlayer.Active ?? new RlAutoSlayer();
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));
            bool ok = await slayer.StartRunAsync(seed, cts.Token);
            string ack = RlAutoSlayer.BuildStartRunAck(
                seed, character, difficulty, ok,
                ok ? "" : "run did not start with the requested seed");
            Logger.Log($"[BridgeServer] start_run ack: {ack}");
            SendState(ack);
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] start_run failed: {ex.Message}");
            SendState(RlAutoSlayer.BuildStartRunAck(
                seed, character, difficulty, false, ex.Message));
        }
    }

    /// <summary>
    /// Process an incoming message from the Python client.
    /// If there's a pending WaitForActionAsync, deliver the message to it.
    /// Otherwise handle special messages (PING).
    /// </summary>
    private async Task ProcessIncomingMessage(string json, bool onGameThread = false)
    {
        try
        {
            // Check for PING
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("action", out var actionProp))
            {
                string action = actionProp.GetString() ?? "";
                if (!onGameThread && action.ToLowerInvariant() is
                    "resume_automation" or "set_headful" or "set_fast_mode" or "start_run")
                {
                    await MainFile.OnGameThreadAsync(() => ProcessIncomingMessage(json, true));
                    return;
                }
                if (action.Equals("ping", StringComparison.OrdinalIgnoreCase))
                {
                    SendState("{\"type\":\"pong\"}");
                    return;
                }
                if (action.Equals("set_fallback", StringComparison.OrdinalIgnoreCase))
                {
                    // Human-parity safe mode: when disabled, handlers must NOT
                    // substitute a random decision for a missing agent response.
                    // Used by evaluation/training so episode outcomes are the
                    // agent's own decisions only.
                    if (root.TryGetProperty("enabled", out var enabledProp))
                    {
                        AllowRandomFallback = enabledProp.GetBoolean();
                        Logger.Log("[BridgeServer] Random fallback "
                            + (AllowRandomFallback ? "ENABLED" : "DISABLED")
                            + " (human-parity safe mode)");
                        SendState("{\"type\":\"ok\",\"fallback\":"
                            + (AllowRandomFallback ? "true" : "false") + "}");
                    }
                    return;
                }
                if (action.Equals("set_agent_timeout", StringComparison.OrdinalIgnoreCase))
                {
                    // Slow reasoning models need more than the default window;
                    // retune every handler's per-decision wait at runtime.
                    if (root.TryGetProperty("seconds", out var secondsProp)
                        && secondsProp.TryGetInt32(out int seconds))
                    {
                        AgentTimeoutSeconds = Math.Clamp(seconds, 10, 300);
                        Logger.Log($"[BridgeServer] Agent timeout set to "
                            + $"{AgentTimeoutSeconds}s per decision");
                        SendState("{\"type\":\"ok\",\"agent_timeout\":"
                            + AgentTimeoutSeconds + "}");
                    }
                    return;
                }
                if (action.Equals("resume_automation", StringComparison.OrdinalIgnoreCase))
                {
                    bool active = RlAutoSlayer.EnsureAutomationActive();
                    Logger.Log($"[BridgeServer] Resume automation requested; active={active}");
                    SendState("{\"type\":\"ok\",\"automation_active\":"
                        + active.ToString().ToLower() + "}");
                    return;
                }
                if (action.Equals("set_headful", StringComparison.OrdinalIgnoreCase))
                {
                    // §31: headful = game stays interactive (BGM/SFX/waits);
                    // headless = legacy NonInteractiveMode suppression.
                    if (root.TryGetProperty("enabled", out var headfulProp))
                    {
                        bool headful = headfulProp.GetBoolean();
                        RlAutoSlayer.ApplyHeadfulMode(headful);
                        Logger.Log($"[BridgeServer] Headful mode set to {headful}");
                        SendState("{\"type\":\"ok\",\"headful\":" + headful.ToString().ToLower() + "}");
                    }
                    return;
                }
                if (action.Equals("start_run", StringComparison.OrdinalIgnoreCase))
                {
                    // B1: explicit, seed-controlled run start for the formal
                    // paired benchmark. Acknowledges requested vs ACTUAL seed
                    // so the runner can only claim seed_applied_to_game when
                    // they match (never a silent "probably paired").
                    await HandleStartRunAsync(root);
                    return;
                }
                if (action.Equals("set_fast_mode", StringComparison.OrdinalIgnoreCase))
                {
                    // §35: FastMode is independent of headful mode.
                    if (root.TryGetProperty("enabled", out var fastProp))
                    {
                        bool fast = fastProp.GetBoolean();
                        RlAutoSlayer.ApplyFastMode(fast);
                        Logger.Log($"[BridgeServer] Fast mode set to {fast}");
                        SendState("{\"type\":\"ok\",\"fast_mode\":" + fast.ToString().ToLower() + "}");
                    }
                    return;
                }
            }

            // Also support legacy "type" field for ping
            if (root.TryGetProperty("type", out var typeProp))
            {
                string type = typeProp.GetString() ?? "";
                if (type.Equals("PING", StringComparison.OrdinalIgnoreCase))
                {
                    SendState("{\"type\":\"pong\"}");
                    return;
                }
            }
        }
        catch
        {
            // If we can't parse, still deliver it to the pending action
        }

        // Deliver to pending WaitForActionAsync
        lock (_pendingLock)
        {
            if (_pendingAction != null)
            {
                try
                {
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    string? requestId = root.TryGetProperty("request_id", out var requestProp)
                        ? requestProp.GetString()
                        : null;
                    if (_pendingRequestId != null && requestId != _pendingRequestId)
                    {
                        Logger.Log(
                            $"[BridgeServer] Dropping stale action for request_id={requestId ?? "null"}, expected {_pendingRequestId}");
                        return;
                    }
                }
                catch
                {
                    if (_pendingRequestId != null)
                    {
                        Logger.Log("[BridgeServer] Dropping unparsable action while waiting for a correlated request.");
                        return;
                    }
                }
                _pendingAction.TrySetResult(json);
                _pendingAction = null;
                _pendingRequestId = null;
                return;
            }
        }

        // No pending action -- log and discard
        Logger.Log($"[BridgeServer] Received action with no handler waiting: {json}");
    }

    private void CancelPendingAction(string reason)
    {
        lock (_pendingLock)
        {
            _pendingAction?.TrySetCanceled();
            _pendingAction = null;
            _pendingRequestId = null;
        }
    }

    private void DisconnectClient()
    {
        lock (_lock)
        {
            // A dropped client can never receive the pending outcome; drop it
            // rather than leak it onto an unrelated future connection.
            _pendingActionResultJson = null;
            _stream?.Close();
            _stream = null;
            _client?.Close();
            _client = null;
        }
    }

    private static string AttachRequestId(string stateJson, string requestId)
    {
        try
        {
            Dictionary<string, object?> payload =
                JsonSerializer.Deserialize<Dictionary<string, object?>>(stateJson)
                ?? new Dictionary<string, object?>();
            payload["request_id"] = requestId;
            return JsonSerializer.Serialize(payload);
        }
        catch
        {
            return stateJson;
        }
    }
}
