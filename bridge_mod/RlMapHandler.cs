// RlMapHandler.cs -- RL-agent-driven map navigation handler.
//
// Replaces AutoSlay's MapScreenHandler. Instead of picking the first child
// of the current map point, this handler:
//   1. Enumerates available map nodes (the reachable next nodes)
//   2. Sends them to Python with their types (Monster, Elite, Shop, etc.)
//   3. Waits for the agent to choose a node index
//   4. Clicks the chosen node
//
// Falls back to random selection if Python is disconnected or times out.

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
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public class RlMapHandler : IScreenHandler, IHandler
{
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    private TaskCompletionSource? _roomEnteredTcs;

    public Type ScreenType => typeof(NMapScreen);
    public TimeSpan Timeout => TimeSpan.FromSeconds(60);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[RlMap] Handling map screen");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        NRun runNode = root.GetNode<NRun>("/root/Game/RootSceneContainer/Run");

        await WaitHelper.Until(
            delegate {
                RlAutoSlayer.CurrentWatchdog?.Reset("Waiting for map screen");
                return runNode.GlobalUi.MapScreen.IsVisibleInTree();
            },
            ct, AutoSlayConfig.mapScreenTimeout, "Map screen not visible");

        List<NMapPoint> allPoints = UiHelper.FindAll<NMapPoint>(runNode.GlobalUi.MapScreen);
        RunState runState = RunManager.Instance.DebugOnlyGetState();
        Logger.Log(
            $"[RlMap] Map visible; allPoints={allPoints.Count},"
            + $" visited={runState.VisitedMapCoords.Count}");

        // Determine available next nodes
        List<NMapPoint> availableNodes;
        if (runState.VisitedMapCoords.Count == 0)
        {
            // First room selection: all nodes in row 0
            availableNodes = allPoints
                .Where(mp => mp.Point.coord.row == 0)
                .ToList();
        }
        else
        {
            // Get the children of the last visited node
            IReadOnlyList<MapCoord> visited = runState.VisitedMapCoords;
            MapCoord lastCoord = visited[visited.Count - 1];
            Logger.Log($"[RlMap] lastCoord=({lastCoord.row},{lastCoord.col})");
            foreach (var mp in allPoints)
            {
                Logger.Log(
                    $"[RlMap] UI point=({mp.Point.coord.row},{mp.Point.coord.col})"
                    + $" enabled={mp.IsEnabled} type={mp.Point.PointType}");
            }
            NMapPoint? lastNode = allPoints.FirstOrDefault(
                mp => mp.Point.coord.Equals(lastCoord));
            if (lastNode == null)
            {
                // Diagnostics invariant: the run's last visited coordinate
                // MUST exist in the visible map UI. A bare First() would
                // throw InvalidOperationException without any context.
                throw new InvalidOperationException(
                    $"MAP_INVARIANT_LAST_COORD_MISSING: "
                    + $"lastCoord=({lastCoord.row},{lastCoord.col}), "
                    + $"allPoints={allPoints.Count}, "
                    + $"visitedCount={visited.Count}, "
                    + $"mapOpen={NMapScreen.Instance?.IsOpen}");
            }
            HashSet<MapCoord> childCoords = new HashSet<MapCoord>(
                lastNode.Point.Children.Select(c => c.coord));
            availableNodes = allPoints
                .Where(mp => childCoords.Contains(mp.Point.coord))
                .ToList();
        }

        if (availableNodes.Count == 0)
        {
            // Never silently "continue" from a broken map state: the run
            // would re-enter the old room and die later with an unrelated
            // error. Fail loudly with the full visible context instead.
            throw new InvalidOperationException(
                $"MAP_INVARIANT_NO_AVAILABLE_NODES: "
                + $"lastCoord=({runState.VisitedMapCoords[^1].row},{runState.VisitedMapCoords[^1].col}), "
                + $"allPoints={allPoints.Count}, "
                + $"visitedCount={runState.VisitedMapCoords.Count}, "
                + $"mapOpen={NMapScreen.Instance?.IsOpen}");
        }

        // Build the state message for Python
        var nodes = new List<Dictionary<string, object>>();
        for (int i = 0; i < availableNodes.Count; i++)
        {
            NMapPoint mp = availableNodes[i];
            nodes.Add(new Dictionary<string, object>
            {
                ["index"] = i,
                ["type"] = mp.Point.PointType.ToString(),
                ["row"] = mp.Point.coord.row,
                ["col"] = mp.Point.coord.col,
            });
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = "map_select",
            ["nodes"] = nodes,
            ["floor"] = runState.TotalFloor,
            ["act"] = runState.CurrentActIndex + 1,
        };

        // Act identity + boss(es): the map screen shows the boss icon(s)
        // for the current act, so this is human-visible information.
        try
        {
            var actInfo = CardSerialization.SerializeActInfo(runState);
            if (actInfo != null)
                stateMsg["act_info"] = actInfo;
        }
        catch (Exception ex)
        {
            Logger.Log("[RlMap] Act info serialization failed: " + ex.Message);
        }

        // Full act map -- everything a human player sees on the map screen:
        // every node (row/col/room type), its child edges, and the visited path.
        try
        {
            var pointByCoord = new Dictionary<(int, int), NMapPoint>();
            foreach (NMapPoint mp in allPoints)
                pointByCoord[(mp.Point.coord.row, mp.Point.coord.col)] = mp;

            var fullMap = new List<Dictionary<string, object>>();
            foreach (NMapPoint mp in allPoints)
            {
                var children = new List<Dictionary<string, object>>();
                foreach (var child in mp.Point.Children)
                {
                    pointByCoord.TryGetValue(
                        (child.coord.row, child.coord.col), out var childPoint);
                    children.Add(new Dictionary<string, object>
                    {
                        ["row"] = child.coord.row,
                        ["col"] = child.coord.col,
                        ["type"] = childPoint != null
                            ? childPoint.Point.PointType.ToString()
                            : "UNKNOWN",
                    });
                }
                fullMap.Add(new Dictionary<string, object>
                {
                    ["row"] = mp.Point.coord.row,
                    ["col"] = mp.Point.coord.col,
                    ["type"] = mp.Point.PointType.ToString(),
                    ["children"] = children,
                });
            }
            stateMsg["full_map"] = fullMap;

            var visited = new List<Dictionary<string, object>>();
            foreach (var coord in runState.VisitedMapCoords)
            {
                visited.Add(new Dictionary<string, object>
                {
                    ["row"] = coord.row,
                    ["col"] = coord.col,
                });
            }
            stateMsg["visited"] = visited;
        }
        catch (Exception ex)
        {
            Logger.Log("[RlMap] Full map serialization failed: " + ex.Message);
        }

        NMapPoint chosenNode;

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
                    var rRoot = doc.RootElement;
                    int chosenIndex = rRoot.GetProperty("index").GetInt32();

                    if (chosenIndex >= 0 && chosenIndex < availableNodes.Count)
                    {
                        chosenNode = availableNodes[chosenIndex];
                        Logger.Log(
                            $"[RlMap] Agent chose node {chosenIndex}: {chosenNode.Point.PointType} at ({chosenNode.Point.coord.row},{chosenNode.Point.coord.col})");
                    }
                    else
                    {
                        Logger.Log($"[RlMap] Invalid index {chosenIndex}, falling back to random");
                        chosenNode = random.NextItem(availableNodes);
                    }
                }
                else
                {
                    Logger.Log("[RlMap] No response from agent, falling back to random");
                    chosenNode = random.NextItem(availableNodes);
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[RlMap] Agent error: {ex.Message}, falling back to random");
                chosenNode = random.NextItem(availableNodes);
            }
        }
        else
        {
            Logger.Log("[RlMap] No agent connected, selecting random node");
            chosenNode = random.NextItem(availableNodes);
        }

        // Wait for the node to be enabled and click it
        // Immediately after the agent's map decision (which can take tens of
        // seconds): refresh the watchdog, or the 30s AutoSlay watchdogTimeout
        // fires on the first Check() inside Until().
        await WaitHelper.Until(
            delegate {
                RlAutoSlayer.CurrentWatchdog?.Reset("Waiting for chosen map point");
                return chosenNode.IsEnabled;
            },
            ct, TimeSpan.FromSeconds(10), "Map point not enabled");

        _roomEnteredTcs = new TaskCompletionSource();
        RunManager.Instance.RoomEntered += OnRoomEntered;
        try
        {
            await UiHelper.Click(chosenNode);
            await WaitHelper.ForTask(_roomEnteredTcs.Task, ct,
                AutoSlayConfig.mapScreenTimeout, "Room not entered after map click");
        }
        finally
        {
            RunManager.Instance.RoomEntered -= OnRoomEntered;
            _roomEnteredTcs = null;
        }

        Logger.Log("[RlMap] Map navigation complete");
    }

    private void OnRoomEntered()
    {
        _roomEnteredTcs?.TrySetResult();
    }
}
