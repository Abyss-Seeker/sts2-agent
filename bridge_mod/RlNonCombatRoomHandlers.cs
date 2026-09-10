// RlNonCombatRoomHandlers.cs -- bridge-driven non-combat room handlers.

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
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

internal static class NonCombatBridgeProtocol
{
    public const string TypeField = "type";
    public const string ResultField = "result";
    public const string MessageField = "message";
    public const string CardSelectState = "card_select";
    public const string CardRewardState = "card_reward";
    public const string GameOverState = "game_over";
    public const string RunCompleteState = "run_complete";
    public const string RestSiteState = "rest_site";
    public const string RewardScreenState = "reward_screen";
    public const string CardBundleState = "card_bundle";
    public const string CrystalSphereState = "crystal_sphere";
    public const string ShopState = "shop";
    public const string EventState = "event";
    public const string TreasureState = "treasure";
    public const string BossRelicState = "boss_relic";
    public const string ChooseAction = "choose";
    public const string SkipAction = "skip";
    public const string RestOptionAction = "rest_option";
    public const string LeaveShopAction = "leave_shop";
    public const string BuyCardAction = "buy_card";
    public const string BuyRelicAction = "buy_relic";
    public const string BuyPotionAction = "buy_potion";
    public const string RemoveCardAction = "remove_card";
    public const string BuyItemAction = "buy_item";
    public const string EventChoiceAction = "event_choice";
    public const string CollectTreasureAction = "collect";
    public const string PickRelicAction = "pick_relic";
    public const string PickRewardAction = "pick_reward";
    public const string PickCardBundleAction = "pick_card_bundle";
    public const string DivineCellAction = "divine_cell";
    public const string ProceedAction = "proceed";
    public const string VictoryResult = "victory";
    public const string TerminatedResult = "terminated";
    public const string GameOverMessage = "Game over";
}

public class RlRestSiteRoomHandler : IRoomHandler, IHandler
{
    private const string RoomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/RestSiteRoom";
    private const int ProceedResponseTimeoutSeconds = 10;
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public RoomType[] HandledTypes => new[] { RoomType.RestSite };
    public TimeSpan Timeout => TimeSpan.FromSeconds(BridgeServer.HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.Action("Waiting for rest site room");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        NRestSiteRoom room = await WaitHelper.ForNode<NRestSiteRoom>(root, RoomPath, ct);
        List<NRestSiteButton> buttons = UiHelper.FindAll<NRestSiteButton>(room)
            .Where(button => button.Option.IsEnabled)
            .ToList();

        if (buttons.Count == 0)
        {
            AutoSlayLog.Warn("No clickable rest site buttons found");
            return;
        }

        NRestSiteButton chosenButton = await ChooseRestSiteButton(buttons, random, ct);
        AutoSlayLog.Action("Selecting rest site option: " + chosenButton.Option.OptionId);
        await UiHelper.Click(chosenButton);

        NProceedButton proceedButton = room.ProceedButton;
        await WaitHelper.Until(delegate
        {
            if (!proceedButton.IsEnabled)
            {
                NOverlayStack? stack = NOverlayStack.Instance;
                return stack != null && stack.ScreenCount > 0;
            }
            return true;
        }, ct, TimeSpan.FromSeconds(ProceedResponseTimeoutSeconds), "Rest site option did not respond");

        NOverlayStack? overlayStack = NOverlayStack.Instance;
        if (overlayStack != null && overlayStack.ScreenCount > 0)
        {
            AutoSlayLog.Action("Overlay screen detected, deferring proceed to drain loop");
            return;
        }

        AutoSlayLog.Action("Clicking proceed");
        await UiHelper.Click(proceedButton);
    }

    private static async Task<NRestSiteButton> ChooseRestSiteButton(
        List<NRestSiteButton> buttons,
        Rng random,
        CancellationToken ct)
    {
        if (!BridgeServer.Instance.IsClientConnected)
        {
            return random.NextItem(buttons);
        }

        try
        {
            string stateJson = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["type"] = NonCombatBridgeProtocol.RestSiteState,
                ["options"] = buttons.Select((button, index) => new Dictionary<string, object>
                {
                    ["index"] = index,
                    ["id"] = button.Option.OptionId,
                    ["action"] = NonCombatBridgeProtocol.RestOptionAction,
                    ["label"] = button.Option.Title.GetFormattedText(),
                    ["description"] = button.Option.Description.GetFormattedText(),
                    ["enabled"] = button.Option.IsEnabled,
                }).ToList(),
                ["floor"] = RunManager.Instance.DebugOnlyGetState().TotalFloor,
                ["act"] = RunManager.Instance.DebugOnlyGetState().CurrentActIndex + 1,
            });
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                stateJson,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return random.NextItem(buttons);
            }
            int chosenIndex = ReadChoiceIndex(responseJson);
            if (chosenIndex >= 0 && chosenIndex < buttons.Count)
            {
                return buttons[chosenIndex];
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlRestSite] Agent error: " + ex.Message);
        }

        return random.NextItem(buttons);
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
}

public class RlTreasureRoomHandler : IRoomHandler, IHandler
{
    private const string RoomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/TreasureRoom";
    private const int ChestOpenDelayMs = 1000;
    private const int RelicPickupDelayMs = 500;
    private const int ProceedTimeoutSeconds = 5;
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public RoomType[] HandledTypes => new[] { RoomType.Treasure };
    public TimeSpan Timeout => TimeSpan.FromSeconds(BridgeServer.HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.Action("Waiting for treasure room");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        NTreasureRoom room = await WaitHelper.ForNode<NTreasureRoom>(root, RoomPath, ct);

        NClickableControl chest = room.GetNode<NClickableControl>("Chest");
        AutoSlayLog.Action("Opening chest");
        await UiHelper.Click(chest);
        await Task.Delay(ChestOpenDelayMs, ct);

        List<NTreasureRoomRelicHolder> holders = UiHelper.FindAll<NTreasureRoomRelicHolder>(room)
            .Where(holder => holder.IsEnabled && holder.Visible)
            .ToList();
        if (holders.Count > 0)
        {
            NTreasureRoomRelicHolder chosenHolder = await ChooseTreasureRelicHolder(holders, random, ct);
            AutoSlayLog.Action("Picking up treasure relic: " + chosenHolder.Relic.Model.Id.Entry);
            await UiHelper.Click(chosenHolder);
            await Task.Delay(RelicPickupDelayMs, ct);
        }

        NProceedButton proceedButton = room.ProceedButton;
        await WaitHelper.Until(
            () => proceedButton.IsEnabled,
            ct,
            TimeSpan.FromSeconds(ProceedTimeoutSeconds),
            "Proceed button not enabled after picking relics");
        AutoSlayLog.Action("Clicking proceed");
        await UiHelper.Click(proceedButton);
    }

    private static async Task<NTreasureRoomRelicHolder> ChooseTreasureRelicHolder(
        List<NTreasureRoomRelicHolder> holders,
        Rng random,
        CancellationToken ct)
    {
        if (!BridgeServer.Instance.IsClientConnected)
        {
            return random.NextItem(holders);
        }

        try
        {
            string stateJson = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["type"] = NonCombatBridgeProtocol.TreasureState,
                ["options"] = holders.Select((holder, index) => RelicOption(
                    index,
                    NonCombatBridgeProtocol.CollectTreasureAction,
                    holder.Relic.Model)).ToList(),
                ["floor"] = RunManager.Instance.DebugOnlyGetState().TotalFloor,
                ["act"] = RunManager.Instance.DebugOnlyGetState().CurrentActIndex + 1,
            });
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                stateJson,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return random.NextItem(holders);
            }
            int chosenIndex = ReadChoiceIndex(responseJson);
            if (chosenIndex >= 0 && chosenIndex < holders.Count)
            {
                return holders[chosenIndex];
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlTreasure] Agent error: " + ex.Message);
        }

        return random.NextItem(holders);
    }

    private static Dictionary<string, object> RelicOption(int index, string action, RelicModel relic)
    {
        return new Dictionary<string, object>
        {
            ["index"] = index,
            ["id"] = relic.Id.Entry,
            ["action"] = action,
            ["label"] = relic.Title.GetFormattedText(),
            ["description"] = relic.DynamicDescription.GetFormattedText(),
            ["enabled"] = true,
        };
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
}

public class RlChooseARelicScreenHandler : IScreenHandler, IHandler
{
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public Type ScreenType => typeof(NChooseARelicSelection);
    public TimeSpan Timeout => TimeSpan.FromSeconds(BridgeServer.HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.EnterScreen("NChooseARelicSelection");
        NChooseARelicSelection currentScreen = AutoSlayer.GetCurrentScreen<NChooseARelicSelection>();
        List<NRelicBasicHolder> holders = UiHelper.FindAll<NRelicBasicHolder>(currentScreen)
            .Where(holder => holder.IsEnabled && holder.Visible)
            .ToList();
        if (holders.Count == 0)
        {
            AutoSlayLog.Warn("No relic holders found in relic selection screen");
            return;
        }

        NRelicBasicHolder chosenHolder = await ChooseBossRelicHolder(holders, random, ct);
        AutoSlayLog.Action("Selecting boss relic: " + chosenHolder.Relic.Model.Id.Entry);
        await UiHelper.Click(chosenHolder);
        AutoSlayLog.ExitScreen("NChooseARelicSelection");
    }

    private static async Task<NRelicBasicHolder> ChooseBossRelicHolder(
        List<NRelicBasicHolder> holders,
        Rng random,
        CancellationToken ct)
    {
        if (!BridgeServer.Instance.IsClientConnected)
        {
            return random.NextItem(holders);
        }

        try
        {
            string stateJson = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["type"] = NonCombatBridgeProtocol.BossRelicState,
                ["options"] = holders.Select((holder, index) => RelicOption(
                    index,
                    NonCombatBridgeProtocol.PickRelicAction,
                    holder.Relic.Model)).ToList(),
                ["floor"] = RunManager.Instance.DebugOnlyGetState().TotalFloor,
                ["act"] = RunManager.Instance.DebugOnlyGetState().CurrentActIndex + 1,
            });
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                stateJson,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return random.NextItem(holders);
            }
            int chosenIndex = ReadChoiceIndex(responseJson);
            if (chosenIndex >= 0 && chosenIndex < holders.Count)
            {
                return holders[chosenIndex];
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlChooseARelic] Agent error: " + ex.Message);
        }

        return random.NextItem(holders);
    }

    private static Dictionary<string, object> RelicOption(int index, string action, RelicModel relic)
    {
        return new Dictionary<string, object>
        {
            ["index"] = index,
            ["id"] = relic.Id.Entry,
            ["action"] = action,
            ["label"] = relic.Title.GetFormattedText(),
            ["description"] = relic.DynamicDescription.GetFormattedText(),
            ["enabled"] = true,
        };
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
}

public class RlShopRoomHandler : IRoomHandler, IHandler
{
    private const string RoomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/MerchantRoom";
    private const int MaxPurchaseAttempts = 50;
    private const int InventoryOpenDelayMs = 500;
    private const int PurchaseSettleDelayMs = 300;
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public RoomType[] HandledTypes => new[] { RoomType.Shop };
    public TimeSpan Timeout => TimeSpan.FromSeconds(Math.Max(120, BridgeServer.HandlerTimeoutSeconds));

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.Action("Waiting for shop room");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        NMerchantRoom room = await WaitHelper.ForNode<NMerchantRoom>(root, RoomPath, ct);

        AutoSlayLog.Action("Opening merchant inventory");
        room.OpenInventory();
        await Task.Delay(InventoryOpenDelayMs, ct);

        int attempts = 0;
        while (attempts < MaxPurchaseAttempts)
        {
            ct.ThrowIfCancellationRequested();
            attempts++;

            List<NMerchantSlot> purchasableSlots = room.Inventory.GetAllSlots()
                .Where(slot => slot.Entry.IsStocked && slot.Entry.EnoughGold)
                .ToList();

            ShopChoice choice = await ChooseShopOption(purchasableSlots, random, ct);
            if (choice.ShouldLeave)
            {
                break;
            }

            AutoSlayLog.Action($"Buying shop option at index {choice.OptionIndex}");
            await choice.Slot!.Entry.OnTryPurchaseWrapper(room.Inventory.Inventory);
            await Task.Delay(PurchaseSettleDelayMs, ct);

            NOverlayStack? overlayStack = NOverlayStack.Instance;
            if (overlayStack != null && overlayStack.ScreenCount > 0)
            {
                AutoSlayLog.Action("Overlay screen opened during shop, deferring to drain loop");
                break;
            }
        }

        if (attempts >= MaxPurchaseAttempts)
        {
            AutoSlayLog.Warn($"Shop hit purchase attempt limit ({MaxPurchaseAttempts})");
        }

        NBackButton? backButton = UiHelper.FindFirst<NBackButton>(room);
        if (backButton != null)
        {
            AutoSlayLog.Action("Closing inventory");
            await UiHelper.Click(backButton);
            await Task.Delay(PurchaseSettleDelayMs, ct);
        }

        AutoSlayLog.Action("Clicking proceed");
        await UiHelper.Click(room.ProceedButton);
    }

    private static async Task<ShopChoice> ChooseShopOption(
        List<NMerchantSlot> purchasableSlots,
        Rng random,
        CancellationToken ct)
    {
        if (purchasableSlots.Count == 0)
        {
            return ShopChoice.Leave();
        }

        if (!BridgeServer.Instance.IsClientConnected)
        {
            return ShopChoice.Buy(random.NextItem(purchasableSlots), 1);
        }

        try
        {
            List<Dictionary<string, object>> options = new()
            {
                new Dictionary<string, object>
                {
                    ["index"] = 0,
                    ["id"] = NonCombatBridgeProtocol.LeaveShopAction,
                    ["action"] = NonCombatBridgeProtocol.LeaveShopAction,
                    ["label"] = "Leave shop",
                    ["enabled"] = true,
                },
            };
            options.AddRange(purchasableSlots.Select((slot, slotIndex) => ShopOption(slot, slotIndex + 1)));

            string stateJson = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["type"] = NonCombatBridgeProtocol.ShopState,
                ["options"] = options,
                ["floor"] = RunManager.Instance.DebugOnlyGetState().TotalFloor,
                ["act"] = RunManager.Instance.DebugOnlyGetState().CurrentActIndex + 1,
            });
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                stateJson,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return ShopChoice.Buy(random.NextItem(purchasableSlots), 1);
            }
            int chosenIndex = ReadChoiceIndex(responseJson);
            if (chosenIndex == 0)
            {
                return ShopChoice.Leave();
            }
            int slotIndex = chosenIndex - 1;
            if (slotIndex >= 0 && slotIndex < purchasableSlots.Count)
            {
                return ShopChoice.Buy(purchasableSlots[slotIndex], chosenIndex);
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlShop] Agent error: " + ex.Message);
        }

        return ShopChoice.Buy(random.NextItem(purchasableSlots), 1);
    }

    private static Dictionary<string, object> ShopOption(NMerchantSlot slot, int index)
    {
        string action = ShopAction(slot);
        Dictionary<string, object> option = new()
        {
            ["index"] = index,
            ["id"] = action,
            ["action"] = action,
            ["label"] = ShopEntryId(slot),
            ["description"] = $"Cost: {slot.Entry.Cost}",
            ["enabled"] = slot.Entry.IsStocked && slot.Entry.EnoughGold,
            ["price"] = slot.Entry.Cost,
        };
        return option;
    }

    private static string ShopAction(NMerchantSlot slot)
    {
        return slot switch
        {
            NMerchantCard => NonCombatBridgeProtocol.BuyCardAction,
            NMerchantRelic => NonCombatBridgeProtocol.BuyRelicAction,
            NMerchantPotion => NonCombatBridgeProtocol.BuyPotionAction,
            NMerchantCardRemoval => NonCombatBridgeProtocol.RemoveCardAction,
            _ => NonCombatBridgeProtocol.BuyItemAction,
        };
    }

    private static string ShopEntryId(NMerchantSlot slot)
    {
        return slot.Entry switch
        {
            MerchantCardEntry cardEntry => cardEntry.CreationResult?.Card.Id.Entry ?? "card",
            MerchantRelicEntry relicEntry => relicEntry.Model?.Id.Entry ?? "relic",
            MerchantPotionEntry potionEntry => potionEntry.Model?.Id.Entry ?? "potion",
            MerchantCardRemovalEntry => NonCombatBridgeProtocol.RemoveCardAction,
            _ => slot.Entry.GetType().Name,
        };
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
        if (action == NonCombatBridgeProtocol.SkipAction)
        {
            return 0;
        }
        return -1;
    }

    private readonly record struct ShopChoice(bool ShouldLeave, NMerchantSlot? Slot, int OptionIndex)
    {
        public static ShopChoice Leave() => new(true, null, 0);

        public static ShopChoice Buy(NMerchantSlot slot, int optionIndex) => new(false, slot, optionIndex);
    }
}

public class RlEventRoomHandler : IRoomHandler, IHandler
{
    private const string RoomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/EventRoom";
    private const int MaxIterations = 50;
    private const int HandlerTimeoutMinutes = 12;
    private const int EventRoomResumeDelayMs = 500;
    private const int EventProceedCloseTimeoutSeconds = 5;
    // The choice result may open sub-screens (card select/upgrade, rewards)
    // whose resolution requires a full agent round-trip (LLM latency +
    // retries). 5s was shorter than one agent decision and aborted the run.
    private const int EventChoiceResultTimeoutSecondsBase = 90;
    private const int EventOptionsLoadTimeoutSeconds = 30;
    private const int EventOptionsLogInterval = 50;
    private const int FakeMerchantProceedTimeoutSeconds = 10;
    private const int AncientDialoguePollDelayMs = 100;
    private const int AncientDialogueClickDelayMs = 500;
    private const int AncientOptionsTimeoutSeconds = 10;
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(BridgeServer.AgentTimeoutSeconds);

    public RoomType[] HandledTypes => new[] { RoomType.Event };
    public TimeSpan Timeout => TimeSpan.FromMinutes(HandlerTimeoutMinutes);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.Action("Waiting for event room");
        Node eventRoom = await WaitForEventRoom(ct);
        if (await WaitForEventOptions(eventRoom, ct))
        {
            AutoSlayLog.Action("Event room completed");
            return;
        }

        int iterations = 0;
        while (iterations < MaxIterations)
        {
            ct.ThrowIfCancellationRequested();
            if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree())
            {
                if (!await TryResumeEventAfterCombat(random, ct))
                {
                    AutoSlayLog.Action("Event room no longer valid, exiting");
                    break;
                }

                Node root = ((SceneTree)Engine.GetMainLoop()).Root;
                Node? resumedRoom = root.GetNodeOrNull(RoomPath);
                if (resumedRoom == null)
                {
                    AutoSlayLog.Action("Event ended after combat");
                    break;
                }
                eventRoom = resumedRoom;
                await Task.Delay(EventRoomResumeDelayMs, ct);
            }

            List<NEventOptionButton> options = UiHelper.FindAll<NEventOptionButton>(eventRoom)
                .Where(option => !option.Option.IsLocked)
                .ToList();
            if (options.Count == 0)
            {
                break;
            }

            NEventOptionButton choice = await ChooseEventOption(options, random, ct);
            AutoSlayLog.Action("Selecting event option: " + choice.Option.TextKey);
            await UiHelper.Click(choice);

            if (choice.Option.IsProceed)
            {
                AutoSlayLog.Action("Clicked proceed, exiting event");
                await WaitHelper.Until(
                    () => !GodotObject.IsInstanceValid(eventRoom)
                        || !eventRoom.IsInsideTree()
                        || (NMapScreen.Instance?.IsOpen ?? false),
                    ct,
                    TimeSpan.FromSeconds(EventProceedCloseTimeoutSeconds),
                    "Event room did not close after clicking proceed");
                break;
            }

            await WaitForEventChoiceResult(eventRoom, ct);
            NOverlayStack? overlayStack = NOverlayStack.Instance;
            if (overlayStack != null && overlayStack.ScreenCount > 0)
            {
                AutoSlayLog.Action("Overlay screen opened during event, deferring to drain loop");
                break;
            }
            iterations++;
        }

        if (iterations >= MaxIterations)
        {
            AutoSlayLog.Warn($"Event room hit iteration limit ({MaxIterations})");
        }
        AutoSlayLog.Action("Event room completed");
    }

    private static async Task<NEventOptionButton> ChooseEventOption(
        List<NEventOptionButton> options,
        Rng random,
        CancellationToken ct)
    {
        if (!BridgeServer.Instance.IsClientConnected)
        {
            return random.NextItem(options);
        }

        try
        {
            // Event narrative: the title and currently visible body text the
            // human reads on the event screen (same LocStrings the UI uses).
            // LocString.Exists guards against multi-page events (e.g. Neow)
            // whose Description loc entry is absent -- calling GetFormattedText
            // on a missing entry returns the raw loc key, which is not
            // human-visible text and must not be sent.
            string eventTitle = "";
            string eventBody = "";
            try
            {
                var ev = options[0].Event;
                if (ev.Title.Exists())
                    eventTitle = CardSerialization.CleanText(ev.Title.GetFormattedText());
                if (ev.Description != null && ev.Description.Exists())
                    eventBody = CardSerialization.CleanText(ev.Description.GetFormattedText());
            }
            catch { }

            string stateJson = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["type"] = NonCombatBridgeProtocol.EventState,
                ["event_title"] = eventTitle,
                ["event_description"] = eventBody,
                ["options"] = options.Select((option, index) => new Dictionary<string, object>
                {
                    ["index"] = index,
                    ["id"] = NonCombatBridgeProtocol.EventChoiceAction,
                    ["action"] = NonCombatBridgeProtocol.EventChoiceAction,
                    ["label"] = option.Option.Title.GetFormattedText(),
                    ["description"] = option.Option.Description.GetFormattedText(),
                    ["enabled"] = !option.Option.IsLocked,
                    ["event_id"] = option.Event.Id.Entry,
                }).ToList(),
                ["floor"] = RunManager.Instance.DebugOnlyGetState().TotalFloor,
                ["act"] = RunManager.Instance.DebugOnlyGetState().CurrentActIndex + 1,
            });
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                stateJson,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return random.NextItem(options);
            }
            int chosenIndex = ReadChoiceIndex(responseJson);
            if (chosenIndex >= 0 && chosenIndex < options.Count)
            {
                return options[chosenIndex];
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlEvent] Agent error: " + ex.Message);
        }

        return random.NextItem(options);
    }

    private static async Task<Node> WaitForEventRoom(CancellationToken ct)
    {
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;
        return await WaitHelper.ForNode<Node>(root, RoomPath, ct);
    }

    private static async Task<bool> WaitForEventOptions(Node eventRoom, CancellationToken ct)
    {
        NAncientEventLayout ancientLayout = UiHelper.FindFirst<NAncientEventLayout>(eventRoom);
        if (ancientLayout != null)
        {
            await HandleAncientEventDialogue(ancientLayout, ct);
            return false;
        }

        NFakeMerchant fakeMerchant = UiHelper.FindFirst<NFakeMerchant>(eventRoom);
        if (fakeMerchant != null)
        {
            AutoSlayLog.Info("Detected custom event: FakeMerchant");
            await HandleFakeMerchantEvent(fakeMerchant, ct);
            return true;
        }

        int waitCycles = 0;
        await WaitHelper.Until(delegate
        {
            waitCycles++;
            if (waitCycles % EventOptionsLogInterval == 0)
            {
                AutoSlayLog.Info($"Waiting for event options: {eventRoom.GetChildCount()} children in event room");
                RlAutoSlayer.CurrentWatchdog?.Reset("Waiting for event options to load");
            }
            return UiHelper.FindAll<NEventOptionButton>(eventRoom).Count > 0;
        }, ct, TimeSpan.FromSeconds(EventOptionsLoadTimeoutSeconds), "Event options not loaded");
        return false;
    }

    private static async Task WaitForEventChoiceResult(Node eventRoom, CancellationToken ct)
    {
        await WaitHelper.Until(delegate
        {
            NOverlayStack? overlayStack = NOverlayStack.Instance;
            if (overlayStack != null && overlayStack.ScreenCount > 0)
            {
                return true;
            }
            if (NMapScreen.Instance?.IsOpen ?? false)
            {
                return true;
            }
            return !GodotObject.IsInstanceValid(eventRoom)
                || !eventRoom.IsInsideTree()
                || UiHelper.FindAll<NEventOptionButton>(eventRoom).Any(option => !option.Option.IsLocked);
        }, ct,
            TimeSpan.FromSeconds(Math.Max(
                EventChoiceResultTimeoutSecondsBase,
                BridgeServer.AgentTimeoutSeconds * 3)),
            "Event options did not reappear after choice");
    }

    private static async Task<bool> TryResumeEventAfterCombat(Rng random, CancellationToken ct)
    {
        RunState runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null || runState.CurrentRoomCount <= 1)
        {
            return false;
        }
        AbstractRoom? baseRoom = runState.BaseRoom;
        if (baseRoom == null || baseRoom.RoomType != RoomType.Event)
        {
            return false;
        }

        AutoSlayLog.Action("Event triggered combat, routing through bridge combat handler");
        await new RlCombatHandler().HandleAsync(random, ct);
        AutoSlayLog.Action("Combat finished, checking if event resumes");
        return true;
    }

    private static async Task HandleFakeMerchantEvent(NFakeMerchant fakeMerchant, CancellationToken ct)
    {
        AutoSlayLog.Action("Handling FakeMerchant event");
        NProceedButton? proceedButton = null;
        await WaitHelper.Until(delegate
        {
            proceedButton = UiHelper.FindFirst<NProceedButton>(fakeMerchant);
            return proceedButton != null && proceedButton.IsEnabled && proceedButton.Visible;
        }, ct, TimeSpan.FromSeconds(FakeMerchantProceedTimeoutSeconds), "FakeMerchant proceed button not available");
        AutoSlayLog.Action("Clicking FakeMerchant proceed button");
        await UiHelper.Click(proceedButton!);
    }

    private static async Task HandleAncientEventDialogue(NAncientEventLayout ancientLayout, CancellationToken ct)
    {
        AutoSlayLog.Info("Detected Ancient event, clicking through dialogue");
        int clicks = 0;
        while (clicks < MaxIterations)
        {
            ct.ThrowIfCancellationRequested();
            if (!GodotObject.IsInstanceValid(ancientLayout))
            {
                break;
            }

            List<NEventOptionButton> options = UiHelper.FindAll<NEventOptionButton>(ancientLayout)
                .Where(button => button.IsEnabled && !button.Option.IsLocked)
                .ToList();
            if (options.Count > 0)
            {
                AutoSlayLog.Info($"Ancient dialogue finished, {options.Count} options available");
                break;
            }

            NButton hitbox = ancientLayout.GetNodeOrNull<NButton>("%DialogueHitbox");
            if (hitbox == null || !hitbox.Visible || !hitbox.IsEnabled)
            {
                await Task.Delay(AncientDialoguePollDelayMs, ct);
                continue;
            }

            AutoSlayLog.Info($"Clicking Ancient dialogue (click {clicks + 1})");
            RlAutoSlayer.CurrentWatchdog?.Reset("Clicking Ancient event dialogue");
            hitbox.EmitSignal(NClickableControl.SignalName.Released, hitbox);
            clicks++;
            await Task.Delay(AncientDialogueClickDelayMs, ct);
        }

        await WaitHelper.Until(
            () => UiHelper.FindAll<NEventOptionButton>(ancientLayout)
                .Any(button => button.IsEnabled && !button.Option.IsLocked),
            ct,
            TimeSpan.FromSeconds(AncientOptionsTimeoutSeconds),
            "Ancient event options did not become available after dialogue");
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
}
