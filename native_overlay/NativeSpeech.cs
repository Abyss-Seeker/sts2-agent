using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Runs;

namespace STS2AgentOverlay;

/// <summary>Use the game's banter/ping VFX, never broadcast multiplayer pings.</summary>
internal sealed class NativeSpeech : IDisposable
{
    private NSpeechBubbleVfx? bubble;
    private NCombatRoom? owner;
    private string current = "";
    private double retryAfter;
    private string tooLarge = "";

    public bool Show(string text, double seconds, float scale)
    {
        if (Time.GetTicksMsec() < retryAfter || text == tooLarge) return false;
        try { return ShowInternal(text, seconds, scale); }
        catch (Exception ex)
        {
            GD.PushWarning($"Agent overlay native speech unavailable: {ex.Message}");
            Clear();
            retryAfter = Time.GetTicksMsec() + 5000;
            return false;
        }
    }

    private bool ShowInternal(string text, double seconds, float scale)
    {
        // This covers selection, inspect, map, modals, capstones and event combat.
        if (NGame.Instance == null) { Clear(); return false; }
        var room = NCombatRoom.Instance;
        if (room == null || !room.IsVisibleInTree() ||
            ActiveScreenContext.Instance.GetCurrentScreen() != room)
        { Clear(); return false; }
        if (owner == room && current == text && GodotObject.IsInstanceValid(bubble))
        {
            var rendered = bubble!.GetNodeOrNull<RichTextLabel>("%Text");
            return rendered != null && rendered.GetContentHeight() <= rendered.Size.Y
                && rendered.GetContentWidth() <= rendered.Size.X;
        }
        Clear();
        var player = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        var creature = room.GetCreatureNode(player?.Creature);
        if (player == null || creature == null || !creature.IsVisibleInTree()) return false;
        bubble = NSpeechBubbleVfx.Create(text.Replace("[", "[lb]"), player.Creature, seconds);
        if (bubble == null) return false;
        room.CombatVfxContainer.AddChild(bubble);
        bubble.Scale = Vector2.One * scale;
        IgnoreMouse(bubble);
        if (!ConfigureText(bubble, text)) { tooLarge = text; Clear(); return false; }
        owner = room;
        current = text;
        return true;
    }

    internal static bool ConfigureText(NSpeechBubbleVfx bubble, string text)
    {
        var label = bubble.GetNodeOrNull<RichTextLabel>("%Text");
        if (label == null) return false;
        using var font = new SystemFont { FontNames = new[] { "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI" } };
        label.AddThemeFontOverride("normal_font", font);
        // Preserve the native frame, fade and positioning. Our explanation
        // needs no per-character fly-in effect or hidden scrolling suffix.
        label.Text = "[center]" + text.Replace("[", "[lb]") + "[/center]";
        label.ScrollActive = false;
        for (int size = label.GetThemeFontSize("normal_font_size"); size >= 18; size--)
        {
            label.AddThemeFontSizeOverride("normal_font_size", size);
            if (label.GetContentHeight() <= label.Size.Y && label.GetContentWidth() <= label.Size.X) return true;
        }
        return false;
    }

    private static void IgnoreMouse(Node node)
    {
        if (node is Control control) control.MouseFilter = Control.MouseFilterEnum.Ignore;
        foreach (var child in node.GetChildren()) IgnoreMouse(child);
    }

    public void Clear()
    {
        if (GodotObject.IsInstanceValid(bubble)) { bubble!.Hide(); bubble.QueueFree(); }
        bubble = null; owner = null; current = "";
    }

    public void Dispose() => Clear();
}
