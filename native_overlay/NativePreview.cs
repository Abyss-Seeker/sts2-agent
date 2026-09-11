#if DEBUG
using Godot;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

namespace STS2AgentOverlay;

// Debug-only fixture: load the real speech scene without starting a game/run.
public partial class NativePreview : Node
{
    public override void _Ready()
    {
        foreach (string original in new[] {
            string.Concat(Enumerable.Repeat("这是完整的一句解释，接下来再考虑药水。", 80)) + "最后一个字。",
            string.Concat(Enumerable.Repeat("Use this potion before choosing the next room. ", 30)) + "FINAL WORD.",
            "This extraordinarywordmustneverbesplitinthemiddle stays intact. 👨‍👩‍👧‍👦e\u0301" })
        {
            var pages = Overlay.Paginate(original);
            if (string.Concat(pages) != original) throw new Exception("Pagination lost text");
            if (pages.Any(p => p.EndsWith("\ud83d"))) throw new Exception("Pagination split a grapheme");
        }
        GD.Print("SPEECH_PAGINATION_OK");
        string? pck = System.Environment.GetEnvironmentVariable("STS2_OVERLAY_PREVIEW_PCK");
        if (string.IsNullOrEmpty(pck)) return;
        if (!ProjectSettings.LoadResourcePack(pck)) throw new InvalidOperationException("Could not load game PCK");
        Godot.Bridge.ScriptManagerBridge.LookupScriptsInAssembly(typeof(NSpeechBubbleVfx).Assembly);
        MegaCrit.Sts2.Core.Saves.SaveManager.Instance.InitPrefsDataForTest();
        var bubble = NSpeechBubbleVfx.Create("先稳住阵脚，用格挡抵消这次攻击，再把剩余能量留给下一次反击。", DialogueSide.Left, new Vector2(450, 430), 15);
        if (bubble == null) throw new InvalidOperationException("Native speech factory returned null");
        bubble.Name = "NativeExample";
        AddChild(bubble);
        string sample = "先稳住阵脚，用格挡抵消这次攻击，再把剩余能量留给下一次反击。";
        if (!NativeSpeech.ConfigureText(bubble, sample)) throw new Exception("Native fit failed");
        if (bubble.GetNode<RichTextLabel>("%Text").GetParsedText() != sample) throw new Exception("Native text lost suffix");
        GD.Print("NATIVE_SPEECH_FACTORY_OK");
    }
}
#endif
