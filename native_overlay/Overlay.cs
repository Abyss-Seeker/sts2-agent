using Godot;
using MegaCrit.Sts2.Core.Modding;
using System.Globalization;
using System.Text.Json;

namespace STS2AgentOverlay;

[ModInitializer(nameof(Initialize))]
public partial class Overlay : CanvasLayer
{
    public static void Initialize() => Callable.From(() =>
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        if (tree.Root.GetNodeOrNull("AgentRecordingOverlay") == null)
            tree.Root.AddChild(new Overlay { Name = "AgentRecordingOverlay" });
    }).CallDeferred();

    private OverlaySettings settings = new();
    private const string SettingsPath = "user://agent_overlay.cfg";
    private readonly NativeSpeech nativeSpeech = new();
    private HttpRequest request = null!, languageRequest = null!;
    private Label subtitle = null!, feedStatus = null!;
    private RichTextLabel feed = null!;
    private PanelContainer panel = null!;
    private readonly List<JsonElement> history = new();
    private double poll, sinceResponse = 100;
    private long bubbleSeq = -1, cursor;
    private string streamId = "", language = "zh";
    private List<string> speechPages = new();
    private readonly Queue<string> speechQueue = new();
    private int speechPage;
    private double pageAge;
    private bool busy, catchUp, feedDirty = true, forceFeedRefresh;
    private Control menu = null!;

    public override void _Ready()
    {
        Layer = 100;
        ProcessMode = ProcessModeEnum.Always;
        using var preferences = new ConfigFile();
        if (preferences.Load(SettingsPath) == Error.Ok)
        {
            try { settings = JsonSerializer.Deserialize<OverlaySettings>(preferences.GetValue("overlay", "json", "{}").AsString()) ?? new(); }
            catch (JsonException) { settings = new(); }
        }
        settings.Normalize();
        subtitle = new Label { Name = "Subtitle", MouseFilter = Control.MouseFilterEnum.Ignore,
            AutowrapMode = TextServer.AutowrapMode.WordSmart, HorizontalAlignment = HorizontalAlignment.Center,
            MaxLinesVisible = -1, ClipText = false };
        subtitle.AddThemeConstantOverride("outline_size", 6);
        subtitle.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0, 0.85f));
        SetFont(subtitle, "font");
        AddChild(subtitle);
        panel = new PanelContainer { Name = "LogPanel", MouseFilter = Control.MouseFilterEnum.Ignore };
        AddChild(panel);
        var column = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        panel.AddChild(column);
        feedStatus = new Label { MouseFilter = Control.MouseFilterEnum.Ignore };
        SetFont(feedStatus, "font");
        column.AddChild(feedStatus);
        feed = new RichTextLabel { Name = "Feed", BbcodeEnabled = false, ScrollActive = true, ScrollFollowing = false,
            SelectionEnabled = true, SizeFlagsVertical = Control.SizeFlags.ExpandFill,
            MouseFilter = Control.MouseFilterEnum.Ignore };
        SetFont(feed, "normal_font");
        column.AddChild(feed);
        request = new HttpRequest { Timeout = 3, BodySizeLimit = 16 * 1024 * 1024 };
        AddChild(request);
        request.RequestCompleted += OnResponse;
        languageRequest = new HttpRequest { Timeout = 3, BodySizeLimit = 4096 };
        AddChild(languageRequest);
        languageRequest.RequestCompleted += OnLanguageResponse;
        BuildMenu();
        ApplyAppearance();
    }

    private static void SetFont(Control control, string name)
    {
        using var font = new SystemFont { FontNames = new[] { "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI" } };
        control.AddThemeFontOverride(name, font);
    }

    public override void _Process(double delta)
    {
        poll += delta; sinceResponse += delta;
        if (!busy && poll >= (catchUp ? 0.05 : 0.75))
        {
            poll = 0;
            var builder = new UriBuilder(settings.Url) { Query = "after=" + cursor };
            busy = request.Request(builder.Uri.AbsoluteUri) == Error.Ok;
        }
        var viewport = GetViewport().GetVisibleRect().Size;
        var size = new Vector2(viewport.X * settings.PanelWidth, viewport.Y * settings.PanelHeight);
        panel.Size = size;
        panel.Position = new Vector2(Math.Clamp(viewport.X * settings.PanelX, 0, Math.Max(0, viewport.X - size.X)),
            Math.Clamp(viewport.Y * settings.PanelY, 0, Math.Max(0, viewport.Y - size.Y)));
        panel.Visible = settings.Enabled && settings.LogPanel;
        bool interactive = menu.Visible || settings.InteractWithLogs;
        feed.MouseFilter = interactive ? Control.MouseFilterEnum.Stop : Control.MouseFilterEnum.Ignore;
        feed.GetVScrollBar().MouseFilter = feed.MouseFilter;
        feed.SelectionEnabled = interactive;
        // While reading older entries, keep the rendered document stable until
        // the user returns to the top or changes a setting.
        if (feedDirty && (forceFeedRefresh || settings.FollowLatest || feed.GetVScrollBar().Value <= 0)) RenderFeed();
        feedStatus.Text = (language == "en" ? "Agent log" : "Agent 决策日志") + $" · {history.Count}" +
            (sinceResponse > 4 ? (language == "en" ? " · disconnected" : " · 已断开") : catchUp ? " · …" : "");
        LayoutMenu(viewport);
        // Reading time pauses in the settings menu. A newer decision never
        // replaces a page that the audience has not finished reading.
        if (!menu.Visible && settings.Enabled && settings.Speech) pageAge += delta;
        if (speechPages.Count > 0 && pageAge >= PageDuration(speechPages[speechPage]))
        {
            pageAge = 0; speechPage++; nativeSpeech.Clear();
            if (speechPage >= speechPages.Count) speechPages.Clear();
        }
        if (speechPages.Count == 0 && speechQueue.Count > 0)
        {
            speechPages = Paginate(speechQueue.Dequeue());
            speechPage = 0; pageAge = 0;
        }
        bool speaking = settings.Enabled && settings.Speech && speechPages.Count > 0;
        subtitle.Visible = false;
        if (!speaking) { nativeSpeech.Clear(); return; }
        string text = speechPages[speechPage];
        double remaining = PageDuration(text) - pageAge;
        bool native = !settings.SubtitlesOnly && !menu.Visible &&
            nativeSpeech.Show(text, Math.Max(2, remaining + 1), settings.SpeechScale);
        if (!native)
        {
            nativeSpeech.Clear();
            subtitle.Text = text;
            for (int fontSize = settings.SubtitleFontSize; fontSize >= 14; fontSize--)
            {
                subtitle.AddThemeFontSizeOverride("font_size", fontSize);
                subtitle.Size = new Vector2(viewport.X * settings.SubtitleWidth, 0);
                if (subtitle.GetMinimumSize().Y <= viewport.Y * 0.2f) break;
            }
            subtitle.Size = new Vector2(subtitle.Size.X, subtitle.GetMinimumSize().Y);
            subtitle.Position = new Vector2((viewport.X - subtitle.Size.X) / 2, viewport.Y * settings.SubtitleY);
            subtitle.Visible = !menu.Visible;
        }
    }

    private void OnResponse(long result, long code, string[] headers, byte[] body)
    {
        busy = false;
        if (result != (long)HttpRequest.Result.Success || code != 200) { catchUp = false; return; }
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            string incoming = root.GetProperty("stream_id").GetString() ?? "";
            if (streamId != incoming)
            {
                bool reconnect = streamId.Length > 0;
                streamId = incoming;
                history.Clear(); cursor = 0; bubbleSeq = -1; feedDirty = true; forceFeedRefresh = true;
                speechPages.Clear(); speechQueue.Clear();
                nativeSpeech.Clear();
                if (reconnect) { poll = 1; catchUp = true; return; }
            }
            language = root.GetProperty("language").GetString() ?? "zh";
            long nextBubble = root.GetProperty("bubble_seq").GetInt64();
            if (bubbleSeq < 0)
            {
                bubbleSeq = nextBubble;
                QueueSpeech(root.GetProperty("bubble").GetString() ?? "");
            }
            foreach (var entry in root.GetProperty("logs").EnumerateArray())
            {
                if (entry.GetProperty("seq").GetInt64() <= cursor) continue;
                history.Add(entry.Clone()); feedDirty = true;
                long seq = entry.GetProperty("seq").GetInt64();
                string kind = Value(entry, "kind");
                if (seq > bubbleSeq && kind is "decision" or "model_plan" or "error" or "warning")
                {
                    QueueSpeech(kind is "error" or "warning"
                        ? (language == "en" ? "Let me think a little more…" : "我得再想想……")
                        : Value(entry, "text"));
                    bubbleSeq = seq;
                }
            }
            cursor = root.GetProperty("next_seq").GetInt64();
            if (history.Count > 2000) history.RemoveRange(0, history.Count - 2000);
            catchUp = root.GetProperty("has_more").GetBoolean();
            sinceResponse = 0;
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException or KeyNotFoundException)
        { catchUp = false; }
    }

    private void QueueSpeech(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        if (!settings.QueueCommentary)
        {
            speechQueue.Clear();
            speechPages = Paginate(text);
            speechPage = 0; pageAge = 0;
            nativeSpeech.Clear();
        }
        else speechQueue.Enqueue(text);
    }

    private void SetQueueCommentary(bool queue)
    {
        settings.QueueCommentary = queue;
        // Switching to immediacy also drops the existing backlog right away.
        if (!queue && speechQueue.Count > 0) QueueSpeech(speechQueue.Last());
        SaveSettings();
    }

    private double PageDuration(string text) => Math.Max(settings.PageSeconds,
        Math.Max(settings.LifetimeSeconds / Math.Max(1, speechPages.Count),
            1.5 + StringInfo.ParseCombiningCharacters(text).Length / (text.Any(c => c >= 0x2e80) ? 6.0 : 16.0)));

    internal static List<string> Paginate(string text)
    {
        var pages = new List<string>();
        int[] offsets = StringInfo.ParseCombiningCharacters(text);
        int limit = text.Any(c => c >= 0x2e80) ? 32 : 90;
        int start = 0;
        while (start < offsets.Length)
        {
            int end = Math.Min(start + limit, offsets.Length);
            if (end < offsets.Length)
            {
                // Prefer complete clauses. English words and combining emoji
                // remain intact; an exceptionally long word uses the subtitle.
                int boundary = -1;
                for (int i = start; i < end; i++)
                    if ("。！？；，.!?;,\n".Contains(text[offsets[i]])) boundary = i + 1;
                if (boundary > start + limit / 3) end = boundary;
                else if (!text.Any(c => c >= 0x2e80))
                {
                    boundary = -1;
                    for (int i = start; i < end; i++) if (char.IsWhiteSpace(text[offsets[i]])) boundary = i + 1;
                    if (boundary > start) end = boundary;
                    else while (end < offsets.Length && end - start < 160 && !char.IsWhiteSpace(text[offsets[end]])) end++;
                }
            }
            pages.Add(text[offsets[start]..(end < offsets.Length ? offsets[end] : text.Length)]);
            start = end;
        }
        return pages;
    }

    private static string Value(JsonElement entry, string key) => entry.TryGetProperty(key, out var value)
        ? value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : value.GetRawText() : "";

    private void RenderFeed()
    {
        feedDirty = false;
        forceFeedRefresh = false;
        double scroll = feed.GetVScrollBar().Value;
        feed.Clear();
        int shown = 0;
        foreach (var entry in history.AsEnumerable().Reverse())
        {
            string kind = Value(entry, "kind");
            bool diagnostic = kind is "error" or "warning";
            if (!settings.Errors && diagnostic) continue;
            if (settings.Filter == "decisions" && kind is not ("decision" or "model_plan")) continue;
            if (settings.Filter == "diagnostics" && kind is not ("error" or "warning" or "info")) continue;
            if (++shown > settings.HistoryCount) break;
            feed.PushColor(diagnostic ? new Color(1, 0.75f, 0.60f, settings.ErrorOpacity) : new Color(0.9f, 0.9f, 0.84f));
            feed.AddText($"{Value(entry, "ts")} · {kind} · #{Value(entry, "seq")}\n");
            feed.AddText(Value(entry, "text") + "\n");
            foreach (var property in entry.EnumerateObject())
            {
                if (property.Name is "seq" or "ts" or "kind" or "text") continue;
                if (property.Name is "action" or "actions")
                {
                    if (settings.Actions) feed.AddText(property.Name + ": " + Value(entry, property.Name) + "\n");
                }
                else if (settings.Details) feed.AddText(property.Name + ": " + Value(entry, property.Name) + "\n");
            }
            feed.AddText("\n"); feed.Pop();
        }
        Callable.From(() =>
        {
            if (IsInstanceValid(feed)) feed.GetVScrollBar().Value = settings.FollowLatest ? 0 : scroll;
        }).CallDeferred();
    }

    private void ApplyAppearance()
    {
        feed.AddThemeFontSizeOverride("normal_font_size", settings.LogFontSize);
        feedStatus.AddThemeFontSizeOverride("font_size", Math.Max(12, settings.LogFontSize - 2));
        subtitle.AddThemeFontSizeOverride("font_size", settings.SubtitleFontSize);
        feed.Modulate = new Color(1, 1, 1, settings.Opacity);
        feedStatus.Modulate = new Color(1, 1, 1, settings.Opacity);
        using var style = new StyleBoxFlat { BgColor = new Color(0.03f, 0.04f, 0.05f, settings.BackgroundOpacity),
            ContentMarginLeft = 10, ContentMarginRight = 10, ContentMarginTop = 8, ContentMarginBottom = 8,
            CornerRadiusTopLeft = 8, CornerRadiusTopRight = 8, CornerRadiusBottomLeft = 8, CornerRadiusBottomRight = 8 };
        panel.AddThemeStyleboxOverride("panel", style);
        feedDirty = true; forceFeedRefresh = true;
    }

    private void SaveSettings()
    {
        settings.Normalize();
        using var preferences = new ConfigFile();
        preferences.SetValue("overlay", "json", JsonSerializer.Serialize(settings));
        var result = preferences.Save(SettingsPath);
        if (result != Error.Ok) menuStatus.Text = "设置保存失败 / Could not save: " + result;
        ApplyAppearance(); nativeSpeech.Clear();
    }

    public override void _Input(InputEvent input)
    {
        if (input is InputEventKey { Pressed: true, Echo: false } key &&
            ((key.AltPressed && key.Keycode == Key.F8) || (menu.Visible && key.Keycode == Key.Escape)))
        {
            menu.Visible = !menu.Visible;
            if (menu.Visible) menu.MoveToFront();
            GetViewport().SetInputAsHandled();
        }
    }

    public override void _UnhandledInput(InputEvent input)
    {
        if (menu.Visible) GetViewport().SetInputAsHandled();
    }

    public override void _ExitTree() => nativeSpeech.Dispose();
}
