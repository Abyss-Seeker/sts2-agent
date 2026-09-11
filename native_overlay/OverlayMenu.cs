using Godot;
using System.Text.Json;

namespace STS2AgentOverlay;

public partial class Overlay
{
    private Label menuStatus = null!;
    private VBoxContainer menuRows = null!;
    private bool languageBusy;

    private void BuildMenu()
    {
        menu = new PanelContainer { Name = "SettingsMenu", Visible = false, MouseFilter = Control.MouseFilterEnum.Stop };
        using var style = new StyleBoxFlat { BgColor = new Color(0.06f, 0.075f, 0.10f, 0.98f),
            ContentMarginLeft = 22, ContentMarginRight = 22, ContentMarginTop = 16, ContentMarginBottom = 16,
            CornerRadiusTopLeft = 12, CornerRadiusTopRight = 12, CornerRadiusBottomLeft = 12, CornerRadiusBottomRight = 12 };
        menu.AddThemeStyleboxOverride("panel", style);
        using var theme = new Theme();
        using var font = new SystemFont { FontNames = new[] { "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI" } };
        theme.DefaultFont = font;
        theme.DefaultFontSize = 17;
        menu.Theme = theme;
        AddChild(menu);
        var layout = new VBoxContainer();
        menu.AddChild(layout);
        var heading = new HBoxContainer();
        layout.AddChild(heading);
        heading.AddChild(new Label { Text = "Agent 展示设置 / Overlay settings", SizeFlagsHorizontal = Control.SizeFlags.ExpandFill });
        var close = new Button { Text = "关闭 / Close · Alt+F8" };
        close.Pressed += () => menu.Hide();
        heading.AddChild(close);
        var scroller = new ScrollContainer { HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
            SizeFlagsVertical = Control.SizeFlags.ExpandFill };
        layout.AddChild(scroller);
        menuRows = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        menuRows.AddThemeConstantOverride("separation", 9);
        scroller.AddChild(menuRows);

        Section("显示与语言 / Display & language");
        Check("启用展示层 / Enable overlay", settings.Enabled, v => settings.Enabled = v);
        var languageRow = Row("角色讲解语言 / Commentary language");
        foreach (var (text, code) in new[] { ("简体中文", "zh"), ("English", "en") })
        {
            var button = new Button { Text = text };
            button.Pressed += () => SetLanguage(code);
            languageRow.AddChild(button);
        }
        Note("讲解语言保存到控制台，下一次启动 Agent 生效；游戏语言不变。\nLanguage applies on next Agent start. This menu does not pause the Agent.");
        var urlRow = Row("控制台地址 / Local endpoint");
        var url = new LineEdit { Text = settings.Url, SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        urlRow.AddChild(url);
        var connect = new Button { Text = "连接 / Connect" };
        urlRow.AddChild(connect);
        connect.Pressed += () =>
        {
            if (!Uri.TryCreate(url.Text, UriKind.Absolute, out var uri) || !uri.IsLoopback || uri.Scheme != "http")
            { menuStatus.Text = "请输入本机 HTTP 地址 / Loopback HTTP required"; return; }
            request.CancelRequest(); busy = false;
            settings.Url = url.Text;
            streamId = ""; cursor = 0; history.Clear(); speechPages.Clear(); speechQueue.Clear(); bubbleSeq = -1;
            sinceResponse = 100; poll = 1;
            SaveSettings();
        };

        Section("原生角色气泡与字幕 / Native speech & subtitles");
        Check("显示角色讲解 / Show commentary", settings.Speech, v => settings.Speech = v);
        var playback = new OptionButton { Name = "SpeechPlayback" };
        playback.AddItem("即时性：显示最新说明 / Latest immediately");
        playback.AddItem("可阅读性：排队完整播放 / Queue for readability");
        playback.Selected = settings.QueueCommentary ? 1 : 0;
        playback.ItemSelected += i => SetQueueCommentary(i == 1);
        Row("讲解播放方式 / Commentary playback").AddChild(playback);
        Note("即时性会替换当前说明并清空积压；可阅读性会依次播完。完整记录始终保留在右侧日志。\nLatest replaces current speech and clears backlog; queued mode finishes each explanation.");
        Check("始终使用顶部字幕 / Always use subtitles", settings.SubtitlesOnly, v => settings.SubtitlesOnly = v);
        Note("默认复用游戏气泡；选卡、地图和其他覆盖页面自动切换为顶部居中字幕。\nNative game bubbles in combat; centered subtitles over covered screens.");
        Number("讲解保留秒数 / Duration", settings.LifetimeSeconds, 3, 120, 1, v => settings.LifetimeSeconds = (float)v);
        Number("长讲解每页秒数 / Seconds per page", settings.PageSeconds, 2, 15, 1, v => settings.PageSeconds = (float)v);
        Number("原生气泡缩放 / Native bubble scale", settings.SpeechScale, 0.5, 1.5, 0.05, v => settings.SpeechScale = (float)v);
        Number("字幕距顶部比例 / Subtitle top", settings.SubtitleY, 0, 0.3, 0.01, v => settings.SubtitleY = (float)v);
        Number("字幕宽度比例 / Subtitle width", settings.SubtitleWidth, 0.2, 0.9, 0.01, v => settings.SubtitleWidth = (float)v);
        Number("字幕字号 / Subtitle font", settings.SubtitleFontSize, 14, 42, 1, v => settings.SubtitleFontSize = (int)v);

        Section("右侧日志与历史 / Log panel & history");
        Check("显示日志栏 / Show log panel", settings.LogPanel, v => settings.LogPanel = v);
        var filter = new OptionButton();
        foreach (string option in new[] { "全部日志 / All", "决策与计划 / Decisions", "信息与异常 / Diagnostics" }) filter.AddItem(option);
        string[] filters = { "all", "decisions", "diagnostics" };
        filter.Selected = Array.IndexOf(filters, settings.Filter);
        filter.ItemSelected += i => { settings.Filter = filters[i]; SaveSettings(); };
        Row("日志类型 / Log types").AddChild(filter);
        Check("显示报错与警告 / Errors and warnings", settings.Errors, v => settings.Errors = v);
        Check("显示 actions JSON / Action JSON", settings.Actions, v => settings.Actions = v);
        Check("显示结果及所有附加字段 / All record details", settings.Details, v => settings.Details = v);
        Number("展示最近条数 / Recent entries", settings.HistoryCount, 1, 2000, 1, v => settings.HistoryCount = (int)v);
        Check("新日志自动回到顶部 / Follow newest", settings.FollowLatest, v => settings.FollowLatest = v);
        Check("菜单关闭后仍可滚动日志 / Mouse interaction", settings.InteractWithLogs, v => settings.InteractWithLogs = v);
        Note("最新在上；打开本菜单时可滚动右侧历史。关闭自动跟随可停留阅读。\n本地保留最近 2000 条；初次连接获取控制台现存的最多 1000 条，不截断正文。\nNewest first. Scroll the right panel; disable Follow newest to read older entries.");
        Number("日志字号 / Log font", settings.LogFontSize, 10, 30, 1, v => settings.LogFontSize = (int)v);
        Number("日志文字不透明度 / Text opacity", settings.Opacity, 0, 1, 0.05, v => settings.Opacity = (float)v);
        Number("面板背景不透明度 / Background opacity", settings.BackgroundOpacity, 0, 1, 0.05, v => settings.BackgroundOpacity = (float)v);
        Number("异常相对不透明度 / Error opacity", settings.ErrorOpacity, 0, 1, 0.05, v => settings.ErrorOpacity = (float)v);
        Number("面板横向位置 / Panel X", settings.PanelX, 0, 0.95, 0.01, v => settings.PanelX = (float)v);
        Number("面板纵向位置 / Panel Y", settings.PanelY, 0, 0.95, 0.01, v => settings.PanelY = (float)v);
        Number("面板宽度 / Panel width", settings.PanelWidth, 0.12, 0.65, 0.01, v => settings.PanelWidth = (float)v);
        Number("面板高度 / Panel height", settings.PanelHeight, 0.15, 0.9, 0.01, v => settings.PanelHeight = (float)v);
        menuStatus = new Label { Text = "设置即时生效并自动保存 / Changes apply and save immediately",
            AutowrapMode = TextServer.AutowrapMode.WordSmart };
        layout.AddChild(menuStatus);
    }

    private void Section(string text) => menuRows.AddChild(new Label { Text = "\n" + text });
    private void Note(string text) => menuRows.AddChild(new Label { Text = text, AutowrapMode = TextServer.AutowrapMode.WordSmart });
    private HBoxContainer Row(string text)
    {
        var row = new HBoxContainer();
        row.AddChild(new Label { Text = text, SizeFlagsHorizontal = Control.SizeFlags.ExpandFill });
        menuRows.AddChild(row);
        return row;
    }
    private void Check(string text, bool value, Action<bool> change)
    {
        var control = new CheckButton { Text = text, ButtonPressed = value };
        control.Toggled += v => { change(v); SaveSettings(); };
        menuRows.AddChild(control);
    }
    private void Number(string text, double value, double min, double max, double step, Action<double> change)
    {
        var control = new SpinBox { MinValue = min, MaxValue = max, Step = step, Value = value,
            CustomMinimumSize = new Vector2(130, 0) };
        control.ValueChanged += v => { change(v); SaveSettings(); };
        Row(text).AddChild(control);
    }
    private void LayoutMenu(Vector2 viewport)
    {
        menu.Size = new Vector2(Math.Min(880, viewport.X * 0.66f), viewport.Y * 0.88f);
        menu.Position = new Vector2(viewport.X * 0.025f, viewport.Y * 0.06f);
    }
    private void SetLanguage(string code)
    {
        if (languageBusy) return;
        var uri = new UriBuilder(settings.Url) { Path = "/api/overlay/language", Query = "" };
        var result = languageRequest.Request(uri.Uri.AbsoluteUri, new[] { "Content-Type: application/json" },
            Godot.HttpClient.Method.Post, JsonSerializer.Serialize(new { language = code }));
        languageBusy = result == Error.Ok;
        menuStatus.Text = languageBusy ? "正在保存 / Saving…" : "请求失败 / Request failed: " + result;
    }
    private void OnLanguageResponse(long result, long code, string[] headers, byte[] body)
    {
        languageBusy = false;
        menuStatus.Text = result == (long)HttpRequest.Result.Success && code == 200
            ? "语言已保存，下次启动 Agent 生效 / Saved; restart Agent to apply"
            : "语言保存失败，请确认控制台已更新并运行 / Could not save language";
    }
}
