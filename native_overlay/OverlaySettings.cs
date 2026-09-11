namespace STS2AgentOverlay;

public sealed class OverlaySettings
{
    public string Url { get; set; } = "http://127.0.0.1:8770/api/overlay";
    public bool Enabled { get; set; } = true;
    public bool Speech { get; set; } = true;
    public bool QueueCommentary { get; set; } = true;
    public bool SubtitlesOnly { get; set; }
    public float LifetimeSeconds { get; set; } = 20;
    public float PageSeconds { get; set; } = 5;
    public float SpeechScale { get; set; } = 1;
    public float SubtitleY { get; set; } = 0.075f;
    public float SubtitleWidth { get; set; } = 0.55f;
    public int SubtitleFontSize { get; set; } = 24;
    public bool LogPanel { get; set; } = true;
    public bool Errors { get; set; } = true;
    public bool Actions { get; set; } = true;
    public bool Details { get; set; } = true;
    public string Filter { get; set; } = "all";
    public int HistoryCount { get; set; } = 40;
    public int LogFontSize { get; set; } = 16;
    public float Opacity { get; set; } = 0.8f;
    public float BackgroundOpacity { get; set; } = 0.22f;
    public float ErrorOpacity { get; set; } = 0.45f;
    public float PanelX { get; set; } = 0.74f;
    public float PanelY { get; set; } = 0.18f;
    public float PanelWidth { get; set; } = 0.25f;
    public float PanelHeight { get; set; } = 0.62f;
    public bool InteractWithLogs { get; set; }
    public bool FollowLatest { get; set; } = true;

    public void Normalize()
    {
        if (!Uri.TryCreate(Url, UriKind.Absolute, out var uri) || !uri.IsLoopback || uri.Scheme != "http")
            Url = "http://127.0.0.1:8770/api/overlay";
        LifetimeSeconds = Clamp(LifetimeSeconds, 3, 120, 20);
        PageSeconds = Clamp(PageSeconds, 2, 15, 5);
        SpeechScale = Clamp(SpeechScale, 0.5f, 1.5f, 1);
        SubtitleY = Clamp(SubtitleY, 0, 0.3f, 0.075f);
        SubtitleWidth = Clamp(SubtitleWidth, 0.2f, 0.9f, 0.55f);
        SubtitleFontSize = Math.Clamp(SubtitleFontSize, 14, 42);
        LogFontSize = Math.Clamp(LogFontSize, 10, 30);
        HistoryCount = Math.Clamp(HistoryCount, 1, 2000);
        Opacity = Clamp(Opacity, 0, 1, 0.8f);
        BackgroundOpacity = Clamp(BackgroundOpacity, 0, 1, 0.22f);
        ErrorOpacity = Clamp(ErrorOpacity, 0, 1, 0.45f);
        PanelX = Clamp(PanelX, 0, 0.95f, 0.74f);
        PanelY = Clamp(PanelY, 0, 0.95f, 0.18f);
        PanelWidth = Clamp(PanelWidth, 0.12f, 0.65f, 0.25f);
        PanelHeight = Clamp(PanelHeight, 0.15f, 0.9f, 0.62f);
        if (Filter is not ("all" or "decisions" or "diagnostics")) Filter = "all";
    }

    private static float Clamp(float value, float min, float max, float fallback) =>
        float.IsFinite(value) ? Math.Clamp(value, min, max) : fallback;
}
