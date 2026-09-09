/* STS2 LLM Agent 控制台前端逻辑 */
"use strict";

const $ = (id) => document.getElementById(id);

const TEXT_FIELDS = [
  "bridge_host", "api_base_url", "api_key", "model", "steam_appid",
];
const NUM_FIELDS = [
  "bridge_port", "temperature", "max_tokens", "llm_timeout", "llm_retries",
  "agent_timeout",
  "max_history_turns", "max_state_chars", "max_context_chars", "action_delay",
  "action_chunk_max_actions",
];
const TEXTAREA_FIELDS = ["system_template", "user_template"];
const SELECT_FIELDS = ["show_thinking", "decision_mode", "failure_policy", "reasoning_effort", "stream_mode", "thinking_enabled"];
const CHECK_FIELDS = ["disable_fallback", "save_log", "auto_launch_game", "delta_observations"];

const DEFAULT_PROMPTS = null; // filled from server DEFAULT_CONFIG on load
let defaults = null;
let lastSeq = 0;

// ---------------- helpers ----------------

function collectConfig() {
  const cfg = {};
  TEXT_FIELDS.forEach((id) => (cfg[id] = $(id).value.trim()));
  NUM_FIELDS.forEach((id) => (cfg[id] = parseFloat($(id).value)));
  TEXTAREA_FIELDS.forEach((id) => (cfg[id] = $(id).value));
  SELECT_FIELDS.forEach((id) => (cfg[id] = $(id).value));
  CHECK_FIELDS.forEach((id) => (cfg[id] = $(id).checked));
  if (Number.isNaN(cfg.bridge_port)) cfg.bridge_port = 9002;
  if (Number.isNaN(cfg.action_delay) || cfg.action_delay < 0) cfg.action_delay = 0;
  if (Number.isNaN(cfg.llm_retries) || cfg.llm_retries < 0) cfg.llm_retries = 0;
  if (Number.isNaN(cfg.agent_timeout)) cfg.agent_timeout = 90;
  cfg.agent_timeout = Math.min(300, Math.max(10, cfg.agent_timeout));
  // thinking_enabled is rendered as a select but must be a real boolean.
  cfg.thinking_enabled = $("thinking_enabled").value === "true";
  if (!Number.isNaN(cfg.action_chunk_max_actions)) {
    cfg.action_chunk_max_actions = Math.min(32, Math.max(1, cfg.action_chunk_max_actions));
  }
  return cfg;
}

function applyConfig(cfg) {
  TEXT_FIELDS.forEach((id) => ($(id).value = cfg[id] ?? ""));
  NUM_FIELDS.forEach((id) => ($(id).value = cfg[id] ?? ""));
  TEXTAREA_FIELDS.forEach((id) => ($(id).value = cfg[id] ?? ""));
  SELECT_FIELDS.forEach((id) => {
    const v = id === "thinking_enabled" ? String(Boolean(cfg[id])) : cfg[id];
    $(id).value = v ?? $(id).value;
  });
  CHECK_FIELDS.forEach((id) => ($(id).checked = Boolean(cfg[id])));
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== null) opts.body = JSON.stringify(body);
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let msg = `${resp.status}`;
    try { msg = (await resp.json()).error || msg; } catch {}
    throw new Error(msg);
  }
  return resp.json();
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ---------------- status polling ----------------

const KIND_LABEL = {
  decision: "决策",
  info: "信息",
  error: "错误",
  state: "局面",
  warning: "警告",
  game_action: "动作",
  model_plan: "计划",
  plan_checkpoint: "检查点",
  benchmark_invalidated: "失效",
};

function renderStatus(st) {
  const chipAgent = $("chip-agent");
  chipAgent.textContent = "Agent: " + (st.running ? "运行中" : "已停止");
  chipAgent.className = "chip " + (st.running ? "ok" : "");
  const chipBridge = $("chip-bridge");
  chipBridge.textContent = "桥接: " + (st.bridge_connected ? "已连接" : "未连接");
  chipBridge.className = "chip " + (st.bridge_connected ? "ok" : "bad");
  const chipState = $("chip-state");
  chipState.textContent = "界面: " + (st.current_state_type || "-");
  chipState.className = "chip " + (st.current_state_type ? "warn" : "");
  $("chip-floor").textContent = `楼层: ${st.floor || "-"}`;
  $("chip-hp").textContent = `HP: ${st.hp ?? "-"}/${st.max_hp ?? "-"}`;
  $("chip-gold").textContent = `金币: ${st.gold ?? "-"}`;
  $("chip-decisions").textContent = `决策: ${st.decision_count}`;
  if ($("chip-plan")) {
    const plan = st.current_plan_id
      ? `${st.current_plan_id} ${st.current_plan_step}/${st.current_plan_total}`
      : (st.last_checkpoint_reason || "-");
    $("chip-plan").textContent = `计划: ${plan}`;
    $("chip-plan").className = "chip " + (st.current_plan_id ? "ok" : "");
  }
  if ($("chip-metrics")) {
    const ratio = st.actions_per_llm_call ? Number(st.actions_per_llm_call).toFixed(1) : "0";
    $("chip-metrics").textContent =
      `调用/动作: ${st.llm_request_count ?? 0}/${st.game_action_count ?? 0} (${ratio}/次)` +
      (st.benchmark_valid === false ? " · 已失效" : "");
    $("chip-metrics").className = "chip " + (st.benchmark_valid === false ? "bad" : "");
  }
  $("btn-start").disabled = st.running;
  $("btn-stop").disabled = !st.running;
}

function renderLogs(logs) {
  const feed = $("feed");
  for (const e of logs) {
    lastSeq = Math.max(lastSeq, e.seq);
    const div = document.createElement("div");
    div.className = "log " + (e.kind || "info");
    const meta = `${e.ts} · ${KIND_LABEL[e.kind] || e.kind}` +
      (e.state_type ? ` · ${e.state_type}` : "") +
      (e.llm_ms ? ` · LLM ${(e.llm_ms / 1000).toFixed(1)}s` : "");
    let html = `<div class="meta"><span class="tag">${KIND_LABEL[e.kind] || e.kind}</span>${escapeHtml(meta)}</div>`;
    html += `<div>${escapeHtml(e.text || "")}</div>`;
    if (e.action) html += `<div class="action">➤ ${escapeHtml(e.action)} ${e.result ? "— " + escapeHtml(e.result) : ""}</div>`;
    div.innerHTML = html;
    feed.prepend(div);
    while (feed.children.length > 200) feed.removeChild(feed.lastChild);
  }
}

async function poll() {
  try {
    const st = await api("/api/status");
    renderStatus(st);
    const { logs } = await api(`/api/logs?after=${lastSeq}`);
    if (logs && logs.length) renderLogs(logs);
    if (st.last_error) {
      // errors already arrive through the log feed; nothing extra here
    }
  } catch (err) {
    console.error("poll failed", err);
  }
}

// ---------------- actions ----------------

async function saveConfig() {
  const cfg = collectConfig();
  await api("/api/config", "POST", cfg);
  return cfg;
}

async function startAgent() {
  try {
    const cfg = collectConfig();
    await api("/api/agent/start", "POST", cfg);
    appendLocal("info", "已请求启动 Agent（自动保存配置）...");
  } catch (err) {
    appendLocal("error", "启动失败: " + err.message);
  }
}

async function stopAgent() {
  try {
    await api("/api/agent/stop", "POST", {});
    appendLocal("info", "已停止 Agent。");
  } catch (err) {
    appendLocal("error", "停止失败: " + err.message);
  }
}

function appendLocal(kind, text) {
  renderLogs([{ seq: 0, ts: new Date().toLocaleTimeString(), kind, text }]);
}

// ---------------- wiring ----------------

$("btn-save").addEventListener("click", async () => {
  try {
    await saveConfig();
    appendLocal("info", "配置已保存到 config.json");
  } catch (err) {
    appendLocal("error", "保存失败: " + err.message);
  }
});
$("btn-start").addEventListener("click", startAgent);
$("btn-stop").addEventListener("click", stopAgent);
$("btn-reset-prompts").addEventListener("click", () => {
  if (!defaults) return;
  $("system_template").value = defaults.system_template;
  $("user_template").value = defaults.user_template;
});

(async function init() {
  try {
    defaults = await api("/api/config");
    applyConfig(defaults);
  } catch (err) {
    appendLocal("error", "加载配置失败: " + err.message);
  }
  await poll();
  setInterval(poll, 1500);
})();
