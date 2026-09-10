/* overlay.js -- 展示叠加层：只轮询状态与最近决策，不做任何控制。
 *
 * URL 参数:
 *   ?n=5        显示最近几条决策（1..20，默认 5）
 *   ?chrome=0   隐藏右上角切换控件（纯净录制）
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);

  var maxLogs = parseInt(params.get("n") || "8", 10);
  if (!maxLogs || maxLogs < 1) maxLogs = 8;
  if (maxLogs > 30) maxLogs = 30;

  if (params.get("chrome") === "0") {
    document.body.classList.add("no-chrome");
  }

  var maxNet = parseInt(params.get("net") || "5", 10);
  if (!maxNet || maxNet < 1) maxNet = 5;
  if (maxNet > 20) maxNet = 20;

  var lastSeq = 0;
  var recent = [];      // kind === "decision"
  var netRecent = [];   // 网络 / 报错 / 提示（error、warning、info）

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function renderLogs() {
    var box = $("log-list");
    // 最新的一条排在最上面：slice 取最近 N 条（旧→新）后反转，
    // 新决策从顶部出现并把旧的向下挤；容器超高时裁掉的是最旧的。
    var shown = recent.slice(-maxLogs).reverse();
    box.innerHTML = "";
    if (shown.length === 0) {
      box.innerHTML = '<div class="entry empty">等待决策…</div>';
      return;
    }
    for (var i = 0; i < shown.length; i++) {
      var e = shown[i];
      var div = document.createElement("div");
      div.className = "entry";
      var html = "";
      if (e.state_type || e.llm_ms) {
        html += '<div class="e-meta">' + escapeHtml(e.state_type || "") +
          (e.llm_ms ? " · LLM " + (e.llm_ms / 1000).toFixed(1) + "s" : "") +
          "</div>";
      }
      html += '<div class="e-thought">' + escapeHtml(e.text || "") + "</div>";
      var renderedAction = e.action || e.actions;
      if (renderedAction) {
        html += '<div class="e-action">➤ ' + escapeHtml(renderedAction);
        if (e.result) html += ' <span class="e-res">— ' + escapeHtml(e.result) + "</span>";
        html += "</div>";
      }
      div.innerHTML = html;
      box.appendChild(div);
    }
    // 最新决策固定在顶部：每次渲染都回到顶部，保证它始终可见
    box.scrollTop = 0;
  }

  function renderNet() {
    var box = $("net-list");
    var shown = netRecent.slice(-maxNet).reverse();   // 最新在最上面
    box.innerHTML = "";
    if (shown.length === 0) {
      box.innerHTML = '<div class="entry empty">暂无异常</div>';
      return;
    }
    var KIND_LABEL = {
      decision: "决策",
      model_plan: "计划",
      info: "信息",
      error: "错误",
      warning: "警告",
    };
    for (var i = 0; i < shown.length; i++) {
      var e = shown[i];
      var div = document.createElement("div");
      div.className = "entry" + (e.kind === "error" ? " err"
        : e.kind === "warning" ? " warn" : "");
      var html = '<div class="e-meta">' + escapeHtml(e.ts || "") +
        " · " + escapeHtml(KIND_LABEL[e.kind] || e.kind || "") + "</div>";
      html += '<div class="e-thought">' + escapeHtml(e.text || "") + "</div>";
      div.innerHTML = html;
      box.appendChild(div);
    }
    box.scrollTop = 0;
  }

  function pollLogs() {
    fetch("/api/logs?after=" + lastSeq)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var added = false;
        var addedNet = false;
        (d.logs || []).forEach(function (e) {
          if (e.seq && e.seq > lastSeq) lastSeq = e.seq;
          // 战斗中的模型认知是 model_plan（ActionChunk 计划的 thought），
          // 与非战斗 decision 一样属于"决策"流；否则战斗内容只剩动作行。
          if (e.kind === "decision" || e.kind === "model_plan") {
            recent.push(e); added = true;
          }
          else if (e.kind === "error" || e.kind === "warning" || e.kind === "info") {
            // 排除 kind === "state"（每回合都产生，太吵）
            netRecent.push(e); addedNet = true;
          }
        });
        // 只留够渲染的量，避免长局无限增长
        if (recent.length > maxLogs * 3) {
          recent = recent.slice(-maxLogs * 3);
        }
        if (netRecent.length > maxNet * 3) {
          netRecent = netRecent.slice(-maxNet * 3);
        }
        if (added) renderLogs();
        if (addedNet) renderNet();
      })
      .catch(function () { /* 服务器忙，下一轮再试 */ });
  }

  function pollStatus() {
    fetch("/api/status")
      .then(function (r) { return r.json(); })
      .then(function (st) {
        $("s-floor").textContent = st.floor != null ? st.floor : "-";
        $("s-hp").textContent = (st.hp != null ? st.hp : "-") + " / " +
          (st.max_hp != null ? st.max_hp : "-");
        $("s-gold").textContent = st.gold != null ? st.gold : "-";
        $("s-decisions").textContent = st.decision_count || 0;
        $("s-state").textContent = st.current_state_type || "-";

        var ag = $("s-agent");
        ag.textContent = "Agent: " + (st.running ? "运行中" : "已停止");
        ag.className = "chip " + (st.running ? "ok" : "");

        var br = $("s-bridge");
        br.textContent = "桥接: " + (st.bridge_connected ? "已连接" : "未连接");
        br.className = "chip " + (st.bridge_connected ? "ok" : "bad");

        // 模型 / 用量 / 费用估算
        var mn = $("m-name");
        if (mn) mn.textContent = st.model || "-";
        var mt = $("m-tokens");
        if (mt) {
          mt.textContent = (st.llm_total_tokens || 0) + " (入 " +
            (st.llm_prompt_tokens || 0) + " / 出 " +
            (st.llm_completion_tokens || 0) + ")";
        }
        // Credits = 累计消耗的 token 总量（不做价格换算：公开标价变动频繁，
        // 换算成金额容易误导，这里只报中性用量）。
        var mc = $("m-cost");
        if (mc) mc.textContent = (st.llm_total_tokens || 0).toLocaleString();
      })
      .catch(function () { /* ignore */ });
  }

  // ---- UI 切换 ----
  Array.prototype.forEach.call(
    document.querySelectorAll("#ui-switch .sw"),
    function (btn) {
      btn.addEventListener("click", function () {
        if (btn.dataset.mode === "console") {
          location.href = "/";           // 回到完整控制台
        }
      });
    }
  );

  // ---- 窗口投屏：把指定窗口（游戏）作为本页背景 ----
  var video = $("bg-video");
  var btnCast = $("btn-cast");
  var btnFit = $("btn-fit");
  var castStream = null;

  function stopCast() {
    if (castStream) {
      castStream.getTracks().forEach(function (t) { t.stop(); });
      castStream = null;
    }
    try { video.srcObject = null; } catch (e) { /* ignore */ }
    document.body.classList.remove("casting");
    document.body.classList.remove("fit-cover");
    if (btnCast) btnCast.textContent = "投屏";
    if (btnFit) { btnFit.hidden = true; btnFit.textContent = "适应"; }
  }

  function startCast() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      alert("当前浏览器不支持窗口捕获（getDisplayMedia）。\n"
        + "请用 Chrome / Edge 打开本页，并通过 127.0.0.1 或 HTTPS 访问。");
      return;
    }
    navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 30 },
      audio: false
    }).then(function (stream) {
      castStream = stream;
      video.srcObject = stream;
      var played = video.play();
      if (played && played.catch) played.catch(function () { /* ignore */ });
      document.body.classList.add("casting");
      if (btnCast) btnCast.textContent = "停止投屏";
      if (btnFit) btnFit.hidden = false;
      // 用户从浏览器原生条停止共享时同步收尾
      var track = stream.getVideoTracks()[0];
      if (track) track.addEventListener("ended", stopCast);
    }).catch(function () {
      // 用户取消选择，或环境不允许 -- 什么都不做
    });
  }

  if (btnCast) {
    btnCast.addEventListener("click", function () {
      if (castStream) stopCast(); else startCast();
    });
  }
  if (btnFit) {
    btnFit.addEventListener("click", function () {
      var cover = document.body.classList.toggle("fit-cover");
      btnFit.textContent = cover ? "铺满" : "适应";
    });
  }

  var btnMirror = $("btn-mirror");
  if (btnMirror) {
    btnMirror.addEventListener("click", function () {
      document.body.classList.toggle("mirror");
    });
  }

  pollStatus();
  pollLogs();
  setInterval(pollStatus, 1000);
  setInterval(pollLogs, 1000);
})();
