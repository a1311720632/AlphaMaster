const API = "";
let selectedDataFile = null;
let selectedSymbol = null;
let selectedStrategyFile = null;
let selectedStrategySymbol = null;
let chart = null;
let chartSymbol = null;
let pollTimer = null;
let clientErrors = [];
let debugMode = false;
let lastDebugViewContent = "";

// 分页与回测状态
let currentPage = "train";
let btActive = false;
let btBuster = "";      // 图表缓存刷新键（用 job 时间戳）
let btChartSig = "";    // 图表签名，变化时才重建图库避免闪烁

const $ = (id) => document.getElementById(id);

const CPU_TRAINING_NOTE = `暂无报错

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【为什么用 CPU 训练，不用 GPU？】

你可以把 GPU 想象成一辆超大的货车，CPU 想象成一辆灵活的小电瓶车。

我们这个项目的训练，就像要做很多很多道「小题」：
每道题只算一点点数字，算完马上换下一道。
货车虽然一次能装很多，但每装卸一次都要准备很久才能再出发；
电瓶车一次装的少，但说走就走，一道接一道做，反而更快。

再打个比方：
GPU 像很多厨师一起做大锅饭，适合一次炒一大锅；
我们这个训练更像一道道菜分开炒，而且每道菜份量很小。
大锅饭团队每次开火、洗锅、集合都要时间，小菜一碟反而耽误在「准备」上。

所以具体原因是：
1. 每次要算的数据不多，GPU「启动一次计算」的等待，有时比真正算数还久。
2. 训练是一步接一步、一条公式接一条公式地指挥，GPU 经常闲着等下一道题，没法一直满负荷。
3. 数据还要在 CPU 和 GPU 之间来回搬运，也要花时间。

我们实测过（同样训练 50 步）：GPU 大约 4.5 秒一步，CPU 大约 1.9 秒一步。
这不是显卡坏了，也不是没装驱动，而是这个项目的做题方式，更适合 CPU。

说白了就是这个项目用CPU训练的速度比用GPU训练的速度更快`;

function emptyDebugMessage() {
  return debugMode ? "暂无日志" : CPU_TRAINING_NOTE;
}

function formatApiError(data, status, path) {
  const d = data?.detail;
  let detail = "";
  if (Array.isArray(d)) {
    detail = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  } else if (typeof d === "string") {
    detail = d;
  } else if (d) {
    detail = JSON.stringify(d);
  }
  if (data?.traceback) {
    detail += `\n\n${data.traceback}`;
  }
  return detail || `HTTP ${status} ${path}`;
}

async function logClientError(message, context = {}) {
  const entry = `[${new Date().toLocaleString()}] ${message}`;
  clientErrors.push(entry);
  if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
  renderDebugView();
  try {
    await fetch(API + "/api/debug/client-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level: "error", message, context }),
    });
  } catch (_) {
    /* server may be down */
  }
}

function isViewAtBottom(el, threshold = 40) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function renderDebugView(serverLines = [], errorLines = []) {
  const parts = [];
  if (clientErrors.length) {
    parts.push("=== 前端报错 ===", ...clientErrors);
  }
  if (errorLines.length) {
    parts.push("\n=== 服务端错误日志 (logs/web_errors.log) ===", ...errorLines);
  }
  if (debugMode && serverLines.length) {
    parts.push("\n=== 服务端运行日志 (logs/web_server.log) ===", ...serverLines);
  }
  const el = $("debugView");
  const atBottom = isViewAtBottom(el);
  const next = parts.length ? parts.join("\n") : emptyDebugMessage();
  const changed = next !== lastDebugViewContent;
  el.textContent = next;
  if (changed && atBottom && lastDebugViewContent) {
    el.scrollTop = el.scrollHeight;
  }
  lastDebugViewContent = next;
}

async function setDebugMode(enabled) {
  debugMode = !!enabled;
  $("debugModeCheck").checked = debugMode;
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ debug_mode: debugMode }),
    });
  } catch (e) {
    await logClientError("切换调试模式失败: " + e.message);
  }
  if (!debugMode) {
    renderDebugView([], []);
  } else {
    await refreshDebugLogs();
  }
}

async function refreshDebugLogs() {
  try {
    const data = await fetch(API + "/api/debug/logs?lines=120").then(async (res) => {
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(formatApiError(json, res.status, "/api/debug/logs"));
      return json;
    });
    $("debugLogPaths").textContent = `本地: ${data.error_log}`;
    renderDebugView(data.server_tail || [], data.error_tail || []);
  } catch (e) {
    renderDebugView();
  }
}

async function fetchJSON(path, opts = {}) {
  let res;
  try {
    res = await fetch(API + path, opts);
  } catch (e) {
    const msg = `网络错误 ${path}: ${e.message}`;
    await logClientError(msg, { path });
    throw new Error(msg);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = formatApiError(data, res.status, path);
    await logClientError(`${path} -> ${msg}`, { path, status: res.status });
    await refreshDebugLogs();
    throw new Error(msg);
  }
  return data;
}

function formatScore(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(4);
}

function renderDataFileCard(info) {
  const card = $("dataFileCard");
  const startBtn = $("startBtn");

  if (!info || !info.data_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择数据文件</div>';
    selectedDataFile = null;
    selectedSymbol = null;
    startBtn.disabled = true;
    if ($("exportBtn")) $("exportBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  selectedDataFile = info.data_file;
  selectedSymbol = info.symbol || null;

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.data_file}</div>
    `;
    startBtn.disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  card.className = "data-file-card valid";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol}</span></div>
      <div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>
      <div class="item"><span class="label">K线</span><span class="value">${info.bars?.toLocaleString()}</span></div>
      <div class="item"><span class="label">进度</span><span class="value" id="fileProgressPct">—</span></div>
      <span class="path" title="${info.data_file}">${info.filename || info.data_file}</span>
    </div>
  `;
  startBtn.disabled = false;
}

function updateBtStartBtn() {
  const startBtn = $("btStartBtn");
  if (!startBtn) return;
  startBtn.disabled = btActive || !selectedStrategyFile;
}

function renderStrategyFileCard(info) {
  const card = $("btStrategyCard");
  if (!card) return;

  if (!info || !info.strategy_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择策略文件</div>';
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    updateBtStartBtn();
    return;
  }

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.strategy_file}</div>
    `;
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    updateBtStartBtn();
    return;
  }

  selectedStrategyFile = info.strategy_file;
  selectedStrategySymbol = info.symbol || null;
  card.className = "data-file-card valid";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol || "—"}</span></div>
      <div class="item"><span class="label">分数</span><span class="value">${formatScore(info.best_score)}</span></div>
      <div class="item"><span class="label">版本</span><span class="value">${info.vocab_version || "—"}</span></div>
      <span class="path" title="${info.strategy_file}">${info.filename || info.strategy_file}</span>
    </div>
  `;
  updateBtStartBtn();
}

function updateFileProgress(progress) {
  const el = document.getElementById("fileProgressPct");
  if (!el || !progress) return;
  el.textContent = `${progress.current_step} / ${progress.train_steps} (${progress.progress_pct}%)`;
}

const CHART_SERIES = [
  { key: "best_score", label: "最优分数", borderColor: "#5eead4", fillRGB: "94, 234, 212", yAxisID: "y" },
  { key: "val_score", label: "验证分数", borderColor: "#818cf8", fillRGB: "129, 140, 248", yAxisID: "y" },
  { key: "entropy", label: "策略熵", borderColor: "#fbbf24", fillRGB: "251, 191, 36", yAxisID: "y1" },
];

// 让曲线在填充区形成竖向渐变
function makeGradient(ctx, area, rgb) {
  if (!area) return `rgba(${rgb}, 0.08)`;
  const g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
  g.addColorStop(0, `rgba(${rgb}, 0.28)`);
  g.addColorStop(0.6, `rgba(${rgb}, 0.06)`);
  g.addColorStop(1, `rgba(${rgb}, 0)`);
  return g;
}

// 发光效果：在每条数据线绘制前设置对应颜色的柔和阴影
const glowPlugin = {
  id: "neonGlow",
  beforeDatasetDraw(chart, args) {
    const color = args?.meta?.dataset?.options?.borderColor;
    const ctx = chart.ctx;
    ctx.save();
    if (typeof color === "string") {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
    }
  },
  afterDatasetDraw(chart) {
    chart.ctx.restore();
  },
};
if (window.Chart) Chart.register(glowPlugin);

const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 450, easing: "easeOutQuart" },
  transitions: {
    active: { animation: { duration: 450, easing: "easeOutQuart" } },
  },
  plugins: {
    legend: {
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 16,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      backgroundColor: "rgba(8, 12, 20, 0.92)",
      borderColor: "rgba(94, 234, 212, 0.35)",
      borderWidth: 1,
      titleColor: "#e8edf4",
      bodyColor: "#a9bccf",
      titleFont: { family: "'JetBrains Mono'", size: 11 },
      bodyFont: { family: "'JetBrains Mono'", size: 11 },
      padding: 10,
      cornerRadius: 8,
      displayColors: true,
      usePointStyle: true,
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y1: {
      position: "right",
      ticks: { color: "#fbbf24", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { drawOnChartArea: false },
      border: { color: "rgba(251,191,36,0.2)" },
    },
  },
};

function buildChartDatasets(history) {
  return CHART_SERIES.filter((s) => history?.[s.key]?.length).map((s) => ({
    label: s.label,
    data: history[s.key],
    borderColor: s.borderColor,
    borderWidth: 2,
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBackgroundColor: s.borderColor,
    pointHoverBorderColor: "#05070d",
    fill: true,
    backgroundColor: (context) => {
      const { ctx, chartArea } = context.chart;
      return makeGradient(ctx, chartArea, s.fillRGB);
    },
    yAxisID: s.yAxisID,
  }));
}

function destroyChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
  chartSymbol = null;
}

function createChart(ctx, steps, history) {
  return new Chart(ctx, {
    type: "line",
    data: { labels: steps, datasets: buildChartDatasets(history) },
    options: CHART_OPTIONS,
  });
}

function updateChartInPlace(steps, history) {
  const prevLen = chart.data.labels.length;
  chart.data.labels = steps;

  const next = buildChartDatasets(history);
  for (const ds of next) {
    const existing = chart.data.datasets.find((d) => d.label === ds.label);
    if (existing) {
      existing.data = ds.data;
    } else {
      chart.data.datasets.push(ds);
    }
  }

  const nextLabels = new Set(next.map((d) => d.label));
  chart.data.datasets = chart.data.datasets.filter((d) => nextLabels.has(d.label));

  const grew = steps.length > prevLen;
  chart.update(grew ? "active" : "none");
}

function renderChart(history, label, progress) {
  const ctx = $("mainChart").getContext("2d");
  const steps = history?.step || [];
  if (!steps.length) {
    destroyChart();
    if (progress?.current_step > 0) {
      $("chartHint").textContent = `训练中 第 ${progress.current_step}/${progress.train_steps} 步，曲线每步更新`;
    } else {
      $("chartHint").textContent = "暂无历史数据（首步约需 15–30 秒）";
    }
    return;
  }

  const sameSymbol = chart && chartSymbol === label;
  if (sameSymbol) {
    updateChartInPlace(steps, history);
  } else {
    destroyChart();
    chart = createChart(ctx, steps, history);
    chartSymbol = label;
  }

  $("chartTitle").textContent = `${label} 训练曲线`;
  $("chartHint").textContent = `${steps.length} 个记录点`;
}

async function loadSymbolChart(symbol, progress) {
  if (!symbol) return;
  try {
    const data = await fetchJSON(`/api/symbols/${encodeURIComponent(symbol)}`);
    renderChart(data.history, symbol, progress || data);
    $("formulaText").textContent = data.formula_decoded || "—";
  } catch (e) {
    $("formulaText").textContent = "—";
  }
}

function renderStrategies(rows) {
  const tbody = $("strategiesBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="4">暂无已保存策略</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r) => `
    <tr>
      <td>${r.symbol}</td>
      <td>${r.timeframe || "—"}</td>
      <td>${formatScore(r.best_score)}</td>
      <td><code>${r.formula_decoded || "—"}</code></td>
    </tr>`
    )
    .join("");
}

function updateTrainingUI(training) {
  const job = training?.job;
  const active = training?.active;
  const pill = $("jobPill");
  const startBtn = $("startBtn");
  const stopBtn = $("stopBtn");

  if (!job || job.state === "idle") {
    pill.innerHTML = '<i class="pill-dot"></i>空闲';
    pill.className = "pill";
    startBtn.disabled = !selectedDataFile;
    stopBtn.disabled = true;
    $("logHint").textContent = "—";
    return;
  }

  const stateLabel = {
    running: "训练中",
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
  };
  const label = job.symbol ? `${job.symbol} ${job.timeframe || ""}`.trim() : "训练";
  const stateText = stateLabel[job.state] || job.state;
  pill.innerHTML = `<i class="pill-dot"></i>${stateText} · ${label}`;
  pill.className = "pill " + (job.state === "running" ? "running" : job.state);

  startBtn.disabled = active;
  stopBtn.disabled = !active;
  $("logHint").textContent = job.log_path || "—";

  const logView = $("logView");
  const atBottom = isViewAtBottom(logView);
  logView.textContent = (training.log_tail || []).join("\n") || "等待输出…";
  if (atBottom) logView.scrollTop = logView.scrollHeight;
}

async function refreshOverview() {
  let overview = { data_file: null, progress: null };
  let strategies = { strategies: [] };
  let training = { active: false, job: null, log_tail: [] };

  try {
    overview = await fetchJSON("/api/overview");
  } catch (_) {}

  try {
    strategies = await fetchJSON("/api/strategies");
  } catch (_) {}

  try {
    training = await fetchJSON("/api/training/status");
  } catch (_) {}

  if (overview.data_file) renderDataFileCard(overview.data_file);
  updateFileProgress(overview.progress);
  updateExportBtn(overview.progress, strategies.strategies);
  updateTrainingBtns(overview.progress, training);
  updateTrainingUI(training);
  renderStrategies(strategies.strategies);

  const sym = overview.progress?.symbol || selectedSymbol;
  if (sym && (training?.active || overview.progress)) {
    await loadSymbolChart(sym, overview.progress);
  }

  await refreshDebugLogs();
}

async function loadConfig() {
  const health = await fetch(API + "/api/health").then((r) => r.json()).catch(() => ({}));
  if (!health.version) {
    await logClientError(
      "后端版本过旧或未启动新版服务。请关闭旧进程后重新运行: python run_web.py",
      { health }
    );
  }

  const cfg = await fetchJSON("/api/config");
  debugMode = !!cfg.debug_mode;
  $("debugModeCheck").checked = debugMode;
  $("deviceMeta").textContent = `${cfg.train_steps} steps · batch ${cfg.batch_size} · ${cfg.device}`;
  if (cfg.error_log) {
    $("debugLogPaths").textContent = `本地: ${cfg.error_log}`;
  }
  if (cfg.data_file) renderDataFileCard(cfg.data_file);
  if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
}

async function browseStrategyFile() {
  try {
    const res = await fetchJSON("/api/strategy-file/browse", { method: "POST" });
    if (res.cancelled) return;
    renderStrategyFileCard(res);
  } catch (e) {
    $("debugView")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function loadBacktestStrategyContext() {
  try {
    const cfg = await fetchJSON("/api/config");
    if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  } catch (_) {
    /* ignore */
  }
}

async function browseDataFile() {
  try {
    const res = await fetchJSON("/api/data-file/browse", { method: "POST" });
    if (res.cancelled) return;
    renderDataFileCard(res);
    selectedSymbol = res.symbol;
    await loadSymbolChart(res.symbol);
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function startTraining() {
  if (!selectedDataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  try {
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: selectedDataFile }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function updateExportBtn(progress, strategies) {
  const sym = progress?.symbol || selectedSymbol;
  const hasStrategy = progress?.has_strategy || (strategies || []).some((s) => s.symbol === sym);
  const btn = $("exportBtn");
  if (btn) btn.disabled = !sym || !hasStrategy;
}

function updateTrainingBtns(progress, training) {
  const sym = progress?.symbol || selectedSymbol;
  const active = training?.active;
  const hasCheckpoint = Boolean(progress?.has_checkpoint);
  const exportBtn = $("exportTrainingBtn");
  const importBtn = $("importTrainingBtn");

  let exportTitle = "打包 checkpoint、训练曲线与策略为 zip";
  if (!sym) {
    exportTitle = "请先选择数据文件";
  } else if (active) {
    exportTitle = "训练进行中，请停止后再导出";
  } else if (!hasCheckpoint) {
    exportTitle = "该品种尚无检查点：至少训练满 20 步后才会生成（每 20 步保存一次）";
  }

  if (exportBtn) {
    exportBtn.disabled = !sym || !hasCheckpoint || !!active;
    exportBtn.title = exportTitle;
  }
  if (importBtn) {
    importBtn.disabled = !sym || !!active;
    importBtn.title = active ? "训练进行中，请停止后再导入" : "上传 .zip 或 .pt，下次训练断点续训";
  }
}

async function exportTraining() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/training/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const disp = res.headers.get("Content-Disposition") || "";
    const m = /filename="([^"]+)"/.exec(disp);
    const filename = m ? m[1] : `training_${sym.replace(/\./g, "_")}.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function triggerImportTraining() {
  const input = $("importTrainingFile");
  if (input) {
    input.value = "";
    input.click();
  }
}

async function handleImportTrainingFile(event) {
  const input = event.target;
  const file = input.files?.[0];
  if (!file) return;

  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/api/training/import?symbol=${encodeURIComponent(sym)}`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(formatApiError(data, res.status, "/api/training/import"));
    }
    if (data.symbol && data.symbol !== sym) {
      selectedSymbol = data.symbol;
    }
    clientErrors.push(`[${new Date().toLocaleString()}] ${data.message || "训练文件导入成功"}`);
    if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
    renderDebugView();
    await refreshOverview();
  } catch (e) {
    await logClientError(`导入训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    input.value = "";
  }
}

async function exportStrategy() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/strategies/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `strategy_${sym.replace(/\./g, "_")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出策略失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function stopTraining() {
  try {
    await fetchJSON("/api/training/stop", { method: "POST" });
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ═══════════════════════════════════════════════════════════════════
// 分页切换
// ═══════════════════════════════════════════════════════════════════
function switchPage(page) {
  if (page !== "train" && page !== "backtest") return;
  currentPage = page;
  document.querySelectorAll(".stepper .step").forEach((s) => {
    s.classList.toggle("active", s.dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
  if (page === "backtest") {
    if (!selectedStrategyFile) loadBacktestStrategyContext();
    refreshBacktest();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 回测：格式化辅助
// ═══════════════════════════════════════════════════════════════════
function fmtPct(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
}
function fmtSigned(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(digits);
}

// ═══════════════════════════════════════════════════════════════════
// 回测：状态轮询 + UI 更新
// ═══════════════════════════════════════════════════════════════════
async function refreshBacktest() {
  let st;
  try {
    st = await fetchJSON("/api/backtest/status");
  } catch (_) {
    return;
  }
  btActive = !!st.active;
  const job = st.job;
  const state = job?.state || "idle";

  // 按钮
  const startBtn = $("btStartBtn");
  const stopBtn = $("btStopBtn");
  updateBtStartBtn();
  if (stopBtn) stopBtn.disabled = !btActive;

  // 缓存刷新键：用最近一次任务的结束/开始时间
  btBuster = job?.finished_at || job?.started_at || btBuster;

  // 日志
  const logView = $("btLogView");
  if (logView) {
    const atBottom = isViewAtBottom(logView);
    logView.textContent = (st.log_tail || []).join("\n") || "等待任务…";
    if (atBottom) logView.scrollTop = logView.scrollHeight;
  }
  if ($("btLogHint")) $("btLogHint").textContent = job?.log_path || "—";

  // 阶段进度条
  updateBacktestPhase(st, state);

  // 结果报告（非运行态时刷新，运行态保留上次结果）
  if (!btActive) {
    await refreshBacktestReport();
  }
}

const BT_STATE_LABEL = {
  running: "回测中",
  completed: "已完成",
  failed: "失败",
  stopped: "已停止",
  idle: "待机",
};

function updateBacktestPhase(st, state) {
  const fill = $("btPhaseFill");
  const label = $("btPhaseLabel");
  if (!fill || !label) return;

  const total = st.phase_total || 7;
  const idx = st.phase_index || 0;

  let pct;
  if (btActive) {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = `${st.phase_label || "回测中"}…`;
    fill.classList.add("animate");
  } else if (state === "completed") {
    pct = 100;
    label.textContent = "完成";
    fill.classList.remove("animate");
  } else if (state === "failed" || state === "stopped") {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = BT_STATE_LABEL[state];
    fill.classList.remove("animate");
  } else {
    pct = 0;
    label.textContent = "待机";
    fill.classList.remove("animate");
  }
  fill.style.width = pct + "%";
}

async function refreshBacktestReport() {
  let data;
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/report?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/report";
  try {
    data = await fetchJSON(url);
  } catch (_) {
    return;
  }
  if (!data.available || !data.report) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = "尚未运行回测";
    renderCharts([]);
    return;
  }
  renderPortfolio(data.report);
  renderBacktestTable(data.report.symbols || {});
  renderCharts(data.charts || []);
}

function renderPortfolio(report) {
  const grid = $("btPortfolioGrid");
  if (!grid) return;
  const p = report.portfolio || {};
  const focus = report.focus_symbol || Object.keys(report.symbols || {})[0] || "";
  const symData = focus ? (report.symbols || {})[focus] : null;

  if (!Object.keys(p).length) {
    grid.innerHTML = '<div class="metric-empty">回测结果无绩效数据</div>';
    return;
  }

  const cards = [
    { label: "总收益", value: fmtPct(p.total_return), cls: p.total_return >= 0 ? "pos" : "neg" },
    { label: "Sharpe", value: fmtSigned(p.sharpe), cls: "accent" },
    { label: "Sortino", value: fmtSigned(p.sortino), cls: "accent" },
    { label: "最大回撤", value: (p.max_drawdown * 100).toFixed(2) + "%", cls: "neg" },
    { label: "Calmar", value: fmtSigned(p.calmar), cls: "accent" },
    {
      label: "交易数",
      value: symData?.n_trades ?? p.n_trades ?? "—",
      cls: "",
    },
    {
      label: "胜率",
      value: symData?.win_rate != null ? (symData.win_rate * 100).toFixed(1) + "%" : "—",
      cls: "",
    },
  ];

  grid.innerHTML = cards
    .map(
      (c) => `
    <div class="metric-card ${c.cls === "pos" || c.cls === "neg" ? c.cls : ""}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.value}</div>
    </div>`
    )
    .join("");

  if ($("btPortfolioHint")) {
    $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
  }
}

function renderBacktestTable(symbols) {
  const tbody = $("btTableBody");
  if (!tbody) return;
  const rows = Object.entries(symbols);
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="8">暂无回测结果</td></tr>';
    if ($("btTableHint")) $("btTableHint").textContent = "—";
    return;
  }
  if ($("btTableHint")) $("btTableHint").textContent = rows.length === 1 ? rows[0][0] : `${rows.length} 个品种`;
  tbody.innerHTML = rows
    .map(([sym, d]) => {
      const retCls = (d.total_return || 0) >= 0 ? "pos" : "neg";
      const shCls = (d.sharpe || 0) >= 0 ? "pos" : "neg";
      return `
      <tr>
        <td class="sym-cell">${sym}</td>
        <td class="${retCls}">${fmtPct(d.total_return)}</td>
        <td class="${shCls}">${fmtSigned(d.sharpe)}</td>
        <td>${fmtSigned(d.sortino)}</td>
        <td class="neg">${(d.max_drawdown * 100).toFixed(2)}%</td>
        <td>${fmtSigned(d.calmar, 2)}</td>
        <td>${d.n_trades ?? "—"}</td>
        <td>${d.win_rate != null ? (d.win_rate * 100).toFixed(1) + "%" : "—"}</td>
      </tr>`;
    })
    .join("");
}

function renderCharts(charts) {
  const gallery = $("btChartGallery");
  if (!gallery) return;
  if (!charts.length) {
    gallery.innerHTML = '<div class="metric-empty">暂无图表</div>';
    btChartSig = "";
    return;
  }
  const sig = charts.map((c) => c.name).join(",") + "|" + btBuster;
  if (sig === btChartSig) return; // 无变化，避免图片重载闪烁
  btChartSig = sig;

  const bust = encodeURIComponent(btBuster || "0");
  gallery.innerHTML = charts
    .map(
      (c) => `
    <div class="chart-card ${c.kind === "portfolio" ? "portfolio" : ""}" data-name="${c.name}">
      <div class="chart-card-head">${c.label}</div>
      <img src="${API}/api/backtest/chart/${encodeURIComponent(c.name)}?t=${bust}" alt="${c.label}" loading="lazy" />
    </div>`
    )
    .join("");
  if ($("btChartsHint")) {
    const sym = selectedStrategySymbol || selectedSymbol;
    $("btChartsHint").textContent = sym
      ? `${sym} · ${charts.length} 张图表 · 点击放大`
      : `${charts.length} 张图表 · 点击放大`;
  }
}

async function startBacktest() {
  if (!selectedStrategyFile) {
    await logClientError("请先选择策略文件");
    return;
  }
  const startBtn = $("btStartBtn");
  if (startBtn) startBtn.disabled = true;
  try {
    const res = await fetchJSON("/api/backtest/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy_file: selectedStrategyFile }),
    });
    if (res.strategy_file) renderStrategyFileCard(res.strategy_file);
    btChartSig = ""; // 强制下次重建图库
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
    updateBtStartBtn();
  }
}

async function stopBacktest() {
  try {
    await fetchJSON("/api/backtest/stop", { method: "POST" });
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// 图片灯箱
// ═══════════════════════════════════════════════════════════════════
function openLightbox(src) {
  const lb = $("lightbox");
  const img = $("lightboxImg");
  if (!lb || !img) return;
  img.src = src;
  lb.classList.add("open");
}
function closeLightbox() {
  const lb = $("lightbox");
  if (lb) lb.classList.remove("open");
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    refreshOverview();
    if (currentPage === "backtest" || btActive) refreshBacktest();
  }, 4000);
}

async function init() {
  try {
    await loadConfig();
    await refreshOverview();
  } catch (e) {
    await logClientError("初始化失败: " + e.message);
  }
  $("browseBtn").addEventListener("click", browseDataFile);
  $("startBtn").addEventListener("click", startTraining);
  $("stopBtn").addEventListener("click", stopTraining);
  $("exportBtn").addEventListener("click", exportStrategy);
  $("exportTrainingBtn").addEventListener("click", exportTraining);
  $("importTrainingBtn").addEventListener("click", triggerImportTraining);
  $("importTrainingFile").addEventListener("change", handleImportTrainingFile);
  $("debugModeCheck").addEventListener("change", (e) => setDebugMode(e.target.checked));

  // 步骤导航
  document.querySelectorAll(".stepper .step").forEach((btn) => {
    btn.addEventListener("click", () => switchPage(btn.dataset.page));
  });

  // 回测控制
  if ($("btBrowseStrategyBtn")) $("btBrowseStrategyBtn").addEventListener("click", browseStrategyFile);
  if ($("btStartBtn")) $("btStartBtn").addEventListener("click", startBacktest);
  if ($("btStopBtn")) $("btStopBtn").addEventListener("click", stopBacktest);

  // 图表灯箱
  const gallery = $("btChartGallery");
  if (gallery) {
    gallery.addEventListener("click", (e) => {
      const card = e.target.closest(".chart-card");
      if (!card) return;
      const img = card.querySelector("img");
      if (img) openLightbox(img.src);
    });
  }
  const lb = $("lightbox");
  if (lb) lb.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });

  startPolling();
}

init();
