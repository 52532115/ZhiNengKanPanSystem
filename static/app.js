const $ = (selector) => document.querySelector(selector);

const COLORS = {
  up: "#b34938",
  down: "#176b4d",
  ma20: "#aa7627",
  ma60: "#426d89",
  dif: "#aa7627",
  dea: "#426d89",
  rsi: "#7b4c8f",
  grid: "#e2e5e0",
  axis: "#88938d",
  ink: "#17231d",
  bench: "#426d89",
  stock: "#b34938",
  rs: "#aa7627",
};

const MAX_WATCHLIST = 20;

const state = {
  current: null,
  chart: null,
  rs: null,
  batch: null,
  allocation: null,
  watchlist: loadWatchlist(),
  suggest: { items: [], active: -1, timer: null, seq: 0, cache: new Map() },
  requestId: 0,
  batchRequestId: 0,
};

function loadWatchlist() {
  try {
    const raw = JSON.parse(localStorage.getItem("guanlan-watchlist") || "[]");
    if (!Array.isArray(raw)) return [];
    const seen = new Set();
    return raw
      .filter((item) => item && /^\d{6}$/.test(String(item.symbol || "")))
      .filter((item) => (seen.has(item.symbol) ? false : seen.add(item.symbol)));
  } catch {
    return [];
  }
}

function formatNumber(value, suffix = "", digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function signed(value, suffix = "", digits = 2) {
  const number = Number(value);
  if (value === null || value === undefined || Number.isNaN(number)) return "—";
  return `${number > 0 ? "+" : ""}${number.toFixed(digits)}${suffix}`;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])
  );
}

function status(message, type = "loading") {
  const el = $("#statusMessage");
  if (!message) { el.classList.add("hidden"); return; }
  el.textContent = message;
  el.className = `status-message ${type}`;
}

async function request(url, options) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("Content-Type") || "";
  if (!contentType.includes("json")) {
    // 旧版 python 进程没有新路由时会退回静态服务，返回 HTML 404/501。
    throw new Error(`服务端未返回 JSON（HTTP ${response.status}），可能仍在运行旧版本，请重启 python app.py`);
  }
  let payload;
  try { payload = await response.json(); } catch { throw new Error("服务器返回的 JSON 无法解析"); }
  if (!response.ok || !payload.ok) throw new Error(payload.error || `请求失败 (${response.status})`);
  return payload.data;
}

const getJSON = (url) => request(url);
const postJSON = (url, body) => request(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function changeClass(value) { return Number(value) > 0 ? "up" : Number(value) < 0 ? "down" : ""; }

/* ---------------------------------- 搜索联想 --------------------------------- */

async function fetchSuggest(query) {
  const key = query.trim();
  if (!key) return [];
  if (state.suggest.cache.has(key)) return state.suggest.cache.get(key);
  const data = await getJSON(`/api/search?q=${encodeURIComponent(key)}`);
  const items = data.items || [];
  state.suggest.cache.set(key, items);
  return items;
}

function initSearchSuggest() {
  const input = $("#symbolInput");
  const panel = $("#suggestPanel");
  const close = () => {
    state.suggest.seq += 1;
    panel.classList.add("hidden");
    panel.innerHTML = "";
    state.suggest.active = -1;
    input.setAttribute("aria-expanded", "false");
  };
  const highlight = () => {
    panel.querySelectorAll(".suggest-item").forEach((el, index) =>
      el.classList.toggle("active", index === state.suggest.active));
  };
  const pick = (index) => {
    const item = state.suggest.items[index];
    if (!item) return;
    input.value = item.symbol;
    close();
    analyzeSymbol(item.symbol);
  };
  const show = (items) => {
    if (!items.length) { close(); return; }
    state.suggest.items = items;
    state.suggest.active = -1;
    panel.innerHTML = items.map((item, index) => `
      <button type="button" class="suggest-item" role="option" data-index="${index}">
        <span class="si-name">${escapeHtml(item.name)}</span>
        <span class="si-code">${item.symbol}</span>
        <span class="si-type">${escapeHtml(item.type_label || "")}</span>
      </button>`).join("");
    panel.classList.remove("hidden");
    input.setAttribute("aria-expanded", "true");
  };

  input.addEventListener("input", () => {
    const value = input.value.trim();
    clearTimeout(state.suggest.timer);
    if (!value) { close(); return; }
    const seq = ++state.suggest.seq;
    state.suggest.timer = setTimeout(async () => {
      try {
        const items = await fetchSuggest(value);
        // 输入框已变化、面板已关闭或已发起新查询时，旧联想结果直接丢弃。
        if (seq !== state.suggest.seq || input.value.trim() !== value) return;
        show(items);
      } catch { close(); }
    }, 220);
  });

  input.addEventListener("keydown", (event) => {
    const total = state.suggest.items.length;
    if (panel.classList.contains("hidden") || !total) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      state.suggest.active = (state.suggest.active + 1) % total;
      highlight();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      state.suggest.active = (state.suggest.active - 1 + total) % total;
      highlight();
    } else if (event.key === "Enter" && state.suggest.active >= 0) {
      event.preventDefault();
      pick(state.suggest.active);
    } else if (event.key === "Escape") {
      close();
    }
  });

  panel.addEventListener("mousedown", (event) => {
    const button = event.target.closest("[data-index]");
    if (!button) return;
    event.preventDefault();
    pick(Number(button.dataset.index));
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !target.closest || !target.closest(".search-wrap")) close();
  });
  input.addEventListener("blur", () => setTimeout(() => {
    if (!panel.matches(":hover")) close();
  }, 120));
}

/* ---------------------------------- 主分析 ---------------------------------- */

function renderMetrics(metrics, assetType) {
  const commonItems = [
    ["20 日收益", metrics.return_20d, "%", 2], ["60 日收益", metrics.return_60d, "%", 2],
    ["RSI · 14", metrics.rsi14, "", 1], ["年化波动", metrics.volatility, "%", 1],
    ["最大回撤", metrics.max_drawdown, "%", 1], ["成交量比", metrics.volume_ratio, "×", 2]
  ];
  const items = assetType === "etf" || assetType === "fund"
    ? [...commonItems, ["20 日均价", metrics.ma20, "", 3], ["60 日均价", metrics.ma60, "", 3]]
    : [...commonItems, ["市盈率 TTM", metrics.pe_ttm, "", 2], ["市净率", metrics.pb, "", 2]];
  $("#metrics").innerHTML = items.map(([label, value, suffix, digits], index) =>
    `<div class="metric"><small>${label}</small><strong class="${index < 2 ? changeClass(value) : ""}">${formatNumber(value, suffix, digits)}</strong></div>`
  ).join("");
}

function renderComponents(components, coverage) {
  const box = $("#components");
  const block = $("#factorBlock");
  if (!block || !box) return;
  if (!Array.isArray(components) || !components.length) {
    block.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  block.classList.remove("hidden");
  box.innerHTML = components.map((item) => {
    const value = Number(item.score);
    const known = Number.isFinite(value);
    const tone = !known ? "" : value >= 60 ? "strong" : value <= 40 ? "weak" : "";
    return `<div class="component" title="权重 ${(item.weight * 100).toFixed(0)}%">
      <span class="component-label">${escapeHtml(item.label || item.key)}</span>
      <span class="component-value">${known ? value.toFixed(0) : "—"}</span>
      <span class="component-track"><i class="${tone}" style="width:${known ? Math.max(2, value) : 0}%"></i></span>
    </div>`;
  }).join("");
  const head = block.querySelector(".factor-head small");
  if (head) {
    head.textContent = coverage < 0.99
      ? `各分量 0–100 · 分量覆盖 ${(coverage * 100).toFixed(0)}%`
      : "各分量 0–100，只描述状态";
  }
}

function pctLabel(value) {
  return value === null || value === undefined ? "—" : `${Math.round(value)} 分位`;
}

function renderEvidence(selector, evidence) {
  const node = $(selector);
  if (!node) return;
  const status = evidence?.status || "untested";
  node.className = `evidence-badge ${status}`;
  node.textContent = `验证：${evidence?.label || "未验证"}${evidence?.detail ? ` · ${evidence.detail}` : ""}`;
  node.title = evidence?.detail || "";
}

function renderDescriptor(field) {
  const band = field.band || {};
  const hasField = Boolean(field.available);
  // 同类分位可能被显式关闭（指数没有同类校准）：此时退回"相对自身"分位，
  // 不能把 null 当 0 画成一根空的进度条，也不能继续写"同类"。
  const classPct = field.percentile && field.percentile.available !== false
    ? field.percentile.value : null;
  const hasClass = hasField && classPct !== null && classPct !== undefined;
  const self = field.self_percentile || null;
  const hasSelf = hasField && self && self.value !== null && self.value !== undefined;
  const shown = hasClass ? classPct : (hasSelf ? self.value : null);
  $("#descriptorBand").textContent = hasField ? band.label || "—" : "数据不足";
  $("#descriptorPct").textContent = shown === null ? "" : (hasClass ? `同类 ${pctLabel(shown)}` : `自身 ${pctLabel(shown)}`);
  const track = $("#descriptorTrack");
  track.className = `pct-track ${hasField ? band.key || "" : ""}`;
  const fill = track.querySelector("i");
  if (fill) fill.style.width = shown === null ? "0%" : `${Math.max(0, Math.min(100, shown))}%`;
  const meta = [];
  if (hasClass) {
    const n = field.percentile.n;
    meta.push(`<span>参照系：<b>${escapeHtml(field.percentile.label)}</b>${n ? `（${n} 个样本）` : "（内置校准表）"}</span>`);
    if (hasSelf) {
      meta.push(`<span>相对自身：<b>近 ${self.window} 日 ${pctLabel(self.value)}</b></span>`);
    }
    if (field.pool_percentile) {
      meta.push(`<span>当日同池：<b>${pctLabel(field.pool_percentile.value)}</b> / ${field.pool_percentile.n} 只</span>`);
    }
  } else if (hasSelf) {
    meta.push(`<span>参照系：<b>近 ${self.window} 日自身分位</b>（${escapeHtml(field.percentile?.reason || "无同类校准")}）</span>`);
    meta.push(`<span>档位依据：<b>${band.basis === "self" ? "自身分位" : "同类分位"}</b></span>`);
  } else {
    meta.push(`<span>${escapeHtml(field.reason || "暂无可用状态")}</span>`);
  }
  $("#descriptorMeta").innerHTML = meta.join("");
  $("#descriptorNote").textContent = field.headline || field.usage || "";
  renderComponents(field.components, field.coverage ?? 1);
}

function renderSelection(field) {
  const available = Boolean(field.available) && field.percentile !== null && field.percentile !== undefined;
  $("#selectionPct").textContent = available ? `${Math.round(field.percentile)} 分位` : "—";
  $("#selectionPool").textContent = available
    ? `${field.label || "当日同池"} · ${field.pool_size} 只`
    : "需要池内比较";
  const track = $("#selectionTrack");
  track.className = `pct-track ${available ? "neutral" : ""}`;
  const fill = track.querySelector("i");
  if (fill) fill.style.width = available ? `${Math.max(0, Math.min(100, field.percentile))}%` : "0%";
  $("#selectionMeta").innerHTML = available
    ? `<span>只回答「同一天、同一池子里排第几」</span><span>换池子排名就变，不能跨池比较</span>`
    : `<span>${escapeHtml(field.reason || "单标的查询没有横截面")}</span>`;
  $("#selectionNote").textContent = field.note || field.usage || "";
}

function renderRegime(field) {
  const available = Boolean(field.available);
  $("#regimeState").textContent = available ? field.label : "未知";
  $("#regimeExposure").textContent = available ? `总暴露上限 ${Math.round((field.exposure ?? 0) * 100)}%` : "";
  document.querySelectorAll("#regimeScale span").forEach((node) => {
    node.classList.toggle("on", available && node.dataset.state === field.state);
  });
  const meta = [];
  if (available) {
    const history = field.history || {};
    meta.push(`<span>标准化分数 z = <b>${Number(field.z).toFixed(2)}</b>（切点 ±${field.cut}σ）</span>`);
    meta.push(`<span>近 ${history.window} 个交易日里该状态占 <b>${history.share}%</b>，切换 ${history.switches} 次</span>`);
    meta.push(`<span>基准：<b>${escapeHtml(field.benchmark || "沪深300")}</b></span>`);
  } else {
    meta.push(`<span>${escapeHtml(field.reason || "暂无基准行情")}</span>`);
  }
  $("#regimeMeta").innerHTML = meta.join("");
  $("#regimeNote").textContent = available ? field.guidance : field.usage || "";
}

function renderAnalysis(data) {
  state.current = data;
  $("#stockCode").textContent = data.symbol;
  $("#assetBadge").textContent = data.asset_label;
  $("#asOf").textContent = `数据截至 ${data.as_of}${data.data_source === "fallback" ? " · 备用线路" : ""}`;
  $("#stockName").textContent = data.name;
  $("#stockPrice").textContent = `¥ ${formatNumber(data.price, "", data.price_precision)}`;
  const change = $("#dayChange");
  change.textContent = `${Number(data.day_change) > 0 ? "+" : ""}${formatNumber(data.day_change, "%")}`;
  change.className = `day-change ${changeClass(data.day_change)}`;

  const fieldset = data.fields || {};
  renderDescriptor(fieldset.descriptor || { available: false, reason: "缺少描述字段" });
  renderSelection(fieldset.selection || { available: false });
  renderRegime(fieldset.regime || { available: false });
  renderEvidence("#descriptorEvidence", data.evidence?.descriptor);
  renderEvidence("#selectionEvidence", data.evidence?.selection || fieldset.selection?.evidence);
  renderEvidence("#regimeEvidence", data.evidence?.regime);
  if (fieldset.regime?.available) updateMarketPill(fieldset.regime);

  $("#supportLevel").textContent = formatNumber(data.levels.support, "", data.price_precision);
  $("#resistanceLevel").textContent = formatNumber(data.levels.resistance, "", data.price_precision);
  $("#riskLevel").textContent = formatNumber(data.levels.risk_reference, "", data.price_precision);
  $("#reasons").innerHTML = data.reasons.map((reason) => `<div class="reason ${reason.type}">${escapeHtml(reason.text)}</div>`).join("");
  $("#methodology").textContent = data.methodology;
  renderMetrics(data.metrics, data.asset_type);
  $("#dashboard").classList.remove("hidden");
  initChart(data.chart);
  refreshRS(data.symbol);
  updateWatchButton();
  if (state.watchlist.some((entry) => entry.symbol === data.symbol)) refreshWatchlist();
}

async function resolveSymbol(query) {
  const cleaned = String(query || "").trim().replace(/^(sh|sz|bj)/i, "");
  if (/^\d{6}$/.test(cleaned)) return { symbol: cleaned, label: cleaned };
  const candidates = await fetchSuggest(query);
  if (!candidates.length) return null;
  const exact = candidates.find((item) => item.name === query);
  const fuzzy = candidates.find((item) => item.name.includes(query) || query.includes(item.name));
  const best = exact || fuzzy || candidates[0];
  return { symbol: best.symbol, label: best.name };
}

function syncUrl(symbol, push) {
  const params = new URLSearchParams(location.search);
  params.set("symbol", symbol);
  const next = `${location.pathname}?${params.toString()}`;
  const current = `${location.pathname}${location.search}`;
  if (current === next) return;
  if (push) history.pushState({ symbol }, "", next);
  else history.replaceState({ symbol }, "", next);
}

async function analyzeSymbol(query, { pushHistory = true, scroll = true } = {}) {
  // 每次分析递增序号：后发的请求先返回时，旧响应不能覆盖新结果。
  const requestId = ++state.requestId;
  const text = String(query || "").trim();
  if (!text) { status("请输入股票代码、名称或拼音缩写，例如 600519、茅台、gzmt。", "error"); return; }
  let resolved;
  if (/^\d{6}$/.test(text.replace(/^(sh|sz|bj)/i, ""))) {
    resolved = { symbol: text.replace(/^(sh|sz|bj)/i, ""), label: text };
  } else {
    status(`正在解析「${text}」…`);
    try {
      resolved = await resolveSymbol(text);
    } catch (error) {
      if (requestId === state.requestId) status(error.message, "error");
      return;
    }
    if (requestId !== state.requestId) return;
    if (!resolved) { status(`没有找到与「${text}」匹配的沪深 A 股或场内基金。`, "error"); return; }
  }
  if (requestId !== state.requestId) return;
  status(`正在读取 ${resolved.symbol} 的行情并计算指标…`);
  $("#symbolInput").value = resolved.label;
  try {
    const data = await getJSON(`/api/analyze?symbol=${encodeURIComponent(resolved.symbol)}`);
    if (requestId !== state.requestId) return;
    renderAnalysis(data);
    status("");
    syncUrl(data.symbol, pushHistory);
    if (scroll) $("#dashboard").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (requestId === state.requestId) status(error.message, "error");
  }
}

function formatVolume(value) {
  if (value === null || value === undefined) return "—";
  if (value >= 1e8) return `${(value / 1e8).toFixed(2)}亿`;
  if (value >= 1e4) return `${(value / 1e4).toFixed(2)}万`;
  return Math.round(value).toLocaleString("zh-CN");
}

/* ----------------------------------- 图表 ----------------------------------- */

function initChart(points) {
  const pane = state.chart?.pane || "volume";
  state.chart = { points, range: [0, Math.max(0, points.length - 1)], pane };
  $("#chartReset").classList.add("hidden");
  syncChartTabs();
  drawChart();
}

function syncChartTabs() {
  document.querySelectorAll("#chartTabs [data-pane]").forEach((button) =>
    button.classList.toggle("active", button.dataset.pane === (state.chart?.pane || "volume")));
}

function resetChartRange() {
  if (!state.chart) return;
  state.chart.range = [0, Math.max(0, state.chart.points.length - 1)];
  $("#chartReset").classList.add("hidden");
  drawChart();
}

function drawChart() {
  const chart = state.chart;
  if (!chart) return;
  const canvas = $("#priceChart");
  const shell = canvas?.parentElement;
  if (!canvas || !shell) return;
  const tooltip = $("#chartTooltip");
  const width = shell.clientWidth;
  const height = shell.clientHeight;
  if (!width || !height) return;

  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const points = chart.points;
  const start = chart.range[0];
  const end = chart.range[1];
  const view = points.slice(start, end + 1);
  const n = view.length;
  if (!n) return;

  const pad = { left: 56, right: 16, top: 14, bottom: 22 };
  const innerH = height - pad.top - pad.bottom;
  const subH = Math.max(78, Math.round(innerH * 0.3));
  const paneGap = 24;
  const priceTop = pad.top;
  const priceBottom = pad.top + innerH - subH - paneGap;
  const subTop = priceBottom + paneGap;
  const subBottom = height - pad.bottom;
  const plotW = width - pad.left - pad.right;
  const step = plotW / Math.max(1, n);
  const x = (index) => pad.left + step * (index + 0.5);
  const precision = state.current?.price_precision ?? 2;
  const fmt = (value) => (value === null || value === undefined ? "—" : Number(value).toFixed(precision));

  const priceValues = [];
  view.forEach((point) => {
    const high = point.high ?? Math.max(point.open, point.close);
    const low = point.low ?? Math.min(point.open, point.close);
    [high, low, point.ma20, point.ma60].forEach((value) => {
      if (value !== null && value !== undefined) priceValues.push(value);
    });
  });
  const rawMin = Math.min(...priceValues);
  const rawMax = Math.max(...priceValues);
  const pricePadding = Math.max((rawMax - rawMin) * 0.06, rawMax * 0.002) || 1;
  const minPrice = rawMin - pricePadding;
  const maxPrice = rawMax + pricePadding;
  const priceRange = maxPrice - minPrice || 1;
  const priceY = (value) => priceTop + (maxPrice - value) * (priceBottom - priceTop) / priceRange;

  let maxVolume = 1;
  let macdAbs = 1;
  view.forEach((point) => {
    maxVolume = Math.max(maxVolume, point.volume || 0, point.volume_ma5 || 0, point.volume_ma10 || 0);
    macdAbs = Math.max(macdAbs, Math.abs(point.dif || 0), Math.abs(point.dea || 0), Math.abs(point.macd || 0));
  });
  maxVolume *= 1.08;
  macdAbs *= 1.15;
  const midSub = (subTop + subBottom) / 2;
  const halfSub = (subBottom - subTop) / 2;
  const volumeY = (value) => subBottom - (value / maxVolume) * (subBottom - subTop);
  const macdY = (value) => midSub - (value / macdAbs) * halfSub * 0.92;
  const rsiY = (value) => subBottom - (value / 100) * (subBottom - subTop);
  const bodyW = Math.max(1, Math.min(11, step * 0.66));

  function line(key, color, lineWidth, yMap, dashed = false) {
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.setLineDash(dashed ? [4, 3] : []);
    let started = false;
    view.forEach((point, index) => {
      const value = point[key];
      if (value === null || value === undefined) return;
      const cx = x(index), cy = yMap(value);
      if (started) ctx.lineTo(cx, cy); else ctx.moveTo(cx, cy);
      started = true;
    });
    ctx.stroke();
    ctx.restore();
  }

  function drawCandles() {
    view.forEach((point, index) => {
      const open = point.open ?? point.close;
      const close = point.close;
      const high = point.high ?? Math.max(open, close);
      const low = point.low ?? Math.min(open, close);
      const rising = close >= open;
      const color = rising ? COLORS.up : COLORS.down;
      const cx = x(index);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(cx, priceY(high));
      ctx.lineTo(cx, priceY(low));
      ctx.stroke();
      const yOpen = priceY(open);
      const yClose = priceY(close);
      const top = Math.min(yOpen, yClose);
      const bodyHeight = Math.max(1, Math.abs(yClose - yOpen));
      ctx.fillRect(cx - bodyW / 2, top, bodyW, bodyHeight);
    });
  }

  function drawSubPane() {
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    const hasField = (key) => view.some((point) => point[key] !== null && point[key] !== undefined);
    const explainMissing = (label) => {
      ctx.fillStyle = "#9aa39d";
      ctx.textAlign = "center";
      ctx.font = "12px \"PingFang SC\", \"Microsoft YaHei\", sans-serif";
      ctx.fillText(`当前行情缺少 ${label} 字段，请重启服务后重新分析`, pad.left + plotW / 2, (subTop + subBottom) / 2);
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
    };
    if (chart.pane === "volume" && !hasField("volume")) { explainMissing("成交量"); return; }
    if (chart.pane === "macd" && !hasField("dif")) { explainMissing("MACD"); return; }
    if (chart.pane === "rsi" && !hasField("rsi14")) { explainMissing("RSI"); return; }
    if (chart.pane === "volume") {
      view.forEach((point, index) => {
        const rising = Number(point.pct_change) >= 0;
        ctx.fillStyle = rising ? "rgba(190,73,58,.72)" : "rgba(23,107,77,.72)";
        const top = volumeY(point.volume || 0);
        ctx.fillRect(x(index) - bodyW / 2, top, bodyW, Math.max(1, subBottom - top));
      });
      line("volume_ma5", "#d19a35", 1.15, volumeY);
      line("volume_ma10", "#875b98", 1.15, volumeY);
      ctx.fillStyle = "#78857e";
      ctx.fillText("成交量（手）", pad.left, subTop - 6);
      ctx.fillStyle = COLORS.axis;
      ctx.fillText(formatVolume(maxVolume), 2, subTop + 8);
      ctx.fillText(formatVolume(maxVolume / 2), 2, subTop + (subBottom - subTop) / 2 + 3);
      return;
    }
    if (chart.pane === "macd") {
      view.forEach((point, index) => {
        const value = point.macd;
        if (value === null || value === undefined) return;
        ctx.fillStyle = value >= 0 ? "rgba(190,73,58,.75)" : "rgba(23,107,77,.75)";
        const y = macdY(value);
        const zero = macdY(0);
        ctx.fillRect(x(index) - bodyW / 2, Math.min(y, zero), bodyW, Math.max(1, Math.abs(zero - y)));
      });
      ctx.save();
      ctx.strokeStyle = "#d3d8d2";
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(pad.left, macdY(0));
      ctx.lineTo(width - pad.right, macdY(0));
      ctx.stroke();
      ctx.restore();
      line("dif", COLORS.dif, 1.2, macdY);
      line("dea", COLORS.dea, 1.2, macdY);
      ctx.fillStyle = "#78857e";
      ctx.fillText("MACD(12,26,9)", pad.left, subTop - 6);
      ctx.fillStyle = COLORS.axis;
      ctx.fillText(macdAbs.toFixed(3), 2, subTop + 8);
      ctx.fillText((0).toFixed(3), 2, macdY(0) + 3);
      ctx.fillText((-macdAbs).toFixed(3), 2, subBottom - 1);
      return;
    }
    // RSI
    [[70, "#c9b79b"], [30, "#c9b79b"]].forEach(([level, color]) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pad.left, rsiY(level));
      ctx.lineTo(width - pad.right, rsiY(level));
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = COLORS.axis;
      ctx.fillText(String(level), 2, rsiY(level) + 3);
    });
    line("rsi14", COLORS.rsi, 1.5, rsiY);
    ctx.fillStyle = "#78857e";
    ctx.fillText("RSI(14)", pad.left, subTop - 6);
    ctx.fillStyle = COLORS.axis;
    ctx.fillText("100", 2, subTop + 8);
    ctx.fillText("0", 2, subBottom - 1);
  }

  function render(activeIndex = null, selection = null) {
    ctx.clearRect(0, 0, width, height);
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.lineWidth = 1;

    for (let i = 0; i < 5; i++) {
      const y = priceTop + (i * (priceBottom - priceTop)) / 4;
      ctx.strokeStyle = COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.axis;
      ctx.fillText((maxPrice - (i * priceRange) / 4).toFixed(precision), 2, y + 3);
    }

    ctx.strokeStyle = "#cfd5ce";
    ctx.beginPath();
    ctx.moveTo(pad.left, subTop - 16);
    ctx.lineTo(width - pad.right, subTop - 16);
    ctx.stroke();

    drawSubPane();
    drawCandles();
    line("ma20", COLORS.ma20, 1.3, priceY);
    line("ma60", COLORS.ma60, 1.3, priceY);

    const tickCount = Math.min(4, n);
    const ticks = [];
    for (let i = 0; i < tickCount; i++) {
      ticks.push(tickCount === 1 ? 0 : Math.round((i * (n - 1)) / (tickCount - 1)));
    }
    ctx.fillStyle = COLORS.axis;
    ctx.textAlign = "center";
    [...new Set(ticks)].forEach((index) => ctx.fillText(view[index].date.slice(5), x(index), height - 7));
    ctx.textAlign = "left";

    if (selection) {
      const a = Math.min(selection.from, selection.to);
      const b = Math.max(selection.from, selection.to);
      const left = x(a) - step / 2;
      const right = x(b) + step / 2;
      ctx.fillStyle = "rgba(66,109,137,.14)";
      ctx.fillRect(left, priceTop, Math.max(1, right - left), subBottom - priceTop);
      ctx.strokeStyle = "rgba(66,109,137,.55)";
      ctx.beginPath();
      ctx.moveTo(left, priceTop);
      ctx.lineTo(left, subBottom);
      ctx.moveTo(right, priceTop);
      ctx.lineTo(right, subBottom);
      ctx.stroke();
    }

    if (activeIndex !== null) {
      const point = view[activeIndex];
      const cx = x(activeIndex);
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(70,84,76,.46)";
      ctx.beginPath();
      ctx.moveTo(cx, priceTop);
      ctx.lineTo(cx, subBottom);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = COLORS.ink;
      ctx.beginPath();
      ctx.arc(cx, priceY(point.close), 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const indexAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    return Math.max(0, Math.min(n - 1, Math.floor((mx - pad.left) / step)));
  };

  let dragFrom = null;

  canvas.onmousedown = (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    dragFrom = indexAt(event);
    tooltip.classList.add("hidden");
  };
  canvas.onmousemove = (event) => {
    const index = indexAt(event);
    if (dragFrom !== null) {
      render(null, { from: dragFrom, to: index });
      return;
    }
    render(index);
    const point = view[index];
    const change = Number(point.pct_change);
    tooltip.innerHTML = [
      point.date,
      `开 ${fmt(point.open)}　高 ${fmt(point.high)}`,
      `低 ${fmt(point.low)}　收 ${fmt(point.close)}`,
      `涨跌 ${change > 0 ? "+" : ""}${change.toFixed(2)}%`,
      `量 ${formatVolume(point.volume)} 手`,
      `MACD ${Number(point.macd ?? 0).toFixed(3)}　RSI ${point.rsi14 ?? "—"}`,
    ].join("<br>");
    tooltip.style.left = `${Math.max(82, Math.min(width - 82, x(index)))}px`;
    tooltip.style.top = `${Math.max(96, priceY(point.high ?? point.close))}px`;
    tooltip.classList.remove("hidden");
  };
  canvas.onmouseup = (event) => {
    if (dragFrom === null) return;
    const to = indexAt(event);
    const from = Math.min(dragFrom, to);
    const till = Math.max(dragFrom, to);
    dragFrom = null;
    if (till - from >= 3) {
      chart.range = [start + from, start + till];
      $("#chartReset").classList.remove("hidden");
      drawChart();
    } else {
      render();
    }
  };
  canvas.onmouseleave = () => {
    dragFrom = null;
    tooltip.classList.add("hidden");
    render();
  };
  canvas.ondblclick = () => resetChartRange();

  render();
}

/* --------------------------------- 相对强弱 --------------------------------- */

function renderRS(data) {
  state.rs = data;
  $("#rsBenchmark").textContent = data.benchmark_name;
  const chip = $("#rsVerdict");
  chip.textContent = `${data.verdict} · RS ${Number(data.rs).toFixed(1)}`;
  chip.className = `rs-verdict ${data.tone}`;
  const trendLabel = { up: "走强", down: "走弱", flat: "走平" }[data.rs_trend] || "—";
  const stats = [
    ["个股区间收益", signed(data.stock_return, "%"), changeClass(data.stock_return)],
    ["基准区间收益", signed(data.benchmark_return, "%"), changeClass(data.benchmark_return)],
    ["超额收益", signed(data.excess_return, "%"), changeClass(data.excess_return)],
    ["RS 比率", Number(data.rs).toFixed(2), data.rs >= 100 ? "up" : "down"],
    ["RS 20 日变化", signed(data.rs_change_20d), changeClass(data.rs_change_20d)],
    ["RS 趋势", trendLabel, data.rs_trend === "up" ? "up" : data.rs_trend === "down" ? "down" : ""],
  ];
  $("#rsStats").innerHTML = stats
    .map(([label, value, cls]) => `<div><small>${label}</small><strong class="${cls}">${value}</strong></div>`)
    .join("");
  $("#rsNote").textContent = `${data.start_date} 至 ${data.as_of}（${data.window} 个交易日）· ${data.note}`;
  drawRSChart(data.series);
}

async function refreshRS(symbol) {
  const requestId = state.requestId;
  try {
    const data = await getJSON(`/api/rs?symbol=${encodeURIComponent(symbol)}&days=120&benchmark=000300`);
    if (requestId !== state.requestId || state.current?.symbol !== symbol) return;
    renderRS(data);
  } catch (error) {
    if (requestId !== state.requestId || state.current?.symbol !== symbol) return;
    state.rs = null;
    $("#rsVerdict").textContent = "暂不可用";
    $("#rsVerdict").className = "rs-verdict";
    $("#rsStats").innerHTML = `<p class="rs-empty">${escapeHtml(error.message)}</p>`;
    $("#rsNote").textContent = "";
    const canvas = $("#rsChart");
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function drawRSChart(series) {
  const canvas = $("#rsChart");
  if (!canvas || !series || !series.length) return;
  const shell = canvas.parentElement;
  const width = shell.clientWidth;
  const height = shell.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 46, right: 14, top: 14, bottom: 20 };
  const innerH = height - pad.top - pad.bottom;
  const gap = 22;
  const topPane = Math.round(innerH * 0.64);
  const normTop = pad.top;
  const normBottom = pad.top + topPane;
  const rsTop = normBottom + gap;
  const rsBottom = height - pad.bottom;
  const plotW = width - pad.left - pad.right;
  const n = series.length;
  const x = (index) => pad.left + (plotW / Math.max(1, n - 1)) * index;

  const normValues = series.flatMap((point) => [point.stock, point.benchmark]);
  const normMin = Math.min(...normValues);
  const normMax = Math.max(...normValues);
  const normPad = Math.max((normMax - normMin) * 0.1, 0.5);
  const low = normMin - normPad;
  const high = normMax + normPad;
  const normY = (value) => normTop + ((high - value) / (high - low)) * (normBottom - normTop);

  const rsValues = series.map((point) => point.rs);
  const rsMin = Math.min(...rsValues, 100);
  const rsMax = Math.max(...rsValues, 100);
  const rsPad = Math.max((rsMax - rsMin) * 0.15, 0.4);
  const rsLow = rsMin - rsPad;
  const rsHigh = rsMax + rsPad;
  const rsY = (value) => rsTop + ((rsHigh - value) / (rsHigh - rsLow)) * (rsBottom - rsTop);

  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "left";

  // 归一化面板：网格 + 100 基准线
  for (let i = 0; i < 3; i++) {
    const value = high - (i * (high - low)) / 2;
    const y = normY(value);
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = COLORS.axis;
    ctx.fillText(value.toFixed(1), 2, y + 3);
  }
  ctx.save();
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = "#cfd5ce";
  ctx.beginPath();
  ctx.moveTo(pad.left, normY(100));
  ctx.lineTo(width - pad.right, normY(100));
  ctx.stroke();
  ctx.restore();

  const drawSeries = (key, color, width2, yMap, dashed) => {
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    series.forEach((point, index) => {
      const cx = x(index), cy = yMap(point[key]);
      if (index === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.restore();
  };

  drawSeries("benchmark", COLORS.bench, 1.3, normY, true);
  drawSeries("stock", COLORS.stock, 1.9, normY, false);

  // RS 面板：面积 + 折线
  ctx.save();
  ctx.beginPath();
  series.forEach((point, index) => {
    const cx = x(index), cy = rsY(point.rs);
    if (index === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  });
  ctx.lineTo(x(n - 1), rsBottom);
  ctx.lineTo(x(0), rsBottom);
  ctx.closePath();
  ctx.fillStyle = "rgba(170,118,39,.13)";
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = "#d8c8a6";
  ctx.beginPath();
  ctx.moveTo(pad.left, rsY(100));
  ctx.lineTo(width - pad.right, rsY(100));
  ctx.stroke();
  ctx.restore();

  drawSeries("rs", COLORS.rs, 1.4, rsY, false);

  ctx.fillStyle = "#78857e";
  ctx.fillText("RS 比率（100 = 同步）", pad.left, rsTop - 6);
  ctx.fillStyle = COLORS.axis;
  ctx.fillText(rsHigh.toFixed(1), 2, rsTop + 8);
  ctx.fillText(rsLow.toFixed(1), 2, rsBottom - 1);

  const ticks = [0, Math.floor((n - 1) / 2), n - 1];
  ctx.textAlign = "center";
  ctx.fillStyle = COLORS.axis;
  [...new Set(ticks)].forEach((index) => ctx.fillText(series[index].date.slice(5), x(index), height - 6));
  ctx.textAlign = "left";
}

/* --------------------------------- 自选面板 --------------------------------- */

function saveWatchlist() {
  localStorage.setItem("guanlan-watchlist", JSON.stringify(state.watchlist));
  updateWatchButton();
  refreshWatchlist();
}

function updateWatchButton() {
  if (!state.current) return;
  const exists = state.watchlist.some((item) => item.symbol === state.current.symbol);
  const full = state.watchlist.length >= MAX_WATCHLIST;
  const button = $("#watchButton");
  button.textContent = exists ? "✓ 已在自选" : full ? `自选已满 ${MAX_WATCHLIST}` : "＋ 加入自选";
  button.title = !exists && full ? `自选最多 ${MAX_WATCHLIST} 只，请先移除一只` : "";
  button.classList.toggle("active", exists);
}

function watchCardHtml(entry, item) {
  const name = item?.name || entry.name || entry.symbol;
  const precision = item?.price_precision ?? 2;
  const change = item ? Number(item.day_change) : null;
  const band = item?.descriptor?.band;
  const descriptorPct = item?.descriptor?.percentile?.value;
  const poolPct = item?.selection?.percentile;
  const body = item
    ? `<div class="wc-price"><span>${formatNumber(item.price, "", precision)}</span><em class="${changeClass(change)}">${signed(change, "%")}</em></div>
       <div class="wc-foot"><span class="score-chip ${band?.key || ""}">${band?.label || "—"} ${descriptorPct === null || descriptorPct === undefined ? "" : Math.round(descriptorPct)}</span><span class="wc-view">池内 ${poolPct === null || poolPct === undefined ? "—" : Math.round(poolPct)} 分位</span></div>
       ${item.alert ? `<div class="wc-flag">${escapeHtml(item.alert_reason)}</div>` : ""}`
    : `<div class="wc-pending">计算中…</div>`;
  return `<article class="watch-card ${item?.alert ? "alert" : ""}">
    <button type="button" class="watch-open" data-open="${entry.symbol}">
      <div class="wc-top"><strong>${escapeHtml(name)}</strong><span class="wc-code">${entry.symbol}</span></div>
      ${body}
    </button>
    <button type="button" class="remove" data-remove="${entry.symbol}" title="移除">×</button>
  </article>`;
}

function renderWatchlist(items) {
  const container = $("#watchlistBody");
  if (!state.watchlist.length) {
    container.innerHTML = `<div class="empty-watch">尚未添加自选。加入 2 只以上后，这里会显示趋势状态档位、池内分位与客观异动提示。</div>`;
    return;
  }
  const map = new Map((items || []).map((item) => [item.symbol, item]));
  const rows = state.watchlist.slice(0, MAX_WATCHLIST).map((entry) => ({ entry, item: map.get(entry.symbol) }));
  const over = state.watchlist.length - MAX_WATCHLIST;
  const limitNote = over > 0
    ? `<div class="watch-limit">自选共 ${state.watchlist.length} 只，超过 ${MAX_WATCHLIST} 只上限；仅前 ${MAX_WATCHLIST} 只参与评分，请移除 ${over} 只。</div>`
    : "";
  const rank = (row) => (row.item ? (row.item.selection?.percentile ?? row.item.descriptor?.percentile?.value ?? -1) : -1);
  const byScore = (a, b) => rank(b) - rank(a);
  const alerts = rows.filter((row) => row.item?.alert).sort(byScore);
  const others = rows.filter((row) => !row.item?.alert).sort(byScore);
  let html = limitNote;
  if (alerts.length) {
    html += `<div class="watch-group-label">异动提示 <span>${alerts.length}</span></div>`;
    html += `<div class="watchlist">${alerts.map((row) => watchCardHtml(row.entry, row.item)).join("")}</div>`;
  }
  if (others.length) {
    if (alerts.length) html += `<div class="watch-group-label">其余自选</div>`;
    html += `<div class="watchlist">${others.map((row) => watchCardHtml(row.entry, row.item)).join("")}</div>`;
  }
  container.innerHTML = html;
}

async function refreshWatchlist() {
  const button = $("#watchRefresh");
  const batchId = ++state.batchRequestId;
  if (!state.watchlist.length) {
    button.classList.remove("loading");
    renderWatchlist();
    $("#watchlistMeta").textContent = "保存在当前浏览器中，池内分位需至少 2 只";
    return;
  }
  const symbols = state.watchlist.slice(0, MAX_WATCHLIST).map((item) => item.symbol);
  renderWatchlist(state.batch?.items);
  button.classList.add("loading");
  $("#watchlistMeta").textContent = "正在计算三字段与异动…";
  const over = state.watchlist.length - MAX_WATCHLIST;
  const limitNote = over > 0 ? ` · 仅前 ${MAX_WATCHLIST} 只计算（超 ${over} 只）` : "";
  try {
    const data = await postJSON("/api/batch", { symbols });
    if (batchId !== state.batchRequestId) return;
    state.batch = data;
    renderWatchlist(data.items);
    const time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    const failed = data.failed?.length ? ` · ${data.failed.length} 个失败` : "";
    const alertText = data.alert_count ? ` · ${data.alert_count} 个异动` : "";
    $("#watchlistMeta").textContent = `${time} 更新 · 按池内分位排序${alertText}${failed}${limitNote}`;
  } catch (error) {
    if (batchId !== state.batchRequestId) return;
    state.batch = null;
    renderWatchlist();
    $("#watchlistMeta").textContent = `刷新失败：${error.message}`;
  } finally {
    if (batchId === state.batchRequestId) button.classList.remove("loading");
  }
}

/* --------------------------------- 市场环境 --------------------------------- */

function updateMarketPill(regime) {
  const pill = $("#marketPill");
  if (!pill || !regime?.available) return;
  pill.className = `market-pill ${regime.tone || ""}`;
  pill.querySelector("span").textContent = `市场状态 · ${regime.label} · 暴露≤${Math.round((regime.exposure ?? 0) * 100)}%`;
  pill.title = regime.guidance || "";
}

async function loadMarket() {
  try {
    const data = await getJSON("/api/market");
    updateMarketPill(data);
  } catch { $("#marketPill span").textContent = "市场状态暂不可用"; }
}

/* -------------------------------- ETF 配置 --------------------------------- */
/*
 * 数据来源只有一个：/api/allocation → reports/etf/latest.json。
 * 页面不做任何权重计算，也不把"信号日"当成"成交日"——两者在产物里是两个字段，
 * 这里分开显示，避免读者以为收盘价就能成交。
 */

const ALLOC_RULE_TEXT = {
  enter: "入场",
  hold: "持有",
  stop: "触发止损",
  mom_negative: "动量转负",
  below_ma_slow: "跌破 MA120",
  entry_filter: "未满足入场",
  warmup: "历史不足",
  no_price: "无行情",
};

function allocYesNo(flag) {
  if (flag === true) return '<span class="alloc-yes">是</span>';
  if (flag === false) return '<span class="alloc-no">否</span>';
  return "—";
}

function allocBar(weight, cash = false) {
  const width = Math.max(2, Math.min(100, Math.round(Number(weight || 0) * 100)));
  return `<span class="alloc-bar${cash ? " cash" : ""}" style="width:${width}%"></span>`;
}

function renderAllocation(data) {
  const body = $("#allocationBody");
  const meta = $("#allocMeta");
  if (!data || !data.available) {
    meta.textContent = data?.reason || "配置数据不可用";
    body.innerHTML = `<div class="alloc-empty">${escapeHtml(data?.hint || "运行 python -m etf_allocation 生成 reports/etf/latest.json")}</div>`;
    return;
  }
  const rules = data.rules || {};
  const cash = data.cash || {};
  const backtest = data.backtest || {};
  const bt = backtest.metrics || {};
  const paper = data.paper || {};
  const fill = data.fill_status === "effective"
    ? `已生效 ${escapeHtml(data.fill_date || "")}`
    : "待下一交易日开盘（T+1）";

  meta.innerHTML = [
    `信号日 ${escapeHtml(data.signal_date || "—")}`,
    `执行 ${fill}`,
    `规则 ${escapeHtml(data.rule_version || "—")}`,
    `数据 ${escapeHtml(data.data_version || "—")}`,
  ].join(" · ");

  const pills = [
    ["总暴露", `${formatNumber((data.exposure || 0) * 100, "%", 1)}`, `${(data.holdings || []).length} 条腿入选`],
    ["现金腿", `${formatNumber((cash.weight || 0) * 100, "%", 1)}`, `${cash.symbol || ""} 年化 ${formatNumber((cash.yield_60d_annual || 0) * 100, "%", 2)}`],
    ["权重约束", data.constraints?.ok ? "通过" : "未通过", `和 ${formatNumber(data.weights_sum, "", 3)} · 单腿上限 ${Math.round((rules.max_weight || 0.3) * 100)}%`],
    ["纸面跟踪", data.observation?.status === "ready" ? `${paper.n_periods} 期` : "观察期", paper.available ? `净 ${signed(paper.cumulative_net_pct, "%")} · 成本 ${formatNumber(paper.avg_cost_bps, "bp", 1)}` : (data.observation?.note || paper.note || "")],
  ].map(([label, value, hint]) => `<div class="alloc-pill"><small>${label}</small><strong>${value}</strong><em>${escapeHtml(hint || "")}</em></div>`).join("");

  const rows = (data.holdings || []).map((leg) => {
    const reason = leg.reason || {};
    return `<tr>
      <td><span class="alloc-leg-name">${escapeHtml(leg.name)}</span><span class="alloc-leg-meta">${escapeHtml(leg.symbol)} · ${escapeHtml(leg.class || "")}</span></td>
      <td><span class="alloc-weight">${formatNumber(leg.weight * 100, "%", 1)}</span>${allocBar(leg.weight)}</td>
      <td class="${changeClass((reason.momentum_12_1 || 0) * 100)}">${signed((reason.momentum_12_1 || 0) * 100, "%", 1)}</td>
      <td>${allocYesNo(reason.above_ma60)}</td>
      <td>${allocYesNo(reason.above_ma120)}</td>
      <td>${signed((reason.drawdown_from_high || 0) * 100, "%", 2)}</td>
      <td>${escapeHtml(ALLOC_RULE_TEXT[reason.rule] || reason.rule || "—")}</td>
    </tr>`;
  }).join("");

  const cashRow = `<tr>
    <td><span class="alloc-leg-name">${escapeHtml(cash.name || "现金腿")}</span><span class="alloc-leg-meta">${escapeHtml(cash.symbol || "")} · 未分配部分</span></td>
    <td><span class="alloc-weight">${formatNumber((cash.weight || 0) * 100, "%", 1)}</span>${allocBar(cash.weight, true)}</td>
    <td colspan="5" class="alloc-leg-meta">货币 ETF，按自身 60 日收益计息（年化 ${formatNumber((cash.yield_60d_annual || 0) * 100, "%", 2)}），不假设 0 收益</td>
  </tr>`;

  const changes = (data.changes || []).length
    ? `<p class="alloc-sub">对照 ${escapeHtml(data.changes_basis || "上次调仓")}：</p><div class="alloc-changes">${data.changes.map((c) =>
        `<span class="alloc-change">${escapeHtml(c.name)} <b>${c.action}</b> ${signed(c.delta * 100, "pp", 1)}（${formatNumber(c.before * 100, "%", 0)} → ${formatNumber(c.after * 100, "%", 0)}）</span>`).join("")}</div>`
    : `<p class="alloc-sub">${escapeHtml(data.changes_basis || "首期，无对照")}：权重无变化。</p>`;

  const rejected = (data.rejected || []).length
    ? `<p class="alloc-sub">未入选：${data.rejected.map((r) =>
        `${escapeHtml(r.name)}（${escapeHtml(ALLOC_RULE_TEXT[r.reason] || r.reason || "")}）`).join("、")}</p>`
    : "";

  const stale = Object.keys(data.stale || {}).length
    ? `<p class="alloc-sub">已剔除过期标的：${Object.entries(data.stale).map(([code, last]) => `${escapeHtml(code)}（末行 ${escapeHtml(last || "无数据")}）`).join("、")}</p>`
    : "";

  const btKv = [
    ["区间", `${escapeHtml(backtest.start || "—")} ~ ${escapeHtml(backtest.end || "—")}`],
    ["年化", signed(bt.cagr, "%", 2)],
    ["净 Sharpe", formatNumber(bt.sharpe, "", 2)],
    ["最大回撤", signed(bt.max_drawdown, "%", 2)],
    ["年换手", formatNumber(bt.annual_turnover, "%", 0)],
    ["成本拖累", formatNumber(bt.cost_bps, "bp", 0)],
    ["同池等权 Sharpe", formatNumber((backtest.benchmarks || {}).EW?.sharpe, "", 2)],
    ["最近调仓", data.last_rebalance ? `${escapeHtml(data.last_rebalance.signal_date)} → ${escapeHtml(data.last_rebalance.fill_date || "待成交")}` : "—"],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");

  const paperBlock = paper.available
    ? `<p class="alloc-sub">纸面 ${paper.n_periods} 期：净 ${signed(paper.cumulative_net_pct, "%")}，
        平均单边换手 ${formatNumber(paper.avg_one_way_turnover, "", 3)}，平均成本 ${formatNumber(paper.avg_cost_bps, "bp", 1)}
        （回测均值 ${formatNumber(paper.backtest_avg_cost_bps, "bp", 1)}，口径一致：${paper.cost_consistent ? "是" : "否"}）。</p>`
    : `<p class="alloc-sub">${escapeHtml(paper.note || "纸面跟踪尚未开始")}</p>`;

  const risk = [(data.observation?.status === "observing" ? data.observation.note : ""), ...(data.risk || [])]
    .filter(Boolean)
    .map((note) => `<li>${escapeHtml(note)}</li>`).join("");

  body.innerHTML = `
    <div class="alloc-pills">${pills}</div>
    <div class="alloc-grid">
      <div class="alloc-card">
        <h3>本期目标权重（按 ${escapeHtml(data.signal_date || "")} 收盘信号）</h3>
        <table class="alloc-table">
          <thead><tr><th>标的</th><th>权重</th><th>12–1 动量</th><th>MA60</th><th>MA120</th><th>回撤</th><th>判定</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="7" class="alloc-leg-meta">本期没有腿通过确认，全部在现金腿</td></tr>'}${cashRow}</tbody>
        </table>
        ${changes}
        ${rejected}
        ${stale}
      </div>
      <div class="alloc-card">
        <h3>回测与执行口径</h3>
        <div class="alloc-kv">${btKv}</div>
        ${paperBlock}
        <h3 style="margin-top:18px">风险提示</h3>
        <ul class="alloc-note-list">${risk}</ul>
      </div>
    </div>`;

  const disclaimer = $("#allocDisclaimer");
  if (disclaimer && data.disclaimer) disclaimer.textContent = data.disclaimer;
}

async function loadAllocation() {
  const body = $("#allocationBody");
  try {
    const data = await getJSON("/api/allocation");
    state.allocation = data;
    renderAllocation(data);
  } catch (error) {
    $("#allocMeta").textContent = "配置数据读取失败";
    body.innerHTML = `<div class="alloc-empty">${escapeHtml(error.message)}</div>`;
  }
}


/* ---------------------------------- 事件绑定 --------------------------------- */

$("#searchForm").addEventListener("submit", (event) => {
  event.preventDefault();
  analyzeSymbol($("#symbolInput").value);
});
document.querySelectorAll("[data-symbol]").forEach((button) =>
  button.addEventListener("click", () => analyzeSymbol(button.dataset.symbol)));

$("#watchButton").addEventListener("click", () => {
  if (!state.current) { status("请先分析一只股票或 ETF，再加入自选。", "error"); return; }
  const index = state.watchlist.findIndex((item) => item.symbol === state.current.symbol);
  if (index >= 0) {
    state.watchlist.splice(index, 1);
  } else {
    if (state.watchlist.length >= MAX_WATCHLIST) {
      status(`自选最多 ${MAX_WATCHLIST} 只，请先移除一只后再添加。`, "error");
      return;
    }
    state.watchlist.push({ symbol: state.current.symbol, name: state.current.name });
  }
  saveWatchlist();
});

$("#watchlistBody").addEventListener("click", (event) => {
  const open = event.target.closest("[data-open]");
  const remove = event.target.closest("[data-remove]");
  if (open) analyzeSymbol(open.dataset.open);
  if (remove) {
    state.watchlist = state.watchlist.filter((item) => item.symbol !== remove.dataset.remove);
    saveWatchlist();
  }
});

$("#watchRefresh").addEventListener("click", () => refreshWatchlist());

$("#chartTabs").addEventListener("click", (event) => {
  const button = event.target.closest("[data-pane]");
  if (!button || !state.chart) return;
  state.chart.pane = button.dataset.pane;
  syncChartTabs();
  drawChart();
});
$("#chartReset").addEventListener("click", () => resetChartRange());

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.chart) drawChart();
    if (state.rs) drawRSChart(state.rs.series);
  }, 120);
});

initSearchSuggest();
renderWatchlist();
loadMarket();
loadAllocation();
if (state.watchlist.length) refreshWatchlist();

window.addEventListener("popstate", () => {
  const symbol = new URLSearchParams(location.search).get("symbol");
  if (symbol) analyzeSymbol(symbol, { pushHistory: false, scroll: false });
});

const initialQuery = new URLSearchParams(location.search).get("symbol") || "600519";
analyzeSymbol(initialQuery, { pushHistory: false });
