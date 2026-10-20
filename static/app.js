const $ = (selector) => document.querySelector(selector);
const state = { current: null, chart: null, watchlist: JSON.parse(localStorage.getItem("guanlan-watchlist") || "[]") };

function formatNumber(value, suffix = "", digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function status(message, type = "loading") {
  const el = $("#statusMessage");
  if (!message) { el.classList.add("hidden"); return; }
  el.textContent = message;
  el.className = `status-message ${type}`;
}

async function getJSON(url) {
  const response = await fetch(url);
  let payload;
  try { payload = await response.json(); } catch { throw new Error("服务器返回了无法识别的数据"); }
  if (!response.ok || !payload.ok) throw new Error(payload.error || `请求失败 (${response.status})`);
  return payload.data;
}

function changeClass(value) { return Number(value) > 0 ? "up" : Number(value) < 0 ? "down" : ""; }

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
  $("#scoreValue").textContent = data.score;
  const ringColor = data.tone === "positive" ? "#176b4d" : data.tone === "negative" ? "#b34938" : "#aa7627";
  $("#scoreRing").style.cssText = `--score:${data.score};--ring:${ringColor}`;
  const tag = $("#verdictTag"); tag.textContent = data.view; tag.className = `verdict-tag ${data.tone}`;
  $("#verdictSummary").textContent = data.summary;
  $("#positionGuidance").textContent = data.position_guidance;
  $("#supportLevel").textContent = formatNumber(data.levels.support, "", data.price_precision);
  $("#resistanceLevel").textContent = formatNumber(data.levels.resistance, "", data.price_precision);
  $("#riskLevel").textContent = formatNumber(data.levels.risk_reference, "", data.price_precision);
  $("#reasons").innerHTML = data.reasons.map(reason => `<div class="reason ${reason.type}">${reason.text}</div>`).join("");
  $("#methodology").textContent = data.methodology;
  renderMetrics(data.metrics, data.asset_type);
  $("#dashboard").classList.remove("hidden");
  drawChart(data.chart);
  updateWatchButton();
}

async function analyzeSymbol(symbol) {
  const cleaned = symbol.trim().replace(/^(sh|sz|bj)/i, "");
  if (!/^\d{6}$/.test(cleaned)) { status("请输入 6 位股票或 ETF 代码，例如 600519、510300。", "error"); return; }
  status(`正在读取 ${cleaned} 的行情并计算指标…`);
  $("#symbolInput").value = cleaned;
  try {
    const data = await getJSON(`/api/analyze?symbol=${encodeURIComponent(cleaned)}`);
    renderAnalysis(data);
    status("");
    history.replaceState(null, "", `?symbol=${cleaned}`);
    $("#dashboard").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) { status(error.message, "error"); }
}

function formatVolume(value) {
  if (value === null || value === undefined) return "—";
  if (value >= 1e8) return `${(value / 1e8).toFixed(2)}亿`;
  if (value >= 1e4) return `${(value / 1e4).toFixed(2)}万`;
  return Math.round(value).toLocaleString("zh-CN");
}

function drawChart(points) {
  const canvas = $("#priceChart"), shell = canvas.parentElement, tooltip = $("#chartTooltip");
  const ratio = window.devicePixelRatio || 1;
  const width = shell.clientWidth, height = shell.clientHeight;
  canvas.width = width * ratio; canvas.height = height * ratio;
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const pad = { left: 54, right: 15, top: 16, bottom: 25 };
  const plotWidth = width - pad.left - pad.right;
  const volumeHeight = Math.max(82, height * 0.24);
  const paneGap = 26;
  const priceBottom = height - pad.bottom - volumeHeight - paneGap;
  const volumeTop = priceBottom + paneGap;
  const volumeBottom = height - pad.bottom;
  const precision = state.current?.price_precision ?? 2;
  const allPrices = points.flatMap(p => [p.close, p.ma20, p.ma60]).filter(v => v != null);
  const rawMin = Math.min(...allPrices), rawMax = Math.max(...allPrices);
  const pricePadding = Math.max((rawMax - rawMin) * 0.06, rawMax * 0.002);
  const minPrice = rawMin - pricePadding, maxPrice = rawMax + pricePadding;
  const priceRange = maxPrice - minPrice || 1;
  const maxVolume = Math.max(...points.flatMap(p => [p.volume, p.volume_ma5 || 0, p.volume_ma10 || 0])) * 1.08 || 1;
  const x = i => pad.left + i * plotWidth / Math.max(1, points.length - 1);
  const priceY = value => pad.top + (maxPrice - value) * (priceBottom - pad.top) / priceRange;
  const volumeY = value => volumeBottom - value / maxVolume * (volumeBottom - volumeTop);
  const barWidth = Math.max(1, Math.min(5, plotWidth / points.length * 0.72));

  function drawLine(key, color, lineWidth, yMap) {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = "round"; let started = false;
    points.forEach((point, index) => {
      const value = point[key]; if (value == null) return;
      const xx = x(index), yy = yMap(value);
      started ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); started = true;
    });
    ctx.stroke();
  }

  function render(activeIndex = null) {
    ctx.clearRect(0, 0, width, height);
    ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "left"; ctx.lineWidth = 1;

    for (let i = 0; i < 5; i++) {
      const yy = pad.top + i * (priceBottom - pad.top) / 4;
      ctx.strokeStyle = "#e2e5e0"; ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(width - pad.right, yy); ctx.stroke();
      ctx.fillStyle = "#88938d"; ctx.fillText((maxPrice - i * priceRange / 4).toFixed(precision), 2, yy + 3);
    }

    ctx.strokeStyle = "#cfd5ce"; ctx.beginPath(); ctx.moveTo(pad.left, volumeTop - 13); ctx.lineTo(width - pad.right, volumeTop - 13); ctx.stroke();
    ctx.fillStyle = "#78857e"; ctx.fillText("成交量（手）", pad.left, volumeTop - 17);
    ctx.fillStyle = "#88938d"; ctx.fillText(formatVolume(maxVolume), 2, volumeTop + 4);
    ctx.fillText(formatVolume(maxVolume / 2), 2, volumeTop + (volumeBottom - volumeTop) / 2 + 3);

    points.forEach((point, index) => {
      const rising = Number(point.pct_change) >= 0;
      ctx.fillStyle = rising ? "rgba(190,73,58,.72)" : "rgba(23,107,77,.72)";
      const top = volumeY(point.volume);
      ctx.fillRect(x(index) - barWidth / 2, top, barWidth, Math.max(1, volumeBottom - top));
    });

    drawLine("volume_ma5", "#d19a35", 1.15, volumeY);
    drawLine("volume_ma10", "#875b98", 1.15, volumeY);
    drawLine("close", "#17231d", 2.15, priceY);
    drawLine("ma20", "#aa7627", 1.35, priceY);
    drawLine("ma60", "#426d89", 1.35, priceY);

    const labelIndexes = [0, Math.floor(points.length / 2), points.length - 1];
    ctx.fillStyle = "#88938d"; ctx.textAlign = "center";
    labelIndexes.forEach(index => ctx.fillText(points[index].date.slice(5), x(index), height - 7));

    if (activeIndex !== null) {
      const xx = x(activeIndex);
      ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(70,84,76,.46)";
      ctx.beginPath(); ctx.moveTo(xx, pad.top); ctx.lineTo(xx, volumeBottom); ctx.stroke(); ctx.restore();
      ctx.fillStyle = "#17231d"; ctx.beginPath(); ctx.arc(xx, priceY(points[activeIndex].close), 3.2, 0, Math.PI * 2); ctx.fill();
    }
  }

  render();
  state.chart = { points, x, priceY, volumeY, width, height };
  canvas.onmousemove = event => {
    const rect = canvas.getBoundingClientRect(), mx = event.clientX - rect.left;
    const index = Math.max(0, Math.min(points.length - 1, Math.round((mx - pad.left) / plotWidth * (points.length - 1))));
    const point = points[index], change = Number(point.pct_change);
    render(index);
    tooltip.innerHTML = `${point.date}<br>开盘&nbsp; ${point.open.toFixed(precision)}<br>收盘&nbsp; ${point.close.toFixed(precision)}<br>涨跌&nbsp; ${change > 0 ? "+" : ""}${change.toFixed(2)}%<br>成交量 ${formatVolume(point.volume)} 手<br>VOL5&nbsp; ${formatVolume(point.volume_ma5)} 手<br>VOL10 ${formatVolume(point.volume_ma10)} 手`;
    tooltip.style.left = `${Math.max(82, Math.min(width - 82, x(index)))}px`;
    tooltip.style.top = `${Math.max(105, priceY(point.close))}px`;
    tooltip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tooltip.classList.add("hidden"); render(); };
}

async function loadMarket() {
  try {
    const data = await getJSON("/api/market"); const pill = $("#marketPill");
    pill.className = `market-pill ${data.tone}`; pill.querySelector("span").textContent = `市场环境 · ${data.regime} ${data.score}分`;
    pill.title = data.guidance;
  } catch { $("#marketPill span").textContent = "市场环境暂不可用"; }
}

function saveWatchlist() { localStorage.setItem("guanlan-watchlist", JSON.stringify(state.watchlist)); renderWatchlist(); updateWatchButton(); }
function updateWatchButton() {
  if (!state.current) return; const exists = state.watchlist.some(item => item.symbol === state.current.symbol);
  const button = $("#watchButton"); button.textContent = exists ? "✓ 已在自选" : "＋ 加入自选"; button.classList.toggle("active", exists);
}
function renderWatchlist() {
  const el = $("#watchlist");
  if (!state.watchlist.length) { el.innerHTML = `<div class="empty-watch">尚未添加自选股。在完成一次分析后，点击右上角“加入自选”。</div>`; return; }
  el.innerHTML = state.watchlist.map(item => `<div class="watch-item"><button data-open="${item.symbol}"><strong>${item.name}</strong><small>${item.symbol}</small></button><button class="remove" data-remove="${item.symbol}" title="移除">×</button></div>`).join("");
}

$("#searchForm").addEventListener("submit", event => { event.preventDefault(); analyzeSymbol($("#symbolInput").value); });
document.querySelectorAll("[data-symbol]").forEach(button => button.addEventListener("click", () => analyzeSymbol(button.dataset.symbol)));
$("#watchButton").addEventListener("click", () => {
  if (!state.current) { status("请先分析一只股票或 ETF，再加入自选。", "error"); return; }
  const index = state.watchlist.findIndex(item => item.symbol === state.current.symbol);
  if (index >= 0) state.watchlist.splice(index,1); else state.watchlist.push({ symbol: state.current.symbol, name: state.current.name });
  saveWatchlist();
});
$("#watchlist").addEventListener("click", event => {
  const open = event.target.closest("[data-open]"), remove = event.target.closest("[data-remove]");
  if (open) analyzeSymbol(open.dataset.open);
  if (remove) { state.watchlist = state.watchlist.filter(item => item.symbol !== remove.dataset.remove); saveWatchlist(); }
});
window.addEventListener("resize", () => { if (state.current) drawChart(state.current.chart); });

renderWatchlist(); loadMarket();
const initialSymbol = new URLSearchParams(location.search).get("symbol") || "600519";
analyzeSymbol(initialSymbol);
