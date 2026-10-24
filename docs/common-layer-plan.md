# 公共层收敛方案（reporting / stats / costs）

> 状态：**方案，未执行**。本文件是后续实施清单；每一步落地后回填「实测」列。
> 基线：`v1.5-three-fields`（`47121a5`）。核查时间 2026-10-24。

---

## 0. 先纠正四个前提

核查后发现待办清单里有四项与代码实际不符。方向不变，但**处置方式要改**，否则会做出第三个重复实现。

| 原判断 | 实测 | 结论 |
| --- | --- | --- |
| `render_html` 10 份 | 确实 10 份，但**不是同一个函数**：5 份输出完整页面（`<!DOCTYPE>`：etf_allocation / fundamental_study / regime_study / score_study / wfv2_study），5 份输出可粘贴片段（factor_study / protocol / validate_fields / walkforward / xsec）。内容层又分「Markdown→HTML 通用转换器」（factor_study、validate_fields）与「report dict → 定制表格」（其余 8） | 要拆成 **3 个 API**（页面壳 / 片段壳 / md 转换），不是 1 个函数 |
| `crossasset.py` 与 `etf_allocation.py` 是两套引擎 | `crossasset.py` 已降级为 **234 行兼容壳**，21 处全部转调 `ea.*` 并抛 `DeprecationWarning`；`validate_fields.py` 已改验 `etf_allocation`；生产层无任何 import | 「两套引擎」在 v1.5 已消除。**不建 `allocation_core.py`**，应做的是「删壳」 |
| 成本常数 3 份 | 实际 **4 处定义 + 1 处硬编码公式 + 5 处散文**：`xsec.py:59` 由 bps 分量派生、`score_study.py:82` 字面量、`etf_allocation.py:80` 字面量、`execution.py:134` **默认参数里还有一份**字面量；`factor_study.py:534` 直接写 `(12.6 + 7.6)`；regime/wfv2/score 的文案里写死「7.6/12.6bp」 | `costs.py` 要覆盖 6 类位置，其中 `execution.py` 的默认值是**最容易漏掉**的一份 |
| `ic_stats` 有多份 | 源码里**一份都没有**，只在 `.cache/diag/diagnose.py`（临时脚本）里有；生产与研究代码的 IC 区间一律走 `evalkit.ic_ci` | **不新建 `stats.ic_stats`**。再加一份等于凭空造出第二套区间口径 |

另有两条被低估的问题，比"重复"更严重：

- **`_fmt` 有三种互不兼容的签名，261 处调用**（详见风险 R1）。`walkforward` 是 `(value, suffix, digits)`，`protocol` 是 `(value, digits, suffix)`，`xsec` 是 `(value, suffix="%", digits)`（默认后缀是百分号！），`validate_fields` 干脆没有 `signed`。位置参数调用遍地，统一签名时**不会报错，只会静默改变输出**。
- **`daily_ic` 的最小横截面门槛有三档**：`scoring.py` 是 10、`protocol/factor_study/wfv2_study/xsec` 是 20、`research/fundamentals.py` 是 30。这已经不是重复实现，而是**同一指标在不同报告里不可比**。统一门槛会改变已发布报告的数字。

规模：上述重复项共 **57 个定义、4091 行**，占全部 Python 代码（25069 行）的 16%。

---

## 1. 核查明细

| 名称 | 份数 | 是否同语义 | 处置 |
| --- | ---: | --- | --- |
| `render_html` | 10 | 否（5 页面 / 5 片段 / 2 种输入类型） | 拆成 `html_page` / `html_fragment` / `markdown_to_html` |
| `render_markdown` | 12（根目录 10 + research/ 2） | 否（各自拼装领域表格） | **只收敛输入输出契约，不合并正文**：`md_table()` + 统一的「标题/小节/结论」骨架，正文仍留在 study |
| `_svg_nav` | 6 | 否（两族：4 份 polyline、2 份带网格与刻度；输入 3 种形态） | `svg_line(curves, *, grid=False, ...)`，`Curve` 归一化输入 |
| `_svg_*` 其他 | 4 | 否（calibration / quintiles / folds / periods / exposure 各不相同） | 不进公共层，留在各自 study |
| `_fmt` | 6 | 否（三种签名 + 是否带 `signed`） | `reporting.fmt(value, *, digits, suffix, signed)` |
| `_num` | 5 | 是（仅默认 `digits` 不同：2 / 3） | `reporting.num` |
| `_signed` | 4 | 是（默认 2 / 4） | `reporting.signed` |
| `_inline` | 2 | 是（md 行内转义） | `reporting` 内部私有 |
| `_sma` | 2 | **是，逐字节相同** | `stats.sma`，零风险 |
| `_rolling_max` | 2 | **否**：`factors` 前 `window-1` 个返回 `None`；`fields` 返回**扩张窗口最大值** | `stats.rolling_max(values, window, *, partial=False)` |
| `_solve` | 2 | 近似同（主元阈值 `1e-12` vs `1e-14`，后者多一道除零保护） | `stats.solve`，取严的那个 |
| `daily_ic` | 5 | **否**（门槛 10 / 20 / 30，输入形态 4 种） | `stats.daily_ic(pairs_by_date, *, min_xsec=20)` |
| `nav_metrics` | 2（其中 1 份是壳） | 否（键名 `cagr`/`vol` vs `annual_return`/`volatility`，年数 `(n-1)/252` vs `len/252`，Sharpe 舍入 3 vs 2） | `stats.nav_metrics`，canonical 键取 `cagr`/`vol`（前端已依赖） |
| 成本常数 | 4 + 1 公式 + 5 处文案 | 否（股票 7.6/12.6 vs ETF 7.5/7.5，另有一处重复） | `costs.py` 单一来源 |

---

## 2. 模块边界

### 2.1 `costs.py`（唯一事实来源，约 60 行）

```python
"""交易成本：所有研究 / 回测 / 验证 / 报告文案的唯一来源。"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Cost:
    buy: float          # 单边买入，小数
    sell: float         # 单边卖出，小数
    note: str = ""
    @property
    def buy_bps(self) -> float: ...
    @property
    def sell_bps(self) -> float: ...
    @property
    def round_trip_bps(self) -> float: ...

# A 股个股：佣金 2.5 + 过户 0.1 + 滑点 5 = 7.6bp；卖出加印花 5bp = 12.6bp
STOCK_COST = Cost(buy=0.00076, sell=0.00126, note="佣金2.5+过户0.1+滑点5；卖出加印花5")

# ETF：免印花税，佣金 2.5 + 滑点 5 = 7.5bp 双边
ETF_COST = Cost(buy=0.00075, sell=0.00075, note="佣金2.5+滑点5，免印花")

_BY_ASSET = {"stock": STOCK_COST, "etf": ETF_COST, "index": ETF_COST, "fund": ETF_COST}

def for_asset(asset_type: str) -> Cost: ...
def label(cost: Cost) -> str:      # "7.6/12.6bp"，供报告文案 f-string 用
```

约束：
- 只依赖 stdlib（`dataclasses`），可被 `execution.py` 直接 import，不破坏 `test_isolation`。
- **旧名保留为属性而非副本**：`xsec.BUY_COST = STOCK_COST.buy`（同一 float 对象），不复制数值，避免再次分叉。
- 报告文案禁止再写字面量：`regime_study` / `wfv2_study` / `score_study` 的「7.6/12.6bp」改为 `costs.label(STOCK_COST)`。
- `factor_study.py:534` 的 `(12.6 + 7.6) / 100` 改为 `STOCK_COST.round_trip_bps / 100`（注意原式除以 100 是把 bp 换成 %，置换时保持量纲）。

### 2.2 `stats.py`（决定性数值，约 320 行）

| API | 说明 |
| --- | --- |
| `sma(values, period)` | 与现有两份逐字节等价 |
| `rolling_max(values, window, *, partial=False)` | `partial=False` = 前 `window-1` 为 `None`（factors 语义）；`partial=True` = 扩张窗口（fields 语义） |
| `solve(matrix, vector)` | 高斯-约当 + 部分主元，阈值 `1e-14`，含除零保护 |
| `daily_ic(pairs_by_date, *, min_xsec=20, min_pairs=None)` | 唯一实现；`pairs_by_date: dict[date, list[(score, label)]]` |
| `ic_stats(series, *, block, iters)` | **薄适配器**，内部直接调 `evalkit.ic_ci`，只做「序列 → (mean, ci_low, ci_high, p, n)」的键名整理，不重复实现区间 |
| `nav_metrics(nav, *, calendar_len=None)` | 唯一入口，返回 canonical 键 |

**边界规则（写进模块 docstring，且由测试断言）**：
- `stats.py` **不做**随机化、p 值、多重检验、OLS 拟合——那些属于 `evalkit.py`（已有 `spearman` / `ic_ci` / `bh_fdr` / `deflated_sharpe` / `ols_fit`）。`daily_ic` 内部必须调 `evalkit.spearman`，不允许出现第二份秩相关实现。
- `stats.py` 不做 I/O、不读缓存、不碰 `research`。
- `nav_metrics` canonical 键：`total_return / cagr / vol / sharpe / max_drawdown / calmar / years`，全部按**百分数**返回（与现有 `etf_allocation` 一致，避免一半小数一半百分数）；`xsec.metrics` 变成 12 行适配器，额外输出 `annual_return` / `volatility` 别名并标注 deprecated。

### 2.3 `reporting.py`（渲染，约 480 行）

```python
# --- 数值 ---
fmt(value, *, digits=3, suffix="", signed=True) -> str
pct(value, *, digits=2, signed=True) -> str        # 自动带 %（value 为百分数）
num(value, *, digits=2) -> str
signed(value, *, digits=2) -> str
DASH = "—"

# --- 表格 ---
md_table(headers, rows, *, aligns=None) -> str      # aligns: "lrrr"
html_table(headers, rows, *, aligns=None, classes=None) -> str

# --- 壳 ---
html_page(title, body, *, css=None, subtitle=None) -> str      # 完整页面（含 DOCTYPE）
html_fragment(body, *, css=None, scope="rpt") -> str           # 可粘贴片段（作用域样式）
markdown_to_html(markdown) -> str                              # 通用 md → html body

# --- 图 ---
@dataclass(frozen=True)
class Curve:
    name: str
    values: Sequence[float]
    dates: Sequence[str] | None = None
    color: str | None = None
    dash: str | None = None

svg_line(curves, *, width=900, height=320, grid=False, footer=None, aria="净值曲线") -> str
svg_from_report(extractor, report, **kw) -> str     # 兼容 _svg_nav(report) 的两种旧形态
```

`as_curves()` 负责归一化三种输入：`{name: [{"date","nav"}]}` / `{name: [float]}` / `report dict`。

约束：
- `reporting.py` **不得 import 任何 study、不得 import `stats` 之外的业务模块**，也不得读文件（写文件由调用方做）。
- 不引入模板引擎（jinja 之类）：纯 stdlib + f-string，与项目"零第三方依赖"约定一致。
- **不合并报告正文**。每个 study 的表格是领域知识，硬合并会造出一个巨型条件分支模块——这是本次最容易犯的错。

### 2.4 依赖方向

```
study 脚本 (walkforward/xsec/protocol/validate_fields/...)
        │  只做：数据装配 → 参数 → 结论
        ├──> reporting.py ──> (stdlib)
        ├──> stats.py ──────> evalkit.py (spearman / ic_ci)
        └──> costs.py ──────> (stdlib)

execution.py ──> costs.py            # 默认值改为 import，不再写字面量
xsec.py / protocol.py / regime_study.py ──> costs.STOCK_COST
etf_allocation.py ──> costs.ETF_COST
```

三个新模块都要加入 `tests/test_isolation.py` 的 `PRODUCTION` 元组（它们是纯计算，必须保持零第三方依赖）。

---

## 3. ETF 引擎：不抽 core，直接删壳

| 方案 | 做法 | 代价 | 结论 |
| --- | --- | --- | --- |
| A 抽 `allocation_core.py` | 把选腿/加权/回测内核抽出，`etf_allocation` 与 `crossasset` 都调 core | 新增第 3 个模块名；`etf_allocation` 有 1764 行、`Config`/`UNIVERSE`/`simulate` 互相咬合，抽核等于大手术 | **否决**：问题已不存在，抽核只会制造第三个入口 |
| B 保留 `etf_allocation`，删 `crossasset.py` | 删 234 行壳 + `tests/test_crossasset.py`（或改为「引擎唯一性」断言） | 失去旧签名兼容；但生产层已无 import，只有测试在用 | **采纳** |
| C 维持现状（壳留着） | 不动 | 每读一次代码就要判断"哪个是真的" | 仅作为 B 的回滚缓冲（保留一个 release） |

执行细节：
1. 先 `grep -rn "import crossasset\|crossasset\." --include=*.py` 确认零生产引用。当前**只有测试**在 import：`tests/test_crossasset.py:7`（整套旧签名测试）与 `tests/test_allocation_product.py:292`（验证废弃告警）；`test_isolation.PRODUCTION` 里还有它的名字，删除后要同步移除。
2. `tests/test_crossasset.py` 里真正有价值的断言（旧键映射 `no_control→no_stop`、`overlay→no_voltarget`）迁到 `tests/test_etf_allocation.py`，其余删除；`test_allocation_product.py` 那条"crossasset 应抛 DeprecationWarning"改为"引擎唯一性"断言（`import etf_allocation` 之外不存在第二个配置入口）。
3. `reports/index.json` 里 producer 若写 `crossasset.py`，同步改成 `python -m etf_allocation`。
4. `validate_fields.py` 导出的 `reports/fields/crossasset.json` **文件名保留**（下游可能引用），但内容里的 `engine` 已是 `etf_allocation`——删壳不影响它。

---

## 4. study 脚本收敛后的形态

每个 study 只保留三段，并用注释显式分节：

```python
# --- 1. 数据装配 ---------------------------------------------------------
bundle = load_inputs(pit=args.pit)

# --- 2. 参数 -------------------------------------------------------------
params = dict(horizon=HORIZON, folds=K_FOLDS, cost=costs.STOCK_COST,
              min_xsec=20, iters=args.iters)

# --- 3. 结论 -------------------------------------------------------------
report = {"meta": ..., "params": params, "verdict": ..., "tables": [...]}
write_report(outdir, report, render=my_sections)   # 渲染交给 reporting
```

渲染部分收缩为「本报告特有的 section 函数」，签名统一 `section(report) -> str`，内部只用 `reporting.*`：

```python
def sections(report):
    return [
        ("1. 抽样与覆盖", md_table(...)),
        ("2. IC 与区间", html_table(...) if as_html else md_table(...)),
        ("3. 净值", svg_line(as_curves(report["nav"]), grid=True)),
    ]
```

预计各 study 减少的行数（按第 1 节统计）：protocol −354/−250、walkforward −269/−162、xsec −268/−181、turnover −207、validate_fields −166、pool_validate −157、fundamental −155/−146、factor −140、etf −123、regime −116。

---

## 5. 迁移步骤（每步一个 commit，可独立回滚）

| 步 | 内容 | 门禁（不过不进下一步） |
| --- | --- | --- |
| **P0 冻结基线** | 用现有代码把 10 份报告重跑一遍（固定数据版本 + 固定 `iters`），把 `report.json` / `report.md` / `report.html` 存进 `tests/golden/`（**不入库大文件，只入库 JSON 与 MD 的 hash**）；同时记 `svg_hash` | 重跑两次结果完全一致（确认无隐藏随机性） |
| **P1 costs.py** | 建 3 个成本对象；改 `xsec` / `score_study` / `etf_allocation` / `execution` 默认值 / `factor_study` 公式 / 5 处文案 | 成本单测全绿；`grep -rn "BUY_COST = "` 只剩 `costs.py`；重跑 xsec/etf 的 `report.json` 与新基线**完全相等** |
| **P2 stats.py** | 迁 `sma` / `rolling_max` / `solve` / `daily_ic` / `nav_metrics`；`ic_stats` 只做 `evalkit.ic_ci` 适配 | 单测 + `report.json` 相等（`daily_ic` 门槛未统一时，各 study 显式传自己的 `min_xsec`，保证数值不变） |
| **P3 reporting 数值层** | 迁 `fmt` / `num` / `signed` / `pct` / `_inline`；**跑「双实现比对」脚本**（见 R1） | 261 处调用逐点字符串相等；`report.md` 归一化 diff = 0 |
| **P4 reporting 表格与壳** | 迁 `md_table` / `html_table` / `html_page` / `html_fragment` / `markdown_to_html` | `report.md` diff = 0（表格分隔线与对齐不变）；HTML 结构 diff = 0 |
| **P5 SVG 与删壳** | `svg_line` 统一 6 份；删 `crossasset.py`；更新 `reports/index.json`、CHANGELOG、`APP_VERSION` | SVG 基线一次性接受并冻结（见 6.4）；5 张整页截图人工过目；标签 `v1.6-common-layer` |

**P2 的口径统一单独拆一步（P2.5，可延后）**：把 `daily_ic` 的 `min_xsec` 统一为 20。这一步会改变 `scoring`（10→20）与 `research/fundamentals`（30→20）的 IC 数值，**必须重跑并更新结论**，不能混在重构里。建议 P2 先"传参不改值"，单独发一版并明确写进 CHANGELOG。

---

## 6. 验证方案

### 6.1 三层 golden（替代"重跑后 diff HTML"）

直接 diff HTML 不可行：本轮要统一 SVG 几何、CSS 变量、表格类名，HTML 必然变化。改成三层：

| 层 | 对象 | 判据 |
| --- | --- | --- |
| L1 | `report.json`（结构体） | **字段与数值完全相等**（浮点容差 1e-12）。这是"口径没变"的硬证据 |
| L2 | `report.md`（纯文本） | 正则归一化后 **0 差异**：抹掉 `\d{4}-\d{2}-\d{2}`、`耗时\s*[\d.]+s`、绝对路径、`generated_at`、随机种子回显 |
| L3 | `report.html` | **结构 diff**：去 `<style>`、去 `<svg>.*</svg>` 后按标签序列比对 = 0；另做 5 张整页截图，像素差 ≤0.5%（超出需人工确认并记录） |

L1/L2 是硬门禁，L3 是"需人工确认"的软门禁。

### 6.2 关键：不要"重跑回测再 diff"

重跑要 20–40 分钟/份且依赖网络数据。改为**冻结输入的渲染回归**：P0 把每个 study 的 `report` dict 存成 `tests/golden/<study>.json`，回归时只调 `sections(report)` 渲染，比对 L1/L2/L3。**只有 P1/P2（动计算）时才需要真实重跑**（一份一次，共约 3 小时，可并行）。

### 6.3 grep 门禁（写成测试，不靠人记）

```python
def test_no_duplicate_helpers():
    assert sum(1 for f in ROOT.glob("*.py") if "def render_html(" in f.read_text()) == 0  # 只在 reporting.py
    assert not any("def _svg_nav(" in f.read_text() for f in ROOT.glob("*.py"))
    assert not any("BUY_COST = 0." in f.read_text() for f in ROOT.glob("*.py"))
    # 反向：允许的唯一定义
    assert "def render_html(" in (ROOT / "reporting.py").read_text()
    assert "STOCK_COST = Cost(" in (ROOT / "costs.py").read_text()
```

补充：
- `grep -rn "def render_markdown" *.py` 允许保留在 study（正文是领域知识），但每个 study 内**不得**再出现 `md_table` 手写循环。
- `grep -rn "12\.6\|7\.6" *.py` 只允许出现在 `costs.py` 与测试；报告文案改为 `costs.label(...)`。

### 6.4 SVG 为什么不能字节一致（必须写进验收说明）

6 份 `_svg_nav` 的几何参数各不相同，统一后必然变化：

| 调用点 | pad | stroke-width | legend 间距 | footer |
| --- | ---: | ---: | ---: | --- |
| wfv2 | 42 | 1.8 | 130 | 净值 low–high（各折首尾相接） |
| regime | 42 | 1.7 | 150 | 净值 low–high |
| score | 44 | 1.8 | 130 | 净值区间 low–high |
| etf | 44 | 1.7 | 140 | 净值 low–high |
| protocol / xsec | 无 pad，带 y 网格与月份标签 | — | — | 无 |

处置：**一次性接受基线**。P5 生成后截图确认可读（线宽、图例不重叠、网格与折线不打架），然后把 `svg_hash` 写进 golden 冻结；此后任何 SVG 变化都必须显式更新基线并在 commit message 说明。不参与逐次 diff。

### 6.5 成本单测（覆盖用户要求的三条）

```python
def test_stock_cost_matches_broker_quote():
    assert costs.STOCK_COST.buy_bps == 7.6
    assert costs.STOCK_COST.sell_bps == 12.6          # 卖出多 5bp 印花税
    assert costs.STOCK_COST.sell_bps - costs.STOCK_COST.buy_bps == 5.0

def test_etf_cost_is_symmetric():
    assert costs.ETF_COST.buy_bps == costs.ETF_COST.sell_bps == 7.5

def test_single_source_across_modules():
    """断言的是"同一来源"，不是"数值恰好相等"。"""
    import xsec, etf_allocation, score_study, execution
    # 现状：这四处各自持有常量。改造后它们必须是 costs 的属性（同一 float 对象），
    # 而不是各自复制一份数值——复制会在下次调整费率时再次分叉。
    assert xsec.SELL_COST is costs.STOCK_COST.sell
    assert score_study.BUY_COST is costs.STOCK_COST.buy
    assert etf_allocation.BUY_COST is costs.ETF_COST.buy
    assert execution.DEFAULT_COST is costs.STOCK_COST      # P1 新增：simulate_t1 默认值
    # protocol / regime_study 不持有常量，改为在调用点取 costs.STOCK_COST
    # 反向断言：不允许再出现独立数值
    for mod in (xsec, score_study, etf_allocation):
        src = inspect.getsource(mod)
        assert "0.00076" not in src and "0.00126" not in src and "0.00075" not in src
```

最后一条是防复发的关键：**光断言相等不够，还要断言模块源码里不再出现字面量**，否则下次有人改回字面量测试照样绿。

---

## 7. 风险点（按严重度）

| # | 风险 | 触发条件 | 缓解 |
| --- | --- | --- | --- |
| **R1** | `_fmt` 统一签名，261 处调用静默改输出 | 位置参数 `_fmt(x, 2, '%')` 在两种签名下含义不同（protocol: digits=2；walkforward: suffix=2） | 新函数全**关键字参数**；写一次性 codemod + **双实现比对脚本**：对每个调用点用新旧两套求值，字符串不等则报错列出文件行号；0 差异才删旧函数。**禁止**用 `**kwargs` 兼容层蒙混（那只是把分叉藏起来） |
| **R2** | `rolling_max` 语义不同 | 直接合并后 `fields` 的「距一年高点」在首个 `window` 内回撤变小 | `partial=` 显式参数；默认 `False`（factors 语义）；`fields` 调用点显式 `partial=True` 并加注释说明"不足一年时用扩张窗口" |
| **R3** | `_solve` 主元阈值不同（1e-12 vs 1e-14） | 共线特征下系数跳变 | 统一取 `1e-14`；对 `xsec.fit_rank_model` 与 `scoring` 现有数据集跑系数回归，容差 1e-10；超差则记录为"预期变化"并写入 CHANGELOG |
| **R4** | `daily_ic` 门槛 10/20/30 统一 | 混进重构一起做 → 报告数字变了却没人知道 | **单列一步**；P2 只"搬实现、显式传参"保数值不变；门槛统一另发一版，重跑 `scoring` 与 `altdata` 并更新结论 |
| **R5** | `nav_metrics` 键名变化 | `cagr`→`annual_return` 会让前端 `static/app.js:1141` 显示空 | canonical 保持 `cagr` / `vol`；`xsec.metrics` 适配器额外输出旧名别名，标注 deprecated |
| **R6** | 年数口径 `(n-1)/252` vs `calendar_len/252` | 年化差异在小数点后两位 | 统一 `(n-1)/252`；`xsec` 显式传 `calendar_len`（nav 可能短于日历）；L1 golden 会抓到差异 |
| **R7** | Sharpe 舍入 3 vs 2 位 | 前端 `formatNumber(...,2)` 不受影响，但报告文本会变 | 统一 3 位；L2 diff 里若出现，逐条确认后接受 |
| **R8** | `reporting.py` 被 `app.py` 误 import | 4000 行渲染代码进请求路径，页面变慢 | `test_isolation` 加断言：`app.py` 的 import 集合不含 `reporting`；`/api/allocation` 仍只读 JSON |
| **R9** | 视觉回归无自动化 | P5 之后没人看得出图变丑 | 5 张整页截图 + SVG 基线 hash；首轮人工过目并记录在 `docs/` |
| **R10** | `reports/` 只入库 `index.json` | 重跑了报告但 `index.json` 的 `verdict`/`generated_at` 没更新 → 登记表与实际产物脱节 | 重跑脚本末尾强制刷新 `index.json`；或加测试：`index.json` 中每条 `generated_at` 不早于对应目录最新文件 mtime |
| **R11** | 过度抽象 | 把 10 份报告正文合并成一个大模板 | 明确"只收敛 fmt/表格/壳/图"，正文留在 study；reporting.py 控制在 500 行内 |

---

## 8. 回滚方式

1. **代码回滚**：每步一个 commit，`git revert <sha>` 即回到上一步。因为新模块是**新增文件**，回滚只需撤调用点，`reporting.py` / `stats.py` / `costs.py` 可以留在树上（无人引用，零风险）。
2. **标签回滚**：P5 后打 `v1.6-common-layer`；出问题 `git checkout v1.5-three-fields -- <files>` 回到三字段基线。`tests/test_release_governance.py` 继续守着「`APP_VERSION` == tag」。
3. **数据回滚**：`tests/golden/*.json` 是冻结的输入与期望输出，回滚后直接跑 L1/L2 即可确认恢复正确，不需要重新下载数据。
4. **报告回滚**：`reports/index.json` 的 `producer` 字段是重跑命令，用旧 tag 的代码重跑即可复现旧报告（产物本身不入库）。
5. **最坏情况**：`git revert` 整个 P1–P5 区间（5 个 commit），代码回到 v1.5 行为；因 `reports/` 不入库，唯一外部可见影响是报告暂时回退到旧排版。

---

## 9. 工作量与建议顺序

| 步 | 估时 | 主要成本 |
| --- | ---: | --- |
| P0 基线 | 0.5d | 跑 10 份报告 + 写 golden harness |
| P1 costs | 0.5d | 6 类位置替换 + 单测 |
| P2 stats | 1d | 5 个函数、4 种输入形态归一化 |
| P3 fmt（**风险最高**） | 1d | 261 处调用 + 双实现比对脚本 |
| P4 表格与壳 | 1.5d | 10 份 HTML 拆分页/片段 |
| P5 SVG + 删壳 | 1.5d | 两族几何合并 + 视觉确认 |
| P2.5 口径统一（可选延后） | 1d | 重跑 scoring / altdata 并改结论 |

合计约 **6 个工作日**（不含 P2.5 的重跑）。

建议：**P1 → P2 → P3 先做**。这三步是纯收益（成本单一来源、数值函数单一实现、格式化不再静默出错），且验证最强（L1 精确相等）。P4/P5 属于"观感与可维护性"，可以晚做，也应单独排期——因为它们必然带来 HTML/SVG 的视觉差异，需要人工过目，不适合和计算层改动混在一个窗口。
