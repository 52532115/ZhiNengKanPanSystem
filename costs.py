"""交易成本：研究 / 回测 / 验证 / 报告文案的唯一来源。

以前这套数字散在四个地方（``xsec`` 的 bp 分量推导、``score_study`` 的字面量、
``etf_allocation`` 的字面量、``execution.simulate_t1`` 的默认参数），外加
``factor_study`` 直接把 ``(12.6 + 7.6)`` 写进公式、若干报告文案里硬写「7.6/12.6bp」。
改一次费率要改十处，漏一处就会出现「同一份报告里两个成本口径」。

这里的约定：

- **A 股个股**买 7.6bp / 卖 12.6bp（卖出多 5bp 印花税，ETF 免）；
- **ETF** 双边各 7.5bp（免印花税，佣金 2.5 + 滑点 5）；
- 数值一律以**小数**存放（``0.00076``），因为它直接进 ``simulate_t1``；
  展示用的 bp 由 :attr:`Cost.buy_bps` 等属性给出，避免各处再乘 10000；
- 引用方**只许取属性**（``STOCK_COST.buy``），不许复制数值——复制会在下次调费时再次分叉。

只依赖标准库，可被生产层的 ``execution.py`` 直接 import。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 费率分量（bp，单边）。改口径从这里改，下面的 Cost 自动跟着变。
# --------------------------------------------------------------------------- #

COMMISSION_BPS = 2.5        # 佣金 万 2.5
TRANSFER_BPS = 0.1          # 过户费 0.001%（A 股个股独有）
STAMP_DUTY_BPS = 5.0        # 印花税 0.05%，仅卖出（ETF 免）
SLIPPAGE_BPS = 5.0          # 滑点（含冲击成本）0.05%

ETF_COMMISSION_BPS = 2.5    # ETF 佣金
ETF_SLIPPAGE_BPS = 5.0      # ETF 滑点
ETF_STAMP_DUTY_BPS = 0.0    # 印花税：ETF 免征，写出来是为了让「为什么两边一样」可读

STOCK_BUY_BPS = COMMISSION_BPS + TRANSFER_BPS + SLIPPAGE_BPS                   # 7.6
STOCK_SELL_BPS = COMMISSION_BPS + TRANSFER_BPS + SLIPPAGE_BPS + STAMP_DUTY_BPS  # 12.6
ETF_BUY_BPS = ETF_COMMISSION_BPS + ETF_SLIPPAGE_BPS + ETF_STAMP_DUTY_BPS        # 7.5
ETF_SELL_BPS = ETF_BUY_BPS


@dataclass(frozen=True)
class Cost:
    """一侧一个费率，外加可展示的口径说明。"""

    buy: float
    sell: float
    name: str
    note: str = ""
    components: dict[str, float] = field(default_factory=dict, compare=False, repr=False)

    @property
    def buy_bps(self) -> float:
        return round(self.buy * 10000, 2)

    @property
    def sell_bps(self) -> float:
        return round(self.sell * 10000, 2)

    @property
    def round_trip_bps(self) -> float:
        """一次「卖出 + 买入」的完整换手成本（bp）。"""
        return round(self.buy_bps + self.sell_bps, 2)

    @property
    def label(self) -> str:
        """报告文案用：``7.6/12.6bp``。"""
        return f"{self.buy_bps:g}/{self.sell_bps:g}bp"

    def side(self, side: str) -> float:
        """``side`` 取 ``"buy"`` / ``"sell"``；其他值抛错而不是静默当买入。"""
        if side == "buy":
            return self.buy
        if side == "sell":
            return self.sell
        raise ValueError(f"未知方向 {side!r}，只接受 'buy' / 'sell'")

    def scaled(self, factor: float) -> "Cost":
        """成本压力测试（如 ×2）。保持 note 可追溯。"""
        return Cost(buy=self.buy * factor, sell=self.sell * factor,
                    name=f"{self.name}×{factor:g}",
                    note=f"{self.note}（压力测试 ×{factor:g}）".strip(),
                    components=self.components)


# bp → 小数要 round 到 10 位：``7.6 / 10000`` 在二进制下是 0.0007599999999999999，
# 而报告与回测长期用的是字面量 0.00076。不 round 就会让「同一个成本」出现两种浮点表示，
# 净值差 1e-19 却会让「重构前后完全一致」的比对失败，也会让后来人怀疑改错了。
def _decimal(bps: float) -> float:
    return round(bps / 10000, 10)


STOCK_COST = Cost(
    buy=_decimal(STOCK_BUY_BPS),
    sell=_decimal(STOCK_SELL_BPS),
    name="A股个股",
    note="佣金2.5 + 过户0.1 + 滑点5；卖出另加印花税5",
    components={
        "commission": COMMISSION_BPS, "transfer": TRANSFER_BPS,
        "slippage": SLIPPAGE_BPS, "stamp_duty": STAMP_DUTY_BPS,
    },
)

ETF_COST = Cost(
    buy=_decimal(ETF_BUY_BPS),
    sell=_decimal(ETF_SELL_BPS),
    name="ETF",
    note="佣金2.5 + 滑点5；免印花税，因此双边相同",
    components={
        "commission": ETF_COMMISSION_BPS, "transfer": 0.0,
        "slippage": ETF_SLIPPAGE_BPS, "stamp_duty": ETF_STAMP_DUTY_BPS,
    },
)

_BY_ASSET: dict[str, Cost] = {
    "stock": STOCK_COST,
    "etf": ETF_COST,
    "fund": ETF_COST,
    "index": ETF_COST,
}

# 旧名保留为「指向同一对象」，方便逐模块迁移；数值不复制。
BUY_COST = STOCK_COST.buy
SELL_COST = STOCK_COST.sell


def for_asset(asset_type: str | None) -> Cost:
    """按资产类别取成本；未知类别按个股处理（保守，成本更高）。"""
    return _BY_ASSET.get((asset_type or "").lower(), STOCK_COST)


def label(cost: Cost) -> str:
    return cost.label


def describe(cost: Cost) -> dict[str, Any]:
    """给报告用的结构化口径，避免文案里再写死数字。"""
    return {
        "name": cost.name,
        "buy_bps": cost.buy_bps,
        "sell_bps": cost.sell_bps,
        "round_trip_bps": cost.round_trip_bps,
        "components": dict(cost.components),
        "note": cost.note,
    }


__all__ = [
    "COMMISSION_BPS", "TRANSFER_BPS", "STAMP_DUTY_BPS", "SLIPPAGE_BPS",
    "ETF_COMMISSION_BPS", "ETF_SLIPPAGE_BPS", "ETF_STAMP_DUTY_BPS",
    "STOCK_BUY_BPS", "STOCK_SELL_BPS", "ETF_BUY_BPS", "ETF_SELL_BPS",
    "Cost", "STOCK_COST", "ETF_COST", "BUY_COST", "SELL_COST",
    "for_asset", "label", "describe",
]
