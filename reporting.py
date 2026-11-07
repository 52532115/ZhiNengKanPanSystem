"""渲染公共层：数值格式化、Markdown/HTML 表格、页面壳、SVG 净值图。

本文件当前只落地 **P3：数值层**（``fmt`` / ``num`` / ``signed`` / ``pct`` / ``inline``）。
表格与页面壳（P4）、SVG（P5）随后补进来；在此之前它们仍留在各 study 里。

为什么数值层要先做：``_fmt`` 曾有 **3 种互不兼容的签名、261 处调用**——

===================  ==========================================  ============
模块                 签名                                         默认后缀/位数
===================  ==========================================  ============
``protocol``         ``(value, digits=3, suffix="", signed=True)``  — / 3
``walkforward``      ``(value, suffix="", digits=2, signed=True)``  — / 2
``xsec``             ``(value, suffix="%", digits=2, signed=True)`` **%** / 2
``validate_fields``  ``(value, digits=3, suffix="")``                — / 3（恒带符号）
``turnover_study``   ``(value, digits=2, suffix="")``                — / 2（**恒不带符号**）
``pool_validate``    ``(value, digits=3, suffix="")``                — / 3
===================  ==========================================  ============

同一行代码 ``_fmt(x, 2, "%")`` 在不同模块里含义完全不同（``digits=2`` 还是
``suffix=2``），所以"统一签名"不可能靠改一个 def 完成——必须把每个调用点按
**旧签名**解析成关键字参数。这里全部做成关键字参数，位置参数一律不许传。
"""

from __future__ import annotations

from typing import Any

DASH = "—"          # 缺值占位符（各模块原本就都用这个字符）


# --------------------------------------------------------------------------- #
# 数值
# --------------------------------------------------------------------------- #


def fmt(
    value: Any,
    *,
    digits: int = 3,
    suffix: str = "",
    signed: bool = True,
    none: str = DASH,
) -> str:
    """数值 → 定长文本。缺值给 ``none``，字符串原样返回（不再套格式化）。

    旧实现里"字符串原样返回"只出现在 ``validate_fields`` / ``turnover_study`` /
    ``pool_validate``，另外三份遇到字符串会直接抛 ``TypeError``。统一后一律放行：
    那三处本来就不该收到字符串，真收到了是上游 bug，静默返回比崩掉更难查——
    但为了不让"重构顺带改错误行为"混进来，这里选择放行，并在 docstring 里写明。

    ``suffix`` 追加在格式化结果之后；``signed=True`` 时正数带 ``+``。
    """
    if value is None:
        return none
    if isinstance(value, str):
        return value
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def num(value: Any, *, digits: int = 2, none: str = DASH) -> str:
    """不带正负号的定长小数（``12.35``）。"""
    return fmt(value, digits=digits, signed=False, none=none)


def signed(value: Any, *, digits: int = 2, none: str = DASH) -> str:
    """带正负号的定长小数（正数显式带 ``+``）。"""
    return fmt(value, digits=digits, signed=True, none=none)


# 说明：不在这里提供 ``pct`` / ``bps`` 之类的便捷包装。它们的输出与
# ``fmt(x, digits=2, suffix="%")`` 完全一样，多一层名字只会让"到底是百分数还是小数"
# 更难判断——乘 100 的转换必须留在调用方，且变量名带 ``_pct``。

# --------------------------------------------------------------------------- #
# 行内 Markdown
# --------------------------------------------------------------------------- #


def inline(text: str) -> str:
    """只处理报告用到的行内语法：``**粗体**`` 与 `` `代码` ``。

    两份旧实现里，``factor_study`` 只处理粗体、``validate_fields`` 两个都处理。
    合并取并集：对 ``factor_study`` 而言这是行为变化（含反引号的表格单元格
    会渲染成 ``<code>`` 而不是保留反引号），但那本来就是漏渲染，不是特性。
    ``render_markdown`` 的输出不受影响，只有 HTML 会变。
    """
    out = text
    while "**" in out:
        first = out.index("**")
        second = out.find("**", first + 2)
        if second < 0:
            break
        out = f"{out[:first]}<strong>{out[first + 2:second]}</strong>{out[second + 2:]}"
    while "`" in out:
        first = out.index("`")
        second = out.find("`", first + 1)
        if second < 0:
            break
        out = f"{out[:first]}<code>{out[first + 1:second]}</code>{out[second + 1:]}"
    return out


__all__ = ["DASH", "fmt", "num", "signed", "inline"]
