"""reporting 数值层的口径守卫。

`_fmt` 曾有 3 种互不兼容签名、315 处调用：同一行 ``_fmt(x, 2, '%')`` 在
``protocol`` 里是 ``digits=2``，在 ``walkforward`` / ``xsec`` 里是 ``suffix=2``。
改签名的唯一安全做法是把每个调用点按**旧签名**解析成关键字参数，然后逐位比对输出。

这里做两件事：

1. **把 6 份旧实现原样抄进来当参考**，用网格输入逐项比对 ``reporting.fmt``；
2. **结构断言**：全仓库不许再出现 ``def _fmt/_num/_signed/_inline``，
   并且所有 ``fmt/num/signed/inline`` 调用除第一个参数外必须是关键字参数——
   只要还有人写位置参数，签名一改就又会静默改输出。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import reporting
from reporting import fmt, inline, num, signed

ROOT = Path(__file__).resolve().parent.parent

VALUES = [None, 0.0, 1.234567, -1.234567, 12.3456, -0.0001, 100.0, -3.5, 1e6]
STRINGS = ["abc", "—"]
DIGITS = (0, 1, 2, 3, 4)
SUFFIXES = ("", "%", "bp", "bp/年")
FLAGS = (True, False)


# --------------------------------------------------------------------------- #
# 6 份旧实现（迁移前代码原样）
# --------------------------------------------------------------------------- #


def legacy_protocol(value, digits=3, suffix="", signed_=True):
    if value is None:
        return "—"
    template = f"{{:+.{digits}f}}" if signed_ else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def legacy_walkforward(value, suffix="", digits=2, signed_=True):
    if value is None:
        return "—"
    template = f"{{:+.{digits}f}}" if signed_ else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def legacy_xsec(value, suffix="%", digits=2, signed_=True):
    if value is None:
        return "—"
    template = f"{{:+.{digits}f}}" if signed_ else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def legacy_validate_fields(value, digits=3, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:+.{digits}f}{suffix}"
    return str(value)


def legacy_turnover(value, digits=2, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}{suffix}"


def legacy_pool(value, digits=3, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:+.{digits}f}{suffix}" if digits else f"{value}{suffix}"


def outcome(func, *args, **kwargs):
    """把"抛异常"也当成一种可比较的输出。

    旧的三份实现遇到字符串会抛 ``TypeError``，新的 ``fmt`` 选择放行。这个差异是
    有意的（放行更好查），但测试不能因此假装两边相等——所以把异常类型也纳入比较。
    """
    try:
        return ("value", func(*args, **kwargs))
    except Exception as exc:                     # noqa: BLE001 - 故意比较异常类型
        return ("error", type(exc).__name__)


class LegacyEquivalenceTest(unittest.TestCase):
    """每一份旧实现都要能被 ``fmt`` 用关键字参数精确复现。"""

    def test_protocol_signature(self) -> None:
        for value in VALUES:
            for digits in DIGITS:
                for suffix in SUFFIXES:
                    for flag in FLAGS:
                        with self.subTest(value=value, digits=digits, suffix=suffix, flag=flag):
                            self.assertEqual(
                                fmt(value, digits=digits, suffix=suffix, signed=flag),
                                legacy_protocol(value, digits, suffix, flag),
                            )
        # 字符串：旧实现会崩，新实现放行（有意为之，见 fmt 的 docstring）
        self.assertEqual(fmt("abc"), "abc")
        for module_legacy in (legacy_protocol, legacy_walkforward, legacy_xsec):
            self.assertEqual(outcome(module_legacy, "abc")[0], "error")

    def test_walkforward_signature(self) -> None:
        for value in VALUES:
            for digits in DIGITS:
                for suffix in SUFFIXES:
                    for flag in FLAGS:
                        with self.subTest(value=value, digits=digits, suffix=suffix, flag=flag):
                            self.assertEqual(
                                fmt(value, digits=digits, suffix=suffix, signed=flag),
                                legacy_walkforward(value, suffix, digits, flag),
                            )

    def test_xsec_signature_defaults_to_percent(self) -> None:
        for value in VALUES:
            for digits in DIGITS:
                with self.subTest(value=value, digits=digits):
                    self.assertEqual(fmt(value, digits=digits, suffix="%"),
                                     legacy_xsec(value, digits=digits))

    def test_validate_fields_passes_strings_through(self) -> None:
        for value in VALUES + STRINGS:
            for digits in DIGITS:
                with self.subTest(value=value, digits=digits):
                    self.assertEqual(fmt(value, digits=digits),
                                     legacy_validate_fields(value, digits))

    def test_turnover_is_unsigned(self) -> None:
        # 全仓库唯一一处"恒不带正号"的口径，靠每个调用点显式 signed=False 实现。
        for value in VALUES:
            for digits in DIGITS:
                for suffix in SUFFIXES:
                    with self.subTest(value=value, digits=digits, suffix=suffix):
                        self.assertEqual(
                            fmt(value, digits=digits, suffix=suffix, signed=False),
                            legacy_turnover(value, digits, suffix),
                        )

    def test_pool_signature(self) -> None:
        for value in VALUES + STRINGS:
            for digits in (1, 2, 3, 4):
                with self.subTest(value=value, digits=digits):
                    self.assertEqual(fmt(value, digits=digits),
                                     legacy_pool(value, digits))


class HelperTest(unittest.TestCase):
    def test_num_is_unsigned(self) -> None:
        self.assertEqual(num(12.3456, digits=2), "12.35")
        self.assertEqual(num(-12.3456, digits=2), "-12.35")

    def test_signed_always_shows_sign(self) -> None:
        self.assertEqual(signed(12.3456, digits=2), "+12.35")
        self.assertEqual(signed(-12.3456, digits=2), "-12.35")

    def test_percent_suffix_does_not_scale(self) -> None:
        # 关键口径：传 12.34 就是 12.34%，不再乘 100。乘 100 的转换必须在调用方。
        self.assertEqual(fmt(12.34, digits=2, suffix="%"), "+12.34%")
        self.assertEqual(fmt(12.34, digits=2, suffix="%", signed=False), "12.34%")

    def test_missing_value_uses_dash(self) -> None:
        for helper in (fmt, num, signed):
            with self.subTest(helper=helper.__name__):
                self.assertEqual(helper(None), reporting.DASH)

    def test_dash_is_the_shared_placeholder(self) -> None:
        # 三份旧实现分别用了 DASH / ELLIPSIS / "—" 三个名字，值都是同一个字符。
        self.assertEqual(reporting.DASH, "—")

    def test_zero_digits_keeps_sign(self) -> None:
        # protocol 里 ``_fmt(x, 0, 'bp/年')`` 的语义：0 位小数 + 带符号。
        self.assertEqual(fmt(12.6, digits=0, suffix="bp/年"), "+13bp/年")

    def test_float_precision_is_not_reformatted(self) -> None:
        self.assertEqual(fmt(0.1 + 0.2, digits=10), "+0.3000000000")


class InlineTest(unittest.TestCase):
    def test_bold(self) -> None:
        self.assertEqual(inline("**粗**"), "<strong>粗</strong>")

    def test_code(self) -> None:
        # factor_study 的旧实现只认粗体，这里取并集（漏渲染反引号是 bug 不是特性）。
        self.assertEqual(inline("`x`"), "<code>x</code>")

    def test_unbalanced_markers_are_left_alone(self) -> None:
        self.assertEqual(inline("**未闭合"), "**未闭合")
        self.assertEqual(inline("`未闭合"), "`未闭合")


class StructureTest(unittest.TestCase):
    """结构守卫：比数值更重要的部分。"""

    def test_no_legacy_definitions_left(self) -> None:
        offenders = []
        for path in sorted(ROOT.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for name in ("_fmt", "_num", "_signed", "_inline"):
                if f"def {name}(" in text:
                    offenders.append(f"{path.name}: def {name}(")
        self.assertEqual(offenders, [], "旧实现应已删除：" + "; ".join(offenders))

    def test_all_calls_use_keyword_arguments(self) -> None:
        """除第一个参数外一律关键字——否则下次改签名又会静默改输出。"""
        offenders = []
        for path in sorted(ROOT.glob("*.py")) + sorted((ROOT / "research").glob("*.py")):
            if path.name == "reporting.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else None
                if name in ("fmt", "num", "signed", "inline") and len(node.args) > 1:
                    offenders.append(f"{path.name}:{node.lineno} {name}(...) 传了 "
                                     f"{len(node.args)} 个位置参数")
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_modules_import_from_reporting(self) -> None:
        import importlib

        for name in ("xsec", "walkforward", "protocol", "validate_fields",
                     "turnover_study", "etf_allocation", "regime_study",
                     "score_study", "wfv2_study", "fundamental_study", "factor_study"):
            module = importlib.import_module(name)
            self.assertTrue(hasattr(module, "fmt") or hasattr(module, "num")
                            or hasattr(module, "inline"), name)

    def test_reporting_has_no_business_imports(self) -> None:
        tree = ast.parse((ROOT / "reporting.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported - {"__future__", "typing"}, set())


if __name__ == "__main__":
    unittest.main()
