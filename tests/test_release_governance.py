"""发布治理：版本与标签一致、CHANGELOG 有判定、reports 只入库登记表。"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class VersionTagTest(unittest.TestCase):
    def test_app_version_matches_tag(self):
        import app

        # tag v1.5-three-fields ⇄ APP_VERSION "1.5-three-fields"
        self.assertEqual(app.APP_VERSION, "1.5-three-fields")
        self.assertTrue(re.fullmatch(r"\d+\.\d+-[a-z-]+", app.APP_VERSION))
        self.assertEqual(app.Handler.server_version,
                         f"AShareResearch/{app.APP_VERSION}")

    def test_changelog_declares_same_version(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [1.5-three-fields]", text)
        self.assertIn("`v1.5-three-fields`", text)
        self.assertIn("APP_VERSION", text)

    def test_changelog_has_verdicts_for_every_milestone(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for topic in ("单一评分", "三字段", "v4 打分链路", "regime", "ETF", "PIT"):
            self.assertIn(topic, text, f"CHANGELOG 缺少里程碑：{topic}")
        # 每条里程碑必须带判定，不能只写"已完成"
        self.assertGreaterEqual(text.count("**通过**"), 3)
        self.assertGreaterEqual(text.count("**未通过**"), 4)


class ReportsGovernanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "reports" / "index.json"
        self.index = json.loads(self.path.read_text(encoding="utf-8"))

    def test_index_schema_and_policy(self):
        self.assertEqual(self.index["schema"], "reports-index/1")
        self.assertEqual(self.index["release"], "v1.5-three-fields")
        policy = self.index["policy"]
        for key in ("tracked", "rebuild", "verdict", "vocabulary"):
            self.assertIn(key, policy)
        self.assertIn("通过", policy["vocabulary"])
        self.assertIn("未通过", policy["vocabulary"])

    def test_entries_are_actionable(self):
        entries = self.index["entries"]
        self.assertGreaterEqual(len(entries), 10)
        ids = set()
        for entry in entries:
            for key in ("id", "title", "path", "producer", "dataset",
                        "generated_at", "verdict", "note"):
                self.assertIn(key, entry, f"{entry.get('id')} 缺字段 {key}")
            self.assertNotIn(entry["id"], ids, "id 重复")
            ids.add(entry["id"])
            self.assertIn(entry["verdict"], self.index["policy"]["vocabulary"])
            self.assertTrue(entry["path"].startswith("reports/"))
            self.assertTrue(entry["producer"].startswith(("python", "POSIX")))
            self.assertRegex(entry["generated_at"], r"^\d{4}-\d{2}-\d{2}$")

    def test_generated_artifacts_are_not_tracked(self):
        patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("reports/**", patterns)
        self.assertIn("!reports/index.json", patterns)
        # 产物目录仍在本地（由 producer 生成），但被忽略规则排除在版本库外
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("reports/**", ignore)


if __name__ == "__main__":
    unittest.main()
