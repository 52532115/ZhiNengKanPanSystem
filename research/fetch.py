"""抓取入口：``python -m research.fetch``（见 ``research.bs_source``）。"""

from .bs_source import main

if __name__ == "__main__":
    raise SystemExit(main())
