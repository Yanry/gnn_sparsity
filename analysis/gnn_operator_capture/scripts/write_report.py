from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnn_operator_capture.report import build_report


def main() -> None:
    output_root = PROJECT_ROOT / "outputs"
    report = build_report(output_root)
    report_path = PROJECT_ROOT / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
