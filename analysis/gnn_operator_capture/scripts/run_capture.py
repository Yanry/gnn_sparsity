from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnn_operator_capture.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture GNN operator operands and sparsity statistics.")
    parser.add_argument("--models", nargs="+", default=["gcn"], help="Models to run.")
    parser.add_argument("--datasets", nargs="+", default=["cora"], help="Datasets to run.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for model_name in args.models:
        for dataset_name in args.datasets:
            print(f"[run] model={model_name} dataset={dataset_name}")
            result = run_experiment(
                model_name=model_name,
                dataset_name=dataset_name,
                root_dir=args.data_root,
                output_root=args.output_root,
                hidden_channels=args.hidden_channels,
                device=args.device,
                save_plots=not args.skip_plots,
            )
            results.append(result.__dict__)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
