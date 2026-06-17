from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.tracking.seed_admission_classifier import train_seed_admission_classifier


APP_NAME = "qstr-seed-admission-smoke"
VOLUME_NAME = "qstr-seed-admission"


try:
    import modal
except Exception:  # pragma: no cover - local smoke should not require Modal.
    modal = None  # type: ignore[assignment]


if modal is not None:
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch")
        .add_local_python_source("qstr_dronedet")
    )
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(image=image, volumes={"/data": volume}, timeout=30 * 60, gpu="T4")
    def train_remote(
        dataset: str,
        out: str,
        epochs: int = 25,
        lr: float = 1e-3,
        hidden: int = 32,
        thresholds: list[float] | None = None,
        smoke: bool = True,
    ) -> dict[str, Any]:
        result = train_seed_admission_classifier(
            dataset,
            out,
            epochs=epochs,
            lr=lr,
            hidden=hidden,
            thresholds=thresholds,
            smoke=smoke,
        )
        volume.commit()
        return _summary_payload(result.summary)

    @app.local_entrypoint()
    def modal_main(
        dataset: str = "/data/seg01_04_probation48_seed_admission.csv",
        out: str = "/data/seed_admission_smoke",
        epochs: int = 25,
        lr: float = 1e-3,
        hidden: int = 32,
        thresholds: str = "0.25,0.5,0.75",
        smoke: bool = True,
    ) -> None:
        values = _parse_thresholds(thresholds)
        print(
            json.dumps(
                train_remote.remote(dataset, out, epochs, lr, hidden, values, smoke),
                indent=2,
            )
        )
else:
    app = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a small seed-admission classifier smoke model.")
    parser.add_argument("--dataset", required=True, help="Local seed-admission CSV path")
    parser.add_argument("--out", required=True, help="Local output directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", help="Mark output as smoke-only")
    args = parser.parse_args(argv)
    result = train_seed_admission_classifier(
        args.dataset,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        thresholds=args.thresholds,
        smoke=args.smoke,
    )
    print(json.dumps(_summary_payload(result.summary), indent=2))


def _summary_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "weights": summary["weights"],
        "metrics_json": summary["metrics_json"],
        "threshold_sweep_csv": summary["threshold_sweep_csv"],
        "summary_json": summary["summary_json"],
        "num_rows": summary["num_rows"],
        "positive_rows": summary["positive_rows"],
        "negative_rows": summary["negative_rows"],
        "smoke": summary["smoke"],
        "best": summary["best"],
    }


def _parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
