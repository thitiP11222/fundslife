from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.momentum_oos import summarize_oos_events


def main() -> None:
    results_dir = Path("results")
    events_path = results_dir / "momentum_robustness_events.csv"

    if not events_path.exists():
        raise FileNotFoundError(
            "Run python run_momentum_robustness.py first."
        )

    events = pd.read_csv(events_path)

    yearly, leave_one_out = summarize_oos_events(events)

    yearly_path = results_dir / "momentum_oos_yearly.csv"
    loo_path = results_dir / "momentum_oos_leave_one_out.csv"

    yearly.to_csv(yearly_path, index=False)
    leave_one_out.to_csv(loo_path, index=False)

    print()
    print("=" * 104)
    print("MOMENTUM OOS ROBUSTNESS - YEAR BY YEAR")
    print("=" * 104)

    if yearly.empty:
        print("No yearly results.")
    else:
        print(
            yearly.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    print()
    print("=" * 104)
    print("LEAVE-ONE-SYMBOL-OUT ROBUSTNESS")
    print("=" * 104)

    if leave_one_out.empty:
        print("No leave-one-out results.")
    else:
        print(
            leave_one_out.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    print()
    print(f"Saved yearly : {yearly_path}")
    print(f"Saved LOO    : {loo_path}")


if __name__ == "__main__":
    main()
