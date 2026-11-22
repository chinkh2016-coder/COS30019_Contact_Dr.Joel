import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent


def main():
    model_dir = BASE_DIR / "saved_models"
    output_dir = BASE_DIR / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )

    data = pd.read_excel(
        BASE_DIR / "Scats Data October 2006.xls",
        sheet_name="Data",
        header=1,
    )
    data["SCATS Number"] = pd.to_numeric(
        data["SCATS Number"], errors="raise"
    ).astype(int)
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")

    volume_columns = [f"V{i:02d}" for i in range(96)]
    evaluation_rows = []
    prediction_rows = []

    for item in metadata:
        site = int(item["site"])
        location = item["location"]
        window = int(item.get("window_size", 4))

        # Use only the reserved October 25–31 test period.
        records = data.loc[
            (data["SCATS Number"] == site)
            & (data["Location"] == location)
            & (data["Date"] >= pd.Timestamp("2006-10-25"))
            & (data["Date"] < pd.Timestamp("2006-11-01"))
        ].sort_values("Date")

        if records.empty:
            evaluation_rows.append({
                "site": site,
                "location": location,
                "status": "NO_TEST_DATA",
                "test_samples": 0,
            })
            continue

        if records["Date"].dt.normalize().duplicated().any():
            raise ValueError(
                f"Duplicate daily records: {site}, {location}"
            )

        inputs, targets, sample_labels = [], [], []

        # Build windows separately within each day.
        for _, row in records.iterrows():
            counts = row[volume_columns].to_numpy(dtype=float)

            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError(
                    f"Invalid test data: {site}, {location}"
                )

            for end in range(window, len(counts)):
                inputs.append(counts[end - window:end])
                targets.append(counts[end])
                sample_labels.append((
                    row["Date"].strftime("%Y-%m-%d"),
                    f"V{end:02d}",
                ))

        X = np.asarray(inputs, dtype=np.float32)[..., np.newaxis]
        actual = np.asarray(targets, dtype=float)

        minimum = float(item["minimum"])
        scale_range = float(item["scale_range"])

        if not np.isfinite(scale_range) or scale_range <= 0:
            raise ValueError(f"Invalid scale range: {site}, {location}")

        # Reuse the scaling parameters saved during training.
        model = load_model(
            model_dir / item["filename"],
            compile=False,
        )
        scaled_inputs = (X - minimum) / scale_range

        predicted = (
            np.asarray(
                model(scaled_inputs, training=False)
            ).ravel()
            * scale_range
            + minimum
        )

        if not np.isfinite(predicted).all():
            raise ValueError(f"Invalid predictions: {site}, {location}")

        # Persistence baseline: repeat the last observed count.
        baseline = X[:, -1, 0]

        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        baseline_rmse = float(
            np.sqrt(np.mean((baseline - actual) ** 2))
        )
        actual_mean = float(actual.mean())

        evaluation_rows.append({
            "site": site,
            "location": location,
            "status": "EVALUATED",
            "test_samples": len(actual),
            "LSTM_RMSE": rmse,
            "Baseline_RMSE": baseline_rmse,
            "NRMSE_percent": (
                rmse / actual_mean * 100
                if actual_mean > 0 else np.nan
            ),
            "LSTM_beats_baseline": rmse < baseline_rmse,
        })

        for i, (date, column) in enumerate(sample_labels):
            prediction_rows.append({
                "site": site,
                "location": location,
                "date": date,
                "target_column": column,
                "actual": actual[i],
                "LSTM_prediction": predicted[i],
                "baseline_prediction": baseline[i],
            })

        print(
            f"{site} | {location}\n"
            f"  LSTM RMSE: {rmse:.2f} | "
            f"Baseline RMSE: {baseline_rmse:.2f}"
        )

    evaluation = pd.DataFrame(evaluation_rows)
    predictions = pd.DataFrame(prediction_rows)

    evaluation.to_csv(
        output_dir / "model_evaluation.csv", index=False
    )
    predictions.to_csv(
        output_dir / "test_predictions.csv", index=False
    )

    print("\nSaved model records:", len(metadata))
    print(
        "Models evaluated:",
        int(evaluation["status"].eq("EVALUATED").sum()),
    )
    print("Results saved in:", output_dir)


if __name__ == "__main__":
    main()