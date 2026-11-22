"""Extend the prototype Burke Road corridor."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


# source, destination, distance_km, source measurement location
LINKS = [
    (3120, 4035, 0.532, "BURKE_RD N of CANTERBURY_RD"),
    (4035, 3120, 0.532, "BURKE_RD S of BARKERS_RD"),
    (4035, 4034, 0.710, "BURKE_RD N of MONT ALBERT_RD"),
    (4034, 4035, 0.710, "BURKE_RD S OF WHITEHORSE_RD"),
    (4034, 4032, 1.074, "BURKE_RD N OF WHITEHORSE_RD"),
    (4032, 4034, 1.074, "BURKE_RD S of HARP_RD"),
    (4032, 4030, 0.775, "BURKE_RD N of HARP_RD"),
    (4030, 4032, 0.775, "BURKE_RD S of DONCASTER_RD"),
]

COLUMNS = [
    "source",
    "destination",
    "distance_km",
    "traffic_location",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    base = args.project_dir.resolve()
    roads_path = base / "roads.csv"
    model_dir = base / "saved_models"
    metadata_path = model_dir / "metadata.json"

    original = roads_path.read_bytes()
    roads = pd.read_csv(roads_path)

    if not set(COLUMNS).issubset(roads.columns):
        raise ValueError("roads.csv is missing required columns.")

    if roads.duplicated(["source", "destination"]).any():
        raise ValueError("Resolve duplicate road links first.")

    stations = set(roads.source) | set(roads.destination)
    if 3120 not in stations:
        raise ValueError("Complete the previous Burke bridge first.")

    def missing_models():
        records = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        missing = []

        for source, _, _, location in LINKS:
            found = [
                record
                for record in records
                if int(record["site"]) == source
                and record["location"] == location
            ]

            if len(found) > 1:
                raise ValueError(
                    f"Duplicate model: {source} | {location}"
                )

            if not found:
                missing.append((source, location))
            elif not (
                model_dir / found[0]["filename"]
            ).is_file():
                raise FileNotFoundError(
                    f"Missing saved model file: {source} | {location}"
                )

        return records, missing

    # Check every location before starting any training.
    data = pd.read_excel(
        base / "Scats Data October 2006.xls",
        sheet_name="Data",
        header=1,
    )
    data["SCATS Number"] = pd.to_numeric(
        data["SCATS Number"], errors="raise"
    ).astype(int)
    data["Date"] = pd.to_datetime(
        data["Date"], errors="raise"
    ).dt.normalize()

    volumes = [f"V{i:02d}" for i in range(96)]

    for source, destination, distance, location in LINKS:
        existing = roads[
            (roads.source == source)
            & (roads.destination == destination)
        ]

        # Do not overwrite a different existing configuration.
        if not existing.empty:
            row = existing.iloc[0]

            if (
                not np.isclose(float(row.distance_km), distance)
                or row.traffic_location != location
            ):
                raise ValueError(
                    "Different road configuration exists: "
                    f"{source} -> {destination}"
                )

        selected = data[
            (data["SCATS Number"] == source)
            & (data["Location"] == location)
        ]

        if (
            selected.empty
            or selected["Date"].duplicated().any()
        ):
            raise ValueError(
                "Missing or duplicate daily records: "
                f"{source} | {location}"
            )

        values = selected[volumes].to_numpy(dtype=float)

        if (
            not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(
                f"Invalid traffic values: {source} | {location}"
            )

        dates = selected["Date"]

        masks = [
            dates < "2006-10-21",
            (dates >= "2006-10-21")
            & (dates < "2006-10-25"),
            (dates >= "2006-10-25")
            & (dates < "2006-11-01"),
        ]

        if not all(mask.any() for mask in masks):
            raise ValueError(
                "A calendar split has no observations: "
                f"{source} | {location}"
            )

        counts = [
            int(mask.sum()) * 92
            for mask in masks
        ]

        print(
            f"CHECKED: {source} -> {destination} "
            f"| samples {counts}",
            flush=True,
        )

    records, missing = missing_models()
    print("Missing models:", len(missing), flush=True)

    if args.check_only:
        print(
            "Check only: no models trained "
            "and no files changed."
        )
        return

    trainer = base / "train_location_model.py"

    if missing and not trainer.is_file():
        raise FileNotFoundError(
            "train_location_model.py is required."
        )

    # Reuse existing models; train only missing locations.
    for index, (site, location) in enumerate(missing, 1):
        print(
            f"\nTraining {index}/{len(missing)}: "
            f"{site} | {location}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(trainer),
                "--site",
                str(site),
                "--location",
                location,
                "--validation-start",
                "2006-10-21",
                "--test-start",
                "2006-10-25",
                "--test-end-exclusive",
                "2006-11-01",
            ],
            cwd=base,
            check=True,
        )

    records, missing = missing_models()

    if missing:
        raise RuntimeError(
            "Some required models are still missing; "
            "roads were not changed."
        )

    if roads_path.read_bytes() != original:
        raise RuntimeError(
            "roads.csv changed during training. "
            "Inspect it before retrying."
        )

    # Append missing links without duplicating existing links.
    pairs = set(zip(roads.source, roads.destination))
    additions = [
        link
        for link in LINKS
        if link[:2] not in pairs
    ]

    if additions:
        updated = pd.concat(
            [
                roads,
                pd.DataFrame(additions, columns=COLUMNS),
            ],
            ignore_index=True,
        )

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        backup = base / (
            f"roads_before_burke_north_{stamp}.csv"
        )
        backup.write_bytes(original)

        temporary = base / (
            f"roads_pending_burke_north_{stamp}.csv"
        )
        updated.to_csv(temporary, index=False)
        temporary.replace(roads_path)

        print("Backup:", backup.name)

    else:
        updated = roads
        print(
            "All proposed links already exist. "
            "No roads.csv changes."
        )

    print("Roads saved:", len(updated))
    print(
        "Stations in road graph:",
        len(set(updated.source) | set(updated.destination)),
    )
    print("Saved model records:", len(records))
    print(
        "Added corridor: "
        "3120 <-> 4035 <-> 4034 <-> 4032 <-> 4030"
    )
    print(
        "Distances are straight-line estimates; "
        "source-side assignments remain prototype proxies."
    )


if __name__ == "__main__":
    main()