"""Prepare models and add the prototype Burke Road connections."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Distances are straight-line estimates.
# Measurement assignments follow the existing source-side prototype.
LINKS = [
    (
        3122, 3120, 0.633,
        "CANTERBURY_RD W of STANHOPE_GV",
    ),
    (
        3120, 3122, 0.633,
        "CANTERBURY_RD E of BURKE_RD",
    ),
    (
        3120, 4040, 1.121,
        "BURKE_RD S of CANTERBURY_RD",
    ),
    (
        4040, 3120, 1.121,
        "BURKE_RD N of RIVERSDALE_RD",
    ),
]


def load_metadata(model_dir):
    return json.loads(
        (model_dir / "metadata.json").read_text(
            encoding="utf-8"
        )
    )


def find_model(metadata, site, location, model_dir):
    matches = [
        record
        for record in metadata
        if int(record["site"]) == site
        and record["location"] == location
    ]

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate model records: {site}, {location}"
        )

    if matches:
        model_path = model_dir / matches[0]["filename"]

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file is missing: {model_path.name}"
            )

        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check models and proposed links without changing files.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    model_dir = base / "saved_models"
    roads_path = base / "roads.csv"

    original = roads_path.read_bytes()
    roads = pd.read_csv(roads_path)

    columns = [
        "source",
        "destination",
        "distance_km",
        "traffic_location",
    ]

    if not set(columns).issubset(roads.columns):
        raise ValueError("Unexpected roads.csv format.")

    existing_stations = (
        set(roads["source"])
        | set(roads["destination"])
    )

    if not {3122, 3126, 4040}.issubset(existing_stations):
        raise ValueError(
            "Complete the previous Canterbury and "
            "Warrigal extensions first."
        )

    # Refuse to silently replace a differently configured link.
    for source, target, distance, location in LINKS:
        existing = roads.loc[
            (roads["source"] == source)
            & (roads["destination"] == target)
        ]

        if not existing.empty:
            same_configuration = (
                (existing["distance_km"] == distance)
                & (existing["traffic_location"] == location)
            )

            if not same_configuration.all():
                raise ValueError(
                    "A different configuration exists for "
                    f"{source} -> {target}."
                )

    metadata = load_metadata(model_dir)
    missing = []

    for source, _, _, location in LINKS:
        found = find_model(
            metadata, source, location, model_dir
        )

        status = "READY" if found else "MISSING"
        print(
            f"{status}: {source} | {location}",
            flush=True,
        )

        if not found:
            missing.append((source, location))

    if args.check_only:
        print("Missing models:", len(missing))
        print("Proposed links: 3122 <-> 3120 <-> 4040")
        print("Check only: no files changed.")
        return

    trainer = base / "train_location_model.py"

    if missing and not trainer.is_file():
        raise FileNotFoundError(
            "Place train_location_model.py "
            "in the project folder first."
        )

    # Train only locations that do not already have saved models.
    for index, (site, location) in enumerate(
        missing, start=1
    ):
        print(
            f"\nTraining missing model "
            f"{index}/{len(missing)}: {site} | {location}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(trainer),
                "--site", str(site),
                "--location", location,
                "--validation-start", "2006-10-21",
                "--test-start", "2006-10-25",
                "--test-end-exclusive", "2006-11-01",
            ],
            cwd=base,
            check=True,
        )

    # Confirm all required models were registered.
    metadata = load_metadata(model_dir)

    for source, _, _, location in LINKS:
        if not find_model(
            metadata, source, location, model_dir
        ):
            raise RuntimeError(
                f"Model was not registered: "
                f"{source}, {location}"
            )

    # Avoid overwriting changes made during training.
    if roads_path.read_bytes() != original:
        raise RuntimeError(
            "roads.csv changed during training. "
            "Re-run after checking the changes."
        )

    additions = pd.DataFrame(LINKS, columns=columns)

    updated = pd.concat(
        [roads, additions],
        ignore_index=True,
    ).drop_duplicates(
        subset=["source", "destination"],
        keep="last",
    )

    # Preserve the road table before writing the updated version.
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    backup = base / f"roads_before_burke_bridge_{stamp}.csv"
    backup.write_bytes(original)

    temporary = base / "roads_pending_burke_bridge.csv"
    updated.to_csv(temporary, index=False)
    temporary.replace(roads_path)

    print("\nRoads saved:", len(updated))
    print(
        "Stations in road graph:",
        len(
            set(updated["source"])
            | set(updated["destination"])
        ),
    )
    print("Saved model records:", len(metadata))
    print("Added: 3122 <-> 3120 <-> 4040")
    print("Backup:", backup.name)
    print(
        "Distances remain straight-line estimates; "
        "source-side measurement assignments "
        "remain prototype proxies."
    )


if __name__ == "__main__":
    main()