"""
Audit the configured network, model references and traffic data.

Run:
    python3 audit_network.py

Optional output folder:
    python3 audit_network.py --output network_audit

These checks establish internal consistency.
They do not verify physical road connections or driving permissions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from network_validation import can_reach


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("network_audit"),
    )

    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    # Load the road graph.
    roads = pd.read_csv(
        base / "roads.csv"
    )

    # Load the original traffic data.
    data = pd.read_excel(
        base / "Scats Data October 2006.xls",
        sheet_name="Data",
        header=1,
    )

    # Load model metadata without loading TensorFlow models.
    metadata = json.loads(
        (
            base
            / "saved_models"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    ).dt.normalize()

    dataset_sites = sorted(
        set(
            data["SCATS Number"].astype(int)
        )
    )

    configured_sites = (
        set(roads["source"])
        | set(roads["destination"])
    )

    # Include all dataset sites, including unconfigured sites.
    graph = {
        site: []
        for site in dataset_sites
    }

    for road in roads.itertuples(index=False):
        graph.setdefault(
            road.source,
            [],
        ).append(
            (
                road.destination,
                road.distance_km,
            )
        )

    model_lookup = {
        (
            int(item["site"]),
            item["location"],
        ): item
        for item in metadata
    }

    # Check coverage and directed reachability.
    coverage_rows = []

    for site in dataset_sites:
        outgoing = int(
            (roads["source"] == site).sum()
        )

        incoming = int(
            (roads["destination"] == site).sum()
        )

        if site not in configured_sites:
            status = "UNCONFIGURED"

        elif outgoing == 0:
            status = "NO_OUTGOING"

        elif incoming == 0:
            status = "NO_INCOMING"

        else:
            status = "CONFIGURED"

        reachable_others = sum(
            can_reach(
                graph,
                site,
                destination,
            )
            for destination in dataset_sites
            if destination != site
        )

        coverage_rows.append(
            {
                "site": site,
                "outgoing": outgoing,
                "incoming": incoming,
                "status": status,
                "reachable_other_sites": reachable_others,
            }
        )

    # Check duplicate daily records and invalid counts.
    volume_columns = [
        f"V{i:02d}"
        for i in range(96)
    ]

    measurement_issues = []

    for (site, location), group in data.groupby(
        ["SCATS Number", "Location"]
    ):
        values = group[
            volume_columns
        ].to_numpy(dtype=float)

        duplicate_rows = int(
            group.duplicated(
                ["Date"],
                keep=False,
            ).sum()
        )

        invalid_counts = int(
            (
                ~np.isfinite(values)
                | (values < 0)
            ).sum()
        )

        if duplicate_rows or invalid_counts:
            measurement_issues.append(
                {
                    "site": int(site),
                    "location": location,
                    "duplicate_date_rows": duplicate_rows,
                    "invalid_counts": invalid_counts,
                }
            )

    # Check each road's source measurement and saved model.
    errors = []

    for road in roads.itertuples(index=False):
        label = (
            f"{road.source}->{road.destination}"
        )

        key = (
            road.source,
            road.traffic_location,
        )

        matching_records = data.loc[
            (
                data["SCATS Number"]
                == road.source
            )
            & (
                data["Location"]
                == road.traffic_location
            )
        ]

        if matching_records.empty:
            errors.append(
                f"{label}: unknown source measurement"
            )

        if key not in model_lookup:
            errors.append(
                f"{label}: no model metadata"
            )

        else:
            model_path = (
                base
                / "saved_models"
                / model_lookup[key]["filename"]
            )

            if not model_path.is_file():
                errors.append(
                    f"{label}: model file missing"
                )

        if (
            not np.isfinite(road.distance_km)
            or road.distance_km <= 0
        ):
            errors.append(
                f"{label}: invalid distance"
            )

    duplicate_pairs = int(
        roads.duplicated(
            ["source", "destination"]
        ).sum()
    )

    if duplicate_pairs:
        errors.append(
            f"{duplicate_pairs} duplicate directed pairs"
        )

    summary = {
        "dataset_sites": len(dataset_sites),
        "configured_sites": len(configured_sites),
        "directed_edges": len(roads),
        "unconfigured_sites": sorted(
            set(dataset_sites) - configured_sites
        ),
        "internally_valid": not errors,
        "errors": errors,
        "warning": (
            "Connections and permitted directions remain "
            "unverified; passing this audit does not "
            "verify real roads."
        ),
    }

    # Save evidence.
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        coverage_rows
    ).to_csv(
        args.output / "site_coverage.csv",
        index=False,
    )

    pd.DataFrame(
        measurement_issues,
        columns=[
            "site",
            "location",
            "duplicate_date_rows",
            "invalid_counts",
        ],
    ).to_csv(
        args.output / "measurement_issues.csv",
        index=False,
    )

    summary_text = json.dumps(
        summary,
        indent=2,
    )

    (
        args.output / "summary.json"
    ).write_text(
        summary_text + "\n",
        encoding="utf-8",
    )

    print(summary_text)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()