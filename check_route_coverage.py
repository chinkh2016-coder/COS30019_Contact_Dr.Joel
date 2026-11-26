"""
Export directed reachability for dataset-site pairs.

Run:
    python3 check_route_coverage.py

This checks the configured graph.
It does not verify real-world road connectivity.
It does not retrain models or modify roads.
"""

from pathlib import Path

import pandas as pd

from network_validation import nodes_reaching
from traffic_runtime import TrafficSystem


def export_coverage(system, dates, output):
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    sites = sorted(
        set(
            system.data["SCATS Number"].astype(int)
        )
    )

    # Build a graph without considering missing daily observations.
    configured_graph = {
        site: []
        for site in sites
    }

    for road in system.roads.itertuples(index=False):
        configured_graph[road.source].append(
            (
                road.destination,
                1.0,
            )
        )

    # For each destination, find all origins that can reach it.
    configured_reachable = {
        destination: nodes_reaching(
            configured_graph,
            destination,
        )
        for destination in sites
    }

    summary_rows = []
    pair_rows = []

    for date in dates:
        # Use V04 for this coverage analysis.
        available_graph = system.build_graph(
            date,
            4,
        )

        available_reachable = {
            destination: nodes_reaching(
                available_graph,
                destination,
            )
            for destination in sites
        }

        counts = {}

        for origin in sites:
            for destination in sites:
                # Do not count same-site trips.
                if origin == destination:
                    continue

                if (
                    origin not in system.stations
                    or destination not in system.stations
                ):
                    status = "SITE_UNCONFIGURED"

                elif (
                    origin
                    not in configured_reachable[destination]
                ):
                    status = "NO_CONFIGURED_PATH"

                elif (
                    origin
                    not in available_reachable[destination]
                ):
                    status = "NO_OBSERVED_PATH"

                else:
                    status = "AVAILABLE"

                counts[status] = (
                    counts.get(status, 0) + 1
                )

                pair_rows.append(
                    {
                        "date": date,
                        "origin": origin,
                        "destination": destination,
                        "status": status,
                    }
                )

        summary_rows.append(
            {
                "date": date,
                "distinct_site_pairs": (
                    len(sites) * (len(sites) - 1)
                ),
                **counts,
            }
        )

    # Save every origin-destination pair.
    pd.DataFrame(
        pair_rows
    ).to_csv(
        output / "route_pair_coverage.csv",
        index=False,
    )

    # Save totals for each date.
    summary_table = pd.DataFrame(
        summary_rows
    ).fillna(0)

    summary_table.to_csv(
        output / "route_coverage_summary.csv",
        index=False,
    )

    print(
        summary_table.to_string(index=False)
    )

    return summary_rows


if __name__ == "__main__":
    project_folder = (
        Path(__file__).resolve().parent
    )

    system = TrafficSystem()

    export_coverage(
        system,
        [
            "2006-10-20",
            "2006-10-25",
            "2006-10-29",
        ],
        project_folder / "network_audit",
    )