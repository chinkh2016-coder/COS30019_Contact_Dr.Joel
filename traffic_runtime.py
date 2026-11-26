import json
import math
from datetime import datetime
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from problem import Problem
from algorithms import solve
from network_validation import can_reach, nodes_reaching


BASE_DIR = Path(__file__).resolve().parent

VOLUME_COLUMNS = [f"V{i:02d}" for i in range(96)]

SUPPORTED_METHODS = {
    "TOP5",
    "BFS",
    "AS",
    "CUS1",
    "CUS2",
}


class TrafficSystem:
    def __init__(self):
        # Load default settings.
        self.config = json.loads(
            (BASE_DIR / "config.json").read_text(
                encoding="utf-8"
            )
        )

        # Load the configured directed road network.
        self.roads = pd.read_csv(
            BASE_DIR / "roads.csv"
        )

        required_columns = {
            "source",
            "destination",
            "distance_km",
            "traffic_location",
        }

        if not required_columns.issubset(self.roads.columns):
            raise ValueError(
                "roads.csv is missing required columns."
            )

        if self.roads.empty:
            raise ValueError(
                "roads.csv contains no roads."
            )

        # Validate station IDs before converting to integers.
        for column in ["source", "destination"]:
            values = pd.to_numeric(
                self.roads[column],
                errors="raise",
            )

            if (
                not np.isfinite(
                    values.to_numpy(dtype=float)
                ).all()
                or (values < 0).any()
                or (values % 1 != 0).any()
            ):
                raise ValueError(
                    f"Invalid station IDs in {column}."
                )

            self.roads[column] = values.astype(int)

        # Validate road distances.
        self.roads["distance_km"] = pd.to_numeric(
            self.roads["distance_km"],
            errors="raise",
        )

        distances = self.roads["distance_km"].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(distances).all()
            or (distances <= 0).any()
        ):
            raise ValueError(
                "Road distances must be positive and finite."
            )

        # Every road must reference a measurement location.
        if (
            self.roads["traffic_location"].isna().any()
            or self.roads["traffic_location"]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Every road needs a traffic measurement location."
            )

        # Reject duplicate connections in the same direction.
        if self.roads.duplicated(
            ["source", "destination"]
        ).any():
            raise ValueError(
                "Duplicate directed road pairs in roads.csv."
            )

        # Load historical traffic observations.
        self.data = pd.read_excel(
            BASE_DIR / "Scats Data October 2006.xls",
            sheet_name="Data",
            header=1,
        )

        self.data["SCATS Number"] = pd.to_numeric(
            self.data["SCATS Number"],
            errors="raise",
        ).astype(int)

        self.data["Date"] = pd.to_datetime(
            self.data["Date"],
            errors="raise",
        )

        self.data = self.data.copy()

        # Remove the time component for daily comparisons.
        self.data["_day"] = (
            self.data["Date"].dt.normalize()
        )

        self.stations = sorted(
            set(self.roads["source"])
            | set(self.roads["destination"])
        )

        # Load saved models and matching scaling parameters.
        model_folder = BASE_DIR / "saved_models"

        metadata = json.loads(
            (model_folder / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        self.models = {}
        self.last_query_warnings = []

        for item in metadata:
            key = (
                int(item["site"]),
                item["location"],
            )

            minimum = float(item["minimum"])
            scale_range = float(item["scale_range"])

            if item.get("window_size", 4) != 4:
                raise ValueError(
                    f"Unsupported input window for {key}."
                )

            if not math.isfinite(minimum):
                raise ValueError(
                    f"Invalid minimum for {key}."
                )

            if (
                not math.isfinite(scale_range)
                or scale_range <= 0
            ):
                raise ValueError(
                    f"Invalid scaling range for {key}."
                )

            if key in self.models:
                raise ValueError(
                    f"Duplicate model metadata for {key}."
                )

            self.models[key] = (
                load_model(
                    model_folder / item["filename"],
                    compile=False,
                ),
                minimum,
                scale_range,
            )

        # Check that each configured road has a saved model.
        for road in self.roads.itertuples(index=False):
            key = (
                road.source,
                road.traffic_location,
            )

            if key not in self.models:
                raise ValueError(
                    f"No saved model for {key}."
                )

        # Calculate representative station coordinates.
        # Zero coordinates mean missing coordinates, not (0, 0).
        positions = self.data[
            [
                "SCATS Number",
                "Location",
                "NB_LONGITUDE",
                "NB_LATITUDE",
            ]
        ].copy()

        for column in [
            "NB_LONGITUDE",
            "NB_LATITUDE",
        ]:
            positions[column] = pd.to_numeric(
                positions[column],
                errors="coerce",
            )

        valid_positions = (
            positions["NB_LATITUDE"].between(-39, -37)
            & positions["NB_LONGITUDE"].between(144, 146)
        )

        positions = positions.loc[valid_positions]

        # Give each measurement location equal weight.
        per_location = positions.groupby(
            ["SCATS Number", "Location"]
        )[
            ["NB_LONGITUDE", "NB_LATITUDE"]
        ].mean()

        self.station_coords = per_location.groupby(
            level="SCATS Number"
        ).mean()

        for station in self.stations:
            if station not in self.station_coords.index:
                raise ValueError(
                    f"No coordinates for station {station}."
                )

            coordinates = (
                self.station_coords.loc[station]
                .to_numpy(dtype=float)
            )

            if not np.isfinite(coordinates).all():
                raise ValueError(
                    f"Invalid coordinates for station {station}."
                )

    @staticmethod
    def parse_date(date_text):
        # Require YYYY-MM-DD.
        try:
            parsed = datetime.strptime(
                date_text,
                "%Y-%m-%d",
            )

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Enter a valid date in YYYY-MM-DD format."
            ) from error

        if parsed.strftime("%Y-%m-%d") != date_text:
            raise ValueError(
                "Use YYYY-MM-DD, for example 2006-10-25."
            )

        return pd.Timestamp(parsed)

    @staticmethod
    def travel_seconds(distance_km, hourly_flow):
        # Use the under-capacity branch of the traffic equation.
        if (
            not math.isfinite(hourly_flow)
            or not 0 <= hourly_flow <= 1500
        ):
            raise ValueError(
                "Predicted hourly flow is outside the supported "
                "0–1500 vehicles/hour range."
            )

        if (
            not math.isfinite(distance_km)
            or distance_km <= 0
        ):
            raise ValueError(
                "Road distance must be positive and finite."
            )

        a = 1.4648375
        b = 93.75

        discriminant = (
            b * b - 4 * a * hourly_flow
        )

        speed_kmh = (
            b + math.sqrt(discriminant)
        ) / (2 * a)

        speed_kmh = min(speed_kmh, 60.0)

        # Convert hours to seconds and add intersection delay.
        return (
            distance_km / speed_kmh * 3600
            + 30
        )

    def predict_count(
        self,
        site,
        location,
        selected_date,
        target_slot,
    ):
        rows = self.data.loc[
            (self.data["SCATS Number"] == site)
            & (self.data["Location"] == location)
            & (self.data["_day"] == selected_date)
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected one record for {site}, {location}, "
                f"on {selected_date.date()}. Found {len(rows)}."
            )

        key = (site, location)

        if key not in self.models:
            raise ValueError(
                f"No saved model for {site}: {location}"
            )

        model, minimum, scale_range = self.models[key]

        # V04 uses V00–V03.
        # V05 uses V01–V04, and so on.
        columns = VOLUME_COLUMNS[
            target_slot - 4:target_slot
        ]

        counts = rows.iloc[0][columns].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
        ):
            raise ValueError(
                f"Invalid historical traffic for "
                f"{site}: {location}"
            )

        inputs = (
            (counts - minimum) / scale_range
        ).reshape(1, 4, 1)

        # Direct inference avoids repeated predict() tracing.
        scaled_prediction = np.asarray(
            model(inputs, training=False)
        ).item()

        # Convert the scaled prediction back to vehicle counts.
        prediction = (
            scaled_prediction * scale_range
            + minimum
        )

        if (
            not math.isfinite(prediction)
            or prediction < 0
        ):
            raise ValueError(
                "The model produced an invalid traffic "
                f"count for {site}."
            )

        return float(prediction)

    def build_graph(self, date_text, target_slot):
        if (
            type(target_slot) is not int
            or not 4 <= target_slot <= 95
        ):
            raise ValueError(
                "Select a prediction column from V04 to V95."
            )

        selected_date = self.parse_date(date_text)

        self.last_query_warnings = []

        if not self.data["_day"].eq(selected_date).any():
            raise ValueError(
                "No observations are available "
                "for the selected date."
            )

        graph = {
            station: []
            for station in self.stations
        }

        prediction_cache = {}

        for road in self.roads.itertuples(index=False):
            # Current implementation uses source-side traffic.
            key = (
                road.source,
                road.traffic_location,
            )

            rows = self.data.loc[
                (self.data["SCATS Number"] == road.source)
                & (
                    self.data["Location"]
                    == road.traffic_location
                )
                & (self.data["_day"] == selected_date)
            ]

            if rows.empty:
                self.last_query_warnings.append(
                    f"{road.source} -> {road.destination}: "
                    "no observations for "
                    f"{road.traffic_location} "
                    f"on {selected_date.date()}."
                )
                continue

            # Predict once per measurement location per query.
            if key not in prediction_cache:
                prediction_cache[key] = self.predict_count(
                    road.source,
                    road.traffic_location,
                    selected_date,
                    target_slot,
                )

            predicted_count = prediction_cache[key]

            # Approximate hourly flow from the 15-minute count.
            seconds = self.travel_seconds(
                road.distance_km,
                predicted_count * 4,
            )

            graph[road.source].append(
                (road.destination, seconds)
            )

        return graph

    def make_search_problem(
        self,
        graph,
        origin,
        destination,
    ):
        reference_lat = float(
            self.station_coords["NB_LATITUDE"].mean()
        )

        reference_lon = float(
            self.station_coords["NB_LONGITUDE"].mean()
        )

        coords = {}

        for station in graph:
            lon = float(
                self.station_coords.loc[
                    station,
                    "NB_LONGITUDE",
                ]
            )

            lat = float(
                self.station_coords.loc[
                    station,
                    "NB_LATITUDE",
                ]
            )

            # Approximate displacement in kilometres,
            # then convert to seconds at 60 km/h.
            x = (
                (lon - reference_lon)
                * 111.195
                * math.cos(math.radians(reference_lat))
                * 60
            )

            y = (
                (lat - reference_lat)
                * 111.195
                * 60
            )

            coords[station] = (x, y)

        # Bound heuristic distances by the actual edge costs.
        scale = 1.0

        for source, outgoing in graph.items():
            for target, seconds in outgoing:
                straight_time = math.dist(
                    coords[source],
                    coords[target],
                )

                if straight_time > 0:
                    scale = min(
                        scale,
                        seconds / straight_time,
                    )

        coords = {
            station: (
                x * scale,
                y * scale,
            )
            for station, (x, y) in coords.items()
        }

        return Problem(
            coords=coords,
            adj=graph,
            origin=origin,
            destinations=[destination],
        )

    @staticmethod
    def top_routes(
        graph,
        origin,
        destination,
        limit,
    ):
        # Identify nodes that can reach the destination.
        viable = nodes_reaching(
            graph,
            destination,
        )

        if origin not in viable:
            return []

        # Enumerate simple paths in increasing travel time.
        queue = [
            (0.0, (origin,))
        ]

        results = []
        returned_paths = set()

        while queue and len(results) < limit:
            seconds, path = heappop(queue)

            current = path[-1]

            if current == destination:
                if path not in returned_paths:
                    results.append(
                        (path, seconds)
                    )
                    returned_paths.add(path)

                continue

            for next_station, edge_seconds in graph[current]:
                if (
                    next_station in viable
                    and next_station not in path
                ):
                    heappush(
                        queue,
                        (
                            seconds + edge_seconds,
                            path + (next_station,),
                        ),
                    )

        return results

    def query_selected(
        self,
        origin,
        destination,
        date_text,
        target_slot,
        method="TOP5",
        limit=5,
    ):
        self.last_query_warnings = []
        self.last_route_status = "NOT_QUERIED"

        if method not in SUPPORTED_METHODS:
            raise ValueError(
                "Unknown search algorithm."
            )

        if (
            origin not in self.stations
            or destination not in self.stations
        ):
            raise ValueError(
                "Select stations in the current road network."
            )

        if (
            type(limit) is not int
            or not 1 <= limit <= 5
        ):
            raise ValueError(
                "The route limit must be between 1 and 5."
            )

        if (
            type(target_slot) is not int
            or not 4 <= target_slot <= 95
        ):
            raise ValueError(
                "Select a prediction column from V04 to V95."
            )

        # Validate the date even when origin equals destination.
        selected_date = self.parse_date(date_text)

        if not self.data["_day"].eq(selected_date).any():
            raise ValueError(
                "No observations are available "
                "for the selected date."
            )

        if origin == destination:
            self.last_route_status = "SAME_SITE"

            return [
                ((origin,), 0.0)
            ]

        # First check the configured network without traffic.
        # The weight of 1 only checks whether a path exists.
        configured = {
            site: []
            for site in self.stations
        }

        for road in self.roads.itertuples(index=False):
            configured[road.source].append(
                (road.destination, 1.0)
            )

        if not can_reach(
            configured,
            origin,
            destination,
        ):
            self.last_route_status = "NO_CONFIGURED_PATH"
            return []

        # Build the graph using the selected date's observations.
        graph = self.build_graph(
            date_text,
            target_slot,
        )

        if not can_reach(
            graph,
            origin,
            destination,
        ):
            self.last_route_status = "NO_OBSERVED_PATH"
            return []

        self.last_route_status = "AVAILABLE"

        if method == "TOP5":
            return self.top_routes(
                graph,
                origin,
                destination,
                limit,
            )

        # Reuse the A2A search algorithms.
        problem = self.make_search_problem(
            graph,
            origin,
            destination,
        )

        result = solve(
            problem,
            method,
        )

        if result is None:
            return []

        return [
            (
                tuple(result.path),
                float(result.cost),
            )
        ]

    def query(
        self,
        origin,
        destination,
        date_text,
        target_slot,
        limit=5,
    ):
        # Preserve calls used by the existing notebook.
        return self.query_selected(
            origin,
            destination,
            date_text,
            target_slot,
            method="TOP5",
            limit=limit,
        )


if __name__ == "__main__":
    system = TrafficSystem()

    settings = system.config

    routes = system.query(
        origin=settings["origin"],
        destination=settings["destination"],
        date_text=settings["date"],
        target_slot=settings["target_slot"],
        limit=settings["max_routes"],
    )

    for warning in system.last_query_warnings:
        print(
            "DATA WARNING:",
            warning,
        )

    if not routes:
        print(
            "No route found in the graph "
            "supported by the selected observations."
        )

    for rank, (path, seconds) in enumerate(
        routes,
        start=1,
    ):
        route_text = " → ".join(
            map(str, path)
        )

        print(
            f"{rank}. {route_text} | "
            f"{seconds:.2f} seconds"
        )
