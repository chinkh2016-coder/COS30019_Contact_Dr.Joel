"""Tkinter route guidance with an offline road graph."""

import math
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor

from traffic_runtime import TrafficSystem
from traffic_map_view import TrafficMapView


ALGORITHM_OPTIONS = {
    "Up to 5 fastest available routes (UCS)": "TOP5",
    "BFS - one route": "BFS",
    "A* - one route": "AS",
    "CUS1 / UCS - one route": "CUS1",
    "CUS2 - one route": "CUS2",
}


def project_points(coordinates, width, height, padding=48):
    """Fit coordinates to the canvas with north pointing up."""
    if not coordinates:
        return {}

    if not all(
        math.isfinite(value)
        for pair in coordinates.values()
        for value in pair
    ):
        raise ValueError(
            "Cannot draw non-finite station coordinates."
        )

    reference_lat = sum(
        lat for lon, lat in coordinates.values()
    ) / len(coordinates)

    longitude_scale = math.cos(
        math.radians(reference_lat)
    )

    points = {
        site: (lon * longitude_scale, lat)
        for site, (lon, lat) in coordinates.items()
    }

    xs, ys = zip(*points.values())

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span_x = max_x - min_x
    span_y = max_y - min_y

    usable_width = max(1, width - 2 * padding)
    usable_height = max(1, height - 2 * padding)

    scales = []

    if span_x > 0:
        scales.append(usable_width / span_x)

    if span_y > 0:
        scales.append(usable_height / span_y)

    scale = min(scales) if scales else 1.0

    return {
        site: (
            width / 2
            + (x - (min_x + max_x) / 2) * scale,
            height / 2
            - (y - (min_y + max_y) / 2) * scale,
        )
        for site, (x, y) in points.items()
    }


class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.closed = False
        self.system = None
        self.coordinates = {}
        self.roads = []
        self.routes = []
        self.missing = set()
        self.selected_route = None
        self.completed_query = None

        root.title("Traffic-based Route Guidance")

        window_width = min(
            1500, root.winfo_screenwidth() - 80
        )
        window_height = min(
            960, root.winfo_screenheight() - 100
        )

        root.geometry(
            f"{window_width}x{window_height}"
        )
        root.minsize(950, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.origin = tk.StringVar()
        self.destination = tk.StringVar()
        self.date = tk.StringVar()
        self.slot = tk.StringVar()

        self.method = tk.StringVar(
            value=next(iter(ALGORITHM_OPTIONS))
        )
        self.limit = tk.StringVar(value="5")

        self.status = tk.StringVar(
            value="Loading saved models..."
        )
        self.map_caption = tk.StringVar(
            value="Configured road network"
        )
        self.route_caption = tk.StringVar(
            value="Select a result to highlight its route."
        )

        self.build_widgets()

        future = self.executor.submit(self.load_system)
        self.poll_loading(future)

    def build_widgets(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill="x")

        self.station_boxes = []

        fields = [
            ("Origin SCATS", self.origin),
            ("Destination SCATS", self.destination),
        ]

        for column, (label, variable) in enumerate(fields):
            ttk.Label(
                controls, text=label
            ).grid(
                row=0,
                column=column,
                sticky="w",
            )

            box = ttk.Combobox(
                controls,
                textvariable=variable,
                state="readonly",
                width=16,
            )
            box.grid(
                row=1,
                column=column,
                sticky="ew",
                padx=(0, 12),
                pady=(5, 12),
            )
            self.station_boxes.append(box)

        ttk.Label(
            controls,
            text="Date (YYYY-MM-DD)",
        ).grid(
            row=0,
            column=2,
            sticky="w",
        )

        ttk.Entry(
            controls,
            textvariable=self.date,
            width=18,
        ).grid(
            row=1,
            column=2,
            sticky="ew",
            padx=(0, 12),
            pady=(5, 12),
        )

        ttk.Label(
            controls,
            text="Prediction column",
        ).grid(
            row=0,
            column=3,
            sticky="w",
        )

        ttk.Combobox(
            controls,
            textvariable=self.slot,
            values=[
                f"V{i:02d}" for i in range(4, 96)
            ],
            state="readonly",
            width=12,
        ).grid(
            row=1,
            column=3,
            sticky="ew",
            pady=(5, 12),
        )

        ttk.Label(
            controls,
            text="Search method",
        ).grid(
            row=2,
            column=0,
            sticky="w",
        )

        ttk.Combobox(
            controls,
            textvariable=self.method,
            values=list(ALGORITHM_OPTIONS),
            state="readonly",
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=(0, 12),
            pady=(5, 8),
        )

        ttk.Label(
            controls,
            text="Maximum routes (TOP5)",
        ).grid(
            row=2,
            column=2,
            sticky="w",
        )

        ttk.Combobox(
            controls,
            textvariable=self.limit,
            values=["1", "2", "3", "4", "5"],
            state="readonly",
            width=10,
        ).grid(
            row=3,
            column=2,
            sticky="w",
            pady=(5, 8),
        )

        self.search_button = ttk.Button(
            controls,
            text="Find Routes",
            command=self.search,
            state="disabled",
        )
        self.search_button.grid(
            row=3,
            column=3,
            sticky="e",
            pady=(5, 8),
        )

        for column in range(4):
            controls.columnconfigure(
                column, weight=1
            )

        ttk.Label(
            outer,
            text=(
                "Each prediction uses the four preceding "
                "observed counts. Missing observations "
                "do not mean a road is closed."
            ),
            wraplength=1150,
        ).pack(
            anchor="w",
            pady=(0, 12),
        )

        panels = ttk.Panedwindow(
            outer, orient="horizontal"
        )
        panels.pack(fill="both", expand=True)

        left = ttk.Frame(
            panels, padding=(0, 0, 10, 0)
        )
        right = ttk.Frame(
            panels, padding=(10, 0, 0, 0)
        )

        panels.add(left, weight=1)
        panels.add(right, weight=2)

        ttk.Label(
            left,
            text="Routes - click a row to highlight",
        ).pack(anchor="w")

        route_frame = ttk.Frame(left)
        route_frame.pack(
            fill="x",
            pady=(6, 10),
        )

        self.tree = ttk.Treeview(
            route_frame,
            columns=("seconds",),
            show="tree headings",
            selectmode="browse",
            height=6,
        )

        self.tree.heading(
            "#0", text="Rank / SCATS route"
        )
        self.tree.heading(
            "seconds", text="Seconds"
        )

        self.tree.column(
            "#0",
            width=300,
            minwidth=160,
        )
        self.tree.column(
            "seconds",
            width=85,
            minwidth=70,
            stretch=False,
            anchor="e",
        )

        self.tree.pack(
            side="left",
            fill="both",
            expand=True,
        )

        route_scroll = ttk.Scrollbar(
            route_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        route_scroll.pack(
            side="right", fill="y"
        )

        self.tree.configure(
            yscrollcommand=route_scroll.set
        )
        self.tree.bind(
            "<<TreeviewSelect>>",
            self.select_route,
        )

        ttk.Label(
            left,
            text="Query details and data coverage",
        ).pack(anchor="w")

        detail_frame = ttk.Frame(left)
        detail_frame.pack(
            fill="both",
            expand=True,
            pady=(6, 0),
        )

        self.details = tk.Text(
            detail_frame,
            width=44,
            height=12,
            wrap="word",
            state="disabled",
        )
        self.details.pack(
            side="left",
            fill="both",
            expand=True,
        )

        detail_scroll = ttk.Scrollbar(
            detail_frame,
            command=self.details.yview,
        )
        detail_scroll.pack(
            side="right", fill="y"
        )

        self.details.configure(
            yscrollcommand=detail_scroll.set
        )

        ttk.Label(
            right,
            textvariable=self.map_caption,
            wraplength=650,
        ).pack(anchor="w")

        self.canvas = tk.Canvas(
            right,
            background="#f8fafb",
            highlightthickness=1,
            highlightbackground="#ccd6dd",
        )
        self.canvas.pack(
            fill="both",
            expand=True,
            pady=(6, 8),
        )

        # The separate module manages zoom, dragging and labels.
        self.map_view = TrafficMapView(
            right,
            self.canvas,
            project_points,
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.draw_network(),
        )

        ttk.Label(
            right,
            textvariable=self.route_caption,
            wraplength=650,
        ).pack(anchor="w")

        ttk.Label(
            right,
            text=(
                "Orange: selected route | "
                "Red dashed: missing observations\n"
                "Green node: origin | "
                "Purple node: destination\n"
                "Arrows: configured direction. "
                "Before a query, all links are grey."
            ),
            wraplength=650,
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        ttk.Label(
            right,
            text=(
                "Diagram uses approximate station "
                "coordinates and straight links. "
                "It is not a street map."
            ),
            wraplength=650,
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        ttk.Label(
            outer,
            textvariable=self.status,
            wraplength=1150,
        ).pack(
            anchor="w",
            pady=(12, 0),
        )

    @staticmethod
    def load_system():
        system = TrafficSystem()

        coordinates = {
            site: (
                float(
                    system.station_coords.loc[
                        site, "NB_LONGITUDE"
                    ]
                ),
                float(
                    system.station_coords.loc[
                        site, "NB_LATITUDE"
                    ]
                ),
            )
            for site in system.stations
        }

        roads = [
            (int(road.source), int(road.destination))
            for road in system.roads.itertuples(index=False)
        ]

        return system, coordinates, roads

    def poll_loading(self, future):
        if self.closed:
            return

        if not future.done():
            self.root.after(
                100, self.poll_loading, future
            )
            return

        try:
            (
                self.system,
                self.coordinates,
                self.roads,
            ) = future.result()

            station_values = [
                str(site)
                for site in self.system.stations
            ]

            for box in self.station_boxes:
                box.configure(values=station_values)

            self.origin.set(
                str(self.system.config["origin"])
            )
            self.destination.set(
                str(self.system.config["destination"])
            )
            self.date.set(
                self.system.config["date"]
            )
            self.slot.set(
                f"V{int(self.system.config['target_slot']):02d}"
            )
            self.limit.set(
                str(
                    self.system.config.get(
                        "max_routes", 5
                    )
                )
            )

            self.search_button.configure(
                state="normal"
            )
            self.status.set(
                f"Ready. {len(self.system.models)} "
                f"saved models; {len(self.coordinates)} "
                "configured stations."
            )

            self.draw_network()

        except Exception as error:
            self.display(str(error))
            self.status.set(
                "Could not load the project."
            )

    def display(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", text)
        self.details.configure(state="disabled")

    def clear_results(self):
        self.routes = []
        self.missing = set()
        self.selected_route = None
        self.completed_query = None

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.route_caption.set(
            "No selected route."
        )
        self.map_caption.set(
            "Configured road network - no completed query"
        )

        self.draw_network()

    def search(self):
        if self.system is None:
            return

        try:
            label = self.method.get()

            settings = (
                int(self.origin.get()),
                int(self.destination.get()),
                self.date.get().strip(),
                int(self.slot.get()[1:]),
                ALGORITHM_OPTIONS[label],
                int(self.limit.get()),
            )

        except (ValueError, KeyError):
            self.status.set(
                "Select valid query settings."
            )
            return

        self.clear_results()
        self.search_button.configure(
            state="disabled"
        )
        self.display(
            "Predicting traffic and finding routes..."
        )
        self.status.set("Working...")

        future = self.executor.submit(
            self.run_query, settings
        )

        self.root.after(
            100,
            self.poll_query,
            future,
            settings,
            label,
        )

    def run_query(self, settings):
        # Background work does not access GUI widgets.
        routes = self.system.query_selected(*settings)

        warnings = list(
            self.system.last_query_warnings
        )

        if settings[0] == settings[1]:
            return routes, warnings, set()

        day = self.system.parse_date(settings[2])

        available = self.system.data.loc[
            self.system.data["_day"] == day
        ]

        keys = set(
            zip(
                available["SCATS Number"].astype(int),
                available["Location"],
            )
        )

        missing = {
            (
                int(road.source),
                int(road.destination),
            )
            for road in self.system.roads.itertuples(
                index=False
            )
            if (
                int(road.source),
                road.traffic_location,
            ) not in keys
        }

        return routes, warnings, missing

    def poll_query(self, future, settings, label):
        if self.closed:
            return

        if not future.done():
            self.root.after(
                100,
                self.poll_query,
                future,
                settings,
                label,
            )
            return

        try:
            (
                self.routes,
                warnings,
                self.missing,
            ) = future.result()

            self.completed_query = settings

            lines = [
                f"Search method: {label}",
                (
                    f"{settings[0]} -> {settings[1]} | "
                    f"{settings[2]} | V{settings[3]:02d}"
                ),
            ]

            if warnings:
                lines.append(
                    "PARTIAL DATA: results use only roads "
                    "with observations on the selected day. "
                    "Missing data does not mean a road "
                    "is closed."
                )

            # Distinguish network gaps from missing daily data.
            if not self.routes:
                status = getattr(
                    self.system,
                    "last_route_status",
                    "",
                )

                if status == "NO_CONFIGURED_PATH":
                    lines.append(
                        "No directed path exists in the "
                        "configured network. "
                        "This does not prove that no "
                        "real-world road route exists."
                    )

                elif status == "NO_OBSERVED_PATH":
                    lines.append(
                        "A configured path exists, but "
                        "missing observations prevent "
                        "routing on this date. "
                        "Try another date."
                    )

                else:
                    lines.append(
                        "No route found in the available graph."
                    )

            for rank, (path, seconds) in enumerate(
                self.routes, start=1
            ):
                route = " -> ".join(map(str, path))

                self.tree.insert(
                    "",
                    "end",
                    iid=str(rank - 1),
                    text=f"{rank}. {route}",
                    values=(f"{seconds:.2f}",),
                )

                lines.append(
                    f"{rank}. {route}\n"
                    f"   {seconds:.2f} seconds "
                    f"({seconds / 60:.2f} minutes)"
                )

            if warnings:
                lines.append(
                    "Excluded road segments:\n"
                    + "\n".join(warnings)
                )

            self.display("\n\n".join(lines))

            self.map_caption.set(
                f"Query: {settings[0]} -> {settings[1]} | "
                f"{settings[2]} | V{settings[3]:02d}"
            )

            self.status.set(
                f"Found {len(self.routes)} route(s). "
                f"Excluded {len(self.missing)} configured "
                "road segment(s) with missing observations."
            )

            if self.routes:
                self.tree.selection_set("0")
                self.tree.focus("0")
                self.select_route()
            else:
                self.draw_network()

        except Exception as error:
            self.clear_results()
            self.display(str(error))
            self.status.set(
                "Query failed. Correct the input "
                "or check the data."
            )

        self.search_button.configure(
            state="normal"
        )

    def select_route(self, event=None):
        selection = self.tree.selection()

        if not selection:
            return

        index = int(selection[0])

        if index >= len(self.routes):
            return

        self.selected_route = index

        path, seconds = self.routes[index]

        self.route_caption.set(
            f"Route {index + 1}: "
            f"{' -> '.join(map(str, path))} | "
            f"{seconds:.2f} s "
            f"({seconds / 60:.2f} min)"
        )

        self.draw_network()

    def draw_network(self):
        self.map_view.draw(
            self.coordinates,
            self.roads,
            self.routes,
            self.selected_route,
            self.missing,
            self.completed_query,
        )

    def close(self):
        self.closed = True
        self.root.destroy()

    def shutdown(self):
        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


def main():
    root = tk.Tk()
    app = TrafficApp(root)

    try:
        root.mainloop()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
