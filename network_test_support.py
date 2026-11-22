import pandas as pd
from pathlib import Path
from traffic_runtime import TrafficSystem


def load_snapshot_system():
    system = TrafficSystem()
    snapshot = Path(__file__).resolve().parent / 'regression_roads_snapshot.csv'
    system.roads = pd.read_csv(snapshot)
    system.stations = sorted(set(system.roads.source) | set(system.roads.destination))
    return system
