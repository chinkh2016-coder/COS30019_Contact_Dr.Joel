"""Run once after installing the missing-observation-aware runtime and GUI."""
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def main():
    base = Path(__file__).resolve().parent
    # Ensure that the newly installed runtime supports missing-data notices.
    from traffic_runtime import TrafficSystem
    system = TrafficSystem()
    if not hasattr(system, 'last_query_warnings'):
        raise RuntimeError('Install the new traffic_runtime.py before updating roads.')
    metadata = json.loads((base/'saved_models'/'metadata.json').read_text())
    for location in ['WARRIGAL_RD N of HIGHBURY_RD', 'WARRIGAL_RD S of HIGHBURY_RD']:
        matches = [r for r in metadata if int(r['site']) == 3685 and r['location'] == location]
        if len(matches) != 1 or not (base/'saved_models'/matches[0]['filename']).is_file():
            raise ValueError(f'Missing or ambiguous model: {location}')
    path = base/'roads.csv'
    original = path.read_bytes()
    roads = pd.read_csv(path)
    roads = roads.loc[~((roads.source == 2000) & (roads.destination == 970))].copy()
    links = pd.DataFrame([
        (2000, 3685, .333, 'WARRIGAL_RD S of BURWOOD_HWY'),
        (3685, 970, 1.395, 'WARRIGAL_RD S of HIGHBURY_RD'),
    ], columns=['source','destination','distance_km','traffic_location'])
    roads = pd.concat([roads, links], ignore_index=True).drop_duplicates(['source','destination'],keep='last')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (base/f'roads_before_reverse_{stamp}.csv').write_bytes(original)
    tmp = base/'roads_pending.csv'
    roads.to_csv(tmp, index=False)
    tmp.replace(path)
    print('Roads saved:',len(roads))
    print('Reverse link split: 2000 -> 3685 -> 970')
    print('Distances remain straight-line estimates; measurement assignments remain prototype proxies.')


if __name__ == '__main__':
    main()
