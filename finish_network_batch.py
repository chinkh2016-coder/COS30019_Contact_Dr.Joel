"""Expand a documented prototype corridor graph, preserve regression fixtures, and export evidence.
This does NOT establish real road directions or complete Boroondara coverage.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

CORRIDORS = [(3180, 4057, 'BALWYN_RD'),
 (4057, 4063, 'BALWYN_RD'),
 (4063, 3127, 'BALWYN_RD'),
 (3001, 3002, 'BARKERS_RD'),
 (3002, 4035, 'BARKERS_RD'),
 (4032, 4057, 'BELMORE_RD'),
 (4043, 4040, 'BURKE_RD'),
 (4040, 3120, 'BURKE_RD'),
 (3120, 4035, 'BURKE_RD'),
 (4035, 4034, 'BURKE_RD'),
 (4034, 4032, 'BURKE_RD'),
 (4032, 4030, 'BURKE_RD'),
 (4030, 2825, 'BURKE_RD'),
 (4263, 4264, 'BURWOOD_RD'),
 (4264, 4266, 'BURWOOD_RD'),
 (4040, 3812, 'CAMBERWELL_RD'),
 (3120, 3122, 'CANTERBURY_RD'),
 (3122, 3127, 'CANTERBURY_RD'),
 (3127, 3126, 'CANTERBURY_RD'),
 (4034, 4324, 'COTHAM_RD'),
 (4051, 3180, 'DONCASTER_RD'),
 (4270, 4264, 'GLENFERRIE_RD'),
 (4264, 4324, 'GLENFERRIE_RD'),
 (4032, 4321, 'HARP_RD'),
 (3001, 3662, 'HIGH_ST'),
 (3662, 4335, 'HIGH_ST'),
 (4335, 4321, 'HIGH_ST'),
 (4321, 4030, 'HIGH_ST'),
 (2846, 970, 'HIGH_ST'),
 (3002, 4263, 'POWER_ST'),
 (2820, 3662, 'PRINCESS_ST'),
 (3682, 3804, 'RIVERSDALE_RD'),
 (3804, 4040, 'RIVERSDALE_RD'),
 (4040, 4272, 'RIVERSDALE_RD'),
 (4272, 4270, 'RIVERSDALE_RD'),
 (2000, 4043, 'TOORAK_RD'),
 (4043, 4273, 'TOORAK_RD'),
 (4272, 4273, 'TOORONGA_RD'),
 (3812, 3804, 'TRAFALGAR_RD'),
 (3126, 3682, 'WARRIGAL_RD'),
 (3682, 2000, 'WARRIGAL_RD'),
 (2000, 3685, 'WARRIGAL_RD'),
 (3685, 970, 'WARRIGAL_RD'),
 (4034, 4063, 'WHITEHORSE_RD'),
 (4063, 2200, 'WHITEHORSE_RD')]

COLUMNS = ['source', 'destination', 'distance_km', 'traffic_location']


def normalise(road):
    name = re.sub('[^A-Z0-9]', '', road.upper())
    return {'HIGHSTREETRD': 'HIGHST', 'MAROONDAHHWY': 'WHITEHORSERD'}.get(name, name)


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def run_logged(command, base, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=base, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-25:])
        raise RuntimeError(f'Command failed. Read {log_path.name}\n{tail}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    base = args.project_dir.resolve()
    roads_path = base / 'roads.csv'
    original = roads_path.read_bytes()
    roads = pd.read_csv(roads_path)
    if not set(COLUMNS).issubset(roads.columns) or roads.duplicated(['source','destination']).any():
        raise ValueError('Resolve roads.csv format or duplicate pairs first.')
    model_dir = base / 'saved_models'
    metadata_path = model_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    keys = [(int(m['site']),m['location']) for m in metadata]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate model metadata.')
    for record in metadata:
        if not (model_dir / record['filename']).is_file():
            raise FileNotFoundError(record['filename'])
    data = pd.read_excel(base/'Scats Data October 2006.xls',sheet_name='Data',header=1)
    data['SCATS Number'] = pd.to_numeric(data['SCATS Number'],errors='raise').astype(int)
    data['Date'] = pd.to_datetime(data['Date'],errors='raise').dt.normalize()
    valid = data[data.NB_LATITUDE.between(-39,-37) & data.NB_LONGITUDE.between(144,146)]
    centres = valid.groupby(['SCATS Number','Location'])[['NB_LONGITUDE','NB_LATITUDE']].mean().groupby(level=0).mean()
    locations = data.groupby('SCATS Number')['Location'].unique().to_dict()
    volumes = [f'V{i:02d}' for i in range(96)]

    bad_sources = {}
    for (site, location), group in data.groupby(['SCATS Number','Location']):
        values = group[volumes].to_numpy(dtype=float)
        if group.Date.duplicated().any():
            bad_sources[(int(site),location)] = 'Duplicate daily records; aggregation is not verified.'
        elif not np.isfinite(values).all() or (values<0).any():
            bad_sources[(int(site),location)] = 'Invalid count values.'

    def location_for(source,destination,road):
        a,b = centres.loc[source],centres.loc[destination]
        dx=(b.NB_LONGITUDE-a.NB_LONGITUDE)*math.cos(math.radians(a.NB_LATITUDE))
        dy=b.NB_LATITUDE-a.NB_LATITUDE
        options=[]
        for location in locations[source]:
            if (source,location) in bad_sources:
                continue
            match=re.match(r'^(.*?)\s+(NE|NW|SE|SW|N|S|E|W)\s+OF\s+',location,re.I)
            if not match or normalise(match[1]) != normalise(road):
                continue
            direction=match[2].upper()
            vx=int('E' in direction)-int('W' in direction)
            vy=int('N' in direction)-int('S' in direction)
            score=(dx*vx+dy*vy)/(math.hypot(dx,dy)*math.hypot(vx,vy))
            if score>.35:
                options.append((score,location))
        return max(options)[1] if options else None

    def distance(a,b):
        lon1,lat1=map(math.radians,centres.loc[a])
        lon2,lat2=map(math.radians,centres.loc[b])
        value=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        return round(6371*2*math.asin(math.sqrt(min(1,value))),3)

    existing=set(zip(roads.source,roads.destination))
    additions=[]
    mapping=[]
    for a,b,road in CORRIDORS:
        for source,destination in [(a,b),(b,a)]:
            if (source,destination) in existing:
                continue
            location=location_for(source,destination,road)
            if not location:
                mapping.append(dict(source=source,destination=destination,road=road,
                                    status='NO_USABLE_SOURCE_ARM_REVIEW_REQUIRED',location=''))
                continue
            additions.append((source,destination,distance(source,destination),location))
            existing.add((source,destination))
            mapping.append(dict(source=source,destination=destination,road=road,
                                status='PROTOTYPE_INFERRED_CORRIDOR',location=location))
    updated=pd.concat([roads,pd.DataFrame(additions,columns=COLUMNS)],ignore_index=True) if additions else roads.copy()
    required=sorted(set(zip(updated.source.astype(int),updated.traffic_location)))
    known=set(keys)
    missing=[key for key in required if key not in known]
    splits={}
    for site,location in missing:
        records=data[(data['SCATS Number']==site)&(data.Location==location)].sort_values('Date')
        if records.empty or records.Date.duplicated().any():
            raise ValueError(f'Missing or duplicate records: {site} | {location}')
        counts=records[volumes].to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts<0).any():
            raise ValueError(f'Invalid traffic: {site} | {location}')
        dates=records.Date
        bounds=['2006-10-21','2006-10-25','2006-11-01']
        masks=[dates<bounds[0],(dates>=bounds[0])&(dates<bounds[1]),(dates>=bounds[1])&(dates<bounds[2])]
        if not all(mask.any() for mask in masks):
            days=sorted(dates.unique())
            if len(days)<10:
                raise ValueError(f'Too few daily records for a safe split: {site} | {location}')
            part=max(2,round(len(days)*.15))
            bounds=[pd.Timestamp(days[-2*part]).strftime('%Y-%m-%d'),
                    pd.Timestamp(days[-part]).strftime('%Y-%m-%d'),
                    (pd.Timestamp(days[-1])+pd.Timedelta(days=1)).strftime('%Y-%m-%d')]
        splits[(site,location)]=bounds
    covered=set(updated.source)|set(updated.destination)
    uncovered=sorted(set(data['SCATS Number'])-covered)
    print('Existing roads:',len(roads),'| Added prototype links:',len(additions))
    print('Proposed roads:',len(updated),'| Covered stations:',len(covered),'of',data['SCATS Number'].nunique())
    print('Missing models to train:',len(missing))
    print('Unconfigured stations:',uncovered)
    print('WARNING: corridor inference and source-arm proxies are NOT verified real-world road directions.')
    if args.check_only:
        print('Check only: no files changed and no models trained.')
        return
    trainer=base/'train_location_model.py'
    if missing and not trainer.is_file():
        raise FileNotFoundError('train_location_model.py is required.')
    # Old exact-path tests retain their original graph; new tests use the expanded graph.
    old_tests=['test_traffic_runtime.py','test_canterbury_extension.py','test_burke_bridge.py','test_burke_north.py']
    patches={}
    for name in old_tests:
        path=base/name
        source=path.read_text(encoding='utf-8')
        if 'load_snapshot_system' in source:
            continue
        replacement,number=re.subn(r'(\bcls\.system\s*=\s*)TrafficSystem\(\)',
                                   r'\1load_snapshot_system()',source)
        if number!=1:
            raise ValueError(f'Cannot safely adapt {name}; no files changed yet.')
        patches[name]=(source,'from network_test_support import load_snapshot_system\n'+replacement)
    output=base/'batch_results'
    output.mkdir(exist_ok=True)
    for index,(site,location) in enumerate(missing,1):
        print(f'[{index}/{len(missing)}] Training {site} | {location}',flush=True)
        val,test,end=splits[(site,location)]
        command=[sys.executable,str(trainer),'--project-dir',str(base),'--site',str(site),'--location',location,
                 '--validation-start',val,'--test-start',test,'--test-end-exclusive',end]
        run_logged(command,base,output/f'train_{index:03d}_{site}.log')
        print('  Saved. Training details are in batch_results.',flush=True)
    records=json.loads(metadata_path.read_text(encoding='utf-8'))
    registry={(int(m['site']),m['location']):m for m in records}
    for key in required:
        if key not in registry or not (model_dir/registry[key]['filename']).is_file():
            raise RuntimeError(f'Model missing after training: {key}; roads not changed.')
    if roads_path.read_bytes()!=original:
        raise RuntimeError('roads.csv changed during training; review and run again.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    fixture=base/'regression_roads_snapshot.csv'
    if not fixture.exists():fixture.write_bytes(original)
    atomic_text(base/'network_test_support.py',SNAPSHOT_HELPER)
    for name,(old,new) in patches.items():
        path=base/name
        if path.read_text(encoding='utf-8')!=old:raise RuntimeError(f'{name} changed during training.')
        (output/f'{Path(name).stem}_before_{stamp}.txt').write_text(old,encoding='utf-8')
        atomic_text(path,new)
    if additions:
        (base/f'roads_before_batch_{stamp}.csv').write_bytes(original)
        temporary=base/'roads_pending_batch.csv'
        updated.to_csv(temporary,index=False)
        temporary.replace(roads_path)
    atomic_text(base/'test_network_batch.py',BATCH_TESTS)
    pd.DataFrame([{'site':site,'location':location,'reason':reason} for (site,location),reason in bad_sources.items()],columns=['site','location','reason']).to_csv(output/'invalid_measurements.csv',index=False)
    pd.DataFrame(mapping,columns=['source','destination','road','status','location']).to_csv(output/'connection_audit.csv',index=False)
    pd.DataFrame({'unconfigured_site':uncovered}).to_csv(output/'unconfigured_stations.csv',index=False)
    summary={'configured_stations':len(covered),'dataset_stations':int(data['SCATS Number'].nunique()),
             'directed_roads':len(updated),'saved_models':len(records),'unconfigured_stations':uncovered,
             'no_outgoing_mapping':sorted(covered-set(updated.source)),
             'no_incoming_mapping':sorted(covered-set(updated.destination)),
             'basis':'Prototype inferred corridors, straight-line estimates and source-side measurement proxies; not a complete or verified physical road network.',
             'legacy_tests':'Original scenarios run on regression_roads_snapshot.csv; test_network_batch uses current roads.csv.'}
    atomic_text(output/'network_summary.json',json.dumps(summary,indent=2))
    print('Running regression and expanded-network tests...',flush=True)
    run_logged([sys.executable,'-m','unittest',*[Path(name).stem for name in old_tests],'test_network_batch','-v'],
               base,output/'all_tests.txt')
    print('\n'.join((output/'all_tests.txt').read_text(encoding='utf-8').splitlines()[-5:]),flush=True)
    evaluator=base/'evaluate_models.py'
    if evaluator.exists():
        print('Evaluating saved models...',flush=True)
        run_logged([sys.executable,str(evaluator)],base,output/'evaluation.log')
    print(json.dumps(summary,indent=2))
    print('DONE. Restart traffic_gui.py. Review batch_results/connection_audit.csv and remaining gaps.')

SNAPSHOT_HELPER = "import pandas as pd\nfrom pathlib import Path\nfrom traffic_runtime import TrafficSystem\n\n\ndef load_snapshot_system():\n    system = TrafficSystem()\n    snapshot = Path(__file__).resolve().parent / 'regression_roads_snapshot.csv'\n    system.roads = pd.read_csv(snapshot)\n    system.stations = sorted(set(system.roads.source) | set(system.roads.destination))\n    return system\n"

BATCH_TESTS = "import heapq\nimport math\nimport unittest\nfrom collections import deque\nfrom traffic_runtime import TrafficSystem\n\n\nclass ExpandedNetworkTests(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls):\n        cls.system = TrafficSystem()\n\n    def test_01_unique_valid_links(self):\n        roads=self.system.roads\n        self.assertFalse(roads.duplicated(['source','destination']).any())\n        self.assertTrue(all(math.isfinite(float(d)) and d>0 for d in roads.distance_km))\n\n    def test_02_all_links_have_models(self):\n        for row in self.system.roads.itertuples():\n            self.assertIn((row.source,row.traffic_location),self.system.models)\n\n    def test_03_expanded_station_coverage(self):\n        self.assertTrue({2200,4057,4063,3180,4051,3001,3002,4263,4264,4324}.issubset(self.system.stations))\n\n    def test_04_same_station(self):\n        routes=self.system.query_selected(2200,2200,'2006-10-29',4)\n        self.assertEqual(tuple(routes[0][0]),(2200,))\n        self.assertEqual(routes[0][1],0)\n\n    def test_05_missing_data_warning(self):\n        self.system.build_graph('2006-10-25',4)\n        self.assertTrue(any('3685 -> 970:' in warning for warning in self.system.last_query_warnings))\n\n    def test_06_route_costs_ranking_and_cycles(self):\n        for origin,destination in [(970,2200),(4030,3126),(3122,3126)]:\n            with self.subTest(origin=origin,destination=destination):\n                routes=self.system.query_selected(origin,destination,'2006-10-29',4)\n                graph=self.system.build_graph('2006-10-29',4)\n                self.assertTrue(routes)\n                self.assertLessEqual(len(routes),5)\n                self.assertEqual([c for _,c in routes],sorted(c for _,c in routes))\n                self.assertEqual(len({tuple(p) for p,_ in routes}),len(routes))\n                for path,cost in routes:\n                    self.assertEqual(len(path),len(set(path)))\n                    self.assertEqual(path[0],origin)\n                    self.assertEqual(path[-1],destination)\n                    total=sum(next(c for node,c in graph[a] if node==b) for a,b in zip(path,path[1:]))\n                    self.assertAlmostEqual(cost,total,places=4)\n\n    @staticmethod\n    def shortest_cost(graph,start,goal):\n        best={start:0.0}\n        queue=[(0.0,start)]\n        while queue:\n            cost,node=heapq.heappop(queue)\n            if cost>best[node]:continue\n            if node==goal:return cost\n            for target,weight in graph[node]:\n                new=cost+weight\n                if new<best.get(target,math.inf):\n                    best[target]=new\n                    heapq.heappush(queue,(new,target))\n        return math.inf\n\n    def test_07_ucs_cost_matches_independent_solver(self):\n        graph=self.system.build_graph('2006-10-29',4)\n        expected=self.shortest_cost(graph,3120,2200)\n        self.assertTrue(math.isfinite(expected))\n        for method in ['CUS1','AS','TOP5']:\n            routes=self.system.query_selected(3120,2200,'2006-10-29',4,method,5)\n            self.assertTrue(routes)\n            self.assertAlmostEqual(routes[0][1],expected,places=4)\n\n    def test_08_bfs_fewest_edges(self):\n        graph=self.system.build_graph('2006-10-29',4)\n        queue=deque([(3120,0)])\n        seen={3120}\n        expected=None\n        while queue:\n            node,hops=queue.popleft()\n            if node==2200:\n                expected=hops\n                break\n            for target,_ in graph[node]:\n                if target not in seen:\n                    seen.add(target)\n                    queue.append((target,hops+1))\n        self.assertIsNotNone(expected)\n        routes=self.system.query_selected(3120,2200,'2006-10-29',4,'BFS',5)\n        self.assertTrue(routes)\n        self.assertEqual(len(routes[0][0])-1,expected)\n\n\nif __name__=='__main__':\n    unittest.main(verbosity=2)\n"

if __name__ == '__main__':
    main()
