"""Add four Canterbury Road prototype links without retraining models."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

LINKS = [
    (3126,3127,1.835,'CANTERBURY_RD W of WARRIGAL_RD'),
    (3127,3126,1.835,'CANTERBURY_RD E of BALWYN_RD'),
    (3127,3122,1.200,'CANTERBURY_RD W of BALWYN_RD'),
    (3122,3127,1.200,'CANTERBURY_RD E of STANHOPE_GV'),
]


def main():
    base = Path(__file__).resolve().parent
    model_dir = base/'saved_models'
    metadata = json.loads((model_dir/'metadata.json').read_text(encoding='utf-8'))
    for source, _, _, location in LINKS:
        matches = [r for r in metadata if int(r['site'])==source and r['location']==location]
        if len(matches)!=1 or not (model_dir/matches[0]['filename']).is_file():
            raise ValueError(f'Saved model missing or ambiguous: {source}, {location}')
    path=base/'roads.csv'
    original=path.read_bytes()
    roads=pd.read_csv(path)
    required={'source','destination','distance_km','traffic_location'}
    if not required.issubset(roads.columns):
        raise ValueError('Missing roads.csv columns.')
    if 3126 not in set(roads.source)|set(roads.destination):
        raise ValueError('Complete the previous Warrigal extension first.')
    for source,target,distance,location in LINKS:
        existing=roads[(roads.source==source)&(roads.destination==target)]
        if not existing.empty and not ((existing.distance_km==distance)&(existing.traffic_location==location)).all():
            raise ValueError(f'Different configuration already exists for {source} -> {target}.')
    additions=pd.DataFrame(LINKS,columns=['source','destination','distance_km','traffic_location'])
    updated=pd.concat([roads,additions],ignore_index=True).drop_duplicates(['source','destination'],keep='last')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=base/f'roads_before_canterbury_{stamp}.csv'
    backup.write_bytes(original)
    temporary=base/'roads_pending_canterbury.csv'
    updated.to_csv(temporary,index=False)
    temporary.replace(path)
    print('Roads saved:',len(updated))
    print('Stations in road graph:',len(set(updated.source)|set(updated.destination)))
    print('Added: 3126 <-> 3127 <-> 3122')
    print('No models retrained. Existing links retained.')
    print('Backup:',backup.name)
    print('Distances are straight-line estimates; source-side measurement assignments remain prototype proxies.')
    print('Site 3127 has no observations on October 25-28. Use October 29 for a complete corridor demo.')


if __name__=='__main__':
    main()
