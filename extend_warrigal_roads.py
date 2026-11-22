"""Add the two prototype links 3682 <-> 3126 without retraining models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd


def main():
    base = Path(__file__).resolve().parent
    model_dir = base/'saved_models'
    metadata_path = model_dir/'metadata.json'
    metadata_original = metadata_path.read_text(encoding='utf-8')
    metadata = json.loads(metadata_original)
    required = [
        (3682, 'WARRIGAL_RD N OF RIVERSDALE_RD'),
        (3126, 'WARRIGAL_RD S of CANTERBURY_RD'),
    ]
    corrections = []
    for site, location in required:
        matches = [item for item in metadata if int(item['site']) == site and item['location'] == location]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one saved model: {site}, {location}')
        item = matches[0]
        if not (model_dir/item['filename']).is_file():
            raise FileNotFoundError(item['filename'])
        # Repair the old script's fixed explanatory note, not model weights.
        if item.get('test_start') == '2006-10-25' and item.get('test_end_exclusive') == '2006-11-01':
            item['evaluation_note'] = 'Same October 25-31 evaluation period as the original benchmark.'
            corrections.append(item['filename'])
    roads_path = base/'roads.csv'
    roads_original = roads_path.read_bytes()
    roads = pd.read_csv(roads_path)
    if not {'source','destination','distance_km','traffic_location'}.issubset(roads.columns):
        raise ValueError('Unexpected roads.csv format.')
    if not {970,2000,3685,3682}.issubset(set(roads.source)|set(roads.destination)):
        raise ValueError('The expected previous road network is missing.')
    # Spherical straight-line distance from valid workbook coordinate centres,
    # rounded to 0.001 km. This is a prototype estimate, not a surveyed road length.
    additions = pd.DataFrame([
        (3682,3126,1.081,'WARRIGAL_RD N OF RIVERSDALE_RD'),
        (3126,3682,1.081,'WARRIGAL_RD S of CANTERBURY_RD'),
    ], columns=['source','destination','distance_km','traffic_location'])
    for row in additions.itertuples(index=False):
        existing = roads[(roads.source==row.source)&(roads.destination==row.destination)]
        if not existing.empty and not ((existing.distance_km==row.distance_km)&(existing.traffic_location==row.traffic_location)).all():
            raise ValueError(f'A different link already exists for {row.source} -> {row.destination}.')
    updated = pd.concat([roads,additions],ignore_index=True).drop_duplicates(['source','destination'],keep='last')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (base/f'roads_before_3126_{stamp}.csv').write_bytes(roads_original)
    temporary = base/'roads_pending_3126.csv'
    updated.to_csv(temporary,index=False)
    temporary.replace(roads_path)
    if corrections:
        (model_dir/f'metadata_before_note_fix_{stamp}.json').write_text(metadata_original,encoding='utf-8')
        temporary_meta = model_dir/'metadata_pending_note_fix.json'
        temporary_meta.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        temporary_meta.replace(metadata_path)
        # Fix the explanatory note in per-model evaluation exports too.
        for filename in corrections:
            metrics_path = model_dir/f"evaluation_{Path(filename).stem.removeprefix('model_')}.json"
            if metrics_path.is_file():
                metrics_original = metrics_path.read_text(encoding='utf-8')
                metrics = json.loads(metrics_original)
                if metrics.get('test_start') == '2006-10-25' and metrics.get('test_end_exclusive') == '2006-11-01':
                    metrics['evaluation_note'] = 'Same October 25-31 evaluation period as the original benchmark.'
                    metrics_path.with_name(metrics_path.stem+f'_before_note_fix_{stamp}.json').write_text(metrics_original,encoding='utf-8')
                    metrics_path.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Roads saved:', len(updated))
    print('Stations in road graph:', len(set(updated.source)|set(updated.destination)))
    print('Added: 3682 -> 3126 and 3126 -> 3682')
    print('No models retrained. Existing links retained.')
    print('Distance remains an approximation; source-location assignments remain prototype proxies.')


if __name__=='__main__':
    main()
