"""Train one location without running notebook cells.
Default: site 3685 south, with its available October 1-22 records.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_samples(data, site, location):
    matching = data.loc[
        (pd.to_numeric(data['SCATS Number'], errors='raise') == site)
        & (data['Location'] == location)
    ].copy()
    if matching.empty:
        raise ValueError(f'No records for {site}: {location}')
    matching['Date'] = pd.to_datetime(matching['Date'], errors='raise').dt.normalize()
    matching = matching.sort_values('Date')
    if matching['Date'].duplicated().any():
        raise ValueError('Duplicate daily records at this location.')
    volumes = matching[[f'V{i:02d}' for i in range(96)]].to_numpy(dtype=np.float32)
    if not np.isfinite(volumes).all() or (volumes < 0).any():
        raise ValueError('Invalid traffic counts; no values have been imputed.')
    X, y, dates = [], [], []
    for day, counts in zip(matching['Date'], volumes):
        for end in range(4, 96):
            X.append(counts[end-4:end])
            y.append(counts[end])
            dates.append(day)
    return (np.asarray(X, dtype=np.float32)[..., np.newaxis],
            np.asarray(y, dtype=np.float32), pd.DatetimeIndex(dates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data-file', default='Scats Data October 2006.xls')
    parser.add_argument('--site', type=int, default=3685)
    parser.add_argument('--location', default='WARRIGAL_RD S of HIGHBURY_RD')
    parser.add_argument('--validation-start', default='2006-10-17')
    parser.add_argument('--test-start', default='2006-10-20')
    parser.add_argument('--test-end-exclusive', default='2006-10-23')
    parser.add_argument('--dry-run', action='store_true', help='Check data and splits without TensorFlow or saving.')
    args = parser.parse_args()
    validation_start = pd.Timestamp(args.validation_start)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end_exclusive)
    if not validation_start < test_start < test_end:
        raise ValueError('Require validation-start < test-start < test-end-exclusive.')
    data = pd.read_excel(args.project_dir / args.data_file, sheet_name='Data', header=1)
    X, y, dates = prepare_samples(data, args.site, args.location)
    train = dates < validation_start
    val = (dates >= validation_start) & (dates < test_start)
    test = (dates >= test_start) & (dates < test_end)
    print(f'Location: {args.site} | {args.location}')
    print(f'Available records: {dates.min().date()} to {dates.max().date()}')
    for label, mask in [('Training', train), ('Validation', val), ('Testing', test)]:
        if not mask.any():
            raise ValueError(f'No {label.lower()} samples; do not bypass this check.')
        print(f'{label} samples: {int(mask.sum())} ({dates[mask].min().date()} to {dates[mask].max().date()})')
    common_period = (
        test_start == pd.Timestamp('2006-10-25')
        and test_end == pd.Timestamp('2006-11-01')
    )
    evaluation_note = (
        'Same October 25-31 evaluation period as the original benchmark.'
        if common_period else
        'Different evaluation period; not directly comparable with October 25-31 models.'
    )
    print(evaluation_note)
    if args.dry_run:
        print('Data check passed. No model trained or files changed.')
        return

    model_dir = args.project_dir / 'saved_models'
    metadata_path = model_dir / 'metadata.json'
    original_text = metadata_path.read_text(encoding='utf-8')
    metadata = json.loads(original_text)
    existing = [item for item in metadata if int(item['site']) == args.site and item['location'] == args.location]
    if existing:
        if len(existing) != 1:
            raise ValueError('Duplicate metadata entries for this location.')
        if not (model_dir / existing[0]['filename']).is_file():
            raise FileNotFoundError('Metadata exists but its model file is missing.')
        print('A saved model already exists for this location. No retraining or overwrite performed.')
        print('Saved model records:', len(metadata))
        return

    import tensorflow as tf
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    minimum = float(min(X[train].min(), y[train].min()))
    scale_range = float(max(X[train].max(), y[train].max()) - minimum)
    if not np.isfinite(scale_range) or scale_range <= 0:
        raise ValueError('Invalid training scaling range.')
    tf.keras.utils.set_random_seed(42)
    model = Sequential([Input(shape=(4, 1)), LSTM(16), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        (X[train] - minimum) / scale_range,
        (y[train] - minimum) / scale_range,
        validation_data=((X[val] - minimum) / scale_range, (y[val] - minimum) / scale_range),
        epochs=30, batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=2,
    )
    predicted = np.asarray(model((X[test] - minimum) / scale_range, training=False)).ravel() * scale_range + minimum
    if not np.isfinite(predicted).all():
        raise ValueError('Model returned non-finite predictions.')
    actual = y[test].astype(float)
    baseline = X[test, -1, 0].astype(float)
    rmse = float(np.sqrt(np.mean((predicted-actual)**2)))
    baseline_rmse = float(np.sqrt(np.mean((baseline-actual)**2)))
    nrmse = rmse / float(actual.mean()) * 100 if actual.mean() > 0 else None
    token = hashlib.sha256(f'{args.site}|{args.location}'.encode()).hexdigest()[:12]
    filename = f'model_{args.site}_{token}.keras'
    model_path = model_dir / filename
    if model_path.exists():
        raise FileExistsError(f'Unregistered file already exists: {filename}')
    if metadata_path.read_text(encoding='utf-8') != original_text:
        raise RuntimeError('Model metadata changed during training; refusing to overwrite it.')
    model.save(model_path)
    entry = {
        'site': args.site, 'location': args.location, 'filename': filename,
        'minimum': minimum, 'scale_range': scale_range, 'window_size': 4,
        'validation_start': args.validation_start, 'test_start': args.test_start,
        'test_end_exclusive': args.test_end_exclusive,
        'evaluation_note': evaluation_note,
    }
    metadata.append(entry)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (model_dir / f'metadata_before_{args.site}_{stamp}.json').write_text(original_text, encoding='utf-8')
    temporary = model_dir / f'metadata_pending_{token}.json'
    temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    temporary.replace(metadata_path)
    metrics = {
        **entry, 'train_samples': int(train.sum()), 'validation_samples': int(val.sum()),
        'test_samples': int(test.sum()), 'LSTM_RMSE': rmse, 'Baseline_RMSE': baseline_rmse,
        'NRMSE_percent': nrmse, 'epochs_run': len(history.history['loss']),
        'best_validation_epoch': int(np.argmin(history.history['val_loss']))+1,
    }
    (model_dir / f'evaluation_{args.site}_{token}.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'LSTM RMSE ({test_start.date()} to {(test_end-pd.Timedelta(days=1)).date()}): {rmse:.2f}')
    print(f'Baseline RMSE: {baseline_rmse:.2f}')
    print('Saved model records:', len(metadata))
    print(f'Model ready: {args.site} {args.location}')
    print('No roads.csv changes.')
    if (dates == pd.Timestamp('2006-10-25')).any():
        print('October 25 observations are available for this location.')
    else:
        print('October 25 observations are unavailable for this location.')


if __name__ == '__main__':
    main()
