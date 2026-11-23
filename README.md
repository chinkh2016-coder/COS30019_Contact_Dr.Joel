# COS30019_Contact_Dr.Joel

# Traffic Flow Prediction and Route Guidance

COS30019 Assignment 2B 
## Team Member
Kian Xiu CHIN -
Kian Hee CHIN - 104392565

## Current recorded status

| Item | Result |
|---|---:|
| Dataset sites | 40 |
| Configured graph stations | 36 |
| Directed links | 87 |
| Saved location-specific LSTM models | 88 |
| Evaluations in the October 25–31 window | 86 |
| Locations with lower LSTM RMSE than persistence | 56 |
| Locations with higher LSTM RMSE | 30 |
| Recorded automated tests | 37 passed in 9.631 seconds |

These are observed results for the supplied development build. Passing tests does not independently verify road connectivity, measurement direction or real-world travel times.

## Required runtime files

Keep these in the same project folder:

- `traffic_gui.py` and `traffic_map_view.py`
- `traffic_runtime.py`
- A2A modules `problem.py`, `algorithms.py` and their imported dependencies, including `search.py` where used
- `config.json` and `roads.csv`
- `Scats Data October 2006.xls`
- `requirements.txt`
- `saved_models/metadata.json` and all model files named by that metadata

Normal queries load the saved models. They do not require retraining or running the notebook.

## Installation and launch

The recorded local environment uses Python 3.14.8, NumPy 2.5.3, pandas 3.0.6, TensorFlow 2.22.0rc0, Matplotlib 3.11.2 and xlrd 2.0.2. Use the supplied requirements file. Tkinter must be available in the Python installation; it is not a separate pip dependency. A clean-machine installation has not been demonstrated.

From the project folder:

```bash
python3 -m pip install -r requirements.txt
python3 traffic_gui.py
```

## Demonstration query

Select origin **3120**, destination **2200**, date **2006-10-29**, prediction column **V04**, method **TOP5**, maximum routes **5**.

The recorded build returns these estimated costs in order:

| Rank | Seconds |
|---|---:|
| 1 | 399.36 |
| 2 | 400.56 |
| 3 | 588.24 |
| 4 | 843.38 |
| 5 | 894.97 |

The first path is `3120 → 3122 → 3127 → 4063 → 2200`. The second is `3120 → 4035 → 4034 → 4063 → 2200`. Values can change after retraining or modifying the road configuration; they are not observed journey times.

Click a route to highlight it. Use **Large map**, **Zoom**, drag-to-pan, **Fit all** and **Fit route** to inspect the diagram. Orange means selected route, red dashed means missing observations for a configured directed edge, green is origin and purple is destination. Red does not mean the road is closed or congested. Very short arrows may only appear after zooming in.

## Evaluation and tests

```bash
python3 evaluate_models.py
python3 -m unittest test_traffic_runtime test_canterbury_extension test_burke_bridge test_burke_north test_network_batch -v
```

Preserve all five test modules and **both** `network_test_support.py` and `regression_roads_snapshot.csv`.

- 29 regression tests use the preserved original road graph through `network_test_support.py`.
- Eight `test_network_batch` cases use the expanded current `roads.csv`.
- GUI operations were checked manually and are not counted as additional automated tests.

Current evaluation outputs belong in `evaluation_results/`. The separate `model_evaluation_current.csv` supplied with the updated report reproduces the pasted 88-record results. Keep the project's original detailed prediction export as evidence.

The two `NO_TEST_DATA` entries are site 3685 south and site 4264 Glenferrie south. Their observations end on October 22. Do not replace absent October 25–31 data with zero or mix their earlier-period scores into that benchmark. Available test sample counts differ by measurement location.

## Training and batch preparation

`train_location_model.py` supports location-specific training using explicit chronological split arguments. Existing model records are reused by the preparation scripts. Training-only minima and ranges must remain paired with their model files.

`finish_network_batch.py` generated the expanded prototype and saved diagnostics under `batch_results/`. It is a preparation utility, not the normal GUI launch command. Existing saved model files avoid the need to repeat training.

## Known scope and assumptions

- Four sites remain unconfigured: **2827, 4262, 4812 and 4821**.
- Site **4335** has no outgoing mapping because its repeated daily measurement records have not been resolved.
- Site **2825** has no incoming mapping. This is not proof of a physical one-way restriction.
- New corridor links and source-arm assignments are inferred from road names and coordinate bearings. They have not been independently verified as actual permitted driving paths.
- Distances are straight-line estimates and may omit road curvature and intervening controlled intersections.
- Source-side flow follows the conversion handout; the assignment brief describes destination-side flow. The report discloses this discrepancy.
- Hourly flow is approximated as four times a predicted 15-minute count, using a constant-rate assumption. The under-capacity conversion uses a 60 km/h cap and a 30-second delay per configured traversed edge.
- Query costs form a snapshot at the selected interval, not a time-dependent prediction at each future arrival time.
- Missing observations exclude affected edges. No live feed or validated imputation is supplied.
- The graph is **not a complete or verified Boroondara road network**, and arbitrary origin–destination coverage is not achieved.

## Submission preparation

Include the actual implemented training and evaluation scripts, test modules, runtime files, model files, data, configuration, README, requirements, final report and supporting evaluation/test evidence. Keep `batch_results/network_summary.json`, `connection_audit.csv`, `invalid_measurements.csv`, `unconfigured_stations.csv` and `all_tests.txt`.

The notebook can document the team's own development work. Unmodified teaching examples should not be presented as the team's solution. Create a submission copy rather than deleting files from the working project.

Fill the report's Team ID, names, student IDs, actual contributions and signatures. Name the final submitted report using the Team ID, as required by the brief. Retain the instructor's two-person-team email permitting LSTM only and the A2A search algorithms. Disclose unresolved features and AI assistance accurately.

Before submission, copy the assembled folder to another location and launch `traffic_gui.py` from that copy. This checks relative resource paths within the existing environment; it is not a fresh-machine compatibility test.
