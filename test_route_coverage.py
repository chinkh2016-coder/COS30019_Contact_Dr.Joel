import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from check_route_coverage import export_coverage


class CoverageExportTests(unittest.TestCase):
    def test_exports_separate_unconfigured_disconnected_and_missing(self):
        # Four dataset sites, but only three are configured.
        #
        # Configured graph: 1 -> 2 -> 3
        # Available graph:  1 -> 2
        # Site 4 is unconfigured.
        system = SimpleNamespace(
            data=pd.DataFrame(
                {
                    "SCATS Number": [
                        1,
                        2,
                        3,
                        4,
                    ]
                }
            ),
            stations=[
                1,
                2,
                3,
            ],
            roads=pd.DataFrame(
                {
                    "source": [
                        1,
                        2,
                    ],
                    "destination": [
                        2,
                        3,
                    ],
                }
            ),
            build_graph=lambda date, slot: {
                1: [(2, 1)],
                2: [],
                3: [],
            },
        )

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)

            summary = export_coverage(
                system,
                ["2006-10-25"],
                output,
            )[0]

            # Four sites produce 4 * 3 distinct ordered pairs.
            self.assertEqual(
                summary["distinct_site_pairs"],
                12,
            )

            self.assertEqual(
                summary["AVAILABLE"],
                1,
            )

            self.assertEqual(
                summary["NO_CONFIGURED_PATH"],
                3,
            )

            self.assertEqual(
                summary["NO_OBSERVED_PATH"],
                2,
            )

            self.assertEqual(
                summary["SITE_UNCONFIGURED"],
                6,
            )

            rows = pd.read_csv(
                output / "route_pair_coverage.csv"
            )

            self.assertEqual(
                len(rows),
                12,
            )

            self.assertFalse(
                (
                    rows["origin"]
                    == rows["destination"]
                ).any()
            )


if __name__ == "__main__":
    unittest.main()