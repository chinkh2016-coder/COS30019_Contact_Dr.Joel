from network_test_support import load_snapshot_system
import math
import unittest

from traffic_runtime import TrafficSystem


class BurkeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = load_snapshot_system()

    def test_01_configured_links(self):
        expected = {
            (3122, 3120): (
                0.633,
                "CANTERBURY_RD W of STANHOPE_GV",
            ),
            (3120, 3122): (
                0.633,
                "CANTERBURY_RD E of BURKE_RD",
            ),
            (3120, 4040): (
                1.121,
                "BURKE_RD S of CANTERBURY_RD",
            ),
            (4040, 3120): (
                1.121,
                "BURKE_RD N of RIVERSDALE_RD",
            ),
        }

        for (source, target), (
            distance, location
        ) in expected.items():
            with self.subTest(
                source=source,
                destination=target,
            ):
                rows = self.system.roads.loc[
                    (self.system.roads["source"] == source)
                    & (
                        self.system.roads["destination"]
                        == target
                    )
                ]

                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(
                    float(rows.iloc[0]["distance_km"]),
                    distance,
                )
                self.assertEqual(
                    rows.iloc[0]["traffic_location"],
                    location,
                )

    def test_02_two_alternative_paths(self):
        routes = self.system.query_selected(
            3122, 3126, "2006-10-29", 4
        )

        paths = [path for path, _ in routes]

        self.assertIn(
            (3122, 3127, 3126),
            paths,
        )
        self.assertIn(
            (3122, 3120, 4040, 3804, 3682, 3126),
            paths,
        )

        # Returned routes must be unique and sorted by cost.
        self.assertEqual(len(paths), len(set(paths)))

        costs = [cost for _, cost in routes]
        self.assertEqual(costs, sorted(costs))

    def test_03_costs_and_simple_paths(self):
        routes = self.system.query_selected(
            3122, 3126, "2006-10-29", 4
        )
        graph = self.system.build_graph(
            "2006-10-29", 4
        )

        self.assertTrue(routes)

        for path, cost in routes:
            # A simple route must not repeat stations.
            self.assertEqual(
                len(path),
                len(set(path)),
            )

            # Check the route cost against its edge costs.
            expected_cost = sum(
                next(
                    edge_cost
                    for neighbor, edge_cost in graph[source]
                    if neighbor == target
                )
                for source, target in zip(
                    path, path[1:]
                )
            )

            self.assertTrue(
                math.isfinite(cost) and cost > 0
            )
            self.assertAlmostEqual(
                cost, expected_cost, places=4
            )

    def test_04_bridge_works_when_3127_data_missing(self):
        routes = self.system.query_selected(
            3120, 3682, "2006-10-25", 4
        )

        self.assertTrue(routes)
        self.assertEqual(
            routes[0][0],
            (3120, 4040, 3804, 3682),
        )
        self.assertTrue(
            any(
                "3127 -> 3122:" in warning
                for warning
                in self.system.last_query_warnings
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)