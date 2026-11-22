from network_test_support import load_snapshot_system
import math
import unittest

from traffic_runtime import TrafficSystem


class CanterburyExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = load_snapshot_system()

    def test_01_four_links_configured(self):
        expected = {
            (3126, 3127): (
                1.835,
                "CANTERBURY_RD W of WARRIGAL_RD",
            ),
            (3127, 3126): (
                1.835,
                "CANTERBURY_RD E of BALWYN_RD",
            ),
            (3127, 3122): (
                1.200,
                "CANTERBURY_RD W of BALWYN_RD",
            ),
            (3122, 3127): (
                1.200,
                "CANTERBURY_RD E of STANHOPE_GV",
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

    def test_02_available_date_both_directions(self):
        queries = [
            (
                3126,
                3122,
                (3126, 3127, 3122),
            ),
            (
                3122,
                3126,
                (3122, 3127, 3126),
            ),
        ]

        for origin, destination, expected_path in queries:
            with self.subTest(origin=origin):
                routes = self.system.query_selected(
                    origin,
                    destination,
                    "2006-10-29",
                    4,
                )

                # The original route must remain available.
                # Additional valid routes are allowed.
                paths = [
                    path for path, _ in routes
                ]
                self.assertIn(expected_path, paths)

                costs = [
                    seconds for _, seconds in routes
                ]
                self.assertEqual(costs, sorted(costs))
                self.assertTrue(
                    all(
                        math.isfinite(cost) and cost > 0
                        for cost in costs
                    )
                )

    def test_03_missing_date_excludes_outgoing_links(self):
        routes = self.system.query_selected(
            3126, 3122, "2006-10-25", 4
        )

        self.assertEqual(routes, [])

        warnings = self.system.last_query_warnings

        self.assertTrue(
            any(
                "3127 -> 3122:" in warning
                for warning in warnings
            )
        )
        self.assertTrue(
            any(
                "3127 -> 3126:" in warning
                for warning in warnings
            )
        )

    def test_04_valid_incoming_link_remains_available(self):
        graph = self.system.build_graph(
            "2006-10-25", 4
        )

        # The prototype uses source-side observations.
        # Missing observations at 3127 exclude its outgoing
        # links, but not a link measured at source 3126.
        self.assertTrue(
            any(
                target == 3127
                for target, _ in graph[3126]
            )
        )
        self.assertEqual(graph[3127], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)