from network_test_support import load_snapshot_system
import unittest

from traffic_runtime import TrafficSystem


class BurkeNorthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = load_snapshot_system()

    def test_01_northbound_corridor(self):
        routes = self.system.query_selected(
            3120, 4030, "2006-10-25", 4
        )

        self.assertTrue(routes)
        self.assertEqual(
            tuple(routes[0][0]),
            (3120, 4035, 4034, 4032, 4030),
        )

    def test_02_southbound_corridor(self):
        routes = self.system.query_selected(
            4030, 3120, "2006-10-25", 4
        )

        self.assertTrue(routes)
        self.assertEqual(
            tuple(routes[0][0]),
            (4030, 4032, 4034, 4035, 3120),
        )

    def test_03_route_cost_matches_edges(self):
        routes = self.system.query_selected(
            4030, 3126, "2006-10-25", 4
        )
        graph = self.system.build_graph(
            "2006-10-25", 4
        )

        self.assertTrue(routes)

        costs = [cost for _, cost in routes]
        self.assertEqual(costs, sorted(costs))

        for path, cost in routes:
            # A returned path must not repeat a station.
            self.assertEqual(len(path), len(set(path)))

            expected = 0.0

            for source, destination in zip(
                path, path[1:]
            ):
                weights = [
                    weight
                    for node, weight in graph[source]
                    if node == destination
                ]

                self.assertEqual(len(weights), 1)
                expected += weights[0]

            self.assertAlmostEqual(
                cost, expected, places=4
            )

    def test_04_missing_data_bypass(self):
        routes = self.system.query_selected(
            4030, 3126, "2006-10-25", 4
        )

        self.assertTrue(routes)
        self.assertEqual(
            tuple(routes[0][0]),
            (
                4030,
                4032,
                4034,
                4035,
                3120,
                4040,
                3804,
                3682,
                3126,
            ),
        )

        self.assertNotIn(3127, routes[0][0])

        self.assertTrue(
            any(
                "3127 -> 3126:" in warning
                for warning
                in self.system.last_query_warnings
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)