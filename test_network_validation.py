import unittest
from unittest.mock import patch

from network_validation import can_reach
from traffic_runtime import TrafficSystem


class NetworkValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the real project once for integration tests.
        cls.system = TrafficSystem()

    def test_direction_is_not_reversed(self):
        graph = {
            1: [(2, 1)],
            2: [],
        }

        self.assertTrue(
            can_reach(graph, 1, 2)
        )

        self.assertFalse(
            can_reach(graph, 2, 1)
        )

    def test_cycles_do_not_create_reachability(self):
        graph = {
            1: [(2, 1)],
            2: [(1, 1)],
            3: [],
        }

        self.assertFalse(
            can_reach(graph, 1, 3)
        )

    def test_dead_end_is_pruned_without_changing_alternatives(self):
        graph = {
            1: [
                (2, 1),
                (3, 2),
                (9, 0.1),
            ],
            2: [(4, 2)],
            3: [(4, 2)],
            4: [],
            9: [(9, 1)],
        }

        result = TrafficSystem.top_routes(
            graph,
            1,
            4,
            5,
        )

        self.assertEqual(
            result,
            [
                ((1, 2, 4), 3.0),
                ((1, 3, 4), 4.0),
            ],
        )

    def test_unreachable_target_does_not_start_path_enumeration(self):
        graph = {
            1: [(2, 1)],
            2: [(1, 1)],
            3: [],
        }

        # Unreachable queries should stop before popping paths.
        with patch(
            "traffic_runtime.heappop",
            side_effect=AssertionError(
                "Path enumeration started"
            ),
        ):
            result = TrafficSystem.top_routes(
                graph,
                1,
                3,
                5,
            )

            self.assertEqual(
                result,
                [],
            )

    def test_configured_gap_has_explicit_status(self):
        result = self.system.query_selected(
            4335,
            2200,
            "2006-10-29",
            4,
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            self.system.last_route_status,
            "NO_CONFIGURED_PATH",
        )

    def test_observation_gap_has_different_status(self):
        result = self.system.query_selected(
            3685,
            970,
            "2006-10-25",
            4,
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            self.system.last_route_status,
            "NO_OBSERVED_PATH",
        )


if __name__ == "__main__":
    unittest.main()