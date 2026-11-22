from network_test_support import load_snapshot_system
import math
import unittest
from traffic_runtime import TrafficSystem

class TrafficSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = load_snapshot_system()

    def query(self, **changes):
        settings = dict(origin=970, destination=3682, date_text='2006-10-25', target_slot=4, method='TOP5', limit=5)
        settings.update(changes)
        return self.system.query_selected(**settings)

    def check_single_route(self, method):
        routes = self.query(method=method)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0][0], (970, 3685, 2000, 3682))
        self.assertAlmostEqual(routes[0][1], 292.68, places=2)

    def test_01_top_routes(self):
        routes = self.query()
        self.assertEqual([p for p, _ in routes], [(970,3685,2000,3682), (970,3685,2000,4043,4040,3804,3682)])
        self.assertAlmostEqual(routes[0][1],292.68,places=2)
        self.assertAlmostEqual(routes[1][1],815.88,places=2)

    def test_06_reverse_with_available_data(self):
        routes=self.query(origin=2000,destination=970,date_text='2006-10-20')
        self.assertEqual(len(routes),1)
        self.assertEqual(routes[0][0],(2000,3685,970))
        self.assertTrue(math.isfinite(routes[0][1]) and routes[0][1]>0)

    def test_07_no_route(self):
        self.assertEqual(self.query(origin=3682,destination=970),[])

    def test_08_same_station(self):
        self.assertEqual(self.query(destination=970),[((970,),0.0)])

    def test_15_cycles(self):
        graph={1:[(3,150.0),(2,60.0)],2:[(1,10.0),(3,60.0)],3:[]}
        self.assertEqual(TrafficSystem.top_routes(graph,1,3,5),[((1,2,3),120.0),((1,3),150.0)])

    def test_16_missing_observations(self):
        self.assertEqual(self.query(origin=2000,destination=970),[])
        self.assertTrue(any('3685 -> 970' in w for w in self.system.last_query_warnings))
        graph=self.system.build_graph('2006-10-25',4)
        self.assertFalse(any(target==970 for target,_ in graph[3685]))
        self.assertTrue(any(target==2000 for target,_ in graph[3685]))

    def test_17_unsupported_date(self):
        with self.assertRaises(ValueError):
            self.query(date_text='2007-01-01')
        with self.assertRaises(ValueError):
            self.query(date_text='2007-01-01',destination=970)

    def test_02_bfs(self):
        self.check_single_route('BFS')

    def test_03_as(self):
        self.check_single_route('AS')

    def test_04_cus1(self):
        self.check_single_route('CUS1')

    def test_05_cus2(self):
        self.check_single_route('CUS2')

    def test_09_invalid_date(self):
        with self.assertRaises(ValueError):
            self.query(date_text='abc')

    def test_10_unknown_station(self):
        with self.assertRaises(ValueError):
            self.query(origin=9999)

    def test_11_low_slot(self):
        with self.assertRaises(ValueError):
            self.query(target_slot=3)

    def test_12_high_slot(self):
        with self.assertRaises(ValueError):
            self.query(target_slot=96)

    def test_13_invalid_limit(self):
        with self.assertRaises(ValueError):
            self.query(limit=0)

    def test_14_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            self.query(method='UNKNOWN')

if __name__ == "__main__":
    unittest.main(verbosity=2)
