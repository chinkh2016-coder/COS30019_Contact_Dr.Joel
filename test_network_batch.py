import heapq
import math
import unittest
from collections import deque
from traffic_runtime import TrafficSystem


class ExpandedNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = TrafficSystem()

    def test_01_unique_valid_links(self):
        roads=self.system.roads
        self.assertFalse(roads.duplicated(['source','destination']).any())
        self.assertTrue(all(math.isfinite(float(d)) and d>0 for d in roads.distance_km))

    def test_02_all_links_have_models(self):
        for row in self.system.roads.itertuples():
            self.assertIn((row.source,row.traffic_location),self.system.models)

    def test_03_expanded_station_coverage(self):
        self.assertTrue({2200,4057,4063,3180,4051,3001,3002,4263,4264,4324}.issubset(self.system.stations))

    def test_04_same_station(self):
        routes=self.system.query_selected(2200,2200,'2006-10-29',4)
        self.assertEqual(tuple(routes[0][0]),(2200,))
        self.assertEqual(routes[0][1],0)

    def test_05_missing_data_warning(self):
        self.system.build_graph('2006-10-25',4)
        self.assertTrue(any('3685 -> 970:' in warning for warning in self.system.last_query_warnings))

    def test_06_route_costs_ranking_and_cycles(self):
        for origin,destination in [(970,2200),(4030,3126),(3122,3126)]:
            with self.subTest(origin=origin,destination=destination):
                routes=self.system.query_selected(origin,destination,'2006-10-29',4)
                graph=self.system.build_graph('2006-10-29',4)
                self.assertTrue(routes)
                self.assertLessEqual(len(routes),5)
                self.assertEqual([c for _,c in routes],sorted(c for _,c in routes))
                self.assertEqual(len({tuple(p) for p,_ in routes}),len(routes))
                for path,cost in routes:
                    self.assertEqual(len(path),len(set(path)))
                    self.assertEqual(path[0],origin)
                    self.assertEqual(path[-1],destination)
                    total=sum(next(c for node,c in graph[a] if node==b) for a,b in zip(path,path[1:]))
                    self.assertAlmostEqual(cost,total,places=4)

    @staticmethod
    def shortest_cost(graph,start,goal):
        best={start:0.0}
        queue=[(0.0,start)]
        while queue:
            cost,node=heapq.heappop(queue)
            if cost>best[node]:continue
            if node==goal:return cost
            for target,weight in graph[node]:
                new=cost+weight
                if new<best.get(target,math.inf):
                    best[target]=new
                    heapq.heappush(queue,(new,target))
        return math.inf

    def test_07_ucs_cost_matches_independent_solver(self):
        graph=self.system.build_graph('2006-10-29',4)
        expected=self.shortest_cost(graph,3120,2200)
        self.assertTrue(math.isfinite(expected))
        for method in ['CUS1','AS','TOP5']:
            routes=self.system.query_selected(3120,2200,'2006-10-29',4,method,5)
            self.assertTrue(routes)
            self.assertAlmostEqual(routes[0][1],expected,places=4)

    def test_08_bfs_fewest_edges(self):
        graph=self.system.build_graph('2006-10-29',4)
        queue=deque([(3120,0)])
        seen={3120}
        expected=None
        while queue:
            node,hops=queue.popleft()
            if node==2200:
                expected=hops
                break
            for target,_ in graph[node]:
                if target not in seen:
                    seen.add(target)
                    queue.append((target,hops+1))
        self.assertIsNotNone(expected)
        routes=self.system.query_selected(3120,2200,'2006-10-29',4,'BFS',5)
        self.assertTrue(routes)
        self.assertEqual(len(routes[0][0])-1,expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
