"""Checks geometry and worker behaviour without opening a Tk window."""
import math
import unittest
from types import SimpleNamespace
import pandas as pd
from traffic_gui import project_points, TrafficApp


class GuiVisualizationTests(unittest.TestCase):
    def test_01_north_up_and_east_right(self):
        positions=project_points({1:(145,-38),2:(145.01,-38),3:(145,-37.99)},600,450)
        self.assertGreater(positions[2][0],positions[1][0])
        self.assertLess(positions[3][1],positions[1][1])
        self.assertAlmostEqual(positions[1][1],positions[2][1])
        for x,y in positions.values():
            self.assertTrue(48-1e-7<=x<=552+1e-7 and 48-1e-7<=y<=402+1e-7)

    def test_02_single_station_and_empty_network(self):
        self.assertEqual(project_points({},600,400),{})
        self.assertEqual(project_points({970:(145,-38)},600,400),{970:(300,200)})

    def test_03_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            project_points({970:(math.nan,-38)},600,400)

    def test_04_worker_keeps_valid_incoming_edge(self):
        class FakeSystem:
            def __init__(self):
                self.calls=0
                self.last_query_warnings=['Missing observations at source 3127']
                self.data=pd.DataFrame({'SCATS Number':[3126], 'Location':['west'], '_day':[pd.Timestamp('2006-10-25')]})
                self.roads=pd.DataFrame([(3126,3127,'west'),(3127,3126,'east')],columns=['source','destination','traffic_location'])
            def query_selected(self,*settings):
                self.calls+=1
                return [((3126,3127),100.0)]
            def parse_date(self,date):
                return pd.Timestamp(date)
        app=TrafficApp.__new__(TrafficApp)
        app.system=FakeSystem()
        routes,warnings,missing=app.run_query((3126,3127,'2006-10-25',4,'TOP5',5))
        self.assertEqual(app.system.calls,1)
        self.assertEqual(routes,[((3126,3127),100.0)])
        self.assertEqual(missing,{(3127,3126)})
        self.assertEqual(warnings,app.system.last_query_warnings)
        self.assertIsNot(warnings,app.system.last_query_warnings)

    def test_05_selected_route_changes_highlight(self):
        app=TrafficApp.__new__(TrafficApp)
        app.routes=[((1,2,3),100.0),((1,4,3),130.0)]
        app.tree=SimpleNamespace(selection=lambda:('1',))
        caption=[]
        app.route_caption=SimpleNamespace(set=caption.append)
        redraw=[]
        app.draw_network=lambda:redraw.append(True)
        app.select_route()
        self.assertEqual(app.selected_route,1)
        self.assertIn('1 -> 4 -> 3',caption[0])
        self.assertIn('130.00',caption[0])
        self.assertEqual(redraw,[True])


if __name__=='__main__':
    unittest.main(verbosity=2)
