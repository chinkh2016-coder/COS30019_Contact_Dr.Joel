"""Zoomable Tkinter network view; all positions are display coordinates only."""
import math
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font


def label_positions(points, width, height, measure, line_height, edges=()):
    """Choose nearby label boxes with minimal node and label overlap."""
    placed = []
    result = {}

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def crosses(start, end, box):
        near, far = 0.0, 1.0
        for origin, delta, low, high in ((start[0], end[0]-start[0], box[0], box[2]),
                                         (start[1], end[1]-start[1], box[1], box[3])):
            if abs(delta) < 1e-10:
                if not low <= origin <= high:
                    return False
            else:
                first, last = sorted(((low-origin)/delta, (high-origin)/delta))
                near, far = max(near, first), min(far, last)
                if near > far:
                    return False
        return True

    segments = [(points[a], points[b]) for a, b in set(edges) if a in points and b in points]
    nodes = [(x-9, y-9, x+9, y+9) for x, y in points.values()]
    for index, (site, (x, y)) in enumerate(sorted(points.items())):
        if not (-70 <= x <= width+70 and -70 <= y <= height+70):
            continue
        tw, th = measure(str(site)) + 6, line_height + 4
        candidates = []
        preferred = (16, 18) if site == 3685 else ((18, -18) if index % 2 == 0 else (-18, -18))
        offsets = [preferred]
        for gap in (18, 32, 48, 66):
            offsets.extend([(gap, -gap), (-gap, -gap), (gap, gap), (-gap, gap),
                            (gap, 0), (-gap, 0), (0, -gap), (0, gap)])
        for order, (dx, dy) in enumerate(offsets):
            cx = x + dx + (tw/2 if dx > 0 else -tw/2 if dx < 0 else 0)
            cy = y + dy
            box = (cx-tw/2, cy-th/2, cx+tw/2, cy+th/2)
            outside = tw*th - overlap(box, (4, 4, width-4, height-4))
            score = (sum(overlap(box, node) for node in nodes)*150
                     + sum(overlap(box, other) for other in placed)*200
                     + outside*250 + 1800*sum(crosses(a, b, box) for a, b in segments)
                     + math.hypot(dx, dy) + order*0.01)
            candidates.append((score, box, cx, cy))
        _, box, cx, cy = min(candidates, key=lambda item: item[0])
        placed.append(box)
        result[site] = (cx, cy, box)
    return result


class TrafficMapView:
    def __init__(self, parent, canvas, projector, allow_window=True):
        self.canvas = canvas
        self.projector = projector
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.drag_start = None
        self.payload = None
        self.windows = []
        self.font = Font(root=canvas, family='TkDefaultFont', size=10, weight='bold')
        bar = ttk.Frame(parent)
        bar.pack(fill='x', before=canvas, pady=(4, 0))
        for text, command in [('+ Zoom', lambda: self.change_zoom(1.25)),
                              ('− Zoom', lambda: self.change_zoom(0.8)),
                              ('Fit all', self.fit_all), ('Fit route', self.fit_route)]:
            ttk.Button(bar, text=text, command=command).pack(side='left', padx=(0, 4))
        if allow_window:
            ttk.Button(bar, text='Large map', command=self.open_window).pack(side='left', padx=(0, 4))
        self.zoom_text = tk.StringVar(master=canvas, value='100%')
        ttk.Label(bar, textvariable=self.zoom_text).pack(side='right')
        self.help_text = ttk.Label(parent, text='Scroll to zoom • Drag to move • Fit all to reset')
        self.help_text.pack(anchor='w', before=canvas)
        canvas.configure(cursor='hand2')
        canvas.bind('<MouseWheel>', self.wheel)
        canvas.bind('<Button-4>', lambda e: self.wheel(e, 1))
        canvas.bind('<Button-5>', lambda e: self.wheel(e, -1))
        canvas.bind('<ButtonPress-1>', self.start_drag)
        canvas.bind('<B1-Motion>', self.drag)
        canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag_start', None))
        canvas.bind('<Double-Button-1>', lambda e: self.fit_all())

    def draw(self, coordinates, roads, routes, selected_route, missing, completed_query):
        self.payload = (coordinates, roads, routes, selected_route, missing, completed_query)
        self.redraw()
        live = []
        for window, view in self.windows:
            if window.winfo_exists():
                view.draw(*self.payload)
                live.append((window, view))
        self.windows = live

    def base_points(self):
        if not self.payload:
            return {}
        return self.projector(self.payload[0], self.canvas.winfo_width(),
                              self.canvas.winfo_height(), padding=65)

    def change_zoom(self, factor, x=None, y=None):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        x = width/2 if x is None else x
        y = height/2 if y is None else y
        new_zoom = max(0.5, min(10.0, self.zoom*factor))
        ratio = new_zoom/self.zoom
        self.pan_x = x-width/2 - (x-width/2-self.pan_x)*ratio
        self.pan_y = y-height/2 - (y-height/2-self.pan_y)*ratio
        self.zoom = new_zoom
        self.redraw()

    def wheel(self, event, direction=None):
        direction = direction if direction is not None else (1 if event.delta > 0 else -1)
        self.change_zoom(1.12 if direction > 0 else 1/1.12, event.x, event.y)
        return 'break'

    def start_drag(self, event):
        self.drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def drag(self, event):
        if self.drag_start:
            x, y, px, py = self.drag_start
            self.pan_x, self.pan_y = px+event.x-x, py+event.y-y
            self.redraw()

    def fit_all(self):
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.redraw()

    def fit_route(self):
        if not self.payload:
            return
        _, _, routes, selected, _, _ = self.payload
        if selected is None or not 0 <= selected < len(routes):
            self.fit_all()
            return
        points = self.base_points()
        chosen = [points[site] for site in routes[selected][0] if site in points]
        if not chosen:
            return
        xs, ys = zip(*chosen)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.zoom = max(1.0, min(8.0, (width-140)/max(1, max(xs)-min(xs)),
                                         (height-140)/max(1, max(ys)-min(ys))))
        self.pan_x = -((min(xs)+max(xs))/2-width/2)*self.zoom
        self.pan_y = -((min(ys)+max(ys))/2-height/2)*self.zoom
        self.redraw()

    def open_window(self):
        window = tk.Toplevel(self.canvas)
        window.title('Traffic network — large map')
        w = min(1250, window.winfo_screenwidth()-80)
        h = min(950, window.winfo_screenheight()-100)
        window.geometry(f'{w}x{h}')
        window.minsize(650, 480)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill='both', expand=True)
        canvas = tk.Canvas(frame, background='#f8fafb', highlightthickness=0)
        canvas.pack(fill='both', expand=True, pady=8)
        view = TrafficMapView(frame, canvas, self.projector, allow_window=False)
        canvas.bind('<Configure>', lambda e: view.redraw())
        self.windows.append((window, view))
        if self.payload:
            window.after_idle(lambda: view.draw(*self.payload))

    def redraw(self):
        canvas = self.canvas
        canvas.delete('all')
        width, height = canvas.winfo_width(), canvas.winfo_height()
        self.zoom_text.set(f'{self.zoom*100:.0f}%')
        if width < 100 or height < 100 or not self.payload:
            return
        coordinates, roads, routes, selected, missing, query = self.payload
        base = self.base_points()
        points = {site: (width/2+(x-width/2)*self.zoom+self.pan_x,
                         height/2+(y-height/2)*self.zoom+self.pan_y)
                  for site, (x, y) in base.items()}
        path = routes[selected][0] if selected is not None and 0 <= selected < len(routes) else ()
        selected_edges = set(zip(path, path[1:]))
        road_set = set(roads)

        def edge(source, target, color, line_width, dash=None):
            if source not in points or target not in points:
                return
            x1, y1 = points[source]
            x2, y2 = points[target]
            dx, dy = x2-x1, y2-y1
            length = math.hypot(dx, dy)
            if length < 2:
                return
            offset = 4 if (target, source) in road_set else 0
            ox, oy = -dy/length*offset, dx/length*offset
            cut = min(13, length/3)
            options = dict(fill=color, width=line_width,
                           arrow=tk.LAST if length > 36 else tk.NONE, arrowshape=(7, 9, 3))
            if dash:
                options['dash'] = dash
            canvas.create_line(x1+dx/length*cut+ox, y1+dy/length*cut+oy,
                               x2-dx/length*cut+ox, y2-dy/length*cut+oy, **options)

        for source, target in roads:
            unavailable = (source, target) in missing
            edge(source, target, '#bb3e3e' if unavailable else '#b3bec8',
                 1.5, (5, 4) if unavailable else None)
        for source, target in selected_edges:
            edge(source, target, '#e48614', 3)
        origin, destination = query[:2] if query else (None, None)
        for site, (x, y) in points.items():
            color = '#e48614' if site in path else '#557c91'
            if site == origin:
                color = '#24834b'
            if site == destination:
                color = '#7955a3'
            radius = 6.5 if site in path else 5.5
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                               fill=color, outline='white', width=1.5)
        labels = label_positions(points, width, height, self.font.measure, self.font.metrics('linespace'), roads)
        for site, (cx, cy, box) in labels.items():
            x, y = points[site]
            if math.hypot(cx-x, cy-y) > 48:
                length = math.hypot(cx-x, cy-y)
                canvas.create_line(x+(cx-x)/length*9, y+(cy-y)/length*9, cx, cy, fill='#a7b4bd', width=0.7)
            canvas.create_rectangle(*box, fill='#f8fafb', outline='')
            canvas.create_text(cx, cy, text=str(site), fill='#152d3e', font=self.font)
        canvas.create_text(width-18, 18, text='N ↑', anchor='ne',
                           fill='#425d6c', font=self.font)
