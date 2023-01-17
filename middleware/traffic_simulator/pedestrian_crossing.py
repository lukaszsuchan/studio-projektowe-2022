from traffic_simulator import Road


class PedestrianCrossing:

    def create_paths(self, location, sin, cos):
        a = 10*sin
        b = 10*cos
        paths = []
        x1 = location[0]
        y1 = location[1] - 3
        for i in range(-3, 4, 1):
            if i == 0: continue
            start = (x1-a+i, y1-b-i)
            end = (x1+a+i, y1+b-i)
            paths.append(Road(start, end))
        return paths

    def __init__(self, location, roads, config={}):
        self.location = location
        self.roads = roads
        self.paths = self.create_paths(self.location, self.roads[0].angle_sin, self.roads[0].angle_cos)

        self.init_properties()

        self.set_default_config()

    def init_properties(self):
        self.roads[0].set_crossing(self)
        self.roads[1].set_crossing(self)

    def set_default_config(self):
        self.is_pedestrian_passing = False

    def update(self):
        self.is_pedestrian_passing = False
        if len(self.paths[0].vehicles) > 0 or len(self.paths[1].vehicles) > 0\
            or len(self.paths[2].vehicles) > 0 or len(self.paths[3].vehicles) > 0 \
            or len(self.paths[4].vehicles) > 0 or len(self.paths[5].vehicles) > 0:
            self.is_pedestrian_passing = True