import random

from traffic_simulator import Vehicle


class PedestrianGenerator:

    def __init__(self, sim, config={}):
        self.sim = sim
        self.set_default_config()

        self.init_properties()

    def set_default_config(self):
        """Set default configuration"""
        f = open("params.txt", "r")
        params_list = f.read().split("\n")
        pedestrian_level = int(params_list[2])
        self.pedestrian_rate = pedestrian_level
        self.pedestrians = [
            (1, {'l': 1, 'v_max': 1, 'path': [2]}),
            (5, {'l': 1, 'v_max': 2, 'path': [0]})
        ]
        self.last_added_time = 0

    def init_properties(self):
        self.upcoming_pedestrian = self.generate_pedestrian()

    def generate_pedestrian(self):
        """Returns a random pedestrian from self.pedestrians with random proportions"""
        self.sim.pedestrian_crossing[0].is_pedestrian_passing = True
        total = sum(pair[0] for pair in self.pedestrians)
        r = random.randint(1, total)
        for (weight, config) in self.pedestrians:
            r -= weight
            if r <= 0:
                return Vehicle(config)

    def update(self):
        """Add pedestrian"""
        if self.sim.t - self.last_added_time >= 60 / self.pedestrian_rate:
            # If time elasped after last added pedestrian is
            # greater than pedestrian_period; generate a pedestrian
            for cross in self.sim.pedestrian_crossing:
                path = cross.paths[self.upcoming_pedestrian.path[0]]
                if len(path.vehicles) == 0 \
                        or path.vehicles[-1].x > self.upcoming_pedestrian.s0 + self.upcoming_pedestrian.l:
                    self.upcoming_pedestrian.time_added = self.sim.t
                    path.vehicles.append(self.upcoming_pedestrian)
                    self.last_added_time = self.sim.t
                self.upcoming_pedestrian = self.generate_pedestrian()