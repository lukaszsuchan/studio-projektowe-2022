from collections import deque
import math

class Road:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.vehicles = deque()

        self.init_properties()

    def euclidean_distance(self, u, v, p=2, w=None):
        if p <= 0:
            raise ValueError("p must be greater than 0")
        u_v = (u[0] - v[0], u[1] - v[1])
        if w is not None:
            if p == 1:
                root_w = w
            elif p == 2:
                # better precision and speed
                root_w = math.sqrt(w)
            elif p == math.inf:
                root_w = (w != 0)
            else:
                root_w = math.pow(w, 1 / p)
            u_v = root_w * u_v
        dist = 0

        for i in u_v:
            dist += abs(i) ** 2

        dist = math.sqrt(dist)
        return dist

    def init_properties(self):
        self.length = self.euclidean_distance(self.start, self.end)
        self.angle_sin = (self.end[1]-self.start[1]) / self.length
        self.angle_cos = (self.end[0]-self.start[0]) / self.length
        # self.angle = np.arctan2(self.end[1]-self.start[1], self.end[0]-self.start[0])
        self.has_traffic_signal = False
        self.has_crossing = False
        self.has_bus_pass = False

    def set_traffic_signal(self, signal, group):
        self.traffic_signal = signal
        self.traffic_signal_group = group
        self.has_traffic_signal = True

    def set_crossing(self, crossing):
        self.crossing = crossing
        self.has_crossing = True

    def set_bus_pass(self, position):
        self.bus_pass_position = position
        self.has_bus_pass = True

    @property
    def traffic_signal_state(self):
        if self.has_traffic_signal:
            i = self.traffic_signal_group
            return self.traffic_signal.current_cycle[i]
        return True

    @property
    def crossing_state(self):
        if self.has_crossing:
            return not self.crossing.is_pedestrian_passing

    def update(self, dt):
        n = len(self.vehicles)

        if n > 0:
            # Update first vehicle
            self.vehicles[0].update(None, dt)
            # Update other vehicles
            for i in range(1, n):
                lead = self.vehicles[i-1]
                self.vehicles[i].update(lead, dt)

             # Check for traffic signal
            if self.traffic_signal_state:
                # If traffic signal is green or doesn't exist
                # Then let vehicles pass
                self.vehicles[0].unstop()
                for vehicle in self.vehicles:
                    vehicle.unslow()
            else:
                # If traffic signal is red
                if self.vehicles[0].x >= self.length - self.traffic_signal.slow_distance:
                    # Slow vehicles in slowing zone
                    self.vehicles[0].slow(self.traffic_signal.slow_factor*self.vehicles[0]._v_max)
                if self.vehicles[0].x >= self.length - self.traffic_signal.stop_distance and\
                   self.vehicles[0].x <= self.length - self.traffic_signal.stop_distance / 2:
                    # Stop vehicles in the stop zone
                    self.vehicles[0].stop()

            if self.has_crossing:
                if self.crossing_state:
                    self.vehicles[0].unstop()
                    for vehicle in self.vehicles:
                        vehicle.unslow()
                else:
                    l = self.length
                    # print(l)
                    if self.vehicles[0].x >= l - 47:
                        # Slow vehicles in slowing zone
                        self.vehicles[0].slow(0.4 * self.vehicles[0]._v_max)
                    if self.vehicles[0].x >= l - 11 and \
                            self.vehicles[0].x <= l - 3.5:
                        # Stop vehicles in the stop zone
                        self.vehicles[0].stop()
