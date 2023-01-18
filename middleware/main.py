from collections import deque
from copy import deepcopy
import math
import pygame
import random
import asyncio
def curve_points(start, end, control, resolution=5):
    # If curve is a straight line
    if (start[0] - end[0]) * (start[1] - end[1]) == 0:
        return [start, end]

    # If not return a curve
    path = []

    for i in range(resolution + 1):
        t = i / resolution
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
        path.append((x, y))

    return path


def curve_road(start, end, control, resolution=15):
    points = curve_points(start, end, control, resolution=resolution)
    return [(points[i - 1], points[i]) for i in range(1, len(points))]


TURN_LEFT = 0
TURN_RIGHT = 1


def turn_road(start, end, turn_direction, resolution=15):
    # Get control point
    x = min(start[0], end[0])
    y = min(start[1], end[1])

    if turn_direction == TURN_LEFT:
        control = (
            x - y + start[1],
            y - x + end[0]
        )
    else:
        control = (
            x - y + end[1],
            y - x + start[0]
        )

    return curve_road(start, end, control, resolution=resolution)


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



class Simulation:
    def __init__(self, config={}):
        # Set default configuration
        self.set_default_config()

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def set_default_config(self):
        self.t = 0.0  # Time keeping
        self.frame_count = 0  # Frame count keeping
        self.dt = 1 / 60  # Simulation time step
        self.roads = []  # Array to store roads
        self.generators = []
        self.traffic_signals = []
        self.pedestrian_crossing = []

    def create_road(self, start, end):
        road = Road(start, end)
        self.roads.append(road)
        return road

    def create_roads(self, road_list):
        for road in road_list:
            self.create_road(*road)

    def create_gen(self, config: object = {}) -> object:
        gen = VehicleGenerator(self, config)
        self.generators.append(gen)
        return gen

    def create_pedestrian_gen(self, config: object = {}) -> object:
        gen = PedestrianGenerator(self, config)
        self.generators.append(gen)
        return gen

    def create_signal(self, roads, config={}):
        roads = [[self.roads[i] for i in road_group] for road_group in roads]
        sig = TrafficSignal(roads, config)
        self.traffic_signals.append(sig)
        return sig

    def create_pedestrian_crossing(self, location, roads, config={}):
        roads = [r for r in self.roads if (r.start, r.end) in roads]
        cross = PedestrianCrossing(location, roads, config)
        self.pedestrian_crossing.append(cross)
        return cross

    def update(self):
        # Update every road
        for road in self.roads:
            road.update(self.dt)

        # Add vehicles
        for gen in self.generators:
            gen.update()

        # Add traffic signals
        for signal in self.traffic_signals:
            signal.update(self)

        # Add pedestrian crossings
        for cross in self.pedestrian_crossing:
            cross.update()
            for road in cross.paths:
                road.update(self.dt)

        # Check roads for out of bounds vehicle
        for road in self.roads:
            # If road has no vehicles, continue
            if len(road.vehicles) == 0: continue
            # If not
            vehicle = road.vehicles[0]

            # if self.roads.index(road) == 46 :
            #     # print("jedzie bus")
            #     vehicle.
            # bus pass
            if vehicle.current_road_index == 2:
                if len(self.roads[27].vehicles) > 0 and len(
                        self.roads[26].vehicles) > 0 and vehicle.x < road.length - 100:
                    vehicle.slow(0.4 * vehicle.v_max)
                    if vehicle.x >= road.length - 100 and vehicle.x <= road.length - 50:
                        vehicle.stop()
            # print(vehicle.path)
            # print(vehicle.current_road_index)
            if vehicle.current_road_index != len(vehicle.path) - 1:
                # print(vehicle.path)
                # print(vehicle.current_road_index)
                next_road_id_in_path = vehicle.path[vehicle.current_road_index + 1]
                next_road = self.roads[next_road_id_in_path]

                # if next_road.length < (len(next_road.vehicles) * 4 + (len(next_road.vehicles) - 1) * 4) + 10:
                if len(next_road.vehicles) > 0 and next_road.vehicles[-1].x < 8:
                    vehicle.slow(0.4 * vehicle.v_max)
                    #print("zwalniam")
                    if vehicle.x >= road.length - 8 and vehicle.x <= road.length - 4:
                        # Stop vehicles in the stop zone
                        #print(("zatrzymałem się"))
                        vehicle.stop()
            # If first vehicle is out of road bounds
            if vehicle.x >= road.length:
                # If vehicle has a next road
                if vehicle.current_road_index + 1 < len(vehicle.path):
                    # Update current road to next road
                    vehicle.current_road_index += 1
                    # Create a copy and reset some vehicle properties
                    new_vehicle = deepcopy(vehicle)
                    new_vehicle.x = 0
                    # Add it to the next road
                    next_road_index = vehicle.path[vehicle.current_road_index]
                    self.roads[next_road_index].vehicles.append(new_vehicle)
                # In all cases, remove it from its road
                road.vehicles.popleft()

        for cross in self.pedestrian_crossing:
            for road in cross.paths:
                # If road has no vehicles, continue
                if len(road.vehicles) == 0: continue
                # If not
                vehicle = road.vehicles[0]
                # next_road = self.roads[vehicle.current_road_index + 1]
                # if next_road.length < (len(next_road.vehicles) * 4 + (len(next_road.vehicles) - 1) * 4) + 10:
                # if len(next_road.vehicles) > 0 and next_road.vehicles[-1].x < 8:
                #     vehicle.slow(0.4 * vehicle.v_max)
                #     if vehicle.x >= road.length - 8 and vehicle.x <= road.length - 4:
                #         # Stop vehicles in the stop zone
                #         vehicle.stop()
                # If first vehicle is out of road bounds
                if vehicle.x >= road.length:
                    # If vehicle has a next road
                    if vehicle.current_road_index + 1 < len(vehicle.path):
                        # Update current road to next road
                        vehicle.current_road_index += 1
                        # Create a copy and reset some vehicle properties
                        new_vehicle = deepcopy(vehicle)
                        new_vehicle.x = 0
                        # Add it to the next road
                        next_road_index = vehicle.path[vehicle.current_road_index]
                        self.roads[next_road_index].vehicles.append(new_vehicle)
                    # In all cases, remove it from its road
                    road.vehicles.popleft()

        # Increment time
        self.t += self.dt
        self.frame_count += 1

    def run(self, steps):
        for _ in range(steps):
            self.update()
class TrafficSignal:
    def __init__(self, roads, config={}):
        # Initialize roads
        self.roads = roads
        # Set default configuration
        self.set_default_config()
        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)
        # Calculate properties
        self.init_properties()

    def set_default_config(self):
        self.cycle = [(False, True), (True, False)]
        self.slow_distance = 50
        self.slow_factor = 0.4
        self.stop_distance = 15
        self.cycle_time = 30
        self.current_cycle_index = 0

        self.last_t = 0

    def init_properties(self):
        for i in range(len(self.roads)):
            for road in self.roads[i]:
                road.set_traffic_signal(self, i)

    @property
    def current_cycle(self):
        return self.cycle[self.current_cycle_index]

    def update(self, sim):
        cycle_length = self.cycle_time
        k = (sim.t // cycle_length) % 2
        self.current_cycle_index = int(k)


class Vehicle:
    def __init__(self, config={}):
        # Set default configuration
        self.set_default_config()

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        # Calculate properties
        self.init_properties()

    def set_default_config(self):
        self.l = 4
        self.s0 = 4
        self.T = 1
        self.v_max = 10
        self.a_max = 1
        self.b_max = 4.61

        self.path = []
        self.current_road_index = 0

        self.x = 0
        self.v = self.v_max
        self.a = 0
        self.stopped = False

    def init_properties(self):
        self.sqrt_ab = 2 * math.sqrt(self.a_max * self.b_max)
        self._v_max = self.v_max

    def update(self, lead, dt):
        # Update position and velocity
        if self.v + self.a * dt < 0:
            self.x -= 1 / 2 * self.v * self.v / self.a
            self.v = 0
        else:
            self.v += self.a * dt
            self.x += self.v * dt + self.a * dt * dt / 2

        # Update acceleration
        alpha = 0
        if lead:
            delta_x = lead.x - self.x - lead.l
            delta_v = self.v - lead.v

            alpha = (self.s0 + max(0, self.T * self.v + delta_v * self.v / self.sqrt_ab)) / delta_x

        self.a = self.a_max * (1 - (self.v / self.v_max) ** 4 - alpha ** 2)

        if self.stopped:
            self.a = -self.b_max * self.v / self.v_max

    def stop(self):
        self.stopped = True

    def unstop(self):
        self.stopped = False

    def slow(self, v):
        self.v_max = v

    def unslow(self):
        self.v_max = self._v_max

class VehicleGenerator:
    def __init__(self, sim, config={}):
        self.sim = sim

        # Set default configurations
        self.set_default_config()

        # Update configurations
        for attr, val in config.items():
            setattr(self, attr, val)

        # Calculate properties
        self.init_properties()

    def set_default_config(self):
        """Set default configuration"""
        self.vehicle_rate = 20
        self.vehicles = [
            (1, {})
        ]
        self.last_added_time = 0

    def init_properties(self):
        self.upcoming_vehicle = self.generate_vehicle()

    def generate_vehicle(self):
        """Returns a random vehicle from self.vehicles with random proportions"""
        total = sum(pair[0] for pair in self.vehicles)
        r = random.randint(1, total)
        for (weight, config) in self.vehicles:
            r -= weight
            if r <= 0:
                return Vehicle(config)

    def update(self):
        """Add vehicles"""
        if self.sim.t - self.last_added_time >= 60 / self.vehicle_rate:
            # If time elasped after last added vehicle is
            # greater than vehicle_period; generate a vehicle
            road = self.sim.roads[self.upcoming_vehicle.path[0]]      
            if len(road.vehicles) == 0 \
               or road.vehicles[-1].x > self.upcoming_vehicle.s0 + self.upcoming_vehicle.l:
                # If there is space for the generated vehicle; add it
                self.upcoming_vehicle.time_added = self.sim.t
                road.vehicles.append(self.upcoming_vehicle)
                # Reset last_added_time and upcoming_vehicle
                self.last_added_time = self.sim.t
            self.upcoming_vehicle = self.generate_vehicle()




class Window:
    def __init__(self, sim, config={}):
        # Simulation to draw
        self.sim = sim

        # Set default configurations
        self.set_default_config()

        # Update configurations
        for attr, val in config.items():
            setattr(self, attr, val)

    def set_default_config(self):
        """Set default configuration"""
        self.width = 1000
        self.height = 700
        self.bg_color = (250, 250, 250)

        self.fps = 60
        self.zoom = 5
        self.offset = (-163, 0)

        self.mouse_last = (0, 0)
        self.mouse_down = False
        self.min_zoom = 0

    async def loop(self, loop=None):
        """Shows a window visualizing the simulation and runs the loop function."""
        # Create a pygame window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.flip()

        # Fixed fps
        clock = pygame.time.Clock()

        # To draw text
        pygame.font.init()
        self.text_font = pygame.font.SysFont('Lucida Console', 16)

        # Draw loop
        running = True
        while running:
            # Update simulation
            if loop: loop(self.sim)

            # Draw simulation
            self.draw()

            # Update window
            pygame.display.update()
            clock.tick(self.fps)

            # Handle all events
            for event in pygame.event.get():
                # Quit program if window is closed
                if event.type == pygame.QUIT:
                    running = False
                # Handle mouse events
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # If mouse button down
                    if event.button == 1:
                        # Left click
                        x, y = pygame.mouse.get_pos()
                        x0, y0 = self.offset
                        self.mouse_last = (x - x0 * self.zoom, y - y0 * self.zoom)
                        self.mouse_down = True
                    if event.button == 4:
                        # Mouse wheel up
                        self.zoom *= (self.zoom ** 2 + self.zoom / 4 + 1) / (self.zoom ** 2 + 1)
                    if event.button == 5:
                        # Mouse wheel down
                        if self.zoom > self.min_zoom:
                            self.zoom *= (self.zoom ** 2 + 1) / (self.zoom ** 2 + self.zoom / 4 + 1)
                elif event.type == pygame.MOUSEMOTION:
                    # Drag content
                    if self.mouse_down:
                        x1, y1 = self.mouse_last
                        x2, y2 = pygame.mouse.get_pos()
                        self.offset = ((x2 - x1) / self.zoom, (y2 - y1) / self.zoom)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_down = False
            await asyncio.sleep(0)

    async def run(self, steps_per_update=1):
        """Runs the simulation by updating in every loop."""

        def loop(sim):
            sim.run(steps_per_update)

        await self.loop(loop)

    def convert(self, x, y=None):
        """Converts simulation coordinates to screen coordinates"""
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return (
            int(self.width / 2 + (x + self.offset[0]) * self.zoom),
            int(self.height / 2 + (y + self.offset[1]) * self.zoom)
        )

    def inverse_convert(self, x, y=None):
        """Converts screen coordinates to simulation coordinates"""
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return (
            int(-self.offset[0] + (x - self.width / 2) / self.zoom),
            int(-self.offset[1] + (y - self.height / 2) / self.zoom)
        )

    def own_convert(self, x, y=None):
        """Converts screen coordinates to simulation coordinates"""
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return (
            int(self.offset[0] + (x - self.width / 2) / self.zoom),
            int(self.offset[1] + (y - self.height / 2) / self.zoom)
        )

    def background(self, r, g, b):
        """Fills screen with one color."""
        self.screen.fill((r, g, b))

    def line(self, start_pos, end_pos, color):
        """Draws a line."""
        # gfxdraw.line(
        #     self.screen,
        #     *start_pos,
        #     *end_pos,
        #     color
        # )

    def rect(self, pos, size, color):
        """Draws a rectangle."""
        pygame.draw.rect(self.screen, color, (*pos, *size))

    def box(self, pos, size, color):
        """Draws a rectangle."""
        pygame.draw.rect(self.screen, color, (*pos, *size))

    def circle(self, pos, radius, color):
        pygame.draw.circle(self.screen, color, *pos, radius)

    def polygon(self, vertices, color):
        pygame.draw.polygon(self.screen, color, vertices)

    def rotated_box(self, pos, size, angle=None, cos=None, sin=None, centered=True, color=(0, 0, 255), filled=True):
        """Draws a rectangle center at *pos* with size *size* rotated anti-clockwise by *angle*."""
        x, y = pos
        l, h = size

        if angle:
            cos, sin = math.cos(angle), math.sin(angle)

        vertex = lambda e1, e2: (
            x + (e1 * l * cos + e2 * h * sin) / 2,
            y + (e1 * l * sin - e2 * h * cos) / 2
        )

        if centered:
            vertices = self.convert(
                [vertex(*e) for e in [(-1, -1), (-1, 1), (1, 1), (1, -1)]]
            )
        else:
            vertices = self.convert(
                [vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]]
            )

        self.polygon(vertices, color)

    def rotated_rect(self, pos, size, angle=None, cos=None, sin=None, centered=True, color=(0, 0, 255)):
        self.rotated_box(pos, size, angle=angle, cos=cos, sin=sin, centered=centered, color=color, filled=False)

    def arrow(self, pos, size, angle=None, cos=None, sin=None, color=(150, 150, 190)):
        if angle:
            cos, sin = math.cos(angle), math.sin(angle)

        self.rotated_box(
            pos,
            size,
            cos=(cos - sin) / math.sqrt(2),
            sin=(cos + sin) / math.sqrt(2),
            color=color,
            centered=False
        )

        self.rotated_box(
            pos,
            size,
            cos=(cos + sin) / math.sqrt(2),
            sin=(sin - cos) / math.sqrt(2),
            color=color,
            centered=False
        )

    def draw_axes(self, color=(100, 100, 100)):
        x_start, y_start = self.inverse_convert(0, 0)
        x_end, y_end = self.inverse_convert(self.width, self.height)
        self.line(
            self.convert((0, y_start)),
            self.convert((0, y_end)),
            color
        )
        self.line(
            self.convert((x_start, 0)),
            self.convert((x_end, 0)),
            color
        )

    def draw_grid(self, unit=50, color=(150, 150, 150)):
        x_start, y_start = self.inverse_convert(0, 0)
        x_end, y_end = self.inverse_convert(self.width, self.height)

        n_x = int(x_start / unit)
        n_y = int(y_start / unit)
        m_x = int(x_end / unit) + 1
        m_y = int(y_end / unit) + 1

        for i in range(n_x, m_x):
            self.line(
                self.convert((unit * i, y_start)),
                self.convert((unit * i, y_end)),
                color
            )
        for i in range(n_y, m_y):
            self.line(
                self.convert((x_start, unit * i)),
                self.convert((x_end, unit * i)),
                color
            )

    def draw_roads(self):
        for road in self.sim.roads:
            # Draw road background
            self.rotated_box(
                road.start,
                (road.length, 3.7),
                cos=road.angle_cos,
                sin=road.angle_sin,
                color=(180, 180, 220),
                centered=False
            )

            # Draw road lines
            # self.rotated_box(
            #     road.start,
            #     (road.length, 0.25),
            #     cos=road.angle_cos,
            #     sin=road.angle_sin,
            #     color=(0, 0, 0),
            #     centered=False
            # )

            def arange(start, end, step):
                i = start
                result = []
                while i <= end:
                    result.append(i)
                    i += step
                return result

            # Draw road arrow
            if road.length > 5:
                for i in arange(-0.5 * road.length, 0.5 * road.length, 10):
                    pos = (
                        road.start[0] + (road.length / 2 + i + 3) * road.angle_cos,
                        road.start[1] + (road.length / 2 + i + 3) * road.angle_sin
                    )

                    self.arrow(
                        pos,
                        (-1.25, 0.2),
                        cos=road.angle_cos,
                        sin=road.angle_sin
                    )

                    # TODO: Draw road arrow

    def draw_vehicle(self, vehicle, road):
        l, h = vehicle.l, 2
        sin, cos = road.angle_sin, road.angle_cos

        x = road.start[0] + cos * vehicle.x
        y = road.start[1] + sin * vehicle.x

        self.rotated_box((x, y), (l, h), cos=cos, sin=sin, centered=True)

    def draw_bus(self, bus, road):
        l, h = 8, 2
        sin, cos = road.angle_sin, road.angle_cos

        x = road.start[0] + cos * bus.x
        y = road.start[1] + sin * bus.x

        self.rotated_box((x, y), (l, h), cos=cos, sin=sin, color=(255, 140, 0), centered=True)

    def draw_pedestrian(self, pedestrian, crossing):
        l, h = 1, 1
        sin, cos = crossing.paths[0].angle_sin, crossing.paths[0].angle_cos

        x = crossing.paths[0].start[0] + cos * pedestrian.x
        y = crossing.paths[0].start[1] + sin * pedestrian.x

        self.rotated_box((x, y), (l, h), cos=cos, sin=sin, color=(255, 0, 213), centered=True)

    def draw_vehicles(self):
        for road in self.sim.roads:
            # Draw vehicles
            for vehicle in road.vehicles:
                if vehicle.l == 4:
                    self.draw_vehicle(vehicle, road)

    def draw_buses(self):
        for road in self.sim.roads:
            # Draw bus
            for vehicle in road.vehicles:
                if vehicle.l == 8:
                    self.draw_bus(vehicle, road)

    def draw_pedestrians(self):
        for cross in self.sim.pedestrian_crossing:
            # Draw pedestrian
            for path in cross.paths:
                for pedestrian in path.vehicles:
                    self.draw_pedestrian(pedestrian, cross)

    def draw_signals(self):
        for signal in self.sim.traffic_signals:
            for i in range(len(signal.roads)):
                color = (0, 255, 0) if signal.current_cycle[i] else (255, 0, 0)
                for road in signal.roads[i]:
                    a = 0
                    position = (
                        (1 - a) * road.end[0] + a * road.start[0],
                        (1 - a) * road.end[1] + a * road.start[1]
                    )
                    self.rotated_box(
                        position,
                        (1, 3),
                        cos=road.angle_cos, sin=road.angle_sin,
                        color=color)

    # def draw_status(self):
    #     text_fps = self.text_font.render(f't={self.sim.t:.5}', False, (0, 0, 0))
    #     text_frc = self.text_font.render(f'n={self.sim.frame_count}', False, (0, 0, 0))
    #
    #     self.screen.blit(text_fps, (0, 0))
    #     self.screen.blit(text_frc, (100, 0))
    def draw_pedestrian_crossing(self):
        for cross in self.sim.pedestrian_crossing:
            for i in range(0, 20, 4):
                self.rotated_box(
                    (cross.location[0], cross.location[1] - 8 + i),
                    (6, 2),
                    cos=cross.roads[0].angle_cos, sin=cross.roads[0].angle_sin,
                    color=(255, 255, 255))
            for i in range(-2, 20, 4):
                self.rotated_box(
                    (cross.location[0], cross.location[1] - 8 + i),
                    (6, 2),
                    cos=cross.roads[0].angle_cos, sin=cross.roads[0].angle_sin,
                    color=(128, 128, 128))

    def draw(self):
        # Fill background
        self.background(*self.bg_color)
        # img = pygame.image.load('assets/background.png')
        # img.convert()
        scale_x = int(self.width * self.zoom)
        scale_y = int(self.height * self.zoom)
        x_end, y_end = self.own_convert(self.width, self.height)
        # img = pygame.transform.scale(img, (scale_x, scale_y))
        # self.screen.blit(img, ((x_end-170)*self.zoom, (y_end-353)*self.zoom))

        # Major and minor grid and axes
        self.draw_grid(4, (220, 220, 220))
        # self.draw_grid(100, (200,200,200))
        # self.draw_axes()

        self.draw_roads()
        self.draw_pedestrian_crossing()
        self.draw_vehicles()
        self.draw_buses()
        self.draw_pedestrians()
        self.draw_signals()

        # Draw status info
        # self.draw_status()

SCALE = 50000

sim = Simulation()

DOWN_RIGHT = (50.0674134, 19.9031611)
UP_LEFT = (50.0699855, 19.9040286)
UP_RIGHT = (50.0658971, 19.9241282)
DOWN_LEFT = (50.0638776, 19.9236128)
TRAFFIC_SIGNALS_AWITEKS = (50.0697536, 19.9060001)
NAWOJKI_FIRST_TURN = (50.069035, 19.910721)
TRAFFIC_SIGNALS_ALEJA_KIJOWSKA = (50.0679708, 19.9136371)
ALEJA_KIJOWSKA = (50.068681, 19.913845)
TRAFFIC_SIGNALS_CZARNOWIEJSKA = (50.0675762, 19.9181620)
MIECHOWKSA = (50.0658352, 19.9140487)
CZARNOWIEJSKA_CROSSING = (50.0664123, 19.9225809)
PEDESTRIAN_CROSSING_NR1_CORD = (50.0693760, 19.9086055)

l1 = round(abs(UP_LEFT[1] - TRAFFIC_SIGNALS_AWITEKS[1]) * SCALE)
l2 = round(abs(TRAFFIC_SIGNALS_AWITEKS[1] - NAWOJKI_FIRST_TURN[1]) * SCALE)
l3 = round(abs(NAWOJKI_FIRST_TURN[1] - TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1]) * SCALE)
l4 = round(abs(ALEJA_KIJOWSKA[1] - TRAFFIC_SIGNALS_CZARNOWIEJSKA[1]) * SCALE) - 80
l5 = round(abs(TRAFFIC_SIGNALS_CZARNOWIEJSKA[1] - CZARNOWIEJSKA_CROSSING[1]) * SCALE) - 100
pedestrian_crossing_position_len = round(abs(CZARNOWIEJSKA_CROSSING[1] - PEDESTRIAN_CROSSING_NR1_CORD[1]) * SCALE) - 540
# print(l1)
# print(l2)
d1 = round(abs(ALEJA_KIJOWSKA[0] - TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0]) * SCALE)
# print(d1)
# print(pedestrian_crossing_position_len)

NAWOJKI_RIGHT_START = (-50 - l1 - 15, 4)
NAWOJKI_LEFT_START = (-50 - l1 - 15, -4)
WEST_RIGHT = (-65, 4)
WEST_LEFT = (-65, -4)
RIGHT_TRAFFIC_SIGNALS_NAWOJKI = (-50, 2)
LEFT_TRAFFIC_SIGNALS_NAWOJKI = (-50, -2)
RIGHT_FIRST_TURN_NAWOJKI = (-50 + l2, 2)
LEFT_FIRST_TURN_NAWOJKI = (-50 + l2, -2)
RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA = (-50 + l2 + l3, 30)
LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA = (-50 + l2 + l3, 26)
RIGHT_ALEJA_KIJOWSKA_START = (-50 + l2 + l3 + 12, 30 - d1 - 17)
LEFT_ALEJA_KIJOWSKA_START = (-50 + l2 + l3 + 8, 30 - d1 - 17)
RIGHT_NAWOJKI_END = (l2 + l3, 50)
LEFT_NAWOJKI_END = (l2 + l3, 46)
RIGHT_CZARNOWIEJSKA_FIRST_PART = (-50 + l2 + l3 + l4, 45)
LEFT_CZARNOWIEJSKA_FIRST_PART = (-50 + l2 + l3 + l4, 41)
RIGHT_MIECHOWSKA_END = (l2 + l3 - 5, 100)
LEFT_MIECHOWSKA_END = (-4 + l2 + l3 - 5, 100)
URZEDNICZNA_START = (-50 + l2 + l3 + l4, 0)
RIGHT_CZARNOWIEJSKA_SECOND_PART = (-50 + l2 + l3 + l4 + l5, 68)
LEFT_CZARNOWIEJSKA_SECOND_PART = (-50 + l2 + l3 + l4 + l5, 64)

NAWOJKI_FIRST_PART_INBOUND = (NAWOJKI_RIGHT_START, WEST_RIGHT)
NAWOJKI_FIRST_PART_OUTBOUND = (WEST_LEFT, NAWOJKI_LEFT_START)
RIGHT_NAWOJKI_FIRST_AND_SECOND_PART = (WEST_RIGHT, (-45, 2))
RIGHT_CROSSING = ((-45, 2), (-30, 2))
PEDESTRIAN_CROSSING_NR1 = ((-45 + pedestrian_crossing_position_len, 2), (-45 + pedestrian_crossing_position_len + 3, 2))
LEFT_NAWOJKI_FIRST_AND_SECOND_PART = ((-45, -2), WEST_LEFT)
LEFT_CROSSING = ((-30, -2), (-45, -2))
NAWOJKI_SECOND_PART_INBOUND = ((-30, 2), (125, 2))
NAWOJKI_SECOND_AND_HALF_PART_INBOUND = ((125, 2), RIGHT_FIRST_TURN_NAWOJKI)
NAWOJKI_SECOND_PART_OUTBOUND = ((132, -2), (-30, -2))
NAWOJKI_SECOND_AND_HALF_PART_OUTBOUND = (LEFT_FIRST_TURN_NAWOJKI, (132, -2))
NAWOJKI_THIRD_PART_INBOUND_I = ((RIGHT_FIRST_TURN_NAWOJKI[0], RIGHT_FIRST_TURN_NAWOJKI[1]), (288, 22))
NAWOJKI_THIRD_PART_INBOUND_II = ((288, 22), RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA)
NAWOJKI_THIRD_PART_OUTBOUND_I = (LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA, (288, 18))
NAWOJKI_THIRD_PART_OUTBOUND_II = ((288, 18), LEFT_FIRST_TURN_NAWOJKI)
NAWOJKI_LAST_PART_INBOUND = ((RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2), RIGHT_NAWOJKI_END)
NAWOJKI_LAST_PART_OUTBOUND = (LEFT_NAWOJKI_END, (LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2))

NAWOJKI_FIRST_TURN_RIGHT = turn_road(RIGHT_FIRST_TURN_NAWOJKI, (RIGHT_FIRST_TURN_NAWOJKI[0] + 4, RIGHT_FIRST_TURN_NAWOJKI[1] + 0.5), TURN_RIGHT, 15)
NAWOJKI_FIRST_TURN_LEFT = turn_road((LEFT_FIRST_TURN_NAWOJKI[0] - 1, LEFT_FIRST_TURN_NAWOJKI[1] - 1), LEFT_FIRST_TURN_NAWOJKI, TURN_LEFT, 10)

RIGHT_ALEJA_KIJOWSKA_ROAD = ((RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 8, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] - 7), RIGHT_ALEJA_KIJOWSKA_START)
LEFT_ALEJA_KIJOWSKA_ROAD = (LEFT_ALEJA_KIJOWSKA_START, (LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 4, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] - 3))

KIJOWSKA_NAWOJKI_TURN_RIGHT = turn_road((LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 4, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] - 3), (RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2), TURN_LEFT, 15)
KIJOWSKA_NAWOJKI_TURN_LEFT = turn_road((LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 4, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] - 3), LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA, TURN_RIGHT, 15)

NAWOJKI_KIJOWSKA_TURN_RIGHT = turn_road((LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2), (RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 8, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] - 7), TURN_RIGHT, 15)
NAWOJKI_KIJOWSKA_TURN_LEFT = turn_road(RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA, (RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 8, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] - 7), TURN_LEFT, 15)

RIGHT_KIJOWSKA_CROSSING_STRAIGHT = (RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA, (RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2))
LEFT_KIJOWSKA_CROSSING_STRAIGHT = ((LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2), LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA)

RIGHT_FIRST_PART_CZARNOWIEJSKA = (RIGHT_NAWOJKI_END, RIGHT_CZARNOWIEJSKA_FIRST_PART)
LEFT_FIRST_PART_CZARNOWIEJSKA = (LEFT_CZARNOWIEJSKA_FIRST_PART, LEFT_NAWOJKI_END)

NAWOJKI_SECOND_TURN_RIGHT = turn_road((RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 10, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2), (RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 11, RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1] + 2), TURN_RIGHT, 15)
NAWOJKI_SECOND_TURN_LEFT = turn_road((LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0] + 1, LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1]), LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA, TURN_LEFT, 15)


RIGHT_MIECHOWSKA_ROAD = (RIGHT_MIECHOWSKA_END, (l2 + l3, 50))
LEFT_MIECHOWSKA_ROAD = ((l2 + l3 - 4, 50), LEFT_MIECHOWSKA_END)

URZEDNICZA_ROAD = (URZEDNICZNA_START, LEFT_CZARNOWIEJSKA_FIRST_PART)

RIGHT_SECOND_PART_CZARNOWIEJSKA = (RIGHT_CZARNOWIEJSKA_FIRST_PART, RIGHT_CZARNOWIEJSKA_SECOND_PART)
RIGHT_THIRD_PART_CZARNOWIEJSKA = ((-50 + l2 + l3 + l4 + l5, 68), (750, 80))
RIGHT_FOURTH_PART_CZARNOWIEJSKA = ((750, 80), (780, 82))
RIGHT_FIFTH_PART_CZARNOWIEJSKA = ((780, 82), (870, 84))
LEFT_SECOND_PART_CZARNOWIEJSKA = (LEFT_CZARNOWIEJSKA_SECOND_PART, LEFT_CZARNOWIEJSKA_FIRST_PART)
LEFT_THIRD_PART_CZARNOWIEJSKA = ((750, 76), (-50 + l2 + l3 + l4 + l5, 64))
LEFT_FOURTH_PART_CZARNOWIEJSKA = ((800, 78), (750, 76))
LEFT_FIFTH_PART_CZARNOWIEJSKA = ((870, 80), (800, 78))

CZARNOWIEJSKA_FIRST_TURN_RIGHT = turn_road(RIGHT_CZARNOWIEJSKA_FIRST_PART, (RIGHT_CZARNOWIEJSKA_FIRST_PART[0] + 1, RIGHT_CZARNOWIEJSKA_FIRST_PART[1] + 1), TURN_RIGHT, 15)
CZARNOWIEJSKA_FIRST_TURN_LEFT = turn_road((LEFT_CZARNOWIEJSKA_FIRST_PART[0] - 1, LEFT_CZARNOWIEJSKA_FIRST_PART[1] - 1), LEFT_CZARNOWIEJSKA_FIRST_PART, TURN_LEFT, 15)

BUS_LINE_NAWOJKI_FIRST_PART_I = ((-10, 6), (85, 6))
BUS_LINE_NAWOJKI_FIRST_PART_II = ((85, 6), (100, 6))
BUS_FIRST_JOIN = ((-29, 2.5), (-8, 6.5))
BUS_FIRST_MERGE = ((100, 6), (125, 2))

BUS_SECOND_JOIN = ((130, 2.5), (158, 6.5))
BUS_LINE_CZARNOWIEJSKA = ((155, 6), (185, 6))
BUS_LINE_CZARNOWIEJSKA_II = ((185, 6), (260, 20))
BUS_SECOND_MERGE = ((260, 20), (288, 22))

BUS_THIRD_JOIN = ((l2 + l3, 50), (400, 53))
BUS_LINE_CZARNOWIEJSKA_III = ((400, 53), (468, 49))
BUS_LINE_CZARNOWIEJSKA_IV = ((467.5, 49), (565, 67.5))
BUS_THIRD_MERGE = ((565, 67.5), (585, 67))

BUS_FOURTH_JOIN = ((585, 67), (610, 74))
BUS_LINE_CZARNOWIEJSKA_V = ((610, 74), (755, 84.5))
BUS_FOURTH_MERGE = ((755, 84.5), (779, 82.5))

MIECHOWSKA_TURN_RIGHT = turn_road(RIGHT_NAWOJKI_END, (l2 + l3 - 4, 50), TURN_RIGHT, 15)

sim.create_roads([
    # index-0
    NAWOJKI_FIRST_PART_INBOUND,
    RIGHT_NAWOJKI_FIRST_AND_SECOND_PART,
    RIGHT_CROSSING,
    NAWOJKI_SECOND_PART_INBOUND,
    NAWOJKI_SECOND_AND_HALF_PART_INBOUND,
    NAWOJKI_THIRD_PART_INBOUND_I,
    NAWOJKI_THIRD_PART_INBOUND_II,
    RIGHT_KIJOWSKA_CROSSING_STRAIGHT,
    NAWOJKI_LAST_PART_INBOUND,

    NAWOJKI_FIRST_PART_OUTBOUND,
    LEFT_NAWOJKI_FIRST_AND_SECOND_PART,
    LEFT_CROSSING,
    NAWOJKI_SECOND_PART_OUTBOUND,
    NAWOJKI_SECOND_AND_HALF_PART_OUTBOUND,
    #index-14
    NAWOJKI_THIRD_PART_OUTBOUND_II,
    NAWOJKI_THIRD_PART_OUTBOUND_I,
    LEFT_KIJOWSKA_CROSSING_STRAIGHT,
    NAWOJKI_LAST_PART_OUTBOUND,

    # index-18
    LEFT_ALEJA_KIJOWSKA_ROAD,
    RIGHT_ALEJA_KIJOWSKA_ROAD,


    RIGHT_MIECHOWSKA_ROAD,
    LEFT_MIECHOWSKA_ROAD,

    RIGHT_FIRST_PART_CZARNOWIEJSKA,
    LEFT_FIRST_PART_CZARNOWIEJSKA,

    URZEDNICZA_ROAD,

    #index-25
    LEFT_SECOND_PART_CZARNOWIEJSKA,
    LEFT_THIRD_PART_CZARNOWIEJSKA,
    LEFT_FOURTH_PART_CZARNOWIEJSKA,
    LEFT_FIFTH_PART_CZARNOWIEJSKA,
    RIGHT_SECOND_PART_CZARNOWIEJSKA,
    RIGHT_THIRD_PART_CZARNOWIEJSKA,
    RIGHT_FOURTH_PART_CZARNOWIEJSKA,
    RIGHT_FIFTH_PART_CZARNOWIEJSKA,

    #index-33
    BUS_LINE_NAWOJKI_FIRST_PART_I,
    BUS_LINE_NAWOJKI_FIRST_PART_II,
    BUS_FIRST_JOIN,
    BUS_FIRST_MERGE,

    BUS_SECOND_JOIN,
    BUS_LINE_CZARNOWIEJSKA,
    BUS_LINE_CZARNOWIEJSKA_II,
    BUS_SECOND_MERGE,

    BUS_THIRD_JOIN,
    BUS_LINE_CZARNOWIEJSKA_III,
    BUS_LINE_CZARNOWIEJSKA_IV,
    BUS_THIRD_MERGE,

    BUS_FOURTH_JOIN,
    BUS_LINE_CZARNOWIEJSKA_V,
    BUS_FOURTH_MERGE,

    # index-48
    *KIJOWSKA_NAWOJKI_TURN_RIGHT,
    *KIJOWSKA_NAWOJKI_TURN_LEFT,

    *NAWOJKI_KIJOWSKA_TURN_RIGHT,
    *NAWOJKI_KIJOWSKA_TURN_LEFT,

    *NAWOJKI_SECOND_TURN_RIGHT,
    *NAWOJKI_SECOND_TURN_LEFT,

    *CZARNOWIEJSKA_FIRST_TURN_RIGHT,
    *CZARNOWIEJSKA_FIRST_TURN_LEFT
    # *MIECHOWSKA_TURN_RIGHT

])

sim.create_signal([[6, 17], [18]])
sim.create_signal([[1, 12]])
sim.create_signal([[28, 31, 47]])

sim.create_pedestrian_crossing((-30 + pedestrian_crossing_position_len, 0), (NAWOJKI_SECOND_PART_INBOUND, NAWOJKI_SECOND_AND_HALF_PART_OUTBOUND))
print(-30 + pedestrian_crossing_position_len)
print(-50 + l2)
sim.create_pedestrian_crossing((288, 22), (NAWOJKI_THIRD_PART_INBOUND_I, NAWOJKI_THIRD_PART_OUTBOUND_I))
sim.create_pedestrian_crossing((288, 22), (NAWOJKI_THIRD_PART_INBOUND_I, NAWOJKI_THIRD_PART_OUTBOUND_I))
sim.create_pedestrian_crossing((-50 + l2 + l3 + l4 + l5, 68), (RIGHT_SECOND_PART_CZARNOWIEJSKA, LEFT_THIRD_PART_CZARNOWIEJSKA))
def road(a): return range(a, a+15)


f = open("params.txt", "r")
params_list = f.read().split("\n")
vmax = int(params_list[0])
bus_vmax = math.ceil(int(params_list[0])/2)
buses_vehicle_rate = int(params_list[1])
traffic_level = int(params_list[3])

sim.create_gen({
    'vehicle_rate': traffic_level,
    'vehicles': [
        [3, {'path': [0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 29 ,30, 31, 32] , 'v_max':vmax}],
        [1, {'path': [18, *road(48 + 15), 15, 14, 13, 12, 11, 10, 9] , 'v_max':vmax}],
        [3, {'path': [18, *road(48), 8, 21] , 'v_max':vmax}],
        [4, {'path': [28, 27, 26, 25, 23, 17, 16, 15, 14, 13, 12, 11, 10] , 'v_max':vmax}]
    ]}
)

sim.create_gen({
    'vehicle_rate': buses_vehicle_rate,
    'vehicles': [
        [4, {'path': [0, 1, 2, 35, 33, 34, 36, 37, 38, 39, 40,6, 7, 8, 41, 42, 43, 44, 45, 46, 47, 32], 'l': 8, 'v_max': bus_vmax}]
    ]}
)

sim.create_pedestrian_gen()

win = Window(sim)
win.zoom = 1.5
asyncio.run(win.run(steps_per_update=5))
