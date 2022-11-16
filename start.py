def curve_points(start, end, control, resolution=5):
	# If curve is a straight line
	if (start[0] - end[0])*(start[1] - end[1]) == 0:
		return [start, end]

	# If not return a curve
	path = []

	for i in range(resolution+1):
		t = i/resolution
		x = (1-t)**2 * start[0] + 2*(1-t)*t * control[0] + t**2 *end[0]
		y = (1-t)**2 * start[1] + 2*(1-t)*t * control[1] + t**2 *end[1]
		path.append((x, y))

	return path

def curve_road(start, end, control, resolution=15):
	points = curve_points(start, end, control, resolution=resolution)
	return [(points[i-1], points[i]) for i in range(1, len(points))]

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


from scipy.spatial import distance
from collections import deque

class Road:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.vehicles = deque()

        self.init_properties()

    def init_properties(self):
        self.length = distance.euclidean(self.start, self.end)
        self.angle_sin = (self.end[1]-self.start[1]) / self.length
        self.angle_cos = (self.end[0]-self.start[0]) / self.length
        # self.angle = np.arctan2(self.end[1]-self.start[1], self.end[0]-self.start[0])
        self.has_traffic_signal = False

    def set_traffic_signal(self, signal, group):
        self.traffic_signal = signal
        self.traffic_signal_group = group
        self.has_traffic_signal = True

    @property
    def traffic_signal_state(self):
        if self.has_traffic_signal:
            i = self.traffic_signal_group
            return self.traffic_signal.current_cycle[i]
        return True

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

from copy import deepcopy

class Simulation:
    def __init__(self, config={}):
        # Set default configuration
        self.set_default_config()

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def set_default_config(self):
        self.t = 0.0            # Time keeping
        self.frame_count = 0    # Frame count keeping
        self.dt = 1/60          # Simulation time step
        self.roads = []         # Array to store roads
        self.generators = []
        self.traffic_signals = []

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

    def create_signal(self, roads, config={}):
        roads = [[self.roads[i] for i in road_group] for road_group in roads]
        sig = TrafficSignal(roads, config)
        self.traffic_signals.append(sig)
        return sig

    def update(self):
        # Update every road
        for road in self.roads:
            road.update(self.dt)

        # Add vehicles
        for gen in self.generators:
            gen.update()

        for signal in self.traffic_signals:
            signal.update(self)

        # Check roads for out of bounds vehicle
        for road in self.roads:
            # If road has no vehicles, continue
            if len(road.vehicles) == 0: continue
            # If not
            vehicle = road.vehicles[0]
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
        cycle_length = 30
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
        self.sqrt_ab = 2*np.sqrt(self.a_max*self.b_max)
        self._v_max = self.v_max

    def update(self, lead, dt):
        # Update position and velocity
        if self.v + self.a*dt < 0:
            self.x -= 1/2*self.v*self.v/self.a
            self.v = 0
        else:
            self.v += self.a*dt
            self.x += self.v*dt + self.a*dt*dt/2
        
        # Update acceleration
        alpha = 0
        if lead:
            delta_x = lead.x - self.x - lead.l
            delta_v = self.v - lead.v

            alpha = (self.s0 + max(0, self.T*self.v + delta_v*self.v/self.sqrt_ab)) / delta_x

        self.a = self.a_max * (1-(self.v/self.v_max)**4 - alpha**2)

        if self.stopped: 
            self.a = -self.b_max*self.v/self.v_max
        
    def stop(self):
        self.stopped = True

    def unstop(self):
        self.stopped = False

    def slow(self, v):
        self.v_max = v

    def unslow(self):
        self.v_max = self._v_max
        


from numpy.random import randint

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
        r = randint(1, total+1)
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
            if len(road.vehicles) == 0\
               or road.vehicles[-1].x > self.upcoming_vehicle.s0 + self.upcoming_vehicle.l:
                # If there is space for the generated vehicle; add it
                self.upcoming_vehicle.time_added = self.sim.t
                road.vehicles.append(self.upcoming_vehicle)
                # Reset last_added_time and upcoming_vehicle
                self.last_added_time = self.sim.t
            self.upcoming_vehicle = self.generate_vehicle()


from pygame import gfxdraw

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
        self.offset = (0, 0)

        self.mouse_last = (0, 0)
        self.mouse_down = False


    def loop(self, loop=None):
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
                        self.mouse_last = (x-x0*self.zoom, y-y0*self.zoom)
                        self.mouse_down = True
                    if event.button == 4:
                        # Mouse wheel up
                        self.zoom *=  (self.zoom**2+self.zoom/4+1) / (self.zoom**2+1)
                    if event.button == 5:
                        # Mouse wheel down 
                        self.zoom *= (self.zoom**2+1) / (self.zoom**2+self.zoom/4+1)
                elif event.type == pygame.MOUSEMOTION:
                    # Drag content
                    if self.mouse_down:
                        x1, y1 = self.mouse_last
                        x2, y2 = pygame.mouse.get_pos()
                        self.offset = ((x2-x1)/self.zoom, (y2-y1)/self.zoom)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_down = False

    def run(self, steps_per_update=1):
        """Runs the simulation by updating in every loop."""
        def loop(sim):
            sim.run(steps_per_update)
        self.loop(loop)

    def convert(self, x, y=None):
        """Converts simulation coordinates to screen coordinates"""
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return (
            int(self.width/2 + (x + self.offset[0])*self.zoom),
            int(self.height/2 + (y + self.offset[1])*self.zoom)
        )

    def inverse_convert(self, x, y=None):
        """Converts screen coordinates to simulation coordinates"""
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return (
            int(-self.offset[0] + (x - self.width/2)/self.zoom),
            int(-self.offset[1] + (y - self.height/2)/self.zoom)
        )


    def background(self, r, g, b):
        """Fills screen with one color."""
        self.screen.fill((r, g, b))

    def line(self, start_pos, end_pos, color):
        """Draws a line."""
        gfxdraw.line(
            self.screen,
            *start_pos,
            *end_pos,
            color
        )

    def rect(self, pos, size, color):
        """Draws a rectangle."""
        gfxdraw.rectangle(self.screen, (*pos, *size), color)

    def box(self, pos, size, color):
        """Draws a rectangle."""
        gfxdraw.box(self.screen, (*pos, *size), color)

    def circle(self, pos, radius, color, filled=True):
        gfxdraw.aacircle(self.screen, *pos, radius, color)
        if filled:
            gfxdraw.filled_circle(self.screen, *pos, radius, color)



    def polygon(self, vertices, color, filled=True):
        gfxdraw.aapolygon(self.screen, vertices, color)
        if filled:
            gfxdraw.filled_polygon(self.screen, vertices, color)

    def rotated_box(self, pos, size, angle=None, cos=None, sin=None, centered=True, color=(0, 0, 255), filled=True):
        """Draws a rectangle center at *pos* with size *size* rotated anti-clockwise by *angle*."""
        x, y = pos
        l, h = size

        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        
        vertex = lambda e1, e2: (
            x + (e1*l*cos + e2*h*sin)/2,
            y + (e1*l*sin - e2*h*cos)/2
        )

        if centered:
            vertices = self.convert(
                [vertex(*e) for e in [(-1,-1), (-1, 1), (1,1), (1,-1)]]
            )
        else:
            vertices = self.convert(
                [vertex(*e) for e in [(0,-1), (0, 1), (2,1), (2,-1)]]
            )

        self.polygon(vertices, color, filled=filled)

    def rotated_rect(self, pos, size, angle=None, cos=None, sin=None, centered=True, color=(0, 0, 255)):
        self.rotated_box(pos, size, angle=angle, cos=cos, sin=sin, centered=centered, color=color, filled=False)

    def arrow(self, pos, size, angle=None, cos=None, sin=None, color=(150, 150, 190)):
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        
        self.rotated_box(
            pos,
            size,
            cos=(cos - sin) / np.sqrt(2),
            sin=(cos + sin) / np.sqrt(2),
            color=color,
            centered=False
        )

        self.rotated_box(
            pos,
            size,
            cos=(cos + sin) / np.sqrt(2),
            sin=(sin - cos) / np.sqrt(2),
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

    def draw_grid(self, unit=50, color=(150,150,150)):
        x_start, y_start = self.inverse_convert(0, 0)
        x_end, y_end = self.inverse_convert(self.width, self.height)

        n_x = int(x_start / unit)
        n_y = int(y_start / unit)
        m_x = int(x_end / unit)+1
        m_y = int(y_end / unit)+1

        for i in range(n_x, m_x):
            self.line(
                self.convert((unit*i, y_start)),
                self.convert((unit*i, y_end)),
                color
            )
        for i in range(n_y, m_y):
            self.line(
                self.convert((x_start, unit*i)),
                self.convert((x_end, unit*i)),
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

            # Draw road arrow
            if road.length > 5: 
                for i in np.arange(-0.5*road.length, 0.5*road.length, 10):
                    pos = (
                        road.start[0] + (road.length/2 + i + 3) * road.angle_cos,
                        road.start[1] + (road.length/2 + i + 3) * road.angle_sin
                    )

                    self.arrow(
                        pos,
                        (-1.25, 0.2),
                        cos=road.angle_cos,
                        sin=road.angle_sin
                    )   
            


            # TODO: Draw road arrow

    def draw_vehicle(self, vehicle, road):
        l, h = vehicle.l,  2
        sin, cos = road.angle_sin, road.angle_cos

        x = road.start[0] + cos * vehicle.x 
        y = road.start[1] + sin * vehicle.x 

        self.rotated_box((x, y), (l, h), cos=cos, sin=sin, centered=True)

    def draw_vehicles(self):
        for road in self.sim.roads:
            # Draw vehicles
            for vehicle in road.vehicles:
                self.draw_vehicle(vehicle, road)

    def draw_signals(self):
        for signal in self.sim.traffic_signals:
            for i in range(len(signal.roads)):
                color = (0, 255, 0) if signal.current_cycle[i] else (255, 0, 0)
                for road in signal.roads[i]:
                    a = 0
                    position = (
                        (1-a)*road.end[0] + a*road.start[0],        
                        (1-a)*road.end[1] + a*road.start[1]
                    )
                    self.rotated_box(
                        position,
                        (1, 3),
                        cos=road.angle_cos, sin=road.angle_sin,
                        color=color)

    def draw_status(self):
        text_fps = self.text_font.render(f't={self.sim.t:.5}', False, (0, 0, 0))
        text_frc = self.text_font.render(f'n={self.sim.frame_count}', False, (0, 0, 0))
        
        self.screen.blit(text_fps, (0, 0))
        self.screen.blit(text_frc, (100, 0))


    def draw(self):
        # Fill background
        self.background(*self.bg_color)

        # Major and minor grid and axes
        self.draw_grid(4, (220,220,220))
        # self.draw_grid(100, (200,200,200))
        # self.draw_axes()

        self.draw_roads()
        self.draw_vehicles()
        self.draw_signals()

        # Draw status info
        self.draw_status()
        
from traffic_simulator import *

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

l1 = round(abs(UP_LEFT[1] - TRAFFIC_SIGNALS_AWITEKS[1]) * SCALE)
l2 = round(abs(TRAFFIC_SIGNALS_AWITEKS[1] - NAWOJKI_FIRST_TURN[1]) * SCALE)
l3 = round(abs(NAWOJKI_FIRST_TURN[1] - TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[1]) * SCALE)
l4 = round(abs(ALEJA_KIJOWSKA[1] - TRAFFIC_SIGNALS_CZARNOWIEJSKA[1]) * SCALE) - 80
l5 = round(abs(TRAFFIC_SIGNALS_CZARNOWIEJSKA[1] - CZARNOWIEJSKA_CROSSING[1]) * SCALE) - 100
print(l1)
print(l2)
d1 = round(abs(ALEJA_KIJOWSKA[0] - TRAFFIC_SIGNALS_ALEJA_KIJOWSKA[0]) * SCALE)
print(d1)

NAWOJKI_RIGHT_START = (-50 - l1, 4)
NAWOJKI_LEFT_START = (-50 - l1, -4)
WEST_RIGHT = (-50, 4)
WEST_LEFT = (-50, -4)
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
RIGHT_NAWOJKI_FIRST_AND_SECOND_PART = (WEST_RIGHT, (-35, 2))
RIGHT_CROSSING = ((-35, 2), (-30, 2))
LEFT_NAWOJKI_FIRST_AND_SECOND_PART = ((-30, -2), WEST_LEFT)
LEFT_CROSSING = ((-30, -2), (-35, -2))
NAWOJKI_SECOND_PART_INBOUND = ((-30, 2), RIGHT_FIRST_TURN_NAWOJKI)
NAWOJKI_SECOND_PART_OUTBOUND = (LEFT_FIRST_TURN_NAWOJKI, (-30, -2))
NAWOJKI_THIRD_PART_INBOUND = ((RIGHT_FIRST_TURN_NAWOJKI[0] + 4, RIGHT_FIRST_TURN_NAWOJKI[1] + 0.5), RIGHT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA)
NAWOJKI_THIRD_PART_OUTBOUND = (LEFT_TRAFFIC_SIGNALS_ALEJA_KIJOWSKA, LEFT_FIRST_TURN_NAWOJKI)
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
LEFT_SECOND_PART_CZARNOWIEJSKA = (LEFT_CZARNOWIEJSKA_SECOND_PART, LEFT_CZARNOWIEJSKA_FIRST_PART)

CZARNOWIEJSKA_FIRST_TURN_RIGHT = turn_road(RIGHT_CZARNOWIEJSKA_FIRST_PART, (RIGHT_CZARNOWIEJSKA_FIRST_PART[0] + 1, RIGHT_CZARNOWIEJSKA_FIRST_PART[1] + 1), TURN_RIGHT, 15)
CZARNOWIEJSKA_FIRST_TURN_LEFT = turn_road((LEFT_CZARNOWIEJSKA_FIRST_PART[0] - 1, LEFT_CZARNOWIEJSKA_FIRST_PART[1] - 1), LEFT_CZARNOWIEJSKA_FIRST_PART, TURN_LEFT, 15)

sim.create_roads([
    # index-0
    NAWOJKI_FIRST_PART_INBOUND,
    RIGHT_NAWOJKI_FIRST_AND_SECOND_PART,
    RIGHT_CROSSING,
    NAWOJKI_SECOND_PART_INBOUND,
    NAWOJKI_THIRD_PART_INBOUND,
    RIGHT_KIJOWSKA_CROSSING_STRAIGHT,
    NAWOJKI_LAST_PART_INBOUND,

    NAWOJKI_FIRST_PART_OUTBOUND,
    LEFT_NAWOJKI_FIRST_AND_SECOND_PART,
    LEFT_CROSSING,
    NAWOJKI_SECOND_PART_OUTBOUND,
    NAWOJKI_THIRD_PART_OUTBOUND,
    LEFT_KIJOWSKA_CROSSING_STRAIGHT,
    NAWOJKI_LAST_PART_OUTBOUND,

    # index-14
    LEFT_ALEJA_KIJOWSKA_ROAD,
    RIGHT_ALEJA_KIJOWSKA_ROAD,


    RIGHT_MIECHOWSKA_ROAD,
    LEFT_MIECHOWSKA_ROAD,

    RIGHT_FIRST_PART_CZARNOWIEJSKA,
    LEFT_FIRST_PART_CZARNOWIEJSKA,

    URZEDNICZA_ROAD,

    LEFT_SECOND_PART_CZARNOWIEJSKA,
    RIGHT_SECOND_PART_CZARNOWIEJSKA,

    # index-23
    *KIJOWSKA_NAWOJKI_TURN_RIGHT,
    *KIJOWSKA_NAWOJKI_TURN_LEFT,

    *NAWOJKI_KIJOWSKA_TURN_RIGHT,
    *NAWOJKI_KIJOWSKA_TURN_LEFT,

    *NAWOJKI_SECOND_TURN_RIGHT,
    *NAWOJKI_SECOND_TURN_LEFT,

    *NAWOJKI_FIRST_TURN_RIGHT,
    # *NAWOJKI_FIRST_TURN_LEFT,
    *CZARNOWIEJSKA_FIRST_TURN_RIGHT,
    *CZARNOWIEJSKA_FIRST_TURN_LEFT

])

sim.create_signal([[4, 13] ,[14]])

def road(a): return range(a, a+15)

sim.create_gen({
    'vehicle_rate': 60,
    'vehicles': [
        [3, {'path': [0, 1, 2, *road(23 + 6*15), 3, 4, 5, 6]}],
        # [1, {'path': [14, *road(23 + 15), 11]}],
        # [3, {'path': [14, *road(23), 6]}],
        # [1, {'path': [4, *road(23 + 45), 15]}]
    ]}
)

win = Window(sim)
win.zoom = 1.5
win.run(steps_per_update=5)

