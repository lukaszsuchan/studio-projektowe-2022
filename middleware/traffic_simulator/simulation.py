from .pedestrian_crossing import PedestrianCrossing
from .pedestrian_generator import PedestrianGenerator
from .road import Road
from copy import deepcopy
from .vehicle_generator import VehicleGenerator
from .traffic_signal import TrafficSignal


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