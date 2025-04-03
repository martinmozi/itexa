from websocket_server import ItexaWebSocketServer
import json
import threading
import time


class Simulation:
    def __init__(self, itexaWebSocketServer: ItexaWebSocketServer):
        self.simulationThread = None
        self.simulationBreakEvent = threading.Event()
        self.itexaWebSocketServer = itexaWebSocketServer

    def _simulate(self, timeStep: float, water_level: float, hole_height: float, hole_diameter: float, tank_width: float, g=9.81):
        """
        Simulate water flowing from a tank through a hole.
        Args:
            timeStep (float): Time step in seconds for example 0.1.
            water_level (float): Water level height in cm.
            hole_height (float): Height of the hole from bottom in cm.
            hole_diameter (float): Diameter of the hole in cm.
            tank_width (float): Width of the tank in cm.
            g (float): Acceleration due to gravity in m/s^2. Default is 9.81 m/s^2.
        """
        # Convert from cm to m for calculations
        water_level_m = water_level / 100
        hole_height_m = hole_height / 100
        hole_diameter_m = hole_diameter / 100
        tank_width_m = tank_width / 100

        current_time = 0.0
        current_water_level_m = water_level_m

        # Calculate hole area in m²
        hole_area = 3.14159 * (hole_diameter_m / 2) ** 2

        # Simulate until water level drops below hole height
        while current_water_level_m > hole_height_m:
            # Calculate height of water above hole in m
            height_above_hole = current_water_level_m - hole_height_m

            # Calculate flow velocity using Torricelli's equation (m/s)
            velocity = (2 * g * height_above_hole) ** 0.5

            # Calculate flow rate (m³/s)
            flow_rate = velocity * hole_area

            # Calculate water level drop
            volume_lost = flow_rate * timeStep  # m³
            tank_area = tank_width_m * tank_width_m  # m² (assuming square tank)
            level_drop = volume_lost / tank_area  # m

            # Update water level
            current_water_level_m -= level_drop

            # Calculate horizontal distance water travels
            # For water coming out horizontally from the hole height
            # Time it takes to fall from hole to ground
            time_to_ground = (2 * hole_height_m / g) ** 0.5

            # Horizontal distance traveled (m)
            distance = velocity * time_to_ground

            # Send data (convert back to cm for display)
            data = {
                "method": "data",
                "time": round(current_time, 2),
                "water_level": round(current_water_level_m * 100, 2),  # Convert back to cm
                "flow_distance": round(distance * 100, 2),  # Convert back to cm
                "flow_rate": round(flow_rate * 1000, 4)  # Convert m³/s to L/s
            }
            json_message = json.dumps(data)
            self.itexaWebSocketServer.send_data(json_message)
            if self.simulationBreakEvent.is_set():
                self.simulationBreakEvent.clear()
                return

            time.sleep(timeStep)
            current_time += timeStep

    def simulate(self, limits, water_level: float, hole_height: float, hole_diameter: float, tank_width: float, timeStep=0.1, g=9.81):
        self.itexaWebSocketServer.send_data(
            json.dumps({
                "method": "init",
                "tank_width": tank_width,
                "tank_height": water_level * 1.2,  # Assuming tank height is 1.2 times the water level
                "hole_diameter": hole_diameter,
                "hole_height": hole_height,
                "water_distance": 2 * ((hole_height * (water_level - hole_height)) ** 0.5),
                "max_water_level": limits["MAX_WATER_LEVEL"],
                "max_hole_diameter": limits["MAX_HOLE_DIAMETER"],
                "max_tank_width": limits["MAX_TANK_WIDTH"]
            })
        )

        if self.simulationThread is not None and self.simulationThread.is_alive():
            self.simulationBreakEvent.set()
            self.simulationThread.join()

        self.simulationThread = threading.Thread(target=self._simulate, args=(timeStep, water_level, hole_height, hole_diameter, tank_width, g))
        self.simulationThread.start()
