import carla
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.tools.misc import distance_vehicle, draw_waypoints
from custom_vehicle_pid_controller import CustomVehiclePIDController


class CustomLocalPlanner(LocalPlanner):
    """
        LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
        The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
        and the other for the longitudinal control (cruise speed).

        When multiple paths are available (intersections) this local planner makes a random choice.
        """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 1.0

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        super(CustomLocalPlanner, self).__init__(vehicle, opt_dict)

    def __del__(self):
        print("No destroy")
        # if self._vehicle:
        #     self._vehicle.destroy()
        # print("Destroying ego-vehicle!")

    def set_target_speed(self, target_speed):
        if self._vehicle:
            self._target_speed = target_speed + 2

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 30.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0,
            'K_I': 2,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = CustomVehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed
        self._dt = 1 / speed

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) <= self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        return control