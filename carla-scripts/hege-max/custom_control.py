#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.
"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    CONTROL MODE 
    ----------------------------------------
    O            : model driving             
    P            : autopilot driving 
    M            : manual driving

    OTHERS 
    ----------------------------------------
    R            : toggle recording images to disk
    U            : change model
    I            : toggle noise
    N            : next spawn point
    B            : previous spawn point
    Backspace    : restart episode/next route
    
    ESC          : quit
    F1           : toggle HUD
    H/?          : toggle help

    SENSORS
    ----------------------------------------
    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import glob
import os
import sys
# This is an alternative to running easy_install <carla egg file>
from typing import Optional, List

from agents.navigation.agent import Agent
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import get_nearest_traffic_light
from utils import init_tensorflow

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

FIXED_DELTA_SECONDS = 0.05

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

from pathlib import Path

from drive_models import LSTMKeras, CNNKeras, SegmentationModel

import cv2
import time
import ast
import pandas as pd

# Import ConfigParser for wheel
from enums import WeatherType, ControlType, EventType, Environment
from configparser import ConfigParser
from agents.navigation.local_planner import RoadOption

from misc import get_distance
from vehicle_spawner import VehicleSpawner
from helpers import is_valid_lane_change, set_green_traffic_light, get_models, get_route_distance

import argparse
import datetime
import logging
import math
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_u
    from pygame.locals import K_w
    from pygame.locals import K_KP1
    from pygame.locals import K_KP3
    from pygame.locals import K_KP4
    from pygame.locals import K_KP5
    from pygame.locals import K_KP6
    from pygame.locals import K_KP7
    from pygame.locals import K_KP8
    from pygame.locals import K_KP9
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [
        x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)
    ]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, client: carla.Client, carla_world: carla.World, hud: 'HUD', environment: Environment,
                 history: 'History', actor_filter, settings, hq_recording=False):
        self.client = client
        self.world = carla_world
        self.map = self.world.get_map()
        self.player = None  # type: Optional[carla.Vehicle]
        self._actor_filter = actor_filter

        # Other classes 
        self.hud = hud
        self.history = history
        self.camera_manager = None  # type: Optional[CameraManager]

        # Weather 
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._change_weather = None
        self._weather_type = None  # type: WeatherType

        self.evaluator = Evaluator(self.hud, self)

        # Default pawnpoints
        self._spawn_point_start = 1
        self._spawn_point_destination = 7

        # Eval mode 
        self._eval_mode = None
        self._eval_num = None
        self._eval_num_current = None
        self._eval_cars = None
        self._eval_cars_idx = None
        self._eval_routes = None
        self._eval_routes_idx = None  # Current evaluating route
        self._eval_route_idx = None  # Current route waypoint
        self._eval_weathers = None
        self._eval_weathers_idx = None
        self._eval_route_canceled = None

        # Route recording 
        self._new_spawn_point = None
        self._routes = None
        self._current_route_num = None
        self._tot_route_num = None
        self._auto_record = None

        # Vehicle spawning 
        self._num_vehicles_min = None
        self._num_vehicles_max = None
        self._num_walkers_min = None
        self._num_walkers_max = None
        self._spawning_radius = None
        self._vehicle_spawner = VehicleSpawner(self.client, self.world)

        # Other settings 
        self._environment = environment
        self._initialize_settings(settings)
        self._hq_recording = hq_recording
        # self.world.on_tick(hud.on_world_tick)
        self._current_traffic_light = 0
        self._client_ap_active = False
        self._quit_next = False
        self._auto_timeout = None

        # Init settings 
        self._initialize_settings(settings)

        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def _initialize_settings(self, settings):
        s = {}

        # Read recording mode settings
        routes = settings.get("Recording", "RecordingRoutes", fallback=None)
        self._auto_record = True if settings.get("Recording", "AutoRecord",
                                                 fallback="No").strip().lower() == "yes" else False

        # Read eval mode settings
        self._eval_mode = True if settings.get("Eval", "EvalMode", fallback="No").strip().lower() == "yes" else False
        self._eval_num = int(settings.get("Eval", "EvalNum", fallback=1))
        self._eval_num_current = 1
        eval_cars = settings.get("Eval", "EvalCars", fallback=None)
        eval_routes = settings.get("Eval", "EvalRoutes", fallback=None)
        eval_weathers = settings.get("Eval", "EvalWeathers", fallback=None)

        # Read vehicle settings 
        self._num_vehicles_min = int(settings.get("Spawning", "NumberOfVehiclesMin", fallback=0))
        self._num_vehicles_max = int(settings.get("Spawning", "NumberOfVehiclesMax", fallback=0))
        self._num_walkers_min = int(settings.get("Spawning", "NumberOfWalkersMin", fallback=0))
        self._num_walkers_max = int(settings.get("Spawning", "NumberOfWalkersMax", fallback=0))
        self._spawning_radius = float(settings.get("Spawning", "SpawnRadius", fallback=0))

        # Read autotimeout: The duration of an episode in server autopilot 
        self._auto_timeout = float(settings.get("Settings", "AutoTimeout", fallback=0))

        # Read weather settings 
        self._change_weather = True if settings.get("Weather", "ChangeWeather", fallback="No") == "Yes" else False
        self._weather_index = -1 if self._change_weather else 0
        self._weather_type = WeatherType[settings.get("Weather", "WeatherType", fallback="ALL")]

        # Parse settings from string to lists 
        if routes:
            routes = routes.split()
            routes = [ast.literal_eval(r) for r in routes]
            self._routes = [[int(r[0]), int(r[1])] for r in routes]
            self._current_route_num = 0
            self._tot_route_num = len(self._routes)
        if eval_routes:
            self._eval_routes_idx = 0
            self._eval_route_idx = 0
            self._eval_routes = []
            eval_routes = eval_routes.split()
            for eval_route in eval_routes:
                eval_route = [ast.literal_eval(e) for e in eval_route.split()][0]
                self._eval_routes.append(eval_route)
        if eval_cars:
            eval_cars = eval_cars.split()
            eval_cars = [ast.literal_eval(c) for c in eval_cars][0]
            self._eval_cars = eval_cars
            self._eval_cars_idx = 0
        if eval_weathers:
            eval_weathers = eval_weathers.split()
            eval_weathers = [ast.literal_eval(w) for w in eval_weathers][0]
            self._eval_weathers = eval_weathers
            self._eval_weathers_idx = 0

    def restart(self):

        self.history._initiate()

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0

        # Get a vehilce blueprint.
        blueprint = self.world.get_blueprint_library().filter(
            'vehicle.audi.etron')[0]
        # Set the vehicle as hero player 
        blueprint.set_attribute('role_name', 'hero')

        # Destroy player 
        if self.player is not None:
            self.destroy()
            self.player = None

        # Choose starting spawnpoint in eval routes
        if self._eval_mode is not None and self._eval_mode != False:
            self._eval_route_canceled = False
            spawn_point = self.map.get_spawn_points()[self._eval_routes[self._eval_routes_idx][0][0]]
            destination_point = self.map.get_spawn_points()[self._eval_routes[self._eval_routes_idx][-1][0]]
            self._spawn_point_start = self._eval_routes[self._eval_routes_idx][0][0]
            if self._eval_cars is not None:
                self._vehicle_spawner.spawn_nearby(
                    self._spawn_point_start,
                    self._eval_cars[self._eval_cars_idx],
                    self._eval_cars[self._eval_cars_idx],
                    0,
                    0,
                    self._spawning_radius)

                # Choose starting spawnpoint in recording routes
        elif self._new_spawn_point is None:
            if self._routes is not None:
                route_len = len(self._routes)
                if len(self._routes) > 0:
                    route = self._routes.pop(0)
                    self._spawn_point_start = route[0]
                    self._spawn_point_destination = route[1]
                else:
                    self._quit_next = True
            if self._current_route_num is not None and route_len > 0:
                self._current_route_num += 1
            spawn_point = self.map.get_spawn_points()[self._spawn_point_start]
            destination_point = self.map.get_spawn_points()[self._spawn_point_destination]

        # Choose starting spawnpoint normally     
        else:
            self._spawn_point_start = self._new_spawn_point
            spawn_point = self.map.get_spawn_points()[self._spawn_point_start]
            destination_point = self.map.get_spawn_points()[
                self._spawn_point_destination]  # TODO: what should be destination?
            self._new_spawn_point = None

        # Spawn player
        while self.player is None:
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # Progress simulation one step to set player position
        self.world.tick()

        # Create Autopilot 
        self.client_ap = BasicAgent(self.player)
        # self._client_ap = BasicAgent(self.player, dt=FIXED_DELTA_SECONDS)
        # self._client_ap._local_planner._dt = FIXED_DELTA_SECONDS
        self.client_ap.set_destination((destination_point.location.x,
                                        destination_point.location.y,
                                        destination_point.location.z))

        # Set up the sensors.
        self.camera_manager = CameraManager(self.player, self.client_ap, self.hud,
                                            self.history, self._eval_mode, self._hq_recording)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager._initiate_recording()

        if self._eval_mode is not None and self._eval_mode != False:
            self.evaluator.initialize_sensors(self.player)
            self.evaluator.new_episode()
            if self._eval_weathers is not None:
                self.set_weather(self._eval_weathers[self._eval_weathers_idx])

        # Turn on recording at new route 
        if self._auto_record:
            self.camera_manager.toggle_recording()
        actor_type = get_actor_display_name(self.player)

        # Spawn other vehicles, but not in eval mode  
        if not self._eval_mode and self._num_vehicles_max != 0 and self._spawning_radius is not None:
            self._vehicle_spawner.spawn_nearby(self._spawn_point_start, self._num_vehicles_min, self._num_vehicles_max,
                                               self._num_walkers_min, self._num_walkers_max, self._spawning_radius)

        # Change weather 
        if self._change_weather == True and self._eval_mode == False:
            self.next_weather(weather_type=self._weather_type)

        self.hud._episode_start_time = self.hud.simulation_time

    def set_weather(self, weather_idx):
        """ Set weather to given id"""
        self._weather_index = weather_idx
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.world.set_weather(preset[0])
        self.history.update_weather_index(self._weather_index)

    def next_weather(self, reverse=False, weather_type: WeatherType = None):
        """ Change weather to next weather """
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)

        # Only choose clear weather 
        if weather_type == WeatherType.CLEAR and self._weather_index > 4:
            self._weather_index = 0

        # Only choose rainy/wet weather 
        if weather_type == WeatherType.RAIN and self._weather_index < 5:
            self._weather_index = 5

        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

        self.history.update_weather_index(self._weather_index)

    def reset_weather(self):
        """ Set weather to first weather """
        preset = self._weather_presets[0]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_spawn_point(self):
        if len(self.map.get_spawn_points()) > self._spawn_point_start:
            self._new_spawn_point = self._spawn_point_start + 1
            self.restart()
            self.hud.notification('Next spawn point')
        else:
            self.hud.notification("No next spawn point")

    def previous_spawn_point(self, ):
        if self._spawn_point_start > 0:
            self._new_spawn_point = self._spawn_point_start - 1
            self.restart()
            self.hud.notification('Next spawn point')
        else:
            self.hud.notification("No previous spawn point")

    def tick(self, clock):

        self.hud.tick(self, clock)
        self.camera_manager.tick()

        if self._eval_mode:
            self.evaluator.tick()

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroySensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager._index = None

    def destroy(self):
        self._vehicle_spawner.destroy_vehicles()
        actors = [
            self.camera_manager.sensor,
            self.player
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        self.camera_manager._destroy_sensors()

        if self.evaluator:
            self.evaluator.destroy_sensors()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, settings, use_steering_wheel=False, models=None):
        self._world = world

        # Model 
        self._models = models
        self._drive_model = None
        self._current_model_idx = None
        self.segmentation_model = None  # type: Optional[SegmentationModel]

        # Settings
        self._steering_wheel_enabled = use_steering_wheel
        self._control_type = None
        self._red_lights_allowed = None
        self._noise_enabled = False
        self._noise_amount = None
        self._environment = None

        # History 
        self._lane_change_activated = None
        self._lane_change_started = False
        self._history_size = 10
        self._steer_history = []
        self._active_hlc = RoadOption.LANEFOLLOW

        # Initalize model 
        if self._models is not None:
            self._current_model_idx = -1
            self.next_model()

        self._initialize_settings(settings)

        self._entered_traffic_light_loc = None

        self._control = carla.VehicleControl()
        self._world.player.set_autopilot(self._control_type == ControlType.SERVER_AP)
        self._steer_cache = 0.0

        # initialize steering wheel
        if self._steering_wheel_enabled:
            self._initialize_steering_wheel()

    def _initialize_model(self):
        init_tensorflow()
        path, seq_length, sampling_interval, model_type, segmentation_network = self._models[self._current_model_idx]
        if model_type == "lstm":
            self._drive_model = LSTMKeras(path, sampling_interval)
        else:
            self._drive_model = CNNKeras(path)

        # Init segmentation model (whose output will be plotted)
        if segmentation_network is not None:
            self.segmentation_model = SegmentationModel(segmentation_network)

        self._world.hud._drive_model_name = '/'.join(path.split('/')[-2:])
        self._world.hud._drive_model_idx = self._current_model_idx
        self._world.hud._drive_model_num = len(self._models)

    def _initialize_settings(self, settings):
        self._control_type = ControlType[settings.get("Settings", "ControlType", fallback="MANUAL")]
        self._red_lights_allowed = True if settings.get("Settings", "RedLights", fallback="No") == "Yes" else False
        self._noise_amount = float(settings.get("Settings", "Noise", fallback="0"))
        if self._noise_amount != 0:
            self._noise_enabled = True
        if self._drive_model is None and self._control_type == ControlType.DRIVE_MODEL:
            self._control_type = ControlType.MANUAL

    def _initialize_steering_wheel(self):
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

    def next_model(self):
        if self._models is not None:
            self._current_model_idx += 1
            if len(self._models) > self._current_model_idx:
                self._initialize_model()
                self._world.evaluator.initialize_model()
                return True
            else:
                print("INFO: No more new models")
                return False
        else:
            print("ERROR: Failed to initialize model, list of models was None or empty")
            return False

    def _add_to_steer_history(self, steer):
        self._steer_history.insert(0, steer)
        if len(self._steer_history) > self._history_size:
            self._steer_history.pop()

    def parse_events(self, client, world: World, clock: pygame.time.Clock):
        if not self._red_lights_allowed and (not world._eval_mode or world._eval_cars[world._eval_cars_idx] == 0):
            set_green_traffic_light(world.player)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.camera_manager.toggle_camera()
                elif event.button == 10:
                    world.next_spawn_point()
                elif event.button == 11:
                    world.previous_spawn_point()
                elif event.button == 2:
                    world.camera_manager.toggle_recording()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == 4:
                    world.restart()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if world._eval_mode:
                        world._eval_route_canceled = True
                    else:
                        world.restart()
                        if self._control_type == ControlType.SERVER_AP:
                            world.player.set_autopilot(self._control_type == ControlType.SERVER_AP)
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h:
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_n:
                    if self._control_type == ControlType.SERVER_AP:
                        world.player.set_autopilot(self._control_type == ControlType.SERVER_AP)

                    world.next_spawn_point()
                elif event.key == K_b:
                    if self._control_type == ControlType.SERVER_AP:
                        world.player.set_autopilot(self._control_type == ControlType.SERVER_AP)
                    world.previous_spawn_point()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_i:
                    self._noise_enabled = not self._noise_enabled
                    noise_status = "Enabled" if self._noise_enabled else "Disabled"
                    world.hud.notification('Noise: ' + noise_status)
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control_type = ControlType.MANUAL
                    world.hud.notification('Control Mode: Manual')
                    world.player.set_autopilot(False)
                elif event.key == K_o:
                    if self._drive_model is not None:
                        self._control_type = ControlType.DRIVE_MODEL
                        world.hud.notification('Control Mode: Drive Model')
                        world.player.set_autopilot(False)
                elif event.key == K_p and pygame.key.get_mods() & KMOD_SHIFT:
                    self._control_type = ControlType.SERVER_AP
                    world.player.set_autopilot(True)
                    world.hud.notification('Control Mode: Server Autopilot')

                elif event.key == K_p:
                    self._control_type = ControlType.CLIENT_AP
                    world.hud.notification('Control Mode: Client Autopilot')
                    world.player.set_autopilot(False)
                    world._client_ap_active = not world._client_ap_active
                elif event.key == K_u:
                    # If multiple model testing is activated and there are remaining models to test 
                    if self._models is not None:
                        if self.next_model():
                            world.hud.notification('Change model')
                        else:
                            world.hud.notification('No more models left')
                elif event.key == K_KP1:
                    if self._control_type == ControlType.DRIVE_MODEL:
                        world.history.update_hlc(RoadOption.CHANGELANELEFT)
                        self._lane_change_activated = (world.hud.simulation_time, np.mean(self._steer_history),
                                                       world.map.get_waypoint(world.player.get_location()).lane_id,
                                                       RoadOption.CHANGELANELEFT)
                elif event.key == K_KP3:
                    if self._control_type == ControlType.DRIVE_MODEL:
                        world.history.update_hlc(RoadOption.CHANGELANERIGHT)
                        self._lane_change_activated = (world.hud.simulation_time, np.mean(self._steer_history),
                                                       world.map.get_waypoint(world.player.get_location()).lane_id,
                                                       RoadOption.CHANGELANERIGHT)
                elif event.key == K_KP8:
                    self._active_hlc = RoadOption.STRAIGHT
                    world.hud.notification('STRAIGHT')
                    world.history.update_hlc(RoadOption.STRAIGHT)

                elif event.key == K_KP4:
                    self._active_hlc = RoadOption.LEFT
                    world.hud.notification('LEFT')
                    world.history.update_hlc(RoadOption.LEFT)

                elif event.key == K_KP5:
                    self._active_hlc = RoadOption.LANEFOLLOW
                    world.hud.notification('LANE FOLLOW')
                    world.history.update_hlc(RoadOption.LANEFOLLOW)

                elif event.key == K_KP6:
                    self._active_hlc = RoadOption.RIGHT
                    world.hud.notification('RIGHT')
                    world.history.update_hlc(RoadOption.RIGHT)
                elif event.key == K_KP7:
                    self._active_hlc = RoadOption.CHANGELANELEFT
                    world.history.update_hlc(RoadOption.CHANGELANELEFT)
                    world.hud.notification('CHANGE LANE LEFT')
                elif event.key == K_KP9:
                    self._active_hlc = RoadOption.CHANGELANERIGHT
                    world.history.update_hlc(RoadOption.CHANGELANERIGHT)
                    world.hud.notification('CHANGE LANE RIGHT')

        world.history.control_type = self._control_type

        # Evaluationg multiple models 
        if world._eval_mode:
            if self._control_type != ControlType.DRIVE_MODEL:
                pass
                # print("ERROR: Evaluation mode expects control type DRIVE_MODEL, received " + self._control_type.name)
                # return True

            next_waypoint = world.map.get_spawn_points()[
                world._eval_routes[world._eval_routes_idx][world._eval_route_idx][0]]

            # Calculate distance to next waypoint 
            loc = world.player.get_transform().location
            dx = next_waypoint.location.x - loc.x
            dy = next_waypoint.location.y - loc.y
            distance = math.sqrt(dx * dx + dy * dy)

            # Vehicle has reached next waypoint or have experienced a catastrophic failure
            if distance < 4 or world._eval_route_canceled:
                # Get road option 
                _, road_option = world._eval_routes[world._eval_routes_idx][world._eval_route_idx]

                world.hud.notification(RoadOption(road_option).name)
                # Update HLC
                world.history.update_hlc(RoadOption(road_option))
                self._active_hlc = RoadOption(road_option)
                world.hud.notification(self._active_hlc.name)

                route_complete = world._eval_route_idx == len(world._eval_routes[world._eval_routes_idx]) - 1
                # Check if this is the last waypoint in route, or the route is canceled
                if route_complete or world._eval_route_canceled:
                    self._drive_model.restart()
                    world.evaluator.episode_complete(route_complete)

                    if len(world._eval_routes) - 1 == world._eval_routes_idx:
                        # At the end of the last route  
                        world._eval_routes_idx = 0
                        world._eval_route_idx = 0

                        # Check if there are available weathers left 
                        if world._eval_weathers is None or world._eval_weathers_idx == len(world._eval_weathers) - 1:
                            world._eval_weathers_idx = 0
                            # Check if there are available car-spawns left 
                            if world._eval_cars is None or world._eval_cars_idx == len(world._eval_cars) - 1:
                                world._eval_cars_idx = 0
                                # Check if model has been tested eval_num times 
                                if world._eval_num_current == world._eval_num:
                                    # If it has, check for available models left 
                                    if self._current_model_idx == len(self._models) - 1:
                                        # Exit
                                        return True
                                    else:
                                        world._eval_num_current = 1
                                        self.next_model()
                                else:
                                    world._eval_num_current += 1

                            else:
                                world._eval_cars_idx += 1

                        else:
                            world._eval_weathers_idx += 1

                        world.restart()



                    else:

                        # Next route 
                        world._eval_route_idx = 0
                        world._eval_routes_idx += 1
                        world.restart()
                else:

                    # Update next waypoint id
                    world._eval_route_idx += 1

                    # Server Autopilot Driving
        if self._control_type == ControlType.SERVER_AP:
            world.player.set_autopilot(self._control_type == ControlType.SERVER_AP)

            # Update lane check 
            is_left = is_valid_lane_change(RoadOption.CHANGELANELEFT, world)
            world.history.update_left_lane_change_valid(is_left)

            is_right = is_valid_lane_change(RoadOption.CHANGELANERIGHT, world)
            world.history.update_right_lane_change_valid(is_right)

            # If RouteRecording is activated  
            if world._routes is not None:
                # Exit game if no more routes are available  
                if world._current_route_num - 1 == world._tot_route_num:
                    return True
                # Change route when it has driven for more than set route duration 
                if world._auto_timeout != 0 and world.hud.simulation_time - world.hud._episode_start_time > world._auto_timeout:
                    world.restart()

                    # Manual Driving
        elif self._control_type == ControlType.MANUAL:

            # Update lane check 
            is_left = is_valid_lane_change(RoadOption.CHANGELANELEFT, world)
            world.history.update_left_lane_change_valid(is_left)

            is_right = is_valid_lane_change(RoadOption.CHANGELANERIGHT, world)
            world.history.update_right_lane_change_valid(is_right)

            # Steering wheel activated 
            if self._steering_wheel_enabled:
                self._parse_vehicle_wheel()
            else:
                self._parse_vehicle_keys(world, pygame.key.get_pressed(),
                                         clock.get_time())
            self._control.reverse = self._control.gear < 0

        # Model Driving 
        elif self._control_type == ControlType.DRIVE_MODEL:
            self._parse_drive_model_commands(world)

        # Client Autopilot Driving 
        elif self._control_type == ControlType.CLIENT_AP:
            # Update HLC to current RoadOption 
            world.history.update_hlc(world.client_ap._local_planner._target_road_option)

            # Update lane check 
            is_left = is_valid_lane_change(RoadOption.CHANGELANELEFT, world)
            world.history.update_left_lane_change_valid(is_left)

            is_right = is_valid_lane_change(RoadOption.CHANGELANERIGHT, world)
            world.history.update_right_lane_change_valid(is_right)

            self._parse_client_ap(world)

            # Calculate distance to destination  
            destination_waypoint = world.map.get_spawn_points()[world._spawn_point_destination]
            loc = world.player.get_transform().location
            dx = destination_waypoint.location.x - loc.x
            dy = destination_waypoint.location.y - loc.y
            distance = math.sqrt(dx * dx + dy * dy)

            # Change route if client AP has reached its destination
            if distance < 25:
                world.hud.notification("Route Complete")
                world.restart()
                # Exit program if all routes are finished 
                if world._quit_next:
                    return True

        # Plot segmentation output
        if self.segmentation_model is not None and "forward_center_rgb" in world.history._latest_images.keys():
            self.segmentation_model.plot_segmentation(world.history._latest_images)

        # Automatic Lane Change
        if self._lane_change_activated != None:
            activated, original_steer, original_lane, option = self._lane_change_activated
            if self._lane_change_started or is_valid_lane_change(option, world):
                self._lane_change_started = True
                direction = 1 if option == RoadOption.CHANGELANELEFT else -1
                current_lane = world.map.get_waypoint(world.player.get_location()).lane_id
                if abs(current_lane) == abs(original_lane):
                    self._control.steer = original_steer - 0.005 * direction
                else:
                    self._lane_change_activated = None
                    self._lane_change_started = False
                    world.history.update_hlc(RoadOption.LANEFOLLOW)
        else:
            self._add_to_steer_history(self._control.steer)

        # Apply control signal to vehicle 
        world.player.apply_control(self._control)

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx]) / 3

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

    def _parse_drive_model_commands(self, world: World):
        images = {}
        info = {}

        if "forward_center_rgb" not in world.history._latest_images:
            return

        images["forward_center_rgb"] = world.history._latest_images[
            "forward_center_rgb"]
        """images["left_center_rgb"] = world.history._latest_images[
            "left_center_rgb"]
        images["right_center_rgb"] = world.history._latest_images[
            "right_center_rgb"]"""

        player = world.player
        player_loc = world.player.get_location()

        closest_wp = world.map.get_waypoint(world.player.get_location())
        lane_yaw = closest_wp.transform.rotation.yaw

        # light, distance = get_nearest_traffic_light(player)
        # red_light = 0 if \
        #     distance <= 40 \
        #     and light is not None \
        #     and light.state != carla.TrafficLightState.Green \
        #     else 1

        red_light = 0 if player.get_traffic_light_state() == carla.TrafficLightState.Red else 1

        """
        if not closest_wp.is_junction:
            self._entered_traffic_light_loc = None
        elif self._entered_traffic_light_loc is None:
            print("yolo")
            self._entered_traffic_light_loc = player_loc
        else:
            print(get_distance(player_loc, self._entered_traffic_light_loc))
            if get_distance(player_loc, self._entered_traffic_light_loc) > 1:
                red_light = 1
        """

        v = player.get_velocity()

        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        info["speed"] = speed
        info["traffic_light"] = red_light
        info["speed_limit"] = player.get_speed_limit() / 3.6
        info["hlc"] = world.history._latest_hlc
        info["environment"] = world._environment

        steer = 0
        throttle = 0
        brake = 0

        if images["forward_center_rgb"] is not None:
            steer, throttle, brake = self._drive_model.get_prediction(
                images, info)

        self._control.steer = float(steer)

        self._control.throttle = max(min(float(throttle), 1), 0)

        self._control.brake = float(brake) if brake > 0.02 else 0 # 1 if float(brake) > 0.3 else 0

    def _parse_vehicle_keys(self, world: World, keys: List[int], milliseconds: int):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_client_ap(self, world: World):
        client_autopilot_control = world.client_ap.run_step(debug=False)
        world.history.update_client_autopilot_control(client_autopilot_control)
        self._control.brake = client_autopilot_control.brake
        self._control.throttle = client_autopilot_control.throttle
        if self._noise_enabled:
            noise = np.random.uniform(-self._noise_amount, self._noise_amount)
            self._control.steer = client_autopilot_control.steer + noise

        else:
            self._control.steer = client_autopilot_control.steer

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q
                                     and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):

        # Display parameters 
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        self._mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(self._mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(self._mono, 24), width, height)

        # Content parameters 
        self.server_fps = 0
        self.server_fps_realtime = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._episode_start_time = 0

        # Print info in eval mode
        self._drive_model_name = None
        self._drive_model_num = None
        self._drive_model_idx = None

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = 1 / timestamp.delta_seconds
        self.server_fps_realtime = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds
        if self._episode_start_time == 0:
            self._episode_start_time = self.simulation_time

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''

        vehicles = world.world.get_actors().filter('vehicle.*')
        speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        speed_limit = world.player.get_speed_limit()

        self._info_text = [
            'Server (realtime):  % 5.0f FPS' % self.server_fps_realtime,
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(), '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Spawn:   % 20d' % world._spawn_point_start,
            'Simulation time: % 12s' %
            datetime.timedelta(seconds=int(self.simulation_time)), '',
            'Speed:   % 15.0f km/h' % speed,
            'Steer:   % 17.2f' % c.steer,
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z, ''
        ]
        if world._auto_record:
            self._info_text += ['Mode: Recording']

        if world._eval_mode:
            self._info_text += ['Mode: Eval']
        if world._current_route_num is not None and world._tot_route_num is not None:
            self._info_text += ['Route status: % 11d/%2d' % (world._current_route_num, world._tot_route_num)]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [('Throttle:', c.throttle, 0.0, 1.0),
                                ('Steer:', c.steer, -1.0, 1.0),
                                ('Brake:', c.brake, 0.0, 1.0),
                                ('Reverse:', c.reverse),
                                ('Hand brake:', c.hand_brake),
                                ('Manual:', c.manual_gear_shift),
                                'Gear:        %s' % {
                                    -1: 'R',
                                    0: 'N'
                                }.get(c.gear, c.gear)]
        self._info_text += [
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if world._eval_mode or True:
            if world._eval_num is not None and world._eval_num_current is not None:
                self._info_text += ['']
                self._info_text += ['Eval run: %d/%d' % (world._eval_num_current, world._eval_num)]

            if self._drive_model_name is not None:
                self._info_text += [
                    'Model: %s (%d/%d)' % (self._drive_model_name, self._drive_model_idx + 1, self._drive_model_num)]

            if world._eval_cars_idx is not None and world._eval_cars is not None:
                self._info_text += ['Spawned vehicles: %d (%d/%d)' % (
                    world._eval_cars[world._eval_cars_idx], world._eval_cars_idx + 1, len(world._eval_cars))]

            if world._eval_weathers_idx is not None and world._eval_weathers is not None:
                self._info_text += ['Eval weather: %d/%d' % (world._eval_weathers_idx + 1, len(world._eval_weathers))]

            if world._eval_routes_idx is not None and world._eval_routes is not None:
                self._info_text += ['Eval route: %d/%d' % (world._eval_routes_idx + 1, len(world._eval_routes))]
                self._info_text += ['']

        self._info_text.append(('Speed: ', '%.0f/%.0f' % (speed, speed_limit)))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            speed = self._info_text[-1]
            for item in self._info_text[:-1]:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False,
                                          points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8),
                                           (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect,
                                         0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8),
                                                  (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border,
                                         1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if (math.isnan(f)):
                            continue
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6),
                                 v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8),
                                               (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True,
                                                     (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
            font = pygame.font.Font(self._mono, 32)
            surface = font.render(speed[0], True, (255, 255, 255))
            display.blit(surface, (8, self.dim[1] - 160))

            font = pygame.font.Font(self._mono, 60)
            surface = font.render(speed[1], True, (255, 255, 255))
            display.blit(surface, (8, self.dim[1] - 110))

        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0],
                    0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor: carla.Actor, client_ap: Agent, hud: HUD, history: 'History', eval=False,
                 hq_recording=False):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._client_ap = client_ap
        self._hud = hud
        self._history = history
        self._recording = False
        self._capture_rate = 1 if hq_recording else 3
        self._frame_number = 1
        self._hq_recording = hq_recording
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=0.5, z=2.3), carla.Rotation(pitch=-5)),
        ]
        self._transform_index = 1
        self.eval = eval
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            [
                'sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'
            ],
            [
                'sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)'
            ],
            [
                'sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'
            ], ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']
        ]

        self._recording_sensors = []

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self._index = None

    def _initiate_recording(self):

        sensor_bp = self._parent.get_world().get_blueprint_library().find(
            'sensor.camera.rgb')
        sensor_bp.set_attribute('image_size_x', "350")
        sensor_bp.set_attribute('image_size_y', "160")

        sensor = self._parent.get_world().spawn_actor(
            sensor_bp,
            carla.Transform(carla.Location(x=0.5, z=2.3), carla.Rotation(pitch=-5)),
            attach_to=self._parent)
        sensor.listen(lambda image: self._history.update_image(
            image, "forward_center", "rgb"))
        self._recording_sensors.append(sensor)
        if not self.eval:
            sensor = self._parent.get_world().spawn_actor(
                sensor_bp,
                carla.Transform(carla.Location(x=0.5, y=-0.7, z=2.3), carla.Rotation(pitch=-5)),
                attach_to=self._parent)
            sensor.listen(lambda image: self._history.update_image(
                image, "forward_left", "rgb"))
            self._recording_sensors.append(sensor)

            sensor = self._parent.get_world().spawn_actor(
                sensor_bp,
                carla.Transform(carla.Location(x=0.5, y=0.7, z=2.3), carla.Rotation(pitch=-5)),
                attach_to=self._parent)
            sensor.listen(lambda image: self._history.update_image(
                image, "forward_right", "rgb"))
            self._recording_sensors.append(sensor)

            sensor = self._parent.get_world().spawn_actor(
                sensor_bp,
                carla.Transform(carla.Location(x=0, y=-0.5, z=1.8), carla.Rotation(pitch=-20, yaw=-90)),
                attach_to=self._parent)
            sensor.listen(lambda image: self._history.update_image(
                image, "left_center", "rgb"))
            self._recording_sensors.append(sensor)

            sensor = self._parent.get_world().spawn_actor(
                sensor_bp,
                carla.Transform(carla.Location(x=0, y=0.5, z=1.8), carla.Rotation(pitch=-20, yaw=90)),
                attach_to=self._parent)
            sensor.listen(lambda image: self._history.update_image(
                image, "right_center", "rgb"))
            self._recording_sensors.append(sensor)

        if self._hq_recording:
            sensor_bp = self._parent.get_world().get_blueprint_library().find(
                'sensor.camera.rgb')
            sensor_bp.set_attribute('image_size_x', "1920")
            sensor_bp.set_attribute('image_size_y', "1080")

            sensor = self._parent.get_world().spawn_actor(
                sensor_bp,
                carla.Transform(carla.Location(x=-0.5, z=2.0)),
                attach_to=self._parent)
            sensor.listen(lambda image: self._history.update_image_hq(
                image, "hq_record", "rgb"))
            self._recording_sensors.append(sensor)

    def _destroy_sensors(self):
        for sensor in self._recording_sensors:
            sensor.destroy()

        if self.sensor is not None:
            self.sensor.destroy()

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(
            self._camera_transforms)
        self.sensor.set_transform(
            self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(
                weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        if self._recording:
            self._hud.notification('Writing data to disk, please wait..')
            self._history.save_to_disk()
            self._hud.notification('Writing complete!')
        else:
            self._history._active = True

        self._recording = not self._recording
        self._hud.notification(
            'Recording %s' % ('On' if self._recording else 'Off'))

    def tick(self):
        if self._recording and self._hud.simulation_time - self._hud._episode_start_time > 1.5:
            if self._frame_number % self._capture_rate == 0:
                self._history.record_frame(self._parent, self._client_ap)
            self._frame_number += 1

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


class History:
    def __init__(self, output_folder: str, environment: Environment):
        self._latest_images = {}
        self._image_history = []
        self._measurements_history = []
        self._frame_number = 0
        self._output_folder = output_folder
        self._timestamp = None
        self._driving_log = None
        self._active = False
        self._latest_client_autopilot_control = None
        self.control_type = None  # type: ControlType
        self._latest_hlc = None
        self._environment = environment
        self._left_lane_change_valid = None
        self._right_lane_change_valid = None

        self._weather_index = 0

    def _initiate(self):
        if self._active:
            self.save_to_disk()

        self._driving_log = pd.DataFrame(columns=[
            "ForwardCenter", "ForwardLeft", "ForwardRight", "LeftCenter",
            "RightCenter", "Location", "Velocity", "Controls", "ClientAutopilotControls", "ControlType",
            "LeftLaneChangeValid", "RightLaneChangeValid", "TrafficLight", "DeprecatedTrafficLight",
            "SpeedLimit", "HLC", "Environment", "WheaterId"
        ])
        self._timestamp = time.strftime("%Y-%m-%d_%H-%M-%S",
                                        time.localtime(time.time()))

        self._latest_images = {}
        self._image_history = []
        self._measurements_history = []
        self._frame_number = 0
        self._latest_client_autopilot_control = None
        self._latest_hlc = RoadOption.LANEFOLLOW

    def update_weather_index(self, weather_index):
        self._weather_index = weather_index

    def update_image(self, image, position, sensor_type):
        if image.raw_data:
            img = np.reshape(np.array(image.raw_data), (160, 350, 4))[:, :, :3]
            self._latest_images[position + "_" + sensor_type] = img

    def update_left_lane_change_valid(self, is_valid):
        self._left_lane_change_valid = is_valid

    def update_right_lane_change_valid(self, is_valid):
        self._right_lane_change_valid = is_valid

    def update_image_hq(self, image, position, sensor_type):
        if image.raw_data:
            img = np.reshape(np.array(image.raw_data), (1080, 1920, 4))[:, :, :3]
            self._latest_images[position + "_" + sensor_type] = img

    def update_client_autopilot_control(self, control):
        self._latest_client_autopilot_control = control

    def update_hlc(self, hlc):
        self._latest_hlc = hlc

    def record_frame(self, player, client_ap):
        images = []
        self._frame_number += 1

        v = player.get_velocity()
        t = player.get_transform()
        c = player.get_control()

        if self.control_type == ControlType.CLIENT_AP:
            if not self._latest_client_autopilot_control:
                return
            client_ap_c = self._latest_client_autopilot_control

            hlc = client_ap._local_planner._target_road_option

        else:
            client_ap_c = None
            hlc = self._latest_hlc

        for name, image in self._latest_images.items():
            images.append((name + "_%08d.png" % self._frame_number, image))

        self._image_history.append(images)

        output_path = Path(self._output_folder + '/' + self._timestamp)
        image_path = output_path / "imgs"

        light = get_nearest_traffic_light(player)[0]
        red_light = 0 if client_ap.is_affected_by_traffic_light and light is not None and light.state != carla.TrafficLightState.Green else 1

        deprecated_red_light = 0 if player.get_traffic_light_state() == carla.TrafficLightState.Red else 1

        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        if math.isnan(c.steer) or (self.control_type == ControlType.CLIENT_AP and math.isnan(client_ap_c.steer)):
            return

        self._driving_log = self._driving_log.append(
            pd.Series([
                "imgs/forward_center_rgb_%08d.png" % self._frame_number,
                "imgs/forward_left_rgb_%08d.png" % self._frame_number,
                "imgs/forward_right_rgb_%08d.png" % self._frame_number,
                "imgs/left_center_rgb_%08d.png" % self._frame_number,
                "imgs/right_center_rgb_%08d.png" % self._frame_number,
                (t.location.x, t.location.y),
                speed,
                (c.throttle, c.steer, c.brake),
                (client_ap_c.throttle, client_ap_c.steer,
                 client_ap_c.brake) if self.control_type == ControlType.CLIENT_AP else -1,
                self.control_type.value,
                1 if self._left_lane_change_valid else 0,
                1 if self._right_lane_change_valid else 0,
                red_light,
                deprecated_red_light,
                player.get_speed_limit() / 3.6,
                hlc.value,
                self._environment.value,
                self._weather_index

            ],
                index=self._driving_log.columns),
            ignore_index=True)

    def save_to_disk(self):

        output_path = Path(self._output_folder + '/' + self._timestamp)
        image_path = output_path / "imgs"
        image_path.mkdir(parents=True, exist_ok=True)

        for frame in self._image_history:
            for name, image in frame:
                cv2.imwrite(str(image_path / name), image)

        csv_path = str(output_path / "driving_log.csv")
        if not os.path.isfile(csv_path):
            self._driving_log.to_csv(csv_path)
        else:
            self._driving_log.to_csv(csv_path, mode="a", header=False)

        self._active = False
        self._initiate()


class Evaluator():

    def __init__(self, hud, world):
        self.hud = hud
        self.sensors = []
        self.event_logs = []
        self.model_summary = None
        self.last_collision = 0
        self.last_invasion = 0
        self.world = world
        self.last_wp = None
        self.current_wp = None
        self.last_dist = None
        self.last_dist_at = None
        self.total_dist_traveled = None
        self.entered_oncoming_lane_at = None
        self.cancel_reason = None
        self.error_counter = None
        self.current_episode_timestamp = None
        self.current_episode_time = None
        self.current_eval_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self._last_col_loc = None
        self._in_junction = False

        self._exited_junction_at = None

    def new_episode(self):
        self.last_wp = None
        self.current_wp = self.world._eval_routes[self.world._eval_routes_idx][0][0]
        self._last_col_loc = None

        self.last_dist = float('inf')
        self.last_dist_at = None
        self.last_collision = 0
        self.last_invasion = 0
        self.total_dist_traveled = 0
        self.entered_oncoming_lane_at = None
        self.cancel_reason = None
        self.current_episode_time = time.time()
        self.current_episode_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.event_logs.append(pd.DataFrame(columns=[
            "Timestamp", "EventType", "ObjectName", "Instensity",
            "Location"
        ]))

        self.error_counter = {}
        self._in_junction = False
        self._exited_junction_at = None
        for t in EventType:
            self.error_counter[t.name] = 0

    def initialize_model(self):
        cols = [
            "EvalNum", "Route", "RouteId", "WeatherId", "NumVehicles",
            "LastWaypointReached", "DistanceCompleted", "TotalRouteDistance",
            "CancelReason", "EventLogPath", "StartedAt", "EndedAt"
        ]

        for t in EventType:
            cols.append(t.name)
        self.model_summary = pd.DataFrame(columns=cols)

    def tick(self):

        closest_wp = self.world.map.get_waypoint(self.world.player.get_location())
        lane_yaw = closest_wp.transform.rotation.yaw
        if closest_wp.is_junction:
            self._in_junction = True
        else:
            if self._in_junction:
                self._exited_junction_at = time.time()
            self._in_junction = False

        wp = self.world._eval_routes[self.world._eval_routes_idx][self.world._eval_route_idx][0]

        if self.current_wp != wp:
            if self.last_wp != None:
                self.total_dist_traveled += get_distance(self.world.map.get_spawn_points()[self.current_wp].location,
                                                         self.world.map.get_spawn_points()[self.last_wp].location)
            self.last_dist = float('inf')
            self.last_wp = self.current_wp
            self.current_wp = wp

        hero_transform = self.world.player.get_transform()
        dist = get_distance(self.world.map.get_spawn_points()[wp].location, hero_transform.location)
        hero_location = hero_transform.location
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        if self.last_dist > dist:
            self.last_dist = dist
            self.last_dist_at = time.time()
        elif self.last_dist < dist - 20:
            event_type = EventType.HLC_IGNORE
            self.error_counter[event_type.name] += 1
            self.cancel_reason = event_type
            self.world._eval_route_canceled = True
            self.event_logs[-1] = self.event_logs[-1].append(
                pd.Series([
                    timestamp,
                    event_type,
                    None,
                    None,
                    (hero_location.x, hero_location.y)
                ],
                    index=self.event_logs[-1].columns),
                ignore_index=True)
            return

        if self.last_dist_at and self.last_dist_at + 120 < time.time():
            # Model is stuck
            event_type = EventType.STUCK
            self.error_counter[event_type.name] += 1
            self.cancel_reason = event_type
            self.world._eval_route_canceled = True
            self.event_logs[-1] = self.event_logs[-1].append(
                pd.Series([
                    timestamp,
                    event_type,
                    None,
                    None,
                    (hero_location.x, hero_location.y)
                ],
                    index=self.event_logs[-1].columns),
                ignore_index=True)
            return

        closest_wp = self.world.map.get_waypoint(self.world.player.get_location())
        # print("Total distance traveled: ", self.total_dist_traveled, " Distance to next wp: ", dist)
        lane_yaw = closest_wp.transform.rotation.yaw
        hero_yaw = hero_transform.rotation.yaw
        angle_diff = lane_yaw - hero_yaw
        angle_diff = abs((angle_diff + 180) % 360 - 180)

        if angle_diff > 45 and not closest_wp.is_junction:
            if self.entered_oncoming_lane_at is None:
                self.entered_oncoming_lane_at = time.time()
            elif time.time() - self.entered_oncoming_lane_at > 5:
                # Vehicle has entered oncoming lane without recovery, cancel route
                event_type = EventType.ONCOMING_LANE_WITHOUT_RECOVERY
                self.error_counter[event_type.name] += 1
                self.cancel_reason = event_type
                self.world._eval_route_canceled = True
                self.event_logs[-1] = self.event_logs[-1].append(
                    pd.Series([
                        timestamp,
                        event_type,
                        None,
                        None,
                        (hero_location.x, hero_location.y)
                    ],
                        index=self.event_logs[-1].columns),
                    ignore_index=True)
                return
        else:
            if self.entered_oncoming_lane_at and self.entered_oncoming_lane_at < time.time():
                # Vehicle has entered oncoming lane, but recovered
                print("Recovery")
                event_type = EventType.ONCOMING_LANE_WITH_RECOVERY
                self.error_counter[event_type.name] += 1
                self.entered_oncoming_lane_at = None
                self.event_logs[-1] = self.event_logs[-1].append(
                    pd.Series([
                        timestamp,
                        event_type,
                        None,
                        None,
                        (hero_location.x, hero_location.y)
                    ],
                        index=self.event_logs[-1].columns),
                    ignore_index=True)

    def episode_complete(self, route_completed):
        eval_num = self.world._eval_num_current
        route = [x[0] for x in self.world._eval_routes[self.world._eval_routes_idx]]
        routeId = self.world._eval_routes_idx
        weatherId = self.world._eval_weathers[self.world._eval_weathers_idx]
        numberOfVehicles = self.world._eval_cars[self.world._eval_cars_idx]
        route_dist = get_route_distance(route, self.world.map)

        data = [
            eval_num,
            '-'.join(str(x) for x in route),
            routeId,
            weatherId,
            numberOfVehicles,
            self.last_wp if not route_completed else route[-1],
            "{:.1f}".format(self.total_dist_traveled if not route_completed else route_dist),
            "{:.1f}".format(route_dist),
            self.cancel_reason,
            'EventLogs/' + self.current_episode_timestamp + '.csv',
            self.current_episode_time,
            time.time()
        ]
        for t in EventType:
            data.append(self.error_counter[t.name])

        self.model_summary = self.model_summary.append(pd.Series(data,
                                                                 index=self.model_summary.columns),
                                                       ignore_index=True
                                                       )
        # Write eventlog to csv
        model_name = '_'.join(self.hud._drive_model_name.split('/'))
        dir_path = Path("EvalResults") / self.current_eval_timestamp / model_name / "EventLogs"
        csv_path = dir_path / (self.current_episode_timestamp + ".csv")
        dir_path.mkdir(parents=True, exist_ok=True)
        self.event_logs[-1].to_csv(csv_path)

        model_summary_path = Path("EvalResults") / self.current_eval_timestamp / model_name / "summary.csv"
        self.model_summary.to_csv(str(model_summary_path))

    def initialize_sensors(self, parent_actor):
        weak_self = weakref.ref(self)
        world = parent_actor.get_world()

        # Lane Invasion
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        sensor = world.spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        sensor.listen(lambda event: Evaluator._on_lane_invasion(weak_self, event))
        self.sensors.append(sensor)

        # Collision
        bp = world.get_blueprint_library().find('sensor.other.collision')
        sensor = world.spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        sensor.listen(lambda event: Evaluator._on_collision(weak_self, event))
        self.sensors.append(sensor)

    @staticmethod
    def _on_lane_invasion(weak_self, event):

        self = weak_self()
        if not self:
            return
        if (time.time() - self.last_invasion < 2):
            self.last_invasion = time.time()
            return

        if self._in_junction or (self._exited_junction_at and self._exited_junction_at + 2 > time.time()):
            return

        lane_type = (str(event.crossed_lane_markings[-1].type)).lower()
        location = event.actor.get_transform().location
        event_type = EventType.SIDEWALK_TOUCH if lane_type == 'none' else EventType.LANE_TOUCH
        self.error_counter[event_type.name] += 1

        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.event_logs[-1] = self.event_logs[-1].append(
            pd.Series([
                timestamp,
                event_type,
                lane_type,
                None,
                (location.x, location.y)
            ],
                index=self.event_logs[-1].columns),
            ignore_index=True)
        self.last_invasion = time.time()

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        if (time.time() - self.last_collision < 3):
            self.last_collision = time.time()
            return

        actor_type = get_actor_display_name(event.other_actor)
        event_type = None
        location = event.actor.get_transform().location
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

        if self._last_col_loc != None:
            dist = get_distance(location, self._last_col_loc)
            if dist < 5:
                return
            else:
                self._last_col_loc = location
        else:
            self._last_col_loc = location

        if 'vehicle' in event.other_actor.type_id:
            other_actor_yaw = event.other_actor.get_transform().rotation.yaw
            hero_yaw = event.actor.get_transform().rotation.yaw
            angle_diff = other_actor_yaw - hero_yaw

            if abs((angle_diff + 180) % 360 - 180) <= 90:
                event_type = EventType.REAR_END_VEHICLE_COLLISION
            else:
                # Front end collision - canceling route
                event_type = EventType.FRONT_END_VEHICLE_COLLISION
                self.cancel_reason = event_type
                self.world._eval_route_canceled = True
                self.event_logs[-1] = self.event_logs[-1].append(
                    pd.Series([
                        timestamp,
                        event_type,
                        actor_type,
                        intensity,
                        (location.x, location.y)
                    ],
                        index=self.event_logs[-1].columns),
                    ignore_index=True)
                self.error_counter[event_type.name] += 1
                return
        else:
            event_type = EventType.OBJECT_COLLISION

        self.error_counter[event_type.name] += 1
        self.hud.notification(event_type.name + ' with %r, intensity %f' % (actor_type, intensity))

        self.event_logs[-1] = self.event_logs[-1].append(
            pd.Series([
                timestamp,
                event_type,
                actor_type,
                intensity,
                (location.x, location.y)
            ],
                index=self.event_logs[-1].columns),
            ignore_index=True)
        self.last_collision = time.time()

    def destroy_sensors(self):
        for sensor in self.sensors:
            sensor.destroy()
        self.sensors = []


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    client.load_world("Town01")
    sim_world = client.get_world()  # type: carla.World
    map_name = sim_world.get_map().name

    # Enable synchronous mode
    sim_settings = sim_world.get_settings()
    sim_settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    sim_settings.synchronous_mode = True
    sim_world.apply_settings(sim_settings)

    from glob import glob
    for settings_path in sorted(glob(str("settings/*.ini"))):
        try:
            settings = ConfigParser()
            settings.read(settings_path)
            print("Running with settings file ", settings_path)

            # Get environment
            if map_name == "Town01" or map_name == "Town02":
                environment = Environment.RURAL
            elif map_name == "Town04":
                environment = Environment.HIGHWAY
            else:
                environment = Environment.RURAL

            # Set display
            display = pygame.display.set_mode((args.width, args.height),
                                              pygame.HWSURFACE | pygame.DOUBLEBUF)

            hud = HUD(args.width, args.height)
            history = History(args.output, environment)
            world = World(
                client,
                sim_world,
                hud,
                environment,
                history,
                args.filter,
                settings,
                hq_recording=False)

            models = None

            # Program can be called with a folder of models to test
            if args.models is not None:
                models = get_models(Path(args.models))
                if models == False:
                    continue

            controller = KeyboardControl(world, settings, use_steering_wheel=args.joystick, models=models)

            clock = pygame.time.Clock()

            while True:
                sim_world.tick()
                # clock.tick(int(hud.server_fps_realtime))
                clock.tick()
                if controller.parse_events(client, world, clock):
                    break
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
        finally:
            if world is not None:
                # Reset weather so program always starts in CLEAR NOON
                world.reset_weather()
                print("Destroying world")
                world.destroy()

    pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-o',
        '--output',
        metavar='O',
        dest='output',
        default='output',
        help='output-folder for recordings')
    argparser.add_argument(
        '-m',
        '--models',
        dest='models',
        default=None,
        help='folders with models to test, models must be in separate folders with its own config-file')
    argparser.add_argument(
        '-j',
        '--joystick',
        action='store_true',
        default=False,
        help='use steering wheel to control vehicle')
    argparser.add_argument(
        '-hqr',
        dest='hq_recording',
        action='store_true',
        default=False,
        help='record high quality images')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
