#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import math
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random


class  VehicleSpawner(object):

    def __init__(self, client, world): 
        self.client = client
        self.world = world
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        self.blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        self._spawned_vehicles = []
        self._walkers_list = []
        self._all_id = []
        self._all_actors = []
        self._bad_colors = [
            "255,255,255", "183,187,162", "237,237,237", 
            "134,134,134", "243,243,243", "127,130,135", 
            "109,109,109", "181,181,181", "140,140,140", 
            "181,178,124", "171,255,0", "251,241,176",
            "158,149,129", "233,216,168", "233,216,168",
            "108,109,126", "193,193,193", "227,227,227",
            "151,150,125", "206,206,206", "255,222,218",
            "211,211,211", "191,191,191"
            ]

    
    def spawn_nearby(self, hero_spawn_point_index, number_of_vehicles_min,number_of_vehicles_max, number_of_walkers_min, number_of_walkers_max, radius):

        number_of_vehicles = random.randint(number_of_vehicles_min,number_of_vehicles_max)
        number_of_walkers = random.randint(number_of_walkers_min, number_of_walkers_max)
        print(number_of_vehicles)
        hero_spawn_point = self.spawn_points[hero_spawn_point_index]

        hero_x = hero_spawn_point.location.x
        hero_y = hero_spawn_point.location.y

        self.blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('isetta')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('carlacola')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('t2')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('coupe')]


        valid_spawn_points = []
        for spawn_point in self.spawn_points:
            # Distance between spaw points 
            loc = hero_spawn_point.location
            dx = spawn_point.location.x - loc.x
            dy = spawn_point.location.y - loc.y
            distance = math.sqrt(dx * dx + dy * dy)
            min_distance = 10
            if spawn_point == hero_spawn_point or distance < min_distance: 
                continue
            if radius != 0:
                x = spawn_point.location.x
                y = spawn_point.location.y
                yaw = spawn_point.rotation.yaw
                angle_diff = hero_spawn_point.rotation.yaw - yaw 
                angle_diff = abs((angle_diff + 180) % 360 - 180)
                
                if abs(hero_x-x)<= radius and abs(hero_y-y)<=radius and angle_diff < 50: 
                    valid_spawn_points.append(spawn_point)
            else: 
                valid_spawn_points.append(spawn_point)

            
        number_of_spawn_points = len(valid_spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(valid_spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(valid_spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):

                color = "255,255,255"
                while color in self._bad_colors: 
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            

        for response in self.client.apply_batch_sync(batch):
            if response.error is not None:
                self._spawned_vehicles.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # 1. take all the random locations to spawn
        walker_spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invencible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self._walkers_list.append({"id": results[i].actor_id})
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self._walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self._walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self._walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self._walkers_list)):
            self._all_id.append(self._walkers_list[i]["con"])
            self._all_id.append(self._walkers_list[i]["id"])
        self._all_actors = self.world.get_actors(self._all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        for i in range(0, len(self._all_id), 2):
            # start walker
            self._all_actors[i].start()
            # set walk to random point
            self._all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            self._all_actors[i].set_max_speed(1 + random.random()/2)    # max speed between 1 and 1.5 (default is 1.4 m/s)

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len( self._spawned_vehicles), len(self._walkers_list)))


    def destroy_vehicles(self): 
        print('\ndestroying %d actors' % len(self._spawned_vehicles))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self._spawned_vehicles if x is not None])
        self._spawned_vehicles = []

        # stop walker controllers (list is [controler, actor, controller, actor ...])
        for i in range(0, len(self._all_id), 2):
            self._all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self._walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self._all_id])


