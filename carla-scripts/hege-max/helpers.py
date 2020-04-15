from glob import glob
from pathlib import Path
import re 
import carla
import numpy as np
import math
from configparser import ConfigParser

from agents.navigation.local_planner import RoadOption
from agents.tools.misc import distance_vehicle, get_nearest_traffic_light
from misc import get_distance


def set_green_traffic_light(player):

    # traffic_light_state = player.get_traffic_light_state()
    # if traffic_light_state is not None and traffic_light_state ==  carla.TrafficLightState.Red:
    #     traffic_light = player.get_traffic_light()
    #     traffic_light.set_state(carla.TrafficLightState.Green)

    traffic_light, distance = get_nearest_traffic_light(player)
    traffic_light_state = traffic_light.state if traffic_light else None
    if traffic_light_state is not None and traffic_light_state != carla.TrafficLightState.Green:
        traffic_light.set_state(carla.TrafficLightState.Green)


def get_models(models_folder): 
    """
    input: path to folder where different models has been tested 
    return: 
        models: a list of (model_path, seq_length, sampling_interval)
    """

    models = []

    # For each model 
    for model_folder in sorted(glob(str(models_folder / "*"))):

        # Get model path
        model_path = glob(str(Path(model_folder) / "*.h5"))
        if len(model_path)==0: 
            print("ERROR: No .h5 file found in folder: " + model_folder)
            return False 
        if len(model_path)>1: 
            print("ERROR: More than one .h5 file found in folder: " + model_folder)
            return False 
        model_path = model_path[0]

        # Get config settings
        config_path = glob(str(Path(model_folder) / "*.ini"))
        if len(config_path)==0: 
            print("ERROR: No .ini file found in folder: " + model_folder)
            return False 
        if len(config_path)>1: 
            print("ERROR: More than one .ini file found in folder: " + model_folder)
            return False 
        config_path = config_path[0]

        # Read config file 
        config = ConfigParser()
        config.read(config_path)
        seq_length = int(config.get("ModelConfig", "sequence_length", fallback=1))
        sampling_interval = int(config.get("ModelConfig", "sampling_interval", fallback=1))
        model_type = config.get("ModelConfig", "model", fallback=None)
        segmentation_network = config.get("ModelConfig", "seg_network_path", fallback=None)

        if seq_length is None: 
            print("ERROR: No sequence length found in model config in folder: " + model_folder)
            return False
        if sampling_interval is None: 
            print("ERROR: No sampling interval found in model config in folder: " + model_folder)
            return False

        models.append((model_path, seq_length, sampling_interval, model_type, segmentation_network))

    return models 
        

def get_parameter_text(path):
    f = open(str(path), "r")
    params = []
    for line in f:
        if '___' in line: 
            break
        if 'dataset' not in line and 'epochs' not in line and "batch" not in line:  
            params.append(line.replace("\n", ""))
    return params
    

def is_valid_lane_change(road_option, world, proximity_threshold=5.9): 

        if road_option != RoadOption.CHANGELANELEFT and road_option != RoadOption.CHANGELANERIGHT: 
            print("ERROR: road option is not a lane change")
            return False 
        
        player_waypoint = world.map.get_waypoint(world.player.get_location())
        player_lane_id = player_waypoint.lane_id
        player_transform = world.player.get_transform()
        player_yaw = player_transform.rotation.yaw
        
        vehicle_list = world.world.get_actors().filter('vehicle.*')     

        # Lane change left  
        if road_option == RoadOption.CHANGELANELEFT: 
            # check if lane change is valid 
            if not player_waypoint.lane_change & carla.LaneChange.Left:
                #print("Traffic rules does not allow left lane change here")
                return False  

            # Only look at vehicles in left adjecent lane 
            for vehicle in vehicle_list: 
                #vehicle = actorvehicle_list_list.find(vehicle_id)
                vehicle_lane_id = world.map.get_waypoint(vehicle.get_location()).lane_id
                # Check if lane_id is in the same driving direction 
                if (player_lane_id < 0 and vehicle_lane_id <0) or (player_lane_id > 0 and vehicle_lane_id > 0):
                    if abs(player_lane_id)-abs(vehicle_lane_id) == 1: 
                        # check the vehicle|s proximity to the player
                        vehicle_waypoint = world.map.get_waypoint(vehicle.get_location())
                        if distance_vehicle(vehicle_waypoint, player_transform) < proximity_threshold:   
                            return False  
        # Lane change right 
        else: 
            # check if lane change is valid 
            if not player_waypoint.lane_change & carla.LaneChange.Right:
                #print("Traffic rules does not allow right lane change here")
                return False 
            # Only look for vehicles in right adjencent lane 
            for vehicle in vehicle_list:
                #vehicle = vehicle_list.find(vehicle_id)
                vehicle_lane_id = world.map.get_waypoint(vehicle.get_location()).lane_id
                # Check if lane_id is in the same driving direction 
                if (player_lane_id < 0 and vehicle_lane_id <0) or (player_lane_id > 0 and vehicle_lane_id > 0):
                    if abs(player_lane_id)-abs(vehicle_lane_id) == -1: 

                        # check the vehicle|s proximity to the player
                        vehicle_waypoint = world.map.get_waypoint(vehicle.get_location())
                        if distance_vehicle(vehicle_waypoint, player_transform) < proximity_threshold:   
                            return False           
        return True            

def get_route_distance(waypoint_index_list, current_map):
    """
    Calculates the total distance of a given route.
    Input: 
        waypoint_index_list: A list of waypoint indexes
        map: Current map
    Output: Total distance
    """

    total_distance = 0

    for i in range(len(waypoint_index_list)-1):
        wp1 = waypoint_index_list[i]
        wp2 = waypoint_index_list[i+1]
        total_distance += get_distance(current_map.get_spawn_points()[wp1].location, current_map.get_spawn_points()[wp2].location)

    return total_distance
