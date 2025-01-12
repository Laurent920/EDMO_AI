import numpy as np
import asyncio
import random
import argparse
from math import fabs, sqrt
import os
from pathlib import Path
import copy

from EDMOManual import EDMOManual
from data_analysis import merge_parameter_data, plot_data, compute_speed
from ArUCo_Markers_Pose import pose_data, pose_estimation
from GoPro.wifi.WifiCommunication import WifiCommunication


# region ONLINE LEARNING
def parameters_to_param_list(parameters):
    param_list = [parameters[0]['freq']]

    # Group corresponding values from 'amp', 'off', and 'phb' into sublists
    amps = [p['amp'] for p in parameters.values()]
    offs = [p['off'] for p in parameters.values()]
    phbs = [p['phb'] for p in parameters.values()]

    param_list.append(amps)
    param_list.append(offs)
    param_list.append(phbs)

    return param_list

def 
     
async def get_EDMO_speed(server, parameters):
    print(parameters)
    await server.runInputDict(parameters)
    
    server.GoProOn()
    await asyncio.sleep(1)
    filepaths = await server.GoProStopAndSave()
    await server.reset()
    
    print(filepaths)
    video_path = filepaths[0]
    filespath = os.path.dirname(video_path) 
    aruco_pose = pose_estimation.Aruco_pose(video_path)
    dict_all_pos = aruco_pose.pose_estimation()
    
    return random.uniform(0, 0.5)
    
    pose_d = pose_data.Pose_data(filespath, dict_all_pos)
    succeed = pose_d.get_pose()
    if not succeed:
        return
    edmo_poses = pose_d.edmo_poses
    edmo_rots = pose_d.edmo_rots
    exp_edmo_poses = {}
    exp_edmo_poses[0] = {}
    for frame in range(pose_d.nbFrames):
        if frame in edmo_poses:
            exp_edmo_poses[0][frame] = edmo_poses[frame]
    
    param_list = parameters_to_param_list(parameters)
    exp_edmo_movement = compute_speed(exp_edmo_poses, filespath)
    data = merge_parameter_data(param_list, exp_edmo_movement)   
    print(data)
    speed = data['xy frame speed']*30 # m/s
    return speed


async def Powell(nb_players:int = 2):
    wifi_com = WifiCommunication("GoPro 6665", Path("GoPro/GoPro 6665"))
    await wifi_com.initialize()
    
    # Initialize EDMOManual
    server = EDMOManual()
    asyncio.get_event_loop().create_task(server.run())
    server.GoProOff()
    
    param_ranges = {
        'freq': [1, 1],
        'amp': [0, 90],
        'off': [0, 180],
        'phb': [0, 359]
    }
    n = nb_players * 3 # number of dimensions to explore
    u = {dimension+1: [1 if i == dimension else 0 for i in range(n)] for dimension in range(n)}

    parameters = {}
    Points = {}
    
    # Random initialization of parameters
    for i in range(nb_players):
        parameters[i] = {}
        for j, (key, param_range) in enumerate(param_ranges.items()):
            value = random.randint(*param_range)
            if key == 'freq':
                value = 0.2
            parameters[i][key] = value

    param_history = []
    print(parameters)
    print("random parameters")
    # speed = await get_EDMO_speed(server, parameters)
    # Points[0] = (parameters, speed)

    # TODO Store the parameters in a hash table
    iterations = 1
    while True:
        dim = 1
        for i in range(nb_players):
            for param, value in parameters[i].items():
                if param == 'freq':
                    continue
                new_parameters = parameters
                f = fill_parameters(parameters, i, param)

                print(parameters)
                max_speed, val = await golden(param_ranges[param][0], value, param_ranges[param][0], f, u[dim], server)
                Points[iterations] = (f(val), max_speed)
                iterations += 1
                
                    

        
                await asyncio.sleep(10)
        
        
def fill_parameters(parameters, id, param):
    def setter(value):
        updated_parameters = copy.deepcopy(parameters)
        updated_parameters[id][param] = value
        return updated_parameters
    return setter

# Constants for the golden ratio
R = 0.61803399  # The golden ratio
C = 1.0 - R   
tol = 1e-5
async def golden(ax, bx, cx, p, direction, server):
    """
    Perform a golden section search to find the minimum of the function f.

    Parameters:
        ax, bx, cx: float
            The bracketing triplet of abscissas (bx is between ax and cx).
        f: function
            The function to minimize.
        tol: float
            The tolerance for the fractional precision.

    Returns:
        float: The minimum value of the function.
        float: The abscissa of the minimum.
    """
    x0, x3 = ax, cx  # Initial points
    if fabs(distance(cx - bx)) > fabs(distance(bx - ax)):
        x1 = bx
        x2 = bx + list_mul(C, list_substract(cx - bx))  # x0 to x1 is the smaller segment
    else:
        x2 = bx
        x1 = bx - list_mul(C, list_substract(bx - ax))
    print(x0, x1, x2, x3)
    # Initial function evaluations
    parameters1 = p(x1)
    parameters2 = p(x2)
    print(parameters1)
    print(parameters2)
    f1 = await get_EDMO_speed(server, parameters1)
    f2 = await get_EDMO_speed(server, parameters2)
    f1 = -f1 # We want to compute the maximum speed so the minimum of -f
    f2 = -f2

    # Iteratively refine the search
    while fabs(distance(x3 - x0)) > tol * (fabs(x1) + fabs(x2)):
        if f2 < f1:
            x0, x1, x2 = x1, x2, R * x1 + C * x3
            f1, f2 = f2, await get_EDMO_speed(server, p(x2))
            f2 = -f2
        else:
            x3, x2, x1 = x2, x1, R * x2 + C * x0
            f2, f1 = f1, await get_EDMO_speed(server, p(x1))
            f1 = -f1

    if f1 < f2:
        return -f1, x1  
    else:
        return -f2, x2
    
def list_add(a, b):
    return [ax + bx for ax, bx in zip(a, b)]

def list_substract(a, b):
    return [ax - bx for ax, bx in zip(a, b)]

def list_mul(value, l):
    return [x * value for x in l]

def distance(point1:list, point2:list):
    if len(point1) != len(point2):
        raise ValueError("Both points must have the same dimensions.")
    
    return sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--explore", nargs="?", help="Type of EDMO you want to explore (default: Snake)", const="Snake")
    # args = parser.parse_args()
    
    asyncio.run(Powell())