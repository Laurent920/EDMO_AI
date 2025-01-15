# Implemented Powell method following this paper : https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf
import numpy as np
import asyncio
import random
import argparse
from math import fabs, sqrt, floor, ceil
import os
from pathlib import Path
import copy
import sys
import json

from EDMOManual import EDMOManual
from data_analysis import merge_parameter_data, plot_data, compute_speed
from ArUCo_Markers_Pose import pose_data, pose_estimation
from GoPro.wifi.WifiCommunication import WifiCommunication

#TODO: livestream gopro frames
#TODO: Powell termination criterion adjustment

debug = True

gopro = ["GoPro 4448"]
nb_legs = 2
freq_value = 1.0
param_ranges = {
    'freq': [1, 1],
    'amp': [0, 90],
    'off': [0, 180],
    'phb': [0, 359]
}

# region PARAMETER FORMAT: 
# parameters: {0: {'freq': freq_value, 'amp': amp0, 'off': off0, 'phb': phb0}, 1: {'freq': f, 'amp': amp1, 'off': off1, 'phb': phb1}} needed for EDMOManual
# param_list: [freq_value, [amp0, amp1], [off0, off1], [phb0, phb1]]                                                                  needed for computing speed
# vector:     [amp0, amp1, off0, off1, phb0, phb1]                                                                                    needed for Powell
def parameters_to_param_list(parameters):
    param_list = [parameters[0]['freq']]

    amps = [p['amp'] for p in parameters.values()]
    offs = [p['off'] for p in parameters.values()]
    phbs = [p['phb'] for p in parameters.values()]

    param_list.append(amps)
    param_list.append(offs)
    param_list.append(phbs)

    return param_list

def parameters_to_vector(parameters):
    param_list = parameters_to_param_list(parameters)
    return [value for value_list in param_list[1:] for value in value_list]

def vector_to_parameters(vector):
    parameters = {}
    
    counter = 0
    for key, _ in param_ranges.items():
        for i in range(nb_legs):
            if key == 'freq':
                parameters[i] = {}
                parameters[i][key] = freq_value
            else:  
                parameters[i][key] = vector[counter]
                counter += 1          
    if counter != len(vector):
        print(f"The vector size({len(vector)}) does not match the number of parameters of the EDMO({counter})")
        return None
    return parameters

def vector_to_param_list(vector):
    parameters = vector_to_parameters(vector)
    return parameters_to_param_list(parameters)
    

# region GET EDMO SPEED    
async def get_EDMO_speed(server, parameters):
    vector = parameters_to_vector(parameters)
    if vector[:nb_legs] == [0]*nb_legs or np.nan in vector:
        print(f"{vector} in get_EDMO_speed")
        return 0.0
    if debug:
        print(f"parameters in get_EDMO_speed: {parameters}")
    # Send input to the server
    await server.runInputDict(parameters)
    
    server.GoProOn()
    await asyncio.sleep(10) # How long one set of parameter runs
    filepaths = await server.GoProStopAndSave()
    
    print(filepaths)
    video_path = filepaths[0]
    filespath = os.path.dirname(video_path) 
    aruco_pose = pose_estimation.Aruco_pose(video_path) # Get the positions of the Aruco markers
    dict_all_pos = aruco_pose.pose_estimation() 
        
    pose_d = pose_data.Pose_data(filespath, dict_all_pos=dict_all_pos) # Get the positions of the EDMO
    succeed = pose_d.get_pose()
    if not succeed:
        return
    print("EDMO pose calculation succeeded")
    edmo_poses = pose_d.edmo_poses
    edmo_rots = pose_d.edmo_rots
    exp_edmo_poses = {}
    exp_edmo_poses[0] = {}
    for frame in range(pose_d.nbFrames):
        if frame in edmo_poses:
            exp_edmo_poses[0][frame] = edmo_poses[frame]
    
    param_list = parameters_to_param_list(parameters)
    exp_edmo_movement = compute_speed(exp_edmo_poses, filespath) # Compute EDMO's speed
    data = merge_parameter_data({0:param_list}, exp_edmo_movement)   
    print(data)
    speed = data['xy frame speed'][0]*30 # m/s
    return speed


# region POWELL
ftol = 1e-4
async def Powell(nb_players:int = 2):
    nb_legs = nb_players
    wifi_com = WifiCommunication(gopro[0], Path(f"GoPro/{gopro[0]}"))
    await wifi_com.initialize()
    
    # Initialize EDMOManual
    server = EDMOManual(gopro_list=gopro)
    asyncio.get_event_loop().create_task(server.run())
    server.GoProOff()
    
    n = nb_players * (len(param_ranges)-1) # number of dimensions to explore
    u = {dimension+1: [1 if i == dimension else 0 for i in range(n)] for dimension in range(n)} # Initialize the dimensions with unit vectors

    parameters = {}
    # Random initialization of parameters
    for i in range(nb_players):
        parameters[i] = {}
        for _, (key, param_range) in enumerate(param_ranges.items()):
            value = random.randint(*param_range)
            if key == 'freq':
                value = freq_value
            parameters[i][key] = float(value)

    # TODO Store the parameters in a hash table
    param_history = []

    print(f"random parameters: {parameters}")
    current_speed = await get_EDMO_speed(server, parameters)
    Points = [(parameters_to_vector(parameters), current_speed)]
    current_point = parameters_to_vector(parameters)
    iterations = 1
    last_speed = 0
    for _ in range(100):
        # Golden section search for each dimension
        for i in range(n):
            vector_min, vector_max = line_min_max(current_point, u[i+1])
            max_speed, new_point = await golden(vector_min, current_point, vector_max, current_speed, vector_to_parameters, server)
            new_point = [round(x) for x in new_point]
            
            Points.append((new_point, max_speed))
            current_point = new_point
            current_speed = max_speed
            iterations += 1
                
        param_history.append(Points)
        store_path = f"{server.activeSessions[list(server.activeSessions.keys())[0]].sessionLog.directoryName}/param_history.log"
        f = open(store_path, "w")
        json.dump(param_history, f)
        speeds = [speed for vector, speed in Points]
        vectors = [vector for vector, speed in Points]
        
        # Discarding the Direction of Largest Increase
        max_speed_index = np.argmax(speeds)
        max_speed = -speeds[max_speed_index]
        print(f"Points{Points}")
        print(f"max speed index: {max_speed_index}")
        
        # Powell's conditions
        f_0 = -speeds[0]
        f_N = -speeds[-1]
        extra_point = vector_sub(vector_mul(2, vectors[-1]), vectors[0]) # 2*P_N-P_0 
        print(f"f0:{f_0}, f_N:{f_N}, max_speed:{max_speed}, extrapoint:{extra_point}")
        
        PN_P0_direction = vector_sub(vectors[-1], vectors[0])
        extra_point = line_min_max(current_point, PN_P0_direction, extra_point)
        extrapolated_speed = await get_EDMO_speed(server, vector_to_parameters(extra_point))
        f_E = -extrapolated_speed
        print(f"f_E: {f_E}")
        if f_E >= f_0 or 2*(f_0-2*f_N+f_E)*((f_0-f_N)-max_speed)**2 >= max_speed*(f_0-f_E)**2:
            pass
        else:
            u[max_speed_index] = PN_P0_direction
            print(f"PN-P0:{u[max_speed_index]}")
        
        Points = []
        
        if 2.0 * (speeds[-1] - last_speed) <= ftol * (fabs(last_speed) + fabs(speeds[-1])):
            print(f"Convergence reached, maximum found! old: {last_speed}, current: {speeds[-1]}")
        else:
            print(f"{2.0 * (speeds[-1] - last_speed)} > {ftol * (fabs(last_speed) + fabs(speeds[-1]))}")
        last_speed = speeds[-1]


def line_min_max(point, direction, extra_point=None):
    '''
    This method takes a point's coordinate and computes the maximum and minimum points along the direction in the restricted parameter space; 
    if extra_point is given: compute if that extra_point is inside the parameter space otherwise give the furthest point allowed
    '''
    
    # Get absolute max and min
    abs_min_vector, abs_max_vector = [], []
    for key, value_range in param_ranges.items():
        for i in range(nb_legs):
            if key == 'freq':
                continue
            abs_min_vector.append(value_range[0])
            abs_max_vector.append(value_range[1])

    # Get how much we can add/remove our direction to/from the given point on each parameter: (max value for the parameter - value on the point given)/direction value
    upper_diff = [(abs-val)/dir if dir != 0 else np.nan for abs, dir, val in zip(abs_max_vector, direction, point)]        
    lower_diff = [(val-abs)/dir if dir != 0 else np.nan for abs, dir, val in zip(abs_min_vector, direction, point)]        
    # We can only increase/decrease by the most restrictive value
    min_upper_diff = np.nanmin(upper_diff)
    min_lower_diff = np.nanmin(lower_diff)
    
    if debug:
        print("Factors in line_min_max")
        print(f"direction:{direction}, upper_diff:{upper_diff}, lower_diff:{lower_diff}, point:{point}")
        print(min_lower_diff, min_upper_diff)
    
    min_point, max_point = [], []
    for i, dim in enumerate(direction):
        min_point.append(float(point[i]+(min_upper_diff*dim)))
        max_point.append(float(point[i]-(min_lower_diff*dim)))
    
    if extra_point is not None:
        interval_length = distance(min_point, max_point)
        if distance(min_point, extra_point) >= interval_length:
            return max_point
        elif distance(max_point, extra_point) >= interval_length:
            return min_point
        else:
            return extra_point
    
    return min_point, max_point


# Constants for the golden ratio
R = 0.61803399  # The golden ratio
C = 1.0 - R   
tol = 1e-5
async def golden(ax, bx, cx, fb, p, server):
    """
    Perform a golden section search to find the minimum of the function.
    Our function is minus the speed of the EDMO(get_EDMO_speed).
    Maximizing the speed == Minimizing minus the speed

    Parameters:
        ax, bx, cx: vector of floats on a single line in n dimensions
            The bracketing triplet of abscissas (bx is between ax and cx).
        f: function
            The function to minimize.
        tol: float
            The tolerance for the fractional precision.

    Returns:
        float: The minimum minus speed.
        float: The position(vector) of the minimum.
    """
    x0, x3 = ax, cx  
    parameters0, parameters3 = p(x0), p(x3)
    print(f"Initial parameters computation ax, bx, cx:{ax}, {bx}, {cx}")
    f0 = await get_EDMO_speed(server, parameters0)
    f3 = await get_EDMO_speed(server, parameters3)
    print(f0, f3)
    # Return the maximum speed point bx is not a maximum because one end of the line is a maximum
    if fb < f0 and fb < f3:
        if f0 <= f3:
            return f3, x3
        else:
            return f0, x0
    elif fb < f3:
        return f3, x3
    elif fb < f0:
        return f0, x0
    else:
        pass

    f0, f3, fb = -f0, -f3, -fb
    if fabs(distance(cx, bx)) > fabs(distance(bx, ax)):
        x1 = bx
        f1 = fb
        x2 = vector_add(bx, vector_mul(C, vector_sub(cx, bx)))  # x0 to x1 is the smaller segment
        print("f2 computation...")
        f2 = await get_EDMO_speed(server, parameters=p(x2))
        f2 = -f2
    else:
        x2 = bx
        f2 = fb
        x1 = vector_sub(bx, vector_mul(C, vector_sub(bx, ax)))
        print("f1 computation...")
        f1 = await get_EDMO_speed(server, parameters=p(x1))
        f1 = -f1
        
    print(x0, x1, x2, x3)

    # Iteratively refine the search
    itera = 0
    while fabs(distance(x3, x0)) > tol * (fabs(norm(x1)) + fabs(norm(x2))):
        print(f"in while loop:{itera}")
        if f2 < f1:
            x0, x1, x2 = x1, x2, vector_add(vector_mul(R, x1), vector_mul(C, x3))
            f1, f2 = f2, await get_EDMO_speed(server, p(x2))
            f2 = -f2
        else:
            x3, x2, x1 = x2, x1, vector_add(vector_mul(R, x2), vector_mul(C, x0))
            f2, f1 = f1, await get_EDMO_speed(server, p(x1))
            f1 = -f1
        itera += 1
    if f1 < f2:
        return -f1, x1  
    else:
        return -f2, x2
    
def vector_add(a:list, b:list):
    return [ax + bx for ax, bx in zip(a, b)]

def vector_add_val(l:list, value:int):
    return [x + value for x in l]

def vector_sub(a:list, b:list):
    return [ax - bx for ax, bx in zip(a, b)]

def vector_mul(value, l:list):
    return [x * value for x in l]

def norm(l:list):
    return sqrt(sum(x**2 for x in l))

def distance(point1:list, point2:list):
    if len(point1) != len(point2):
        raise ValueError("Both points must have the same dimensions.")
    
    return sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def fill_parameters(parameters, id, param):
    def setter(value):
        updated_parameters = copy.deepcopy(parameters)
        updated_parameters[id][param] = value
        return updated_parameters
    return setter


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--explore", nargs="?", help="Type of EDMO you want to explore (default: Snake)", const="Snake")
    # args = parser.parse_args()
    
    asyncio.run(Powell())