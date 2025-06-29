# Implemented Powell method following this paper : https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf
import numpy as np
import asyncio
import random
import argparse
import math
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

debug = True

gopro = []
nb_legs = 0
freq_value = 1.0
param_ranges = {
    'freq': [1, 1],
    'amp': [0, 90],
    'off': [0, 180],
    'phb': [0, 180]
}
experiment_duration = 10 # How long one set of parameter runs (Seconds)

param_dict = {}
param_dict_path = ""

# region PARAM FORMAT 
""" 
PARAMETER FORMAT: 
    parameters for EDMOManual:     
    {
        0: {'freq': freq_value, 'amp': amp0, 'off': off0, 'phb': phb0}, 
        1: {'freq': freq_value, 'amp': amp1, 'off': off1, 'phb': phb1}
    }
    param_list for computing speed: [freq_value, [amp0, amp1], [off0, off1], [phb0, phb1]]                                                                  
    vector for Powell:              [amp0, amp1, off0, off1, phb0, phb1]    
    
The following functions convert between these formats                                                                               
"""
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
    return np.array([value for value_list in param_list[1:] for value in value_list])

def vector_to_parameters(vector):
    parameters = {}
    
    counter = 0
    for key, _ in param_ranges.items():
        for i in range(nb_legs):
            print(key, i)
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
    """
    Get the speed of the EDMO by sending the parameters to the EDMO and computing the speed from the video.
    Args:
        server     : EDMOManual 
        parameters : Set of parameters to be sent to the edmo
        
    Returns:
        displacement: speed computed using the positions of the EDMO on the first and last frame.
        or
        speed       : speed computed by averaging the difference in position of the EDMO on all frames.
    """
    vector = parameters_to_vector(parameters)
    if 0 in vector[:nb_legs] or np.nan in vector:
        print(f"{vector} in get_EDMO_speed")
        return 0.0
    if debug: print(f"parameters in get_EDMO_speed: {parameters}")
    
    # Look up the parameters dictionnary if this set already exists
    key = generate_unique_key(parameters_to_vector(parameters), param_ranges, nb_legs)
    if key in param_dict.keys():
        speed, displacement, confidence = param_dict[key]
        if confidence >= 200:
            print(f"speed: {speed}, displacement: {displacement}")
            return displacement
        
    # Send input to the server
    await server.runInputDict(parameters)
    
    server.GoProOn() # Start the GoPro recording
    await asyncio.sleep(experiment_duration) 
    filepaths = await server.GoProStopAndSave() # Stop the GoPro recording and save the video
    
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
    exp_edmo_poses = {}
    exp_edmo_poses[0] = {}
    valid_frames = pose_d.nbFrames
    print(f"Valid frames: {valid_frames}")
    if valid_frames <= 0:
        return 0.0
    
    for frame in range(valid_frames):
        if frame in edmo_poses:
            exp_edmo_poses[0][frame] = edmo_poses[frame]
    
    param_list = parameters_to_param_list(parameters)
    exp_edmo_movement = compute_speed(exp_edmo_poses, filespath) # Compute EDMO's speed
    data = merge_parameter_data({0:param_list}, exp_edmo_movement).to_dict(orient='records')   
    speed = data[0]['xy frame speed']*30 # m/s
    displacement = data[0]['xy frame displacement']*30
    print(f"speed: {speed}, displacement: {displacement}")
    print(data)
    # Store the parameters and speed in a hash table
    param_dict[key] = (speed, displacement, valid_frames)
    return displacement


# region POWELL
async def Powell(nb_players:int = 2, path:str=None):
    wifi_com = WifiCommunication(gopro[0], Path(f"GoPro/{gopro[0]}"))
    await wifi_com.initialize()
    
    nb_legs = nb_players
    
    # Initialize EDMOManual
    server = EDMOManual(gopro_list=gopro)
    asyncio.get_event_loop().create_task(server.run())
    server.GoProOff()
    
    n = nb_players * (len(param_ranges)-1) # number of dimensions to explore
    u = {dimension+1: [1 if i == dimension else 0 for i in range(n)] for dimension in range(n)} # Initialize the dimensions with unit vectors
    
    # Load past data
    param_history = []
    if path is not None:
        with open(path, 'r') as f:
            param_history = json.load(f)

    # Random initialization of parameters
    parameters = {}
    for i in range(nb_players):
        parameters[i] = {}
        for _, (key, param_range) in enumerate(param_ranges.items()):
            value = random.randint(*param_range)
            if key == 'freq':
                value = freq_value
            parameters[i][key] = float(value)
            
    # parameters = {0:{'freq':1.0, 'amp':80.0, 'off':60.0,'phb':0.0}, 1:{'freq':1.0, 'amp':60.0, 'off':120.0,'phb':0.0}}
    # [27.0, 58.0, 127.0, 100.0, 227.0, 62.0]done
    # [87.0, 78.0, 168.0, 106.0, 154.0, 124.0]
    # [45.0, 5.0, 7.0, 92.0, 113.0, 174.0]done
    # [43.0, 46.0, 57.0, 58.0, 44.0, 34.0]
    # [0.0, 84.0, 79.0, 89.0, 50.0, 80.0]
    # [17.0, 38.0, 137.0, 81.0, 36.0, 173.0]
    # [43.0, 30.0, 45.0, 131.0, 132.0, 69.0]
    # [63.0, 77.0, 115.0, 168.0, 122.0, 178.0]
    # [63.0, 77.0, 115.0, 168.0, 122.0, 178.0]
    # [70.0, 11.0, 141.0, 15.0, 87.0, 126.0]done
    # [60.0, 24.0, 105.0, 3.0, 150.0, 102.0]
    # [80.0, 77.0, 10.0, 168.0, 120.0, 12.0]done
    
    # init_random_point = [88, 59, 72, 100, 0, 62]
    # [62, 90, 43, 75, 91, 0]
    # [65, 63, 57, 137, 75, 0]
    # [57, 63, 57, 137, 75, 12]
    # parameters = vector_to_parameters(init_random_point)
    print(f"random parameters: {parameters}")
    
    current_speed = await get_EDMO_speed(server, parameters)
    Points = [(parameters_to_vector(parameters).tolist(), current_speed)]
    current_point = parameters_to_vector(parameters)
    iterations = 1
    last_speed = 0
    random_modified = False
    for _ in range(100):
        # Golden section search for each dimension
        for i in range(n):
            vector_min, middle, vector_max = line_min_max(current_point, u[i+1])
            max_speed, new_point = await golden(vector_min, middle, vector_max, current_speed, vector_to_parameters, server)
            new_point = np.array([round(x) for x in new_point])
            
            Points.append((new_point.tolist(), max_speed))
            current_point = new_point
            current_speed = max_speed
            iterations += 1

        param_history.append((Points, u))
        
        # Storing the param history
        store_path = f"{server.activeSessions[list(server.activeSessions.keys())[0]].sessionLog.directoryName}/param_history.log"
        print(f"Storing param history in {store_path}")
        f = open(store_path, "w")
        json.dump(param_history, f)
        f.close()
        speeds = [speed for vector, speed in Points]
        vectors = [np.array(vector) for vector, speed in Points]
        
        # Discarding the Direction of Largest Increase
        max_speed_index = np.argmax(speeds)
        max_speed = -speeds[max_speed_index]
        print(f"Points{Points}")
        print(f"max speed index: {max_speed_index}")
        
        # Powell's conditions
        f_0 = -speeds[0]
        f_N = -speeds[-1]
        extra_point = np.array((2 * vectors[-1]) - vectors[0]) # 2*P_N-P_0 
        print(f"f0:{f_0}, f_N:{f_N}, max_speed:{max_speed}, extrapoint:{extra_point}")
        
        PN_P0_direction = np.array(vectors[-1] - vectors[0])
        extra_point = line_min_max(current_point, PN_P0_direction, extra_point)
        extrapolated_speed = await get_EDMO_speed(server, vector_to_parameters(extra_point))
        f_E = -extrapolated_speed
        print(f"f_E: {f_E}")
        if f_E >= f_0 or 2*(f_0-2*f_N+f_E)*((f_0-f_N)-max_speed)**2 >= max_speed*(f_0-f_E)**2:
            print("Direction not replaced")
            pass
        else:
            u[max_speed_index] = PN_P0_direction
            print(f"PN-P0:{u[max_speed_index]}")
        
        Points = []
        f = open(param_dict_path, "w")
        json.dump(param_dict, f)
        f.close()
        
        # Ending conditions
        if 2.0 * (speeds[-1] - last_speed) <= 0.1 * (math.fabs(last_speed) + math.fabs(speeds[-1])):
            print(f"Convergence reached, maximum found! old: {last_speed}, current: {speeds[-1]}")
            
            if speeds[-1] == last_speed and not random_modified:
                random_modifications = np.random.randint(-10, 10, size=current_point.shape)
                modified_arr = current_point + random_modifications
                clamped_arr = modified_arr.copy()
                for i in range(nb_legs):
                    clamped_arr[i] = np.clip(modified_arr[i], *param_ranges['amp'])    # amp1, amp2
                    clamped_arr[i + nb_legs] = np.clip(modified_arr[i + nb_legs], *param_ranges['off'])  # off1, off2
                    clamped_arr[i + (2*nb_legs)] = np.clip(modified_arr[i + (2*nb_legs)], *param_ranges['phb'])  # phb1, phb2

                current_point = clamped_arr
                random_modified = True
            else:
                break
        else:
            print(f"{2.0 * (speeds[-1] - last_speed)} > {0.1 * (math.fabs(last_speed) + math.fabs(speeds[-1]))}")
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
    
    # if the point is already at an extremum along the direction we take a random point in order to get out of a potential local maximum
    rand = random.randint(1, 10)
    if min_lower_diff == 0:
        min_lower_diff = -rand
    elif min_upper_diff == 0:
        min_upper_diff = -rand
    else:
        pass
    
    min_point, max_point = [], []
    for i, dim in enumerate(direction):
        min_point.append(float(point[i]+(min_upper_diff*dim)))
        max_point.append(float(point[i]-(min_lower_diff*dim)))
        
    min_point, max_point, point = np.array(min_point), np.array(max_point), np.array(point)        
    if extra_point is not None:
        interval_length = distance(min_point, max_point)
        if distance(min_point, extra_point) >= interval_length:
            return max_point
        elif distance(max_point, extra_point) >= interval_length:
            return min_point
        else:
            return np.array(extra_point)
    
    if distance(min_point, max_point) >= distance(point, max_point):
        return min_point, point, max_point
    else: # the point is at an extremum but which one
        if distance(min_point, point) >= distance(max_point, point):
            return min_point, max_point, point
        else:
            return point, min_point, max_point


# region GOLDEN SEARCH
# Constants for the golden ratio
R = 0.61803399  # The golden ratio
C = 1.0 - R   
tol = 1e-5
async def golden(ax, bx, cx, fb, p, server):
    """
    Perform a golden section search to find the minimum of the function.
    Our function is minus the speed of the EDMO (get_EDMO_speed()).
    Maximizing the speed == Minimizing minus the speed

    Parameters:
        ax, bx, cx: vector of floats on a single line in n dimensions
            The bracketing triplet of abscissas (bx is between ax and cx).
        fb: Speed of the EDMO at bx
        p: The function that turns a vector into parameters
        server: an instance of EDMOManual
        
    Returns:
        float: The maximum speed.
        float: The position (vector) of the maximum.
    """
    
    x0, x3 = np.array(ax), np.array(cx)  
    parameters0, parameters3, parametersb = p(x0), p(x3), p(bx)
    if debug:
        print(f"Initial parameters computation ax, bx, cx:{ax}, {bx}, {cx}")
    f0 = await get_EDMO_speed(server, parameters0)
    fb = await get_EDMO_speed(server, parametersb)
    f3 = await get_EDMO_speed(server, parameters3)
    # print(f0, f3)
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
    if math.fabs(distance(cx, bx)) > math.fabs(distance(bx, ax)):
        x1 = bx
        f1 = fb
        x2 = np.array(bx + (C * (cx - bx)))  # x0 to x1 is the smaller segment
        if debug:
            print("f2 computation...")
        f2 = await get_EDMO_speed(server, parameters=p(x2))
        f2 = -f2
    else:
        x2 = bx
        f2 = fb
        x1 = np.array(bx - (C * (bx - ax)))
        if debug:
            print("f1 computation...")
        f1 = await get_EDMO_speed(server, parameters=p(x1))
        f1 = -f1
        
    # print(x0, x1, x2, x3)

    # Iteratively refine the search
    itera = 0
    while math.fabs(distance(x3, x0)) > tol * (math.fabs(norm(x1)) + math.fabs(norm(x2))):
        if debug:
            print(f"in while loop:{itera}")
        if f2 < f1:
            x0, x1, x2 = x1, x2, np.array((R* x1) + (C * x3))
            f1, f2 = f2, await get_EDMO_speed(server, p(x2))
            f2 = -f2
        else:
            x3, x2, x1 = x2, x1, np.array((R * x2) + (C * x0))
            f2, f1 = f1, await get_EDMO_speed(server, p(x1))
            f1 = -f1
        itera += 1
    if f1 < f2:
        return -f1, x1  
    else:
        return -f2, x2

# region helpers
def norm(l:list):
    return math.sqrt(sum(x**2 for x in l))

def distance(point1:list, point2:list):
    if len(point1) != len(point2):
        raise ValueError("Both points must have the same dimensions.")
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def generate_unique_key(values, param_ranges, nb_legs):
    """
    Generate a unique key for a list of parameter values using bitwise encoding,
    skipping the 'freq' parameter.

    Args:
        values (list): List of parameter values.
        param_ranges (dict): Dictionary defining the ranges for each parameter.
        num_sets (int): Number of sets of parameters.

    Returns:
        int: A unique key for the parameter combination.
    """
    bitwise_key = 0
    total_bits = 0
    
    ordered_params = ["amp", "off", "phb"]  # Define the order of parameters explicitly
    params_to_encode = {}

    for param in ordered_params:
        for i in range(1, nb_legs + 1):
            if param in param_ranges:
                params_to_encode[f"{param}{i}"] = tuple(param_ranges[param])

    for value, param in zip(values, params_to_encode):        
        min_val, max_val = params_to_encode[param]
        # Normalize the value
        normalized_value = int(value - min_val)
        
        # Calculate the number of bits required for this parameter
        range_size = max_val - min_val + 1
        bits = math.ceil(math.log2(range_size))
        
        # Shift the existing key and add the new parameter value
        bitwise_key |= (normalized_value << total_bits)
        
        # Update the total number of bits used so far
        total_bits += bits

    return bitwise_key

def closest_line(point, lines):
    """
    Finds the closest line to a given point in 6D space.
    
    Args:
        point: A numpy array of shape (6,), the point in 6D space.
        lines: A list of tuples, where each tuple contains:
               - A numpy array (6,) representing a point on the line.
               - A numpy array (6,) representing the direction of the line.
               
    Returns:
        The index of the closest line and the shortest distance.
    """
    min_distance = float('inf')
    closest_line_idx = -1
    
    for idx, (p0, d) in enumerate(lines):
        v = point - p0
        t = np.dot(v, d) / np.dot(d, d)  # Projection scalar
        proj_point = p0 + t * d          # Projected point on the line
        distance = np.linalg.norm(point - proj_point)  # Distance to the line
        
        if distance < min_distance:
            min_distance = distance
            closest_line_idx = idx
            
    return closest_line_idx, min_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="?", help="Path to the parameter history", const=None)
    args = parser.parse_args()
    
    path = args.path    

    gopro = ["GoPro 4448"] 
    nb_legs = 4
    
    param_dict_path = f"./Utilities/parameters_dictionnary_Snake1.log"            
    with open(param_dict_path, 'r') as f:
        param_dict = json.load(f)

    asyncio.run(Powell(nb_players=nb_legs, path=path))