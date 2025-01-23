import os
import re
from cv2 import exp
import numpy as np
import matplotlib.pyplot as plt
from ArUCo_Markers_Pose import pose_data, pose_estimation
from Utilities.Helpers import toTime
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from experiments import get_all_input
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import heapq
from DataCleaner import clean


debug = True
fps = 30
plot_data = None
def data_analysis(dir, nbPlayers: int = 2):
    '''
    - Extract the aruco positions from the video
    - Compute the edmo's movement from the aruco positions and store in pose_d
    - For each experiment get the time stamp where the input values are reached
    - Compute the frame number corresponding to the timestamp and retrieve the frames from pose_d
    parse the frame's x, y, z and filter out the moment when the edmo was moved
    '''
    
    all_input = get_all_input(nbPlayers) # (freq, (amp0, amp1), (off0, off1), (phb0, phb1))
    
    param_speed_dict: dict[int, list[list, list]] = {}
    for folder in os.listdir(dir):
        if os.path.splitext(folder)[1] != '':
            continue
        print(f'folder: {folder}')
        filepath = f'{dir}/{folder}'
  
        # Get the edmo's position and rotation for each frame
        print('Getting the edmo\'s positions...')
        pose_d = pose_data.Pose_data(filepath)
        succeed = pose_d.get_pose()
        if not succeed:
            print('error: marker pose missing')
            continue
        edmo_poses = pose_d.edmo_poses
        edmo_rots = pose_d.edmo_rots
        
        # Matching the motor data with the input data and matching it with the corresponding frames
        print('Matching the edmo\'s movement with the input data...')
        motor0 = open(f"{filepath}/Motor0.log", 'r').readlines()
        motor1 = open(f"{filepath}/Motor1.log", 'r').readlines()

        exp_nbs = folder.split('-', 2)
        exp_start, exp_end = int(exp_nbs[0]), int(exp_nbs[1])
        input_range = all_input[exp_start:exp_end+1]
        
        motor_ranges = []
        exp_edmo_poses = {}
        skips = 0
        for i, exp_param in enumerate(input_range): 
            experiment_nb = exp_start+i
            motor_range = find_exp_time_frames(input_range[i], (motor0, motor1))
            if not motor_range:
                skips += 1
                continue
            motor_ranges.append(motor_range)
            
            time_start = toDatetime(motor1[motor_range[0]].split(',')[0])
            time_end = toDatetime(motor1[motor_range[1]].split(',')[0])
            frame_start = time_start.seconds * fps + round(time_start.microseconds/1e6 * fps)
            frame_end = time_end.seconds * fps + round(time_end.microseconds/1e6 * fps)

            for frame in range(frame_start, frame_end):
                if frame in edmo_poses:
                    if experiment_nb not in exp_edmo_poses:
                        exp_edmo_poses[experiment_nb] = {}
                    exp_edmo_poses[experiment_nb][frame] = edmo_poses[frame]
        print(f'inputs skipped : {skips}')
        with open(f'{filepath}/edmo_pose.log', 'w') as f:
            json.dump(exp_edmo_poses, f)
        print('Computing the edmo\'s speed ...')
        exp_edmo_movement = compute_speed(exp_edmo_poses, filepath)
        merge_parameter_data(all_input, exp_edmo_movement)        
    return plot_data


def compute_speed(exp_edmo_poses:dict[int, dict[int, list]], filepath):
    '''
        Speed: units=meter per frame, sum of speed over the course of one experiment
        Displacement: units=meters, maximum displacement over the course of one experiment
    '''
    exp_edmo_movement = {}
    x_all_diff, y_all_diff, z_all_diff = [], [], []
    for exp_nb, positions in exp_edmo_poses.items():
        per_frame_x_speed, per_frame_y_speed, per_frame_z_speed = 0.0, 0.0, 0.0
        
        frame_keys = list(positions.keys())
        print(len(frame_keys))
        if len(frame_keys) <= 0:
            print("In compute_speed: zero valid frame")
            return {}
        
        nb_frames = frame_keys[-1] - frame_keys[0] + 1
        previous_el = None
        first_pose, last_pose = positions[frame_keys[0]], positions[frame_keys[-1]]
        x_displacement = abs(last_pose[0] - first_pose[0])
        y_displacement = abs(last_pose[1] - first_pose[1])
        z_displacement = abs(last_pose[2] - first_pose[2])
        
        # For each succession of frame and positions we compute the difference of position
        for i, pos_frame in enumerate(positions.items()): 
            if previous_el is None:
                previous_el = pos_frame
                continue
            frame_diff = pos_frame[0] - previous_el[0]

            x, y, z = previous_el[1]
            x_next, y_next, z_next = pos_frame[1]
            x_diff = (x_next-x)/frame_diff
            y_diff = (y_next-y)/frame_diff
            z_diff = (z_next-z)/frame_diff
            
            threshold = 0.1
            if abs(x_diff) > threshold or abs(y_diff) > threshold: # Too big of a movement per frame means the edmo was moved by an external force
                previous_el = pos_frame
                nb_frames -= frame_diff
                x_displacement -= abs(x_diff*frame_diff)
                y_displacement -= abs(y_diff*frame_diff)
                z_displacement -= abs(z_diff*frame_diff)
                continue
            x_all_diff.append(x_diff)
            y_all_diff.append(y_diff)
            z_all_diff.append(z_diff)
            
            per_frame_x_speed += x_diff
            per_frame_y_speed += y_diff
            per_frame_z_speed += z_diff
            
            previous_el = pos_frame
        
        per_frame_x_speed /= nb_frames
        per_frame_y_speed /= nb_frames
        per_frame_z_speed /= nb_frames
        x_displacement /= nb_frames
        y_displacement /= nb_frames
        z_displacement /= nb_frames
        per_frame_global_speed = float(np.sqrt(per_frame_x_speed**2 + per_frame_y_speed**2 + per_frame_z_speed**2))            
        per_frame_xy_speed = float(np.sqrt(per_frame_x_speed**2 + per_frame_y_speed**2))   
        per_frame_global_displacement = float(np.sqrt(x_displacement**2 + y_displacement**2 + z_displacement**2))            
        per_frame_xy_displacement = float(np.sqrt(x_displacement**2 + y_displacement**2)) 
        print(f"per_frame_global_displacement: {per_frame_global_displacement}")
        
        exp_edmo_movement[exp_nb] =[per_frame_x_speed, per_frame_y_speed, per_frame_z_speed, per_frame_global_speed, per_frame_xy_speed, 
                                    x_displacement, y_displacement, z_displacement, per_frame_global_displacement, per_frame_xy_displacement, exp_nb, nb_frames]
        f = open(f"{filepath}/speed_data.log", "w")
        json.dump(exp_edmo_movement, f)
    if debug:# and len(x_all_diff) != 0 and len(y_all_diff) != 0 and len(z_all_diff) != 0:
        n = 3
        print(f'avg x displacement: {sum(x_all_diff) / len(x_all_diff)}, max : {heapq.nlargest(n, x_all_diff)}, min : {heapq.nsmallest(n, x_all_diff)}')
        print(f'avg y displacement: {sum(y_all_diff) / len(y_all_diff)}, max : {heapq.nlargest(n, y_all_diff)}, min : {heapq.nsmallest(n,y_all_diff)}')
        print(f'avg z displacement: {sum(z_all_diff) / len(z_all_diff)}, max : {heapq.nlargest(n, z_all_diff)}, min : {heapq.nsmallest(n,z_all_diff)}')
    return exp_edmo_movement


def merge_parameter_data(all_input, exp_edmo_movement): 
    '''
    Where speed_type is an integer:
    1,2,3 : x,y,z per frame speed
    4,5 : xyz per frame speed, xy per frame speed
    6,7,8 : x,y,z displacement per frame
    9, 10 : experiment number, number of frames
    '''
    global plot_data
    amp1, amp2, off1, off2, phb_diff, speeds = [], [], [], [], [], {}
    for i in range(12): 
        speeds[i] = []
    for exp_nb, speed in exp_edmo_movement.items():
        inputs = all_input[exp_nb] # (freq, (amp0, amp1), (off0, off1), (phb0, phb1))
        amp1.append(inputs[1][0])
        amp2.append(inputs[1][1])
        off1.append(inputs[2][0])
        off2.append(inputs[2][1])
        phb_diff.append(abs(inputs[3][0]-inputs[3][1]))
        for i in range(12): 
            speeds[i].append(speed[i]) 
        
        
    data = pd.DataFrame({
        'Amp_motor_1': amp1,
        'Amp_motor_2': amp2,
        'Offset_motor_1': off1,
        'Offset_motor_2': off2,
        'Phase_difference': phb_diff,
        'x frame speed' : speeds[0],
        'y frame speed' : speeds[1],
        'z frame speed' : speeds[2],
        'xyz frame speed' : speeds[3],
        'xy frame speed' : speeds[4],
        'x frame displacement' : speeds[5],
        'y frame displacement' : speeds[6],
        'z frame displacement' : speeds[7],
        'xyz frame displacement' : speeds[8],
        'xy frame displacement' : speeds[9],
        'exp nb' : speeds[10],
        'nb frames' : speeds[11]
    })
    
    plot_data = pd.concat([plot_data, data], ignore_index=True) if plot_data is not None else data  
    return data
    

def toDatetime(time):
    t = datetime.strptime(time,"%H:%M:%S.%f")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

def find_exp_time_frames(input, motors_data):
    # Get the range of motor data that corresponds to the input 
    start_ends = []
    for i, motor in enumerate(motors_data):
        start_ends.append([])
        start_end = start_ends[i]
        j = 1
        while j < len(motor):
            motor_data = motor[j]
            data = motor_data.split(',')
            freq = abs(input[0] - float(data[1]))
            amp = abs(input[1][i] - float(data[2]))
            off = abs(input[2][i] - float(data[3]))        
            phb = abs(input[3][i] - float(data[4]))
            
            # start = when the motor data is close enough to the input
            if not len(start_end)%2:
                if freq < 0.1 and amp < 2 and off < 3 and phb < 0.07:
                    start_end.append(j)
                    
             # end = when the motor data is not close enough to the input anymore
            if len(start_end)%2:
                if freq > 0.1 or amp > 2 or off > 3 or phb > 0.07:
                    start_j = [start_end[i] for i in range(0, len(start_end)-1, 2)]
                    
                    if j - start_end[-1] <= 10: # ignore motor transitions (end and start too close)
                        del start_end[-1]
                        continue
                    elif len(start_j) > 0 and all(j < sj for sj in start_j): # ignore repeated motor data
                        del start_end[-1]
                        continue
                    else:
                        start_end.append(j)
            j += 1
    
    if len(start_ends) != len(motors_data):
        print(f'start ends: {start_ends}, input: {input}')
        return None

    start, end = 0, 0
    for i, se1 in enumerate(start_ends):
        for j, se2 in enumerate(start_ends):
            if i >= j:
                continue
            for index1 in range(0, len(se1)-1, 2):
                start1 = se1[index1]
                end1 = se1[index1+1]
                for index2 in range(0, len(se2)-1, 2):
                    start2 = se2[index2]
                    end2 = se2[index2+1]
                    
                    if (start1 > start2 and start1 > end2) or (start2 > start1 and start2 > end1):
                        continue
                    else:
                        if start != 0 and end != 0:
                            print(f'Found a second interval, verify the data !!! First: {start, end} second: {se1, se2}, indexes: {index1, index2}')
                        start = max(start1, start2)
                        end = min(end1, end2)
            
                        if start >= int(end):
                            start, end = 0, 0
    if start == 0 and end == 0:
        if debug:
            print(f'Skipped input: {input}, start ends: {start_ends}')
        return None
    return (start, int(end))
    

        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--path", type=str, help='Path to exploreData/"EDMO" (default: exploreData/Snake)', default="exploreData/Snake")
    # args = parser.parse_args()
     
    # path = args.path
    # path = 'exploreData/Snake/'
    # path = 'cleanData/Snake/'
    # plot_data = data_analysis(path)
    
    plot_data = None
    path = './exploreData/Snake/'
    # for folder in os.listdir(path):
    #     print(f'folder: {folder}')
    #     filepath = f'{path}/{folder}'
    #     files = os.listdir(filepath)
    #     # Extract the poses from the video
    #     if 'marker_pose.log' not in files:
    #         video = None
    #         for file in files:
    #             if os.path.splitext(file)[1].lower() == '.mp4':
    #                 video = f'/{file}'
    #         print("analyzing the video...")
    #         aruco_pose = pose_estimation.Aruco_pose(filepath+video)
    #         aruco_pose.pose_estimation()

    # print('Cleaning ...')
    # clean(path, True)

    print('Processing the data...')
    path = './cleanData/Snake/'
    plot_data = data_analysis(path)

    plot_data['Phase_difference'] = plot_data['Phase_difference'] * 180 / np.pi # radians to degrees

    with open(f'cleanData/Snake/plot data.log', 'w') as f:
        json.dump(plot_data.to_dict(orient='records'), f)
    
    
    
    
    
