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
from ArUCo_Markers_Pose import *
from experiments import get_all_input
from datetime import datetime, timedelta


fps = 30

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
        print(f'folder: {folder}')
        filepath = f'{dir}/{folder}'
        files = os.listdir(filepath)
        # Extract the poses from the video
        if 'marker_pose.log' not in files:
            video = None
            for file in files:
                if os.path.splitext(file)[1].lower() == '.mp4':
                    video = f'/{file}'
            print("analyzing the video...")
            aruco_pose = pose_estimation.Aruco_pose(filepath+video)
            aruco_pose.pose_estimation()
        
        # Get the edmo's position and rotation for each frame
        print('Getting the edmo\'s positions...')
        pose_d = pose_data.Pose_data(filepath)
        pose_d.get_pose()
        edmo_poses = pose_d.edmo_poses
        edmo_rots = pose_d.edmo_rots
                
        # Matching the motor data with the input data and matching it with the corresponding frames
        print('Matching the edmo\'s movement with the input data')
        motor0 = open(f"{filepath}/Motor0.log", 'r').readlines()
        motor1 = open(f"{filepath}/Motor1.log", 'r').readlines()

        exp_nbs = folder.split('-', 2)
        exp_start, exp_end = int(exp_nbs[0]), int(exp_nbs[1])
        input_range = all_input[exp_start:exp_end+1]
        
        motor_ranges = []
        exp_edmo_poses = {}
        for i, exp_param in enumerate(input_range): 
            motor_range = find_exp_time_frames(input_range[i], (motor0, motor1))
            if not motor_range:
                continue
            motor_ranges.append(motor_range)
            
            time_start = toDatetime(motor0[motor_range[0]].split(',')[0])
            time_end = toDatetime(motor0[motor_range[1]].split(',')[0])
            frame_start = time_start.seconds * fps + round(time_start.microseconds/1e6 * fps)
            frame_end = time_end.seconds * fps + round(time_end.microseconds/1e6 * fps)
            # print(time_start, time_end, frame_start, frame_end)
            f = 0
            for frame in range(frame_start, frame_end):
                if frame in edmo_poses:
                    f += 1
                    if exp_start+i not in exp_edmo_poses:
                        exp_edmo_poses[exp_start+i] = []
                    exp_edmo_poses[exp_start+i].append(edmo_poses[frame])
            # print(f'{f}/{frame_end-frame_start}')
        with open(f'{filepath}/edmo_pose.log', 'w') as f:
            json.dump(exp_edmo_poses, f)
            
        # print(len(motor_ranges))
        # print(exp_edmo_poses.keys())
        exp_edmo_movement = {}
        for exp_nb, positions in exp_edmo_poses.items():
            '''
            Speed: units=meter per frame, sum of speed over the course of one experiment
            abs_speed: units=meter per frame, averaged sum of absolute speed over the course of one experiment
            Movement: units=meters, maximum displacement over the course of one experiment
            '''
            x_mov, y_mov = 0.0, 0.0,  
            x_speed, y_speed, z_speed, speed = 0.0, 0.0, 0.0, 0.0
            abs_x_speed, abs_y_speed, abs_z_speed, abs_speed = 0.0, 0.0, 0.0, 0.0
            minX, minY, maxX, maxY = 2.0, 2.0, 0.0, 0.0
            nb_frames = len(positions)
            for i in range(nb_frames - 1):
                x, y, z = positions[i]
                x_next, y_next, z_next = positions[i+1]
                x_diff = x_next-x
                y_diff = y_next-y
                z_diff = z_next-z
                
                # if x_diff > 
                
                x_speed += x_diff
                y_speed += y_diff
                z_speed += z_diff
                
                abs_x_speed += abs(x_diff)
                abs_y_speed += abs(y_diff)
                abs_z_speed += abs(z_diff)
                
                minX = min(minX, min(x, x_next))
                minY = min(minY, min(y, y_next))
                maxX = max(maxX, max(x, x_next))
                maxY = max(maxY, max(y, y_next))
            x_mov = maxX - minX
            y_mov = maxY - minY
            speed = float(np.sqrt(x_speed**2 + y_speed**2 + z_speed**2))            
            abs_speed = float(np.sqrt(abs_x_speed**2 + abs_y_speed**2 + abs_z_speed**2))            
            exp_edmo_movement[exp_nb] =[x_mov, y_mov,\
                                        x_speed, y_speed, z_speed, speed,\
                                        abs_x_speed/nb_frames, abs_y_speed/nb_frames, abs_z_speed/nb_frames, abs_speed/nb_frames]
            # f = open(f"{filepath}/speed_data.log", "w")
            # json.dump(exp_edmo_movement, f)
        plot_parameter_space(all_input, exp_edmo_movement)        
                
            
def plot_parameter_space(all_input, exp_edmo_movement): 
    amp1, amp2, off1, off2, phb_diff, speeds = [], [], [], [], [], []
    for frame, speed in exp_edmo_movement.items():
        inputs = all_input[frame] # (freq, (amp0, amp1), (off0, off1), (phb0, phb1))
        amp1.append(inputs[1][0])
        amp2.append(inputs[1][1])
        off1.append(inputs[2][0])
        off2.append(inputs[2][1])
        phb_diff.append(abs(inputs[3][0]-inputs[3][1]))
        speeds.append(speed[-1])
        # print(inputs)
        # print(speeds)
        # return
    
    
    fig = plt.figure(figsize=(12, 7))  # Adjust the figure size for better spacing

    # Left subplot
    ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, 1st plot
    for i in range(len(off1)):
        ax1.scatter(off1[i], off2[i], phb_diff[i], s=speeds[i]*3000, c=amp1[i], marker='o', cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Offset motor 1')
    ax1.set_ylabel('Offset motor 2')
    ax1.set_zlabel('Phase difference')
    ax1.set_title('Graph 1')
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1, pad=0.1)
    cbar1.set_label('Speed')

    # Right subplot
    ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, 2nd plot
    for i in range(len(off1)):
        ax2.scatter(off1[i], off2[i], phb_diff[i], s=speeds[i]*3000, c=amp2[i], marker='o', cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Offset motor 1')
    ax2.set_ylabel('Offset motor 2')
    ax2.set_zlabel('Phase difference')
    ax2.set_title('Graph 2')
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2, pad=0.1)
    cbar2.set_label('Speed')

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
        

def toDatetime(time):
    t = datetime.strptime(time,"%H:%M:%S.%f")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

def find_exp_time_frames(input, motors_data):
    # Get the range of motor data that corresponds to the input 
    start_end = []
    for i, motor in enumerate(motors_data):
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
                    # if len(start_j) > 1:
                    #     print(f'j: {j}, start_j: {start_j}')
                    #     print(all(j < sj for sj in start_j))
                    if j - start_end[-1] <= 2: # ignore motor transitions
                        del start_end[-1]
                        continue
                    elif len(start_j) > 0 and all(j < sj for sj in start_j): # ignore repeated motor data
                        # print(f'j: {j}, start_j: {start_j}')
                        del start_end[-1]
                        continue
                    else:
                        start_end.append(j)
                        break
            j += 1
    
    # print(start_end)
    start, end = 0, float('inf')
    for indexes in range(0, len(start_end), 2):
        start = max(start, start_end[indexes])
        end = min(end, start_end[indexes+1])
    if len(start_end) != 2*len(motors_data) or start >= int(end):
        return None
    return (start, int(end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help='Path to exploreData/"EDMO" (default: exploreData/Snake)', default="exploreData/Snake")
    args = parser.parse_args()
     
    path = args.path
    path = 'exploreData/Snake/'
    path = 'cleanData/Snake/'
    data_analysis(path)
