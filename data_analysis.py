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
import seaborn as sns
import pandas as pd
import plotly.express as px
import heapq


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
        succeed = pose_d.get_pose()
        if not succeed:
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
        fig, axes = plt.subplots(18, 10, figsize=(18, 12)) # plot all movements for one set of experiment (180)
        axes = axes.flatten()
        for i, exp_param in enumerate(input_range): 
            experiment_nb = exp_start+i
            motor_range = find_exp_time_frames(input_range[i], (motor0, motor1))
            if not motor_range:
                continue
            motor_ranges.append(motor_range)
            
            time_start = toDatetime(motor0[motor_range[0]].split(',')[0])
            time_end = toDatetime(motor0[motor_range[1]].split(',')[0])
            frame_start = time_start.seconds * fps + round(time_start.microseconds/1e6 * fps)
            frame_end = time_end.seconds * fps + round(time_end.microseconds/1e6 * fps)
            # print(time_start, time_end, frame_start, frame_end)

            for frame in range(frame_start, frame_end):
                if frame in edmo_poses:
                    if experiment_nb not in exp_edmo_poses:
                        exp_edmo_poses[experiment_nb] = {}
                    exp_edmo_poses[experiment_nb][frame] = edmo_poses[frame]

            if experiment_nb in exp_edmo_poses:
                plot_2D_poses(exp_edmo_poses[experiment_nb], axes=axes[i], title=f'experiments nb {exp_start+i}')

                # if experiment_nb == 2306:
                #     plot_2D_poses(exp_edmo_poses[experiment_nb])
        plt.tight_layout()
        # plt.show()
        
        with open(f'{filepath}/edmo_pose.log', 'w') as f:
            json.dump(exp_edmo_poses, f)
        print('Computing the edmo\'s speed ...')
        exp_edmo_movement = compute_speed(exp_edmo_poses)
        merge_parameter_data(all_input, exp_edmo_movement)        
    return plot_data


def compute_speed(exp_edmo_poses:dict[int, dict[int, list]]):
    '''
        Speed: units=meter per frame, sum of speed over the course of one experiment
        abs_speed: units=meter per frame, averaged sum of absolute speed over the course of one experiment
        Movement: units=meters, maximum displacement over the course of one experiment
    '''
    exp_edmo_movement = {}
    x_all_diff, y_all_diff, z_all_diff = [], [], []
    for exp_nb, positions in exp_edmo_poses.items():
        x_mov, y_mov = 0.0, 0.0,  
        x_speed, y_speed, z_speed, speed = 0.0, 0.0, 0.0, 0.0
        abs_x_speed, abs_y_speed, abs_z_speed, abs_speed = 0.0, 0.0, 0.0, 0.0
        minX, minY, maxX, maxY = 2.0, 2.0, 0.0, 0.0
        nb_frames = len(positions)
        previous_el = None
        for i, pos_frame in enumerate(positions.items()):
            if previous_el is None:
                previous_el = pos_frame
                continue
            frame_diff = pos_frame[0] - previous_el[0]

            x, y, z = previous_el[1]
            x_next, y_next, z_next = pos_frame[1]
            x_diff = x_next-x
            y_diff = y_next-y
            z_diff = z_next-z
            
            x_all_diff.append(x_diff/frame_diff)
            y_all_diff.append(y_diff/frame_diff)
            z_all_diff.append(z_diff/frame_diff)
            if x_diff/frame_diff > 0.01 or y_diff/frame_diff > 0.01 :
                previous_el = pos_frame
                continue
            
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
            previous_el = pos_frame
        x_mov = maxX - minX
        y_mov = maxY - minY
        speed = float(np.sqrt(x_speed**2 + y_speed**2 + z_speed**2))            
        abs_speed = float(np.sqrt(abs_x_speed**2 + abs_y_speed**2 + abs_z_speed**2))            
        exp_edmo_movement[exp_nb] =[x_mov, y_mov,\
                                    x_speed, y_speed, z_speed, speed,\
                                    abs_x_speed/nb_frames, abs_y_speed/nb_frames, abs_z_speed/nb_frames, abs_speed/nb_frames]
        # f = open(f"{filepath}/speed_data.log", "w")
        # json.dump(exp_edmo_movement, f)
    n = 3
    print(f'avg x displacement: {sum(x_all_diff) / len(x_all_diff)}, max : {heapq.nlargest(n, x_all_diff)}, min : {heapq.nsmallest(n, x_all_diff)}')
    print(f'avg y displacement: {sum(y_all_diff) / len(y_all_diff)}, max : {heapq.nlargest(n, y_all_diff)}, min : {heapq.nsmallest(n,y_all_diff)}')
    print(f'avg z displacement: {sum(z_all_diff) / len(z_all_diff)}, max : {heapq.nlargest(n, z_all_diff)}, min : {heapq.nsmallest(n,z_all_diff)}')
    return exp_edmo_movement
      
def merge_parameter_data(all_input, exp_edmo_movement): 
    global plot_data
    amp1, amp2, off1, off2, phb_diff, speeds = [], [], [], [], [], []
    for frame, speed in exp_edmo_movement.items():
        inputs = all_input[frame] # (freq, (amp0, amp1), (off0, off1), (phb0, phb1))
        amp1.append(inputs[1][0])
        amp2.append(inputs[1][1])
        off1.append(inputs[2][0])
        off2.append(inputs[2][1])
        phb_diff.append(abs(inputs[3][0]-inputs[3][1]))
        speeds.append(np.sqrt(speed[0]**2 + speed[1]**2))
    
    data = pd.DataFrame({
        'Offset_motor_1': off1,
        'Offset_motor_2': off2,
        'Amp_motor_1': amp1,
        'Amp_motor_2': amp2,
        'Phase_difference': phb_diff,
        'Speed' : speeds
    })
    
    plot_data = pd.concat([plot_data, data], ignore_index=True) if plot_data is not None else data  
    
def double_3D_plot(plot_data):
    off1 = plot_data['Offset_motor_1']
    off2 = plot_data['Offset_motor_2']
    amp1 = plot_data['Amp_motor_1']
    amp2 = plot_data['Amp_motor_2']
    phb_diff = plot_data['Phase_difference']
    speeds = plot_data['Speed']
    
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
        
        
def parallel_coord(plot_data):
    # Create a parallel coordinates plot
    fig = px.parallel_coordinates(
        plot_data,
        dimensions=['Offset_motor_1', 'Offset_motor_2', 'Amp_motor_1', 'Amp_motor_2', 'Phase_difference'],
        color='Speed',
        # color_continuous_scale=["white", "blue", "red", "black"],
        color_continuous_scale=[ "4290fb", "4fc0ff", "4fffd5", "7cff4f", "f6f05c", "ff8068", "ff4e6f", "c645b8", "6563de", "18158e", "000000"],  # Color scale for speed
        labels={'Speed': 'Speed (avg absolute speed)'}
    )

    fig.update_layout(
        width=900,
        height=600
    )

    fig.write_image('parallel_plot.png')


def plot_2D_poses(all_positions:dict[int, list], axes=None, title=None):
    show = True if axes is None else False
    if show:
        # 2d
        # fig, axes = plt.subplots(1, 1, figsize=(12, 10))  

        # 3d
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

    x = [pos[0] for frame, pos in all_positions.items()]
    y = [pos[1] for frame, pos in all_positions.items()]
    z = [frame for frame, pos in all_positions.items()]    

    axes.plot(x, y, z, marker='o', linestyle='-', color='b', label='Trajectory')
    # axes.plot(x, y, marker='o', linestyle='-', color='b', label='Trajectory')
    if title is not None:
        axes.set_title(title, fontsize=5)
    axes.set_xlim(0, 1.7)
    axes.set_ylim(0, 1.1) 
    axes.grid(True)

    # Show the plot and legend
    if show:
        axes.set_xlabel("X Position", fontsize=12)
        axes.set_ylabel("Y Position", fontsize=12)
        axes.legend(fontsize=12)
        plt.tight_layout()
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
                    
                    if j - start_end[-1] <= 2: # ignore motor transitions
                        del start_end[-1]
                        continue
                    elif len(start_j) > 0 and all(j < sj for sj in start_j): # ignore repeated motor data
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

       
def visualize_xyz(self, time=False):         
    z = self.t if time else self.z        

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(self.x, self.y, z, color='blue', label='position', linewidth=1)
    ax.scatter(self.x[0], self.y[0], z[0], color='red', s=10, label='starting point')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    zlabel = 'Frame number' if time else 'Z (m)'
    ax.set_zlabel(zlabel)
    ax.legend()

    plt.show()

    
def vizualize_3D(self):
    # Define the initial rectangle (relative to the origin)
    initial_rectangle = np.array([
        [-0.5, -0.5, 0],  # Bottom-left
        [17.5, -0.5, 0],   # Bottom-right
        [17.5, 6.5, 0],    # Top-right
        [-0.5, 6.5, 0],   # Top-left
        [-0.5, -0.5, 0],  # Close rectangle
    ])
    coord = []
    rot = []
    for i in range(1, self.nbFrames):
        if i in self.edmo_poses:
            coord.append(self.edmo_poses[i])
            rot.append(self.edmo_rots[i])

    def rotate_rectangle(rect, angles):
        """Apply 3D rotation to the rectangle vertices."""
        rx, ry, rz = angles
        
        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ])
        
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ])
        
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ])
        
        # Combined rotation
        rotation_matrix = rot_z @ rot_y @ rot_x
        return rect @ rotation_matrix.T

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(0, 10)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        
        # Get the center and rotation for the current frame
        center = coord[frame]
        rotation = rot[frame]
        
        # Rotate and translate the rectangle
        rect = rotate_rectangle(initial_rectangle, rotation) + center
        
        # Plot the rectangle
        ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], 'b-', linewidth=2)
        ax.scatter(rect[:, 0], rect[:, 1], rect[:, 2], c='r')  # Vertices

    # Animate the rectangle's motion
    ani = FuncAnimation(fig, update, frames=len(coord), interval=100)

    # Show the plot
    plt.show()
    
    
def interactive_plot(self, time=False):   
    z = self.t if time else self.z        
    
    fig = go.Figure()

    # Add line plot
    fig.add_trace(go.Scatter3d(
        x=self.x, y=self.y, z=z,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Line'
    ))

    # Add red dot at the first input coordinate
    fig.add_trace(go.Scatter3d(
        x=[self.x[0]], y=[self.y[0]], z=[z[0]],
        mode='markers',
        marker=dict(color='red', size=6),
        name='First Point (red dot)'
    ))

    # Update layout for better visualization
    zlabel = 'Frame number' if time else 'Z'
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title=zlabel
        ),
        title='Interactive 3D Plot',
        showlegend=True
    )

    fig.write_html("interactive_plot.html")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help='Path to exploreData/"EDMO" (default: exploreData/Snake)', default="exploreData/Snake")
    args = parser.parse_args()
     
    path = args.path
    path = 'exploreData/Snake/'
    path = 'cleanData/Snake/'
    plot_data = data_analysis(path)
    parallel_coord(plot_data)
