# visualize_xyz(pose_datas.x, pose_datas.y, pose_datas.z, pose_datas.t, False)
from EDMOLearning import parameters_to_vector, parameters_to_param_list, param_ranges, generate_unique_key
from data_analysis import compute_speed, merge_parameter_data
from EDMOManual import EDMOManual
from ArUCo_Markers_Pose import pose_data, pose_estimation
from GoPro.wifi.WifiCommunication import WifiCommunication

import matplotlib.pyplot as plt
import asyncio
from pathlib import Path
import json
import numpy as np
import os

debug = False
param_dict = {}
param_dict_path = f"./Utilities/parameters_dictionnary_{2}.log"
with open(param_dict_path, 'r') as f:
    param_dict = json.load(f)
gopro = ["GoPro 4448"]    
        
        
def visualize_xyz(x, y, z, t, time=False):         
    z_axis = t if time else z        

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z_axis, color='blue', label='position', linewidth=1)
    ax.scatter(x[0], y[0], z_axis[0], color='red', s=10, label='starting point')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    zlabel = 'Frame number' if time else 'Z (m)'
    ax.set_zlabel(zlabel)
    ax.legend()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title("X-axis Positions")
    plt.xlabel("frame number")
    plt.ylabel("X (m)")
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.title("Y-axis Positions")
    plt.xlabel("frame number")
    plt.ylabel("Y (m)")
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, z)
    plt.title("Z-axis Positions")
    plt.xlabel("frame number")
    plt.ylabel("Z (m)")
    plt.grid()
    plt.show()
    
    
def visualize_xy(x, y, speed):         
    plt.figure(figsize=(12, 8))
    
    plt.plot(x, y)
    plt.scatter([x[0], x[-1]], [y[0], y[-1]])
    x_diff, y_diff = x[-1]-x[0],y[-1]-y[0]
    plt.legend(['({:.2f}, {:.2f}) => {}'.format(x_diff, y_diff, np.sqrt((x_diff**2+y_diff**2)))] )
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    
    plt.xlim([0, 1.7])
    plt.ylim([0, 1.1])
    
    plt.title(f"Edmo's movement (speed: {speed} m/s) ")
    plt.grid()

    plt.show()


async def get_EDMO_speed(server, parameters, nb_legs):
    vector = parameters_to_vector(parameters)
    if 0 in vector[:nb_legs] or np.nan in vector:
        print(f"{vector} in get_EDMO_speed")
        return 0.0
    if debug:
        print(f"parameters in get_EDMO_speed: {parameters}")
    
    # Look up the parameters dictionnary if this set already exists
    key = generate_unique_key(parameters_to_vector(parameters), param_ranges, nb_legs)
    if key in param_dict.keys():
        speed, confidence = param_dict[key]
        if confidence >= 200:
            print(f"speed: {speed}")
            return speed
        
    # Send input to the server
    await server.runInputDict(parameters)
    
    server.GoProOn()
    await asyncio.sleep(15) # How long one set of parameter runs
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
    print(f"speed: {speed}")
    print(data)
    visualize_xy(pose_d.x, pose_d.y, speed)        

    # Store the parameters and speed in a hash table
    param_dict[key] = (speed, valid_frames)
    return speed


async def main():
    # wifi_com = WifiCommunication(gopro[0], Path(f"GoPro/{gopro[0]}"))
    # await wifi_com.initialize()

    server = EDMOManual(gopro_list=["GoPro 4448"])
    asyncio.get_event_loop().create_task(server.run())
    server.GoProOff()

    parameters = {0:{'freq':1.0, 'amp':90.0, 'off':0.0,'phb':64.0}, 1:{'freq':1.0, 'amp':90.0, 'off':180.0,'phb':12.0}} # 0.04225769274242996
    # parameters = {0:{'freq':1.0, 'amp':90.0, 'off':98.0,'phb':359.0}, 1:{'freq':1.0, 'amp':58.0, 'off':100.0,'phb':62.0}} # 0.039866116498910815
    # parameters = {0:{'freq':1.0, 'amp':90.0, 'off':73.0,'phb':113.0}, 1:{'freq':1.0, 'amp':90.0, 'off':92.0,'phb':174.0}} # 0.03527106322744506

    await get_EDMO_speed(server, parameters, 2)


if __name__ == "__main__":
    asyncio.run(main())