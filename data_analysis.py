import os
import re
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


def data_analysis(dir, nbPlayers: int = 2):
    all_input = get_all_input(nbPlayers)
    
    for folder in os.listdir(dir):
        filepath = f'{dir}/{folder}'
        files = os.listdir(filepath)
        if 'marker_pose.log' not in files:
            aruco_pose = pose_estimation.Aruco_pose(filepath)
            aruco_pose.pose_estimation()
        
        pose_d = pose_data.Pose_data(filepath)
        pose_d.get_pose()
        
        exp_start, exp_end = folder.split('-', 2)
        for file in files:
            pass
            # retrieve experiment values from experiments.py
            # for each experiment get the time stamp where the values are reached
            # compute the frame number corresponding to when the values are reached and retrieve the frames from pose_d
            # parse the frame x, y, z and filter out the manual movement
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help='Path to exploreData/"EDMO" (default: exploreData/Snake)', default="exploreData/Snake")
    args = parser.parse_args()
    
    nb_legs =  args.generate 

    data_analysis(args.path)
