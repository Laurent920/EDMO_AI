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


def data_analysis(dir):
    for folder in os.listdir(dir):
        filepath = f'{dir}/{folder}'
        files = os.listdir(filepath)
        if 'marker_pose.log' not in files:
            aruco_pose = pose_estimation.Aruco_pose(filepath)
            aruco_pose.pose_estimation()
        
        pose_d = pose_data.Pose_data(filepath)
        pose_d.get_pose()
        
        for file in files:
            pass
            # retrieve experiment values from experiments.py
            # for each experiment get the time stamp where the values are reached
            # compute the frame number corresponding to when the values are reached and retrieve the frames from pose_d
            # parse the frame x, y, z and filter out the manual movement
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help='Path to the data file you want to start from', default=None)
    parser.add_argument("--replay", action="store_const", help="List of the edmo files you allow to replay (default: Kumoko Lamarr)", const="Kumoko Lamarr")
    parser.add_argument("--generate", action="store_const", help="Number of legs of the EDMO for which you want to generate the parameters (default: 2)", const=2)
    parser.add_argument("--explore", action="store_const", help="Type of EDMO you want to explore (default: Snake)", const="Snake")
    args = parser.parse_args()

    
    nb_legs =  args.generate 

