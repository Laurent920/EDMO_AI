# Regular EDMO 
Run the `main.py` file as usual


# EDMO AI 
Some of the files might require the usage of a GoPro camera, if not done yet please familiarize yourself with the README file in the GoPro folder.

## EDMO Manual
This file contains the main logic built upon EDMOBackend. Running this file allows you to control the EDMO through command line either by setting some specific values or by replaying a file. Run using:

`python EDMOManual.py`

## Experiments
This file contains the code for 3 usages and it contains the base code for controlling the GoPro and the EDMO so that the EDMO's movement is recorded and stored in the replayed/explored folder alongside the `Input_ManualX.log` files for further analysis.

1. Generate the parameter sets for systematic search of the parameter space specifying the number of legs of the EDMO, storing all parameters in the `exploreData` folder: 
    `python experiments.py --generate 2`

2. Replay: Goes through all the files in SessionLogs to replay the files of the corresponding EDMO, following the instructions in the terminal to whether play the file or skip it:
    `python experiments.py --replay Snake2` 

3. Explore: Explore the parameters file in `exploreData` created by running the code point 1:
    `python experiments.py --explore` 

The parameter `-p` can be used to specify the starting file at which we want to start replay/explore. 

## EDMO learning
This file contains the implementation of Powell's method. This file contains 2 very important functions:

1. `get_EDMO_speed` which computes the speed of the EDMO (Use this function as example on how to record with the GoPro, analyse the video and get the output speed).

2. `golden` which performs the golden search and line minimization required for Powell's method

Before running this file you should specify the number of legs of the edmo and the GoPro's ID in the main function then run:

`python EDMOLearning.py`

You can also set the experiment duration in the begining of the file.

## Data Cleaner
Cleans the files in the `SessionLogs` folder by removing duplicates and unsynced data so that the files `MotorX.log`, `Input_PlayerX.log`, `Input_ManualX.log` and `IMU.log` contain synced data to the precision specified when running `DataCleaner`.


## Data Analysis
