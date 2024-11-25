import asyncio
from csv import Error
from datetime import time, timedelta

from requests import session

from EDMOManual import EDMOManual
import os
import platform
from pathlib import Path
from colorama import Fore, Style, init
import argparse
import math
from GoPro.wifi.WifiCommunication import WifiCommunication

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime
        

async def experiment_setup():        
    wifi_com = WifiCommunication("GoPro 6665", Path("GoPro/GoPro 6665"))
    await wifi_com.initialize()
    server = EDMOManual()
    asyncio.get_event_loop().create_task(server.run())
    await asyncio.sleep(1)
    await server.close()
    await asyncio.sleep(5)
    return server

async def experiment_replay(startFilePath: str=None, edmo_list:list[str]=[]):
    skip = True if startFilePath else False    
    
    server = await experiment_setup()    

    cwd = os.getcwd()
    date_dir = cwd + '\\cleanData'
    for date_folder in os.listdir(date_dir):
        edmo_dir = date_dir + '\\' + date_folder 
        for edmo_folder in os.listdir(edmo_dir):
            time_dir = edmo_dir + '\\' + edmo_folder
            if edmo_folder not in edmo_list:
                continue
            for time_folder in os.listdir(time_dir):
                data_path = time_dir + '\\' + time_folder
                if skip:
                    if startFilePath not in data_path:
                        print(f'Skipping: {data_path}')
                        continue
                    else:
                        skip = False
                    
                if await wait_for_input(server, data_path):
                    return
                    
                    
async def experiment_explore(startFilePath:str =None, edmo_type:str =None):
    skip = True if startFilePath else False    
    
    server = await experiment_setup()

    cwd = os.getcwd()
    edmo_dir = cwd + '\\exploreData'
    for edmo in os.listdir(edmo_dir):
        experiment_dir = edmo_dir + '\\' + edmo
        if edmo != edmo_type:
            continue
        for experiment_folder in os.listdir(experiment_dir):
            data_path = experiment_dir + '\\' + experiment_folder
            if skip:
                if startFilePath not in data_path:
                    print(f'Skipping: {experiment_folder}')
                    continue
                else:
                    skip = False
            if await wait_for_input(server, data_path, True):
                return
            # else:
            #     while True:
            #         rerun_input = input("Type rerun or quit: ")
            #         match rerun_input:
            #             case 'rerun':
            #                 await wait_for_input(server, data_path, True)
            #             case 'quit':
            #                 return
            #             case _:
                            # pass
                        


async def wait_for_input(server: EDMOManual, data_path:str, explore:bool=False):
    try:
        cont = True
        while cont:
            human_in = input(Fore.BLUE + f"""To play data from {Path(*data_path.split('\\')[-3:])} press y\nTo skip this folder press s : """)
            match human_in:
                case 'y':        
                    print(Style.RESET_ALL + f"Running {Path(*data_path.split('\\')[-3:])}...")
                    print("press ctrl c to stop the run if you have to move it")
                    await server.initialize(data_path)
                    await server.run(explore)
                    print("Finished running")
                    cont = False
                case 's':
                    cont = False
                case 'quit':
                    print(Style.RESET_ALL) 
                    return True
                case _:
                    pass
        return False
    except Exception as e:
        print(e)
        print("run stopping...")
        return False

def generate_exploration_files(nbPlayers: int = 2):
    explorePath = './exploreData'
    if not os.path.exists(explorePath):
        os.makedirs(explorePath)
        print(f'Creating new folder {explorePath}')

    edmo_type:str
    match nbPlayers:
        case 2:
            edmo_type = "Snake"
        case 4:
            edmo_type = "Spider" 
        case _:
            print(f"EDMOs with {nbPlayers} has yet to be defined")
            return
    explorePath += f'/{edmo_type}'
    if not os.path.exists(explorePath):
        os.makedirs(explorePath)
        print(f'Creating new folder {explorePath}')
        
    
    all_input:list = get_all_input(nbPlayers)
    if not all_input:
        return
        
    players_filename = [] 
    for i in range(nbPlayers):
        players_filename.append(f"Input_Manual{i}.log")

        
    init_time = timedelta(microseconds=1)
    cur_time = init_time
    episode_length = timedelta(seconds=10) # How long do we want to run each parameter change
    session_length = 180 # How many parameter change do we run in one session
    end_time = session_length*episode_length + timedelta(microseconds=1)
    filepath:str
    skip = False
    for i, instructions in enumerate(all_input):
        if not (i % session_length):
            cur_time = init_time
            filepath = explorePath + f'/{i}-{i+session_length-1}'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                print(f'Creating new folder {filepath}')
                skip = False
            else:
                print(f'{filepath} already exists, skipping...')
                skip = True
        if skip:
            continue
                    
        for j, filename in enumerate(players_filename):
            file = f'{filepath}/{filename}'
            
            with open(file, 'a+') as log:
                log.writelines(f"{cur_time}: freq {instructions[0]}\n")          
                log.writelines(f"{cur_time}: amp {instructions[1][j]}\n")
                log.writelines(f"{cur_time}: off {instructions[2][j]}\n")
                log.writelines(f"{cur_time}: phb {instructions[3][j]*(math.pi/180)}\n")
                
                if end_time - cur_time < episode_length + timedelta(seconds=1):
                    log.writelines(f"{end_time}: freq 0\n")          
                    log.writelines(f"{end_time}: amp 0\n")
                    log.writelines(f"{end_time}: off 90\n")
                    log.writelines(f"{end_time}: phb 0\n")
        cur_time += episode_length


def get_all_input(nbPlayers: int = 2):
    freq, amp, off, phb = [], [], [], []
    freq.append(1)
    # for frequency in range(5, 20, 5):  
    #     freq.append(frequency/10.0)
    for amplitude in range(20, 90, 20):
        amp.append(amplitude)
    for offset in range(0, 180, 30):
        off.append(offset)
    for phase in range(0, 360, 40):
        phb.append(phase)
    
    match nbPlayers:
        case 2:
            return players2(amp, freq, off, phb)
        case 4:
            print("To be implemented")
            return None
        case _:
            print(f"Parameters for {nbPlayers} players has to be implemented")
            return None
    


def players2(amp, freq, off, phb):
    all_amp = get_all_amp(amp)
    all_off = get_all_off(off)
    all_phase = get_all_phase(phb)
    all_input = []
    for f in freq:
        for amps in all_amp:
            for offs in all_off:
                for phases in all_phase:
                    all_input.append((f, amps, offs, phases))
    return all_input
                                                      
def get_all_amp(amp):
    all_amp = []
    for amp1 in amp:
        for amp2 in amp:
            # if amp2 < amp1:
            #     continue
            all_amp.append((amp1, amp2))
    return all_amp

def get_all_off(off):
    all_off = []
    for off1 in off:
        for off2 in off:
            if off2 < off1:
                continue
            all_off.append((off1, off2))
    return all_off

def get_all_phase(phb):
    all_phb = []
    phase_diff = []
    for phase1 in phb:
        for phase2 in phb:
            phb_diff = abs(phase1 - phase2)
            if phb_diff in phase_diff:
                continue
            else:
                phase_diff.append(phb_diff)
                all_phb.append((phase1, phase2))
    return all_phb

def main_replay(startFilePath: str=None, edmos:str=None):
    if startFilePath:
        startFilePath.replace('/', '\\')
    
    edmo_list = edmos.split(" ")    
        
    asyncio.run(experiment_replay(startFilePath, edmo_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help='Path to the data file you want to start from', default=None)
    parser.add_argument("--replay", action="store_const", help="List of the edmo files you allow to replay (default: Kumoko Lamarr)", const="Kumoko Lamarr")
    parser.add_argument("--generate", action="store_const", help="Number of legs of the EDMO for which you want to generate the parameters (default: 2)", const=2)
    parser.add_argument("--explore", action="store_const", help="Type of EDMO you want to explore (default: Snake)", const="Snake")
    args = parser.parse_args()

    
    nb_legs =  args.generate 
    edmo_files = args.replay
    edmo_type= args.explore
    
    # nb_legs =  None
    # edmo_files = None
    # edmo_type= "Snake"
    
    if nb_legs:
        generate_exploration_files(nb_legs)
    elif edmo_files:
        main_replay(args.path, edmo_files)
    elif edmo_type:
        asyncio.run(experiment_explore(args.path, edmo_type))
    else:
        print("Decide what you want to do by setting either flags -replay, -generate or -explore (only one flag can be set at a time)")
        exit(0) 