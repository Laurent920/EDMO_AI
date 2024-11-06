import asyncio
from datetime import date
from EDMOManual import EDMOManual
import os
import platform
from pathlib import Path
from colorama import Fore, Style
import argparse


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
        

def experiment_replay(startFilePath: str=None):
    skip = True if startFilePath else False    
    
    cwd = os.getcwd()
    date_dir = cwd + '\\cleanData'
    for date_folder in os.listdir(date_dir):
        edmo_dir = date_dir + '\\' + date_folder 
        for edmo_folder in os.listdir(edmo_dir):
            time_dir = edmo_dir + '\\' + edmo_folder
            for time_folder in os.listdir(time_dir):
                data_path = time_dir + '\\' + time_folder
                if skip:
                    print(data_path)
                    print(startFilePath not in data_path)
                    print(f'skip: {skip}')
                    if startFilePath not in data_path:
                        continue
                    else:
                        skip = False
                    
                cont = True
                while cont:
                    human_in = input(Fore.BLUE + f"""To play data from {Path(*data_path.split('\\')[-3:])} press y\nTo skip this folder press s : """)
                    match human_in:
                        case 'y':        
                            print(Style.RESET_ALL + f"Running {Path(*data_path.split('\\')[-3:])}...")
                            server = EDMOManual(data_path)
                            print("running")
                            asyncio.run(server.run())
                            print("Finished running")
                        case 's':
                            cont = False
                        case 'stop':
                            print(Style.RESET_ALL) 
                            return
                        case _:
                            pass


def main(startFilePath: str=None):
    startFilePath.replace('/', '\\')
    experiment_replay(startFilePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help='Path to the data file you want to start from', default=None)
    args = parser.parse_args()
    main(args.path)
