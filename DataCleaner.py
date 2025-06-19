import os
import re

import numpy as np
import matplotlib.pyplot as plt
from Utilities.Helpers import toTime
import time
start_time = time.time()

printInfo = False
printDebugInfo = False
# Clean the data by removing the packets that were lost in at least one of the data logs
# and remove the possible duplicates

precision = 0.08 # The margin in seconds from which we consider a packet to be out of sync
def readLog(location):
    """ 
    Read the log files of IMU and Motors
    and return the data in a list of lists.
    Each list contains the data of one log file.
    """
    Regex = r"(\d+:\d{2}:\d{2})(?!\.\d{6})"

    motorData = []
    for filename in os.listdir(location):
        clean = False
        pattern = r"^(IMU|Motor[0-9]*).log$"
        file = ''
        if re.match(pattern, filename):
            file = filename
            clean = True

        if clean:
            f = open(os.path.join(location, file), "r").read()

            logs = f.split('\n')[:-1]
            cleanedLogs = []
            for j in range(len(logs)):
                l = logs[j]
                if re.match(Regex, l):
                    if printInfo: print(f"Log file format wrong: {l}")
                    l = re.sub(Regex, r"\1.000000", l)
                    if printInfo: print(l)

                cleanedLogs.append(l.split(' '))
                cleanedLogs[j][0] = cleanedLogs[j][0][:-1]
            motorData.append(cleanedLogs)
            if printInfo: print(f'Original size: {len(cleanedLogs)}')
    return motorData


def removeLogDuplicates(motorData):
    for j, log in enumerate(motorData):  # Search for duplicates
        i = 0
        length = len(log)
        originalSize = length
        while i < length - 1:
            if log[i][0] == log[i + 1][0]:
                # print(f'log number {j} has duplicates:')
                # print(f'{log[i][0]}  {log[i + 1][0]}')
                del motorData[j][i+1]
                length -= 1
                continue
            i += 1
        newSize = len(log)
        if newSize < originalSize:
            print(f'nb logs duplicates removed: {originalSize - newSize}')


def cleanLog(motorData):
    if len(motorData) <= 0:
        return
    shortestLog = min(len(log) for log in motorData)

    timeLog = []
    for i, logs in enumerate(motorData):  # Get timestamps only
        timeLog.append({row[0] for row in logs})
        if len(logs) != len(timeLog[i]):
            print(f'{i}: {len(logs) - len(timeLog[i])} duplicates')

    diff = []
    for i, tLog in enumerate(timeLog):  # Get timestamps that are not exactly the same
        for j, log in enumerate(timeLog):
            if j < i:
                continue
            if i != j:
                d = tLog ^ log
                diff.append(d)
                if printDebugInfo: print(f'{i}: {j}: {len(d)}')
                if printDebugInfo: print(sorted(d))
    if printInfo: print(f'Number of packets not well synced :{sum(len(nb) for nb in diff)}')

    timesToRemove = set()
    for sets in diff:  # Record timestamps that have a bigger difference than the precision
        sets = sorted(sets)
        times = [toTime(t) for t in sets]
        i = 0
        while i < len(sets):
            curT = times[i]
            if i == len(sets) - 1:
                timesToRemove.add(sets[i])
                break
            nextT = times[i + 1]

            if (nextT - curT).total_seconds() < precision:
                i += 2
            else:
                timesToRemove.add(sets[i])
                i += 1
            
    if printInfo: print(f'Packets time to drop: {sorted(timesToRemove)}') 
    if printInfo: print(f'Number of packets to remove = {len(timesToRemove)}')
    return timesToRemove


def nearTimeToRemove(t, timesToRemove):
    times = sorted([toTime(time) for time in timesToRemove]) 
    curT = toTime(t)
    
    low, high = 0, len(times) - 1
    
    while low <= high:
        mid = (low + high) // 2
        diff = (curT - times[mid]).total_seconds()

        if abs(diff) < precision:
            print(f'time we check for:{curT}, time to close in times to remove: {times[mid]}')
            return True

        if diff > 0:  
            low = mid + 1
        else:
            high = mid - 1
    return False


def writeToLog(motorData, timesToRemove, path):
    slashPos = 0
    for i, s in enumerate(path):
        if s == r'/':
            slashPos += 1
        if slashPos == 2:
            slashPos = i
            break
    cleanPath = path[slashPos:]
    cleanPath = './cleanData' + cleanPath
    print(f'clean path: {cleanPath}')
    if not os.path.exists(cleanPath):
        os.makedirs(cleanPath)
        print(f'Creating new folder {cleanPath}')

    pattern = r"^(IMU|Motor[0-9]*).log$"
    index = 1
    deletedRows = np.empty(5, dtype=object)
    for i in range(5):
        deletedRows[i] = []
    for filename in os.listdir(path):
        with open(f"{cleanPath}/{filename}", "w") as newFile:
            count = 0
            if not re.match(pattern, filename):
                # Video
                if os.path.splitext(filename)[1].lower() == '.mp4':
                    # ignore video
                    continue
                    # Copy video
                    video_path = f'{cleanPath}/{filename}'
                    if not os.path.exists(video_path):
                        os.replace(f'{path}/{filename}', video_path)
                    continue
                if os.path.splitext(filename)[1].lower() != '.txt':
                    # Skip all non text files
                    continue
                # Copy files that don't need to be cleaned
                with open(f'{path}/{filename}', 'r') as src:
                    for line in src:
                        newFile.write(line)
            elif filename == 'IMU.log':
                # Write clean IMU log 
                f = open(os.path.join(path, filename), "r").read()
                logs = f.split('\n')[:-1]
                for i, row in enumerate(motorData[0]):
                    if row[0] not in timesToRemove and not nearTimeToRemove(row[0], timesToRemove):
                        size = len(logs[i].split(' ')[0])
                        log = f'{row[0]}{logs[i][size:]}\n'
                        newFile.write(log)
                    else:
                        deletedRows[0].append(row[0])
                        count += 1
                print(f'Row deleted for IMU log: {count}, final size: {len(motorData[0])-count}')
            else:
                # Write clean motor logs
                newFile.write('Time, Frequency, Amplitude, Offset, PhaseShift, Phase\n')
                for row in motorData[index]:
                    if row[0] not in timesToRemove and not nearTimeToRemove(row[0], timesToRemove):
                        newFile.write(f'{row[0]}, {row[2][:-1]}, {row[4][:-1]}, {row[6][:-1]}, {row[9][:-1]}, {row[11]}\n')
                    else:
                        deletedRows[index].append(row[0])
                        count += 1
                print(f'Row deleted for index {index-1}: {count}, final size: {len(motorData[index])-count}')
                index += 1
    # Plot removed timestamps
    # timesToRemove = (list(sorted(timesToRemove)))
    # for i, r in enumerate(deletedRows):
    #     deleted_indices = [timesToRemove.index(time) for time in r]
    #     plt.eventplot(np.array(deleted_indices)[:,np.newaxis], lineoffsets=[i+1] * len(deleted_indices), orientation="vertical", colors='r', linelengths=0.1)
    # plt.xticks([1,2,3,4,5], ['IMU', 'Motor0', 'Motor1', 'Motor2', 'Motor3'])
    # plt.yticks(np.arange(0, len(timesToRemove), 1), np.array(timesToRemove))
    # plt.title(path)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
def clean(path, readMulti, folderName=None):
    if readMulti:
        readFile = True

        for folder in os.listdir(path):  # Read folders of folders
            if folderName is not None and folder != folderName: # Restrict to a single folder eg. "2025.01.07"
                continue 
            print(folder)
            for edmo_folder in os.listdir(f'{path}/{folder}'):
                print(edmo_folder)
                newPath = f'{path}/{folder}/{edmo_folder}/'
                for filename in os.listdir(newPath):
                    if readFile:
                        location = newPath + filename
                        print(f'location: {location}')
                        motorData = readLog(location)
                        timesToRemove = cleanLog(motorData)
                        removeLogDuplicates(motorData)
                        writeToLog(motorData, timesToRemove, location)
                        print('__________________')
                    else:
                        print('skipping')
    else:
        location = path
        motorData = readLog(location)
        removeLogDuplicates(motorData)
        timesToRemove = cleanLog(motorData)
        writeToLog(motorData, timesToRemove, location)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    readMulti = True # Clean all the files in the specified folder or just the one specified in path

    if readMulti:
        path = './SessionLogs/'
        # clean(path, readMulti, '2025.01.07') # Restrict to a single date folder eg. "2025.01.07"
        clean(path, readMulti)
    else:
        path = './SessionLogs/2024.11.18/Athena/15.30.55'
        path = './exploreData/Snake/2700-2879' 
        clean(path, readMulti=readMulti)

