import os
from datetime import datetime
import re


# Clean the data by removing the packets that were lost in at least one of the data logs
# and remove the possible duplicates

def readLog(location):
    Regex = '(?P<hours>\d*):(?P<minutes>\d*):(?P<seconds>\d*)(.(?P<decimals>\d*))*'
    
    motorData = []
    for filename in os.listdir(location):
        clean = False
        file = 'IMU.log'
        for i in range(4):  # Clean motor and IMU data
            motorFile = f'Motor{str(i)}.log'
            if filename == motorFile:
                file = motorFile
                clean = True

        if filename == 'IMU.log':
            clean = True

        if clean:
            f = open(os.path.join(location, file), "r").read()
            if not re.match(Regex, f):
                print("No match found")
            
            logs = f.split('\n')[:-1]
            cleanedLogs = []
            for j in range(len(logs)):
                l = logs[j]
                if not re.match(Regex, l):
                    print(f"Log file format wrong: {l}")
                    return
                
                cleanedLogs.append(l.split(' '))
                cleanedLogs[j][0] = cleanedLogs[j][0][:-5]
                if len(cleanedLogs[j][0]) != 10:
                    print(cleanedLogs[j][0])
                    print('Format problem')
                    
            motorData.append(cleanedLogs)
            print(len(cleanedLogs))
    return motorData


def getLogDuplicates(motorData):
    for j, log in enumerate(motorData):  # Search for duplicates
        for i in range(len(log) - 1):
            if log[i][0] == log[i + 1][0]:
                print(f'log number :{j}')
                print(i)


def cleanLog(motorData):
    shortestLog = min(len(log) for log in motorData)

    timeLog = []
    for i, logs in enumerate(motorData): # Get timestamps only
        timeLog.append({row[0] for row in logs})
        if len(logs) != len(timeLog[i]):
            print(f'{i}: {len(logs) - len(timeLog[i])} duplicates')

    diff = []
    for i, tLog in enumerate(timeLog): # Get timestamps that are not exactly the same
        for j, log in enumerate(timeLog):
            if j < i:
                continue
            if i != j:
                d = tLog ^ log
                diff.append(d)
                # print(f'{i}: {j}: {len(d)}')
                # print(sorted(d))
    print(f'Number of packets not well synced :{sum(len(nb) for nb in diff)}')

    timesToRemove = set()
    precision = 0.3
    for sets in diff:
        sets = sorted(sets)
        today = datetime.today()
        times = [datetime.combine(today, datetime.strptime(t, "%H:%M:%S.%f").time()) for t in sets]

        for i in range(len(sets)):
            curT = times[i]
            nextT = times[i+1] if i < len(sets)-1 else None
            prevT = times[i-1] if i > 0 else None
                
            if nextT and prevT:
                if ((nextT - curT).total_seconds() < precision or
                (curT - prevT).total_seconds() < precision):
                    continue
            elif prevT:
                if (curT - prevT).total_seconds() < precision:
                    continue
            elif nextT:
                if (nextT - curT).total_seconds() < precision:
                    continue
            timesToRemove.add(sets[i])
            
    print(f'Packets time to drop: {sorted(timesToRemove)}')
    print(f'Number = {len(timesToRemove)}')
    
    
    


if __name__ == "__main__":
    currentDir = os.getcwd()  
    readMulti = True
    readOneFile = not readMulti
            
    if readMulti:
        # currentDir += '/Data/2024.09.10/'
        # currentDir += '/SessionLogsHuge/2024.09.10/'
        currentDir += '/SessionLogs/2024.09.13/'
        # currentDir += '/2024.09.07/2024.09.07/'
            
        for folder in os.listdir(currentDir): # Read folders of folders
            print(folder)
            newDir = currentDir + folder + '/'
            for filename in os.listdir(newDir):
                readFile = True
                if False: #folder != 'Ramirez': # Restrict read to one folder
                    readFile = False 
                    
                if readFile:
                    location = newDir + filename
                    motorData = readLog(location)
                    cleanLog(motorData)
                    # getLogDuplicates(motorData)
                    print('__________________')
                
    if readOneFile:
        location = currentDir + '/SessionLogs/2024.09.13/Bloom/14.45.53/'
        motorData = readLog(location)
        cleanLog(motorData)

