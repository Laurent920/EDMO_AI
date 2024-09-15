import os
from datetime import datetime
import re


# Clean the data by removing the packets that were lost in at least one of the data logs
# and remove the possible duplicates

def readLog(location):
    # Regex = r'(?P<hours>\d*):(?P<minutes>\d*):(?P<seconds>\d*)(\.(?P<decimals>\d*))*'
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
                    print(f"Log file format wrong: {l}")
                    l = re.sub(Regex, r"\1.000000", l)
                    print(l)

                cleanedLogs.append(l.split(' '))
                cleanedLogs[j][0] = cleanedLogs[j][0][:-1]
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
                # print(f'{i}: {j}: {len(d)}')
                # print(sorted(d))
    print(f'Number of packets not well synced :{sum(len(nb) for nb in diff)}')

    timesToRemove = set()
    precision = 0.1
    count = 0
    for sets in diff:  # Record timestamps that have a bigger difference than the precision
        sets = sorted(sets)
        today = datetime.today()
        times = [datetime.combine(today, datetime.strptime(t, "%H:%M:%S.%f").time()) for t in sets]
        i = 0
        while i < len(sets):
            curT = times[i]
            if i == len(sets) - 1:
                timesToRemove.add(sets[i])
                break
            nextT = times[i + 1]

            a = (nextT - curT).total_seconds() 
            if (nextT - curT).total_seconds() < precision:
                i += 2
            else:
                timesToRemove.add(sets[i])
                i += 1
    # print(f'Packets time to drop: {sorted(timesToRemove)}')
    print(f'Number of packets to remove = {len(timesToRemove)}')
    return timesToRemove


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
    if not os.path.exists(cleanPath):
        os.makedirs(cleanPath)
        print(f'Creating new folder {cleanPath}')
    # print(cleanPath)

    pattern = r"^(IMU|Motor[0-9]*).log$"
    index = 1
    for filename in os.listdir(location):
        with open(f"{cleanPath}/{filename}", "w") as newFile:
            count = 0
            if not re.match(pattern, filename):
                with open(f'{path}/{filename}', 'r') as src:
                    for line in src:
                        newFile.write(line)
            elif filename == 'IMU.log':
                f = open(os.path.join(location, filename), "r").read()
                logs = f.split('\n')[:-1]
                for i, row in enumerate(motorData[0]):
                    if row[0] not in timesToRemove:
                        size = len(logs[i].split(' ')[0])
                        log = f'{row[0]}{logs[i][size:]}\n'
                        newFile.write(log)
                    else:
                        count += 1
                print(f'Row deleted for IMU log: {count}, final size: {len(motorData[0])-count}')
            else:
                newFile.write('Time, Frequency, Amplitude, Offset, PhaseShift, Phase\n')
                for row in motorData[index]:
                    if row[0] not in timesToRemove:
                        newFile.write(f'{row[0]}, {row[2][:-1]}, {row[4][:-1]}, {row[6][:-1]}, {row[9][:-1]}, {row[11]}\n')
                    else:
                        count += 1
                print(f'Row deleted for index {index-1}: {count}, final size: {len(motorData[index])-count}')
                index += 1


if __name__ == "__main__":
    readMulti = False
    readOneFile = not readMulti

    if readMulti:
        # path = './DataBloom/2024.09.13/'
        path = './SessionLogsHuge/2024.08.28/'
        # path = './SessionLogs/2024.09.13/'
        # path = './DataSmallEDMO/2024.09.07/'

        for folder in os.listdir(path):  # Read folders of folders
            print(folder)
            newPath = path + folder + '/'
            for filename in os.listdir(newPath):
                readFile = True
                if False:  #folder != 'Ramirez': # Restrict read to one folder
                    readFile = False

                if readFile:
                    print(filename)
                    location = newPath + filename
                    motorData = readLog(location)
                    timesToRemove = cleanLog(motorData)
                    # getLogDuplicates(motorData)
                    writeToLog(motorData, timesToRemove, location)
                    print('__________________')

    if readOneFile:
        path = './DataBloom/2024.09.13/Bloom/14.45.53'
        path = './SessionLogsHuge/2024.08.28/Bloom/15.17.50'
        location = path
        motorData = readLog(location)
        getLogDuplicates(motorData)
        timesToRemove = cleanLog(motorData)
        writeToLog(motorData, timesToRemove, path)

