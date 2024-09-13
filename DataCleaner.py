import os
from datetime import datetime


# Clean the data by removing the packets that were lost in at least one of the data logs
# and remove the possible duplicates

def readLog(location):
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
            logs = f.split('\n')[:-1]
            cleanedLogs = []
            for j, log in enumerate(logs):
                cleanedLogs.append(logs[j].split(' '))
                cleanedLogs[j][0] = cleanedLogs[j][0][:-5]
            motorData.append(cleanedLogs)
            print(len(cleanedLogs))
    return motorData


def getLogDuplicates(motorData):
    for log in motorData:  # Search for duplicates
        for i in range(len(log) - 1):
            if log[i][0] == log[i + 1][0]:
                print(i)


def cleanLog(motorData):
    shortestLog = min(len(log) for log in motorData)

    timeLog = []
    print(len(motorData))
    for i, logs in enumerate(motorData):
        timeLog.append({row[0] for row in logs})
        if len(logs) != len(timeLog[i]):
            print(f'{i}: {len(logs) - len(timeLog[i])} duplicates')

    diff = set()
    for i, tLog in enumerate(timeLog):
        for j, log in enumerate(timeLog):
            if i != j:
                d = tLog ^ log
                diff.update(d)
                print(f'{i}: {j}: {len(d)}')
    diff = sorted(diff)
    print(diff)
    print(len(diff))

    timesToRemove = set()
    for i in range(1, len(diff) - 1):
        today = datetime.today()
        prevT = datetime.combine(today, datetime.strptime(diff[i - 1], "%H:%M:%S.%f").time())
        curT = datetime.combine(today, datetime.strptime(diff[i], "%H:%M:%S.%f").time())
        nextT = datetime.combine(today, datetime.strptime(diff[i + 1], "%H:%M:%S.%f").time())

        if ((nextT - curT).total_seconds() < 0.05 or
            (curT - prevT).total_seconds() < 0.05):
            continue
        else:
            timesToRemove.add(diff[i])
    print(sorted(timesToRemove))


if __name__ == "__main__":
    currentDir = os.getcwd()
    location = currentDir + '/Data/2024.09.10/Bloom/11.29.13/'
    motorData = readLog(location)
    cleanLog(motorData)

# time_obj = datetime.strptime(motorData[0][0][0], "%H:%M:%S.%f").time()
