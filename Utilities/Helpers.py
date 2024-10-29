from typing import MutableSequence
from datetime import datetime


def removeIfExist(list: MutableSequence, item):
    if item in list:
        list.remove(item)


def appendIfNotExist(list: MutableSequence, item):
    if item in list:
        return

    list.append(item)

def toTime(t):
    today = datetime.today()
    try:
        return datetime.combine(today, datetime.strptime(t, "%H:%M:%S.%f").time())
    except ValueError as e: 
        print(f'error: {e} for time {t}')