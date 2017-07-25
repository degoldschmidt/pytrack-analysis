import os, sys
import os.path, time

def get_created(filepath):
    """
    Returns created timestamp for given filepath

    args:
    * filepath [str] : full path of given file
    """
    try:
        return time.ctime(os.path.getctime(filepath))
    except OSError:
        return 0

def get_modified(filepath):
    """
    Returns last modified timestamp for given filepath

    args:
    * filepath [str] : full path of given file
    """
    try:
        return time.ctime(os.path.getmtime(filepath))
    except OSError:
        return 0

def check_base(_dict, _dir):
    """
    Checks and returns boolean, whether database is in given directory
    """
    for key in _dict.keys():
        print("[O.K.]" if key in os.listdir(_dir) else "[FAILED]")
    return (key in os.listdir(_dir))

def check_meta(_dict, _dir):
    """
    Checks and returns boolean, whether all meta-data files are in given directory
    """
    flag = 0
    for key,val in _dict.items():
        for expkeys in val.keys():
            if not expkeys in os.listdir(_dir):
                flag = 1
    print("[O.K.]" if flag == 0 else "[FAILED]")
    return (flag == 0)

def check_data(_dict, _dir):
    """
    Checks and returns boolean, whether all data files are in given directory
    """
    flag = 0
    for key,val in _dict.items():
        for expkeys, v2 in val.items():
            for session in v2:
                if not session in os.listdir(_dir):
                    flag = 1
    print("[O.K.]" if flag == 0 else "[FAILED]")
    return (flag == 0)

def check_time(_tstamps, _dir):
    """
    Checks and returns boolean, whether all files have valid timestamps
    """
    flag = 0
    for filename, times in _tstamps.items():
        mod = times["modified"]
        cre = times["created"]
        if not cre == get_created(os.path.join(_dir, filename)) and mod == get_modified(os.path.join(_dir, filename)):
            flag = 1
    print("[O.K.]" if flag == 0 else "[FAILED]")
    return (flag == 0)

def test(_dir, _dict, _tstamps, _VERBOSE=True):
    """
    Returns flags of several data integrity tests for given file structure from database file
    """
    if _VERBOSE:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(os.devnull, 'w')

    print("STARTING DATA INTEGRITY TEST...")
    print("-------------------------------")
    print("CHECKING DATABASE...\t\t\t", end='')
    basefl = check_base(_dict, _dir)
    print("CHECKING METAFILES...\t\t\t", end='')
    metafl = check_meta(_dict, _dir)
    print("CHECKING DATAFILES...\t\t\t", end='')
    datafl = check_data(_dict, _dir)
    print("CHECKING TIMESTAMPS...\t\t\t", end='')
    timefl = check_time(_tstamps, _dir)
    timefl = 0

    sys.stdout = sys.__stdout__
    return [basefl, metafl, datafl, timefl]
