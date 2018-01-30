import os, sys
from datetime import datetime as date
from functools import wraps
import tkinter as tk
from tkinter import messagebox, filedialog
from ._globals import *
from pytrack_analysis.cli import query_yn, flprint

"""
profile.py
AUTHOR: degoldschmidt
DATE: 17/07/2017

Contains functions for creating a project profile for analysis.
"""

###
# GLOBAL CONSTANTS (based on OS)
###
PROFILE, SYSNAME, OS = get_globals()
print(PROFILE, SYSNAME, OS)

def get_profile(_id, _user, script="", VERBOSE=True):
    """
    Returns profile as dictionary. If the given project name or user name is not in the profile, it will create new entries.

    Arguments:
    * _name: project id or '%all' (all projects)
    * _user: username
    * script: scriptname
    """
    if not VERBOSE:
        flprint("Setting up profile...")
    tk.Tk().withdraw()    # Tkinter window suppression

    # Profile object from file
    profile = Profile(PROFILE)

    # system registration
    profile.set_system(SYSNAME)

    # project registration
    profile.set_project(_id, script)

    # user registration
    #profile.set_project(_user)

    # submit profile
    with io.open(PROFILE, 'w+', encoding='utf8') as f:
        yaml.dump(profile.dict, f, default_flow_style=False, allow_unicode=True, canonical=False)

    return profile

class Profile(object):
    def __init__(self, _file):
        self.file = _file
        ### Read YAML profile file
        try:
            with open(_file, 'r') as stream:
                self.dict = yaml.load(stream)
            if self.dict is None:
                self.dict = {}
        except FileNotFoundError:
            self.dict = {}

    def __del__(self):
        with io.open(self.file, 'w+', encoding='utf8') as f:
            yaml.dump(self.dict, f, default_flow_style=False, allow_unicode=True, canonical=False)

    def __str__(self):
        outstr = ""
        for k,v in self.dict.items():
            outstr += str(k) + ':\t' + str(v) + '\n'
        return outstr

    def get_folders(self):
        system = self.dict["SYSTEMS"][self.activesys]
        project = self.dict["PROJECTS"][self.active]
        return {
                    "raw": os.path.join(system['base'], project['raw']),
                    "videos": os.path.join(system['base'], project['videos']),
                    "manual": os.path.join(system['base'], project['manual']),
                    "out": os.path.join(system['base'], project['out']),
                    "processed": os.path.join(system['base'], project['processed']),
            }

    def Nvids(self):
        system = self.dict["SYSTEMS"][self.activesys]
        project = self.dict["PROJECTS"][self.active]
        folder = os.path.join(system['base'], project['videos'])
        return 72##len([i for i in os.listdir(folder) if ".avi" in i])

    def set_project(self, _name, _script):
        system = self.dict["SYSTEMS"][self.activesys]
        if 'PROJECTS' not in self.dict.keys():
            self.dict["PROJECTS"] = {}
        if _name in self.dict['PROJECTS'].keys():
            print("Project \'{:}\' found.".format(_name))
            projects = self.dict["PROJECTS"]
            projects[_name]['last modified'] = date.now().strftime("%Y-%m-%d %H:%M:%S")
            if _script not in projects[_name]['scripts']:
                projects[_name]['scripts'].append(_script)
            base = os.path.join(self.experiments[_name], "data") #set_dir('experiment folder', forced=True)
            for each in ['raw', 'videos', 'manual', 'out', 'processed']:
                projects[_name][each] = os.path.join(base, each)
            video = os.path.join(system['base'], projects[_name]['videos'])
            if not os.path.isdir(video):
                projects[_name]['videos'] = "/Volumes/DATA_BACKUP/data/tracking/all_videos/"
        else:
            print("Project \'{:}\' does not seem to exist in the profile.".format(_name))
            projects = self.dict["PROJECTS"]
            base = os.path.join(self.experiments[_name], "data") #set_dir('experiment folder', forced=True)
            projects[_name] = {
                'base': os.path.dirname(base),
                'raw': os.path.join(base, 'raw'),
                'videos': os.path.join(base, 'videos'),
                'manual': os.path.join(base, 'manual'),
                'processed': os.path.join(base, 'processed'),
                'out': os.path.join(base, 'out'),
                'scripts': [_script],
                'created':  date.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last modified':  date.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if not os.path.isdir(projects[_name]['videos']):
                projects[_name]['videos'] = "/Volumes/DATA_BACKUP/data/tracking/all_videos/"
        self.active = _name

    def set_system(self, _name):
        if 'SYSTEMS' not in self.dict.keys():
            self.dict["SYSTEMS"] = {}
        if _name in self.dict['SYSTEMS'].keys():
            print("System \'{:}\' found.".format(_name))
            systems = self.dict["SYSTEMS"]
            systems[_name]['python'] = sys.version
        else:
            print("System \'{:}\' does not seem to exist in the profile.".format(_name))
            systems = self.dict["SYSTEMS"]
            systems[_name] = {'base': set_dir('system directory', forced=True), 'python': sys.version}
        self.experiments = read_exps(self.dict["SYSTEMS"][_name]['base'])
        if len(self.experiments) == 0:
            del self.dict["SYSTEMS"][_name]
            self.set_system(_name)
        self.activesys = _name

def read_exps(_dir):
    try:
        with open(os.path.join(_dir, 'list_experiments.txt'), 'r') as f:
            lns = f.readlines()
    except FileNotFoundError:
        print("list_experiments.txt not found in base folder {}.".format(_dir))
        lns = []
    outdict = {}
    for ln in lns:
        splitln = ln.split(" ")
        outdict[splitln[0]] = splitln[1][:-1]
    return outdict

def set_dir(_title, forced=False):
    """ Returns base directory chosen from TKinter filedialog GUI """
    if not forced:
        asksave = messagebox.askquestion(_title, "Are you sure you want to set a new path?", icon='warning')
        if asksave == "no":
            return None
    flprint("Set {}...".format(_title))
    base = filedialog.askdirectory(title=_title)
    print(base)
    return base

def get_db(profile):
    """ Returns active system's database file location """
    if 'database' in profile.dict['PROJECTS'][profile.active].keys():
        dbfile = os.path.join(profile.dict['SYSTEMS'][profile.activesys]['base'], profile.dict['PROJECTS'][profile.active]['database'])
        if os.path.exists(dbfile):
            print("Found database: {}".format(dbfile))
            return dbfile
    print("No database file found.")
    dbfile = filedialog.askopenfilename(title="Load database")
    profile.dict['PROJECTS'][profile.active]['database'] = dbfile
    return dbfile


def get_out(profile):
    """ Returns active system's output path """
    return profile[profile['active']]['systems'][NAME]['output']

def get_log(profile):
    """ Returns active system's logfile location """
    return profile[profile['active']]['systems'][NAME]['log']

def get_plot(profile):
    """ Returns active system's plot path """
    return profile[profile['active']]['systems'][NAME]['plot']



def set_database(forced=False):
    """ Returns database file location and video directory chosen from TKinter filedialog GUI """
    if not forced:
        asksave = messagebox.askquestion("Set database path", "Are you sure you want to set a new path for the database?", icon='warning')
        if asksave == "no":
            return None, None
    print("Set database...")
    dbfile = filedialog.askopenfilename(title="Load database")

    print("Set raw videos location...")
    viddir = filedialog.askdirectory(title="Load directory with raw video files")
    return dbfile, viddir

def set_output(forced=False):
    """ Returns output, log and plot path chosen from TKinter filedialog GUI """
    if not forced:
        asksave = messagebox.askquestion("Set output path", "Are you sure you want to set a new path for the output/logging?", icon='warning')
        if asksave == "no":
            return None, None, None
    print("Set output location...")
    outfolder = filedialog.askdirectory(title="Load directory for output")
    ### IF ANYTHING GIVEN
    if len(outfolder) > 0:
        out = outfolder
        log = os.path.join(outfolder,"main.log")
        plot = os.path.join(outfolder,"plots")
    else:
        out = os.path.join(USER_DATA_DIR, "output")
        log = os.path.join(out,"main.log")
        plot = os.path.join(out,"plots")
    ### CHECK WHETHER FOLDERS EXIST
    for each in [out, plot]:
        check_folder(each)
    ### RETURN
    return out, log, plot

def show_profile(profile):
    """ Command-line output of profile with colored formatting (active project is bold green) """
    ### Colors for terminal
    RED   = "\033[1;31m"
    CYAN  = "\033[1;36m"
    MAGENTA = "\033[1;35m"
    RESET = "\033[0;0m"
    print() # one empty line
    if profile is None:
        profile_dump = yaml.dump(profile, default_flow_style=False, allow_unicode=True)
        thisstr = profile_dump.split("\n")
        sys.stdout.write(RED)
        for lines in thisstr:
            if lines == "$PROJECTS:" or lines == "$USERS:":
                sys.stdout.write(RED)
            elif lines.startswith("-"):
                sys.stdout.write(CYAN)
            else:
                sys.stdout.write(RESET)
            print(lines)
        sys.stdout.write(RESET)
    else:
        current_proj = profile['active']
        current_sys = profile['activesys']
        profile_dump = yaml.dump(profile, default_flow_style=False, allow_unicode=True)
        thisstr = profile_dump.split("\n")
        sys.stdout.write(RED)
        for lines in thisstr:
            if lines == "$PROJECTS:" or lines == "$USERS:":
                sys.stdout.write(RED)
            elif lines.startswith("-"):
                sys.stdout.write(CYAN)
            elif (current_proj in lines or current_sys in lines) and "active" not in lines:
                print()
                sys.stdout.write(MAGENTA)
            else:
                sys.stdout.write(RESET)
            print(lines)
        sys.stdout.write(RESET)

def get_scriptname(name):
    return os.path.basename(name).split('.')[0]
