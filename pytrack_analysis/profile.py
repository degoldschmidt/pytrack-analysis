import os, sys
from datetime import datetime as date
from functools import wraps
import tkinter as tk
from tkinter import messagebox, filedialog
from ._globals import *

"""
profile.py
AUTHOR: degoldschmidt
DATE: 17/07/2017

Contains functions for creating a project profile for analysis.
"""

###
# GLOBAL CONSTANTS (based on OS)
###
PROFILE, NAME, OS = get_globals()

def get_profile( _name, _user, script=""):
    """
    Returns profile as dictionary. If the given project name or user name is not in the profile, it will create new entries.

    Arguments:
    * _name: project name or 'all' (all projects)
    * _user: username
    * script: scriptname
    """
    tk.Tk().withdraw()    # Tkinter window suppression
    nowdate = date.now().strftime("%Y-%m-%d %H:%M:%S")    # current timestamp in given format

    ### Read YAML profile file
    with open(PROFILE, 'r') as stream:
        profile = yaml.load(stream)

    ### If _name is 'all', then just return profile (NO changes)
    if _name == 'all':
        return profile

    ### PROJECT EXISTS (update date, activity, systems, etc.)
    if _name in profile['$PROJECTS']:
        NEW_PROJ = True
        project = profile[_name]
        project["last active"] = nowdate
        profile["active"] = _name

        ### CURRENT COMPUTERNAME IS NOT IN PROFILE (update systems incl. paths)
        try:
            systems = project["systems"]
            if NAME not in systems.keys():
                if query_yn("System \'{:}\' does not seem to exist in the profile. DO you want to set up a new systems profile? (Opens TKinter GUI)".format(NAME)):
                    profile["activesys"] = NAME
                    systems[NAME] = {}
                    system = systems[NAME]
                    system["os"] = OS
                    system["python"] = sys.version
                    ### SET UP DATABASE LOCATION
                    dbfile, viddir = set_database(forced=True)
                    if dbfile is not None and viddir is not None:
                        system["database"] = dbfile
                        system["videos"] = viddir
                    ### SET UP OUTPUT LOCATION
                    out, log, plot = set_output(forced=True)
                    if out is not None:
                        system["output"] = out
                        system["log"] = log
                        system["plot"] = plot
                else:
                    pass
            else:
                profile["activesys"] = NAME
                system = systems[NAME]
                system["python"] = sys.version
        except (AttributeError, TypeError):
            # might be empty
            project["systems"] = {}


    ### CREATE NEW PROJECT
    else:
        NEW_PROJ = query_yn("DO you want to create a new project: {:}?".format(_name))
        if NEW_PROJ:
            profile["$PROJECTS"].append(_name)
            profile[_name] = {}
            project = profile[_name]
            project["users"] = []
            project["users"].append(_user)
            project["created"] = nowdate
            project["last active"] = nowdate
            profile["active"] = _name
            project["systems"] = {}
            systems = project["systems"]
            ### ADD COMPUTERNAME TO PROFILE
            systems[NAME] = {}
            system = systems[NAME]
            system["os"] = OS
            system["python"] = sys.version
            ### SET UP DATABASE LOCATION
            dbfile, viddir = set_database(forced=True)
            if dbfile is not None and viddir is not None:
                system["database"] = dbfile
                system["videos"] = viddir
            ### SET UP OUTPUT LOCATION
            out, log, plot = set_output(forced=True)
            if out is not None:
                system["output"] = out
                system["log"] = log
                system["plot"] = plot
            print("Created [PROJECT] {:}.".format(_name))

    if NEW_PROJ:
        ### CREATE NEW USER
        profile["activeuser"] = _user
        users = profile['$USERS']
        if _user not in users:
            if query_yn("DO you want to create a new user: {:}?".format(_user)):
                users.append(_user)

        ### ADD USER TO PROJECT
        if _user not in project["users"]:
            if query_yn("DO you want to add user to project: {:}?".format(_name)):
                project["users"].append(_user)

        print("Loaded [PROJECT] {:}".format(_name))

    ### RETURN
    with io.open(PROFILE, 'w+', encoding='utf8') as outfile:
        yaml.dump(profile, outfile, default_flow_style=False, allow_unicode=True)
    return profile

def get_db(profile):
    """ Returns active system's database file location """
    return profile[profile['active']]['systems'][NAME]['database']

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
        current = profile['active']
        profile_dump = yaml.dump(profile, default_flow_style=False, allow_unicode=True)
        thisstr = profile_dump.split("\n")
        sys.stdout.write(RED)
        for lines in thisstr:
            if lines == "$PROJECTS:" or lines == "$USERS:":
                sys.stdout.write(RED)
            elif lines.startswith("-"):
                sys.stdout.write(CYAN)
            elif current in lines and "active" not in lines:
                print()
                sys.stdout.write(MAGENTA)
            else:
                sys.stdout.write(RESET)
            print(lines)
        sys.stdout.write(RESET)
