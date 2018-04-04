import io, os, platform, sys, yaml
from datetime import datetime as date
from functools import wraps
import tkinter as tk
from tkinter import messagebox, filedialog
from pytrack_analysis.cli import query_yn, flprint, colorprint, prn

def get_globals():
    """
    get_globals function

    Returns profile directory, computername and os name (tested for Windows/MacOS).
    """
    ### get global constants for OS
    if os.name == 'nt': # Windows
        homedir = os.environ['ALLUSERSPROFILE']
        NAME = os.environ["COMPUTERNAME"]
        OS = os.environ["OS"]
    else:
        homedir = os.environ['HOME']
        NAME = platform.uname()[1].split(".")[0]+'_'+platform.uname()[4]+'_'+os.environ["LOGNAME"]
        OS = os.name
    ### define user data directory
    user_data_dir = os.path.join(homedir, ".pytrack")
    check_folder(user_data_dir)
    ### define profile file path
    PROFILE = os.path.join(user_data_dir, "profile.yaml")
    PROFILE = check_file(PROFILE)
    ### return profile file path, computer name, and OS name
    return PROFILE, NAME, OS

def check_file(_file):
    """ If file _file does not exist, function will create it. Or if the file is empty, it will create necessary keywords. """
    if not os.path.exists(_file):
        write_yaml(_file, {'USERS': None, 'EXPERIMENTS': None, 'SYSTEM': None})
    else:
        test = read_yaml(_file)
        if test is None:
            write_yaml(_file, {'USERS': None, 'EXPERIMENTS': None, 'SYSTEM': None})
        elif any([each not in test.keys() for each in ['USERS', 'EXPERIMENTS', 'SYSTEM']]):
            write_yaml(_file, {'USERS': None, 'EXPERIMENTS': None, 'SYSTEM': None})
    return _file

def check_folder(_folder):
    """ If folder _folder does not exist, function will create it. """
    if not os.path.exists(_folder):
        os.makedirs(_folder)

def read_yaml(_file):
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream)
    return out

def write_yaml(_file, _dict):
    """ Writes a given dictionary '_dict' into file '_file' in YAML format. Uses UTF8 encoding and no default flow style. """
    with io.open(_file, 'w+', encoding='utf8') as outfile:
        yaml.dump(_dict, outfile, default_flow_style=False, allow_unicode=True)


"""
profile.py
AUTHOR: degoldschmidt
DATE: 03/04/2018

Contains functions for creating a project profile for analysis.
"""

###
# GLOBAL CONSTANTS (based on OS)
###
PROFILE, SYSNAME, OS = get_globals()
prn(__name__)
colorprint('profile file:\t', color='profile', sln=True)
print(PROFILE)
prn(__name__)
colorprint('system:\t\t', color='profile', sln=True)
print(SYSNAME)
prn(__name__)
colorprint('OS:\t\t', color='profile', sln=True)
print(OS)

def get_profile(_id, _user, VERBOSE=True):
    """
    Returns profile as dictionary. If the given project name or user name is not in the profile, it will create new entries.

    Arguments:
    * _id: project id
    * _user: username
    Keywords:
    * script: scriptname
    * VERBOSE: verbose printing option
    """
    if not VERBOSE:
        flprint("Setting up profile...")
    tk.Tk().withdraw()    # Tkinter window suppression

    # Profile object from file
    profile = Profile(PROFILE)
    if _id == '' and _user == '':
        print(profile)
    else:
        # system registration
        profile.set_system(SYSNAME)
        # project registration
        profile.set_experiment(_id)
        # user registration
        profile.set_user(_user)
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
            if self.dict is None or type(self.dict) is str:
                self.dict = {}
        except FileNotFoundError:
            self.dict = {}

    def __del__(self):
        with io.open(PROFILE, 'w+', encoding='utf8') as f:
            yaml.dump(self.dict, f, default_flow_style=False, allow_unicode=True, canonical=False)

    def __str__(self):
        outstr = ""
        for k,v in self.dict.items():
            outstr += str(k) + ':\t' + str(v) + '\n'
        return outstr

    def add_to_experiment(self, _to, _key, _val):
        if type(_key) is list:
            for k, v in zip(_key, _val):
                self.dict['EXPERIMENTS'][_to][k] = v
        else:
            self.dict['EXPERIMENTS'][_to][_key] = _val

    def add_to_system(self, _to, _key, _val):
        self.dict['SYSTEM'][_to][_key] = _val

    def get_database(self):
        """ Returns active system's database file location """
        if 'database' in self.dict['EXPERIMENTS'][self.active].keys():
            dbfile = os.path.join(self.dict['SYSTEMS'][self.activesys]['base'], self.dict['EXPERIMENTS'][self.active]['database'])
            if os.path.exists(dbfile):
                return dbfile
        prn(__name__)
        print("No database file found.")
        dbfile = filedialog.askopenfilename(title="Load database")
        self.dict['EXPERIMENTS'][self.active]['database'] = dbfile
        return dbfile

    def set_dir(self, _title, forced=False):
        """ Returns base directory chosen from TKinter filedialog GUI """
        base = None
        if not forced:
            asksave = messagebox.askquestion(_title, "Are you sure you want to set a new path?", icon='warning')
            if asksave == "no":
                return None
        flprint("Set {}...".format(_title))
        while base is None:
            base = filedialog.askdirectory(title=_title)
        print(base)
        return base

    def get_folder(self):
        project = self.dict["EXPERIMENTS"][self.active]
        return project['basedir']

    def get_experiment(self, _name):
        if _name in self.dict['EXPERIMENTS'].keys():
            return self.dict['EXPERIMENTS'][_name]
        else:
            print('Could not find experiment {}'.format(_name))
            return None

    def get_user(self, _name):
        if _name in self.dict['USERS'].keys():
            return self.dict['USERS'][_name]
        else:
            print('Could not find user {}'.format(_name))
            return None

    def Nvids(self):
        system = self.dict["SYSTEMS"][self.activesys]
        project = self.dict["PROJECTS"][self.active]
        folder = os.path.join(project['basedir'], project['videos'])
        return 72##len([i for i in os.listdir(folder) if ".avi" in i])

    def out(self):
        dbfile = self.db()
        dbfolder = os.path.dirname(dbfile)
        outfolder = os.path.join(dbfolder, 'out')
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        return outfolder

    def remove_experiment(self, _name):
        if self.get_experiment(_name) is not None:
            if self.active == _name:
                self.active = None
            del self.dict['EXPERIMENTS'][_name]

    def remove_user(self, _name):
        if self.get_user(_name) is not None:
            if self.active == _name:
                self.activeuser = None
            del self.dict['USERS'][_name]

    def set_modules(self, _name):
        """
        Updates version numbers for required packages
        """
        thissystem = self.dict["SYSTEM"][_name]
        thissystem['python'] = sys.version
        import numpy
        thissystem['numpy'] = numpy.__version__
        import scipy
        thissystem['scipy'] = scipy.__version__
        import pandas
        thissystem['pandas'] = pandas.__version__
        import matplotlib
        thissystem['matplotlib'] = matplotlib.__version__

    def set_experiment(self, _name):
        if self.dict['EXPERIMENTS'] is None:
            self.dict['EXPERIMENTS'] = {}
        projects = self.dict['EXPERIMENTS']
        if _name in self.dict['EXPERIMENTS'].keys():
            projects[_name]['last modified'] = date.now().strftime("%Y-%m-%d %H:%M:%S")
            self.active = _name
        else:
            print("Experiment \'{:}\' does not seem to exist in the profile.".format(_name))
            if query_yn("Do you want to add {} to the existing experiments?".format(_name)):
                projects[_name] = {
                    'created':  date.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'last modified':  date.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'basedir': self.set_dir('Set base directory', forced=True)
                }
                self.active = _name
            else:
                self.active = None

    def set_folder(self, _folder):
        project = self.dict["EXPERIMENTS"][self.active]
        if os.path.isdir(_folder):
            project['basedir'] = _folder
        return _folder

    def set_system(self, _name):
        if self.dict['SYSTEM'] is None:
            self.dict['SYSTEM'] = {}
        system = self.dict["SYSTEM"]
        if _name in system.keys():
            self.set_modules(_name)
            self.activesys = _name
        else:
            print("System \'{:}\' does not seem to exist in the profile.".format(_name))
            if query_yn("Do you want to add {} to the existing systems?".format(_name)):
                if len([key for key in system.keys()]):
                    skeys = [key for key in system.keys()]
                    overwrite = query_yn("Do you want to overwrite existing system \'{:}\' with \'{:}\'".format(skeys[0], _name))
                    if overwrite:
                        del system[skeys[0]]
                system[_name] =  {}
                self.set_modules(_name)
                self.activesys = _name
            else:
                self.activesys = None

    def set_user(self, _name):
        if self.dict['USERS'] is None:
            self.dict['USERS'] = {}
        users = self.dict["USERS"]
        if _name in users.keys():
            self.activeuser = _name
        else:
            print("User \'{:}\' does not seem to exist in the profile.".format(_name))
            if query_yn("Do you want to add {} to the existing users?".format(_name)):
                users[_name] =  {}
                self.activeuser = _name
            else:
                self.activeuser = None

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
    current_proj = profile.active
    current_sys = profile.activesys
    profile_dump = yaml.dump(profile.dict, default_flow_style=False, allow_unicode=True)
    thisstr = profile_dump.split("\n")
    sys.stdout.write(RED)
    for lines in thisstr:
        try:
            if lines == "EXPERIMENTS:" or lines == "SYSTEM:" or lines == "USERS:":
                sys.stdout.write(RED)
            elif current_proj is not None:
                if (current_proj in lines) and "active" not in lines:
                    print()
                    sys.stdout.write(MAGENTA)
            elif current_sys is not None:
                if (current_sys in lines) and "active" not in lines:
                    print()
                    sys.stdout.write(MAGENTA)
            else:
                sys.stdout.write(RESET)
            print(lines)
            sys.stdout.write(RESET)
        except:
            sys.stdout.write(RESET)

def get_scriptname(name):
    return os.path.basename(name).split('.')[0]
