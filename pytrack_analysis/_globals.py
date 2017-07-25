import io
import os
import platform
import sys
import yaml

def query_yn(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    From: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

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
    user_data_dir = os.path.join(homedir, "tracking_user_data")
    check_folder(user_data_dir)
    ### define profile file path
    PROFILE = os.path.join(user_data_dir, "profile.yaml")
    PROFILE = check_file(PROFILE)
    ### return profile file path, computer name, and OS name
    return PROFILE, NAME, OS

def check_file(_file):
    """ If file _file does not exist, function will create it. Or if the file is empty, it will create necessary keywords. """
    if not os.path.exists(_file):
        write_yaml(_file, {'$USERS': [], '$PROJECTS': []})
    else:
        test = read_yaml(_file)
        if test is None:
            write_yaml(_file, {'$USERS': [], '$PROJECTS': []})
        elif '$LINK' in test.keys():
            link = os.path.join(test['$LINK'],'profile.yaml')
            print("Found link to {:}".format(link))
            ### Link to new profile file
            return link
        elif '$USERS' not in test.keys() or '$PROJECTS' not in test.keys():
            write_yaml(_file, {'$USERS': [], '$PROJECTS': []})
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
