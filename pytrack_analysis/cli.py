import argparse
import sys

"""
Returns arguments from CLI (CLI)
"""
def get_args(*args):
    ### parsing arguments
    parser = argparse.ArgumentParser()
    for arg in args:
        if type(arg) is str:
            parser.add_argument('-'+arg, action='store', dest=arg, help='Store variable '+arg)
        elif type(arg) is list:
            parser.add_argument('-'+arg[0], action='store', dest=arg[1], help=arg[2])
        elif type(arg) is tuple:
            parser.add_argument('-'+arg[0], action='store', dest=arg[1], help=arg[2])
    return parser.parse_args()

"""
Ask a yes/no question via raw_input() and return their answer.

"question" is a string that is presented to the user.
"default" is the presumed answer if the user just hits <Enter>.
    It must be "yes" (the default), "no" or None (meaning
    an answer is required of the user).

The "answer" return value is True for "yes" or False for "no".
From: http://code.activestate.com/recipes/577058/
"""
def query_yn(question, default="no"):
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
