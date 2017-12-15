import argparse
import sys

"""
Print function for same line printing
"""
def colorprint(*args, color="default", sln=False):
    out = ""
    for arg in args:
        out += arg

    if color == 'success':
        colorstr = '\x1b[0;32;40m'
    if color == 'warning':
        colorstr = '\x1b[0;30;43m'
    if color == 'error':
        colorstr = '\x1b[1;31;40m'
    if color == 'namespace':
        colorstr = '\x1b[0;36;40m'
    if sln:
        flprint(colorstr + out + '\x1b[0m')
    else:
        print(colorstr + out + '\x1b[0m')

"""
Returns arguments from CLI (CLI)
"""
def get_args(*args, SILENT=True):
    ### parsing arguments
    parser = argparse.ArgumentParser()
    for arg in args:
        if type(arg) is str:
            parser.add_argument('-'+arg, action='store', dest=arg, help='Store variable '+arg)
        elif type(arg) is list:
            parser.add_argument('-'+arg[0], action='store', dest=arg[1], help=arg[2])
        elif type(arg) is tuple:
            parser.add_argument('-'+arg[0], action='store', dest=arg[1], help=arg[2])
    if not SILENT:
        print("Parsing arguments...")
        dict_args = vars(parser.parse_args())
        for k, v in dict_args.items():
            print("{}: {}".format(k, v))
    return parser.parse_args()

"""
Print function for same line printing
"""
def flprint(*args):
    out = ""
    for arg in args:
        out += arg
    print(out, flush=True, end="")

"""
Print function for namespace printing
"""
def prn(name):
    outname = "["+name+"]\t"
    colorprint(outname, color='namespace', sln=True)

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
