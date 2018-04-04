import os
from pytrack_analysis.profile import *

if __name__ == '__main__':
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    print(thisscript)
    # Show all projects in profile file
    PROFILE = get_profile("TEST", "TEST_USER")
    # print PROFILE
    show_profile(PROFILE)
    PROFILE.remove_experiment('TEST')
    PROFILE.remove_user('TEST_USER')
    del PROFILE
