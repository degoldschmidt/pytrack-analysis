import os
from pytrack_analysis.profile import *

if __name__ == '__main__':
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    # Start 'Test project' as user 'degoldschmidt'
    PROFILE = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    show_profile(PROFILE)
    # Show all projects in profile file
    PROFILE = get_profile('all', '*')
    show_profile(PROFILE)
