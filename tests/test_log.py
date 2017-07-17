### Standard Python modules
import os
from pytrack_analysis.profile import *

if __name__ == '__main__':
    # filename of this script
    thisscript = os.path.basename(__file__).split(".")[0]                        # filename of this script
    # get an active project
    PROFILE = get_profile("Vero eLife 2016", "degoldschmidt", script=thisscript) # returns profile dict
    # start logging
    log = Logger(PROFILE, scriptname=thisscript)
    """
    imagine doing something here
    """
    # close logging
    log.close()
