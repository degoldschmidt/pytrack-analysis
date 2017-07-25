import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *

if __name__ == '__main__':
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    # load 'Vero eLife 2016' as user 'degoldschmidt'
    PROFILE = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    DB = Database(get_db(PROFILE)) # database from file
