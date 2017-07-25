import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *

if __name__ == '__main__':
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    # load 'Vero eLife 2016' as user 'degoldschmidt'
    PROFILE = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    DB = Database(get_db(PROFILE)) # database from file
    print("0003A01R01Cam03.avi is in:", DB.find('Videofilename=0003A01R01Cam03.avi'))
    this_session = DB.experiment("CANS").session("005")
    print(this_session.keys())
