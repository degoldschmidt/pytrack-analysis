import os
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench
from pytrack_analysis.viz import set_font

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

RUN_STATS = True

def main():
    """
    --- general parameters
     * thisscript: scriptname
     * experiment: experiment id (4 letters)
     * profile:    profile for given experiment, user and scriptname (from profile file)
     * db:         database from file
     * _sessions:  which sessions to process
     * n_ses:      number of sessions
     * stats:      list for stats
    """
    thisscript = os.path.basename(__file__).split('.')[0]
    experiment = 'DIFF'
    profile = get_profile(experiment, 'degoldschmidt', script=thisscript)
    db = Experiment(profile.db())
    sessions = db.sessions
    n_ses = len(sessions)
    stats = []

    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sessions):
        df, meta = each.load(VERBOSE=False)
        kine = Kinematics(df, meta)
        outfolder = os.path.join(profile.out(), kine.name)
        ### run kinematics
        outdf = kine.run(save_as=outfolder, ret=True)
        ### get stats and append to list
        if RUN_STATS: stats.append(kine.stats())

    ### save stats
    statdf = pd.concat(stats, ignore_index=True)
    print(statdf)
    statfile = os.path.join(outfolder, experiment+'_kinematics_stats.csv')
    statdf.to_csv(statfile, index=False)
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
