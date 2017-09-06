# -*- coding: utf-8 -*-
# author: degoldschmidt
# date: 25/8/2017
import logging
import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Statistics
from pytrack_analysis import Multibench
from example_figures import fig_2

def get_fig_2(_data, _meta):
    f, ax = fig_2(_data, _meta)
    figs = {
                'fig_2': (f, ax),
            }
    return figs

def datahook(_file):
    this_size = os.path.getsize(_file)
    if np.log10(this_size) > 8:
        print("Opening large csv file. Might take a while...")
        chunksize = 10 ** 5
        chunks = pd.read_csv(_file, sep="\t", chunksize=chunksize)
        data = pd.concat([chunk for chunk in chunks])
    else:
        data = pd.read_csv(etho_filename, sep="\t")
    print("Opened datahook in", _file)
    return data

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)

    ### Fig. 2
    print("Process Fig. 2...", flush=True)
    ### select all sesson from CANS
    group = db.experiment("CANS").select()
    # initialize kinematics object
    kinematics = Kinematics(db)
    # initialize statistics object
    stats = Statistics(db)

    ### Kinematic analysis of trials
    data_types = ["etho", "visit", "encounter"]
    filenames = [os.path.join(get_out(profile),"fig2_" + each + "_data.csv") for each in data_types]
    load_data = False
    data = []
    for _file in filenames:
        try:
            data.append(datahook(_file))
        except FileNotFoundError:
            load_data = True
    data = kinematics.run_many(group, _VERBOSE=True)
    for ix, _data in enumerate(data):
        _data.to_csv(filenames[ix], index=False, sep='\t', encoding='utf-8')

    ### Statistical analysis of behavioral discrete-valued time series
    filenames = [os.path.join(get_out(profile),"fig2_" + each + "_segments.csv") for each in data_types]
    segments_data = []
    for ix, _file in enumerate(filenames):
        try:
            segments_data.append(datahook(_file))
        except FileNotFoundError:
            segments_data.append(stats.segments(data[ix]))
            segments_data[-1].to_csv(seq_filename, index=False, sep='\t', encoding='utf-8')

    ### Eventually plotting
    figures = get_fig_2([etho_data, etho_segments], db)
    print("[DONE]")
    log.close()
    log.show()

    del kinematics
    del stats
    del db

    ### SAVE FIGURES TO FILE
    pltdir = get_plot(profile)
    for k,v in figures.items():
        figtitle = k + '.pdf'
        pngtitle = k + '.png'
        print(os.path.join(pltdir, figtitle))
        v[0].savefig(os.path.join(pltdir, figtitle), dpi=300)
        ## TODO: does not work for Windows
        #v[0].savefig(os.path.join(pltdir, pngtitle))

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
