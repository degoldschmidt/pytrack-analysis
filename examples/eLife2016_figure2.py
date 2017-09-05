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

    ### DATAHOOK IMPLEMENTATION
    types = ['etho', 'visit', 'encounter']
    filenames = [os.path.join(get_out(profile),"fig2_" + each + "_data.csv") for each in types]
    data = []
    load_all = False
    for fname in filenames:
        try:
            data.append(datahook(fname))
        except FileNotFoundError:
            load_all = True
    if load_all:
    data = kinematics.run_many(group, _VERBOSE=True)
            etho_data.to_csv(fname, index=False, sep='\t', encoding='utf-8')

    ### Statistical analysis of ethogram sequences
    seq_filename = os.path.join(get_out(profile),"fig2_seq_data.csv")
    try:
        sequence_data = pd.read_csv(seq_filename, sep="\t")
        print("Found datahook for sequence data in", seq_filename)
    except FileNotFoundError:
        etho_segments = stats.segments(etho_data)
        etho_segments = etho_segments.query("state == 4") ## only yeast micromovements
        etho_segments.to_csv(seq_filename, index=False, sep='\t', encoding='utf-8')


    ### Eventually plotting
    figures = get_fig_2([etho_data, sequence_data], db)
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
