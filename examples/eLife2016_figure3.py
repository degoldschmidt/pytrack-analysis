# -*- coding: utf-8 -*-
# author: degoldschmidt
# date: 4/9/2017
import logging
import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Statistics
from pytrack_analysis import Multibench
from example_figures import fig_3

### GLOBAL OPTIONS ON HOW TO RUN THE script
MAKE_IT = False     # overwrites datahook (be careful!)
PLOT_IT = True      # plots figure
SAVE_IT = False     # saves figures
###

def datahook(_file):
    if MAKE_IT:
        raise FileNotFoundError
    else:
        this_size = os.path.getsize(_file)
        if np.log10(this_size) > 8:
            print("[WARNING] Opening large csv file. Might take a while...")
            chunksize = 10 ** 5
            chunks = pd.read_csv(_file, sep="\t", chunksize=chunksize)
            data = pd.concat([chunk for chunk in chunks])
        else:
            data = pd.read_csv(_file, sep="\t")
        print("Opened datahook in", _file)
        return data

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)

    print("***\nProcessing data for Fig. 3...\n", flush=True)
    ### select all sesson from CANS
    group = db.experiment("CANS").select()
    # initialize kinematics object
    kinematics = Kinematics(db)
    # initialize statistics object
    stats = Statistics(db)

    ### Kinematic analysis of trials
    data_types = ["etho", "visit", "encounter"]
    filenames = [os.path.join(get_out(profile), each + "_kinematics.csv") for each in data_types]
    load_data = False
    kinematic_data = {}
    for ix, _file in enumerate(filenames):
        try:
            kinematic_data[data_types[ix]] = datahook(_file)
        except FileNotFoundError:
            print("[ERROR] File not found:", _file)
            load_data = True
    if load_data:
        kinematic_data = kinematics.run_many(group, _VERBOSE=True)
        for ix, _data in enumerate(kinematic_data):
            _data.to_csv(filenames[ix], index=False, sep='\t', encoding='utf-8')

    ### Statistical analysis of behavioral discrete-valued time series
    filenames = [os.path.join(get_out(profile), each + "_segments.csv") for each in data_types]
    segments_data = {}
    for ix, _file in enumerate(filenames):
        try:
            segments_data[data_types[ix]] = datahook(_file)
        except FileNotFoundError:
            segments_data[data_types[ix]] = stats.segments(kinematic_data[ix])
            segments_data[data_types[ix]].to_csv(_file, index=False, sep='\t', encoding='utf-8')
    print("\n[DONE]\n***\n")

    ### Eventually plotting
    if PLOT_IT:
        figures = {}
        plot_data = {
                        'A':    segments_data['visit'].query("state == 1"),   # yeast visits (total durations)
                        'B':    segments_data['encounter'],                   # yeast encounter rate (#encounters/time spent outside)
                        'C':    segments_data['encounter'],                   # probability stopping (#encounters with visit/#encounters)
                        'D':    segments_data['visit'].query("state == 1"),   # yeast visits (avg durations)
                        'E':    segments_data['visit'].query("state == 1"),   # yeast visits (number vs avg duration)
        }
        print(plot_data['A'])
        print("***\nPlotting data for Fig. 3...\n", flush=True)
        figures["fig_3"] = fig_3()
        print("\n[DONE]\n***\n")
    #log.close()
    #log.show()

    del kinematics
    del stats
    del db

    ### SAVE FIGURES TO FILE
    if SAVE_IT:
        pltdir = get_plot(profile)
        for k,v in figures.items():
            figtitle = k + '.pdf'
            pngtitle = k + '.png'
            print(os.path.join(pltdir, figtitle))
            print("***")
            print("Saving figures...\n", flush=True)
            try:
                v[0].savefig(os.path.join(pltdir, figtitle), dpi=300)
            except PermissionError:
                print("[WARNING] Saved into temporary file, because opened pdf viewer denied permission to write file.")
                v[0].savefig(os.path.join(pltdir, "temp_"+figtitle), dpi=300)
            ## TODO: does not work for Windows
            #v[0].savefig(os.path.join(pltdir, pngtitle))
        print("\n[DONE]\n***\n")

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
