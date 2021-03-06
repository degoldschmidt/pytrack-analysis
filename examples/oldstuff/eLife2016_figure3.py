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
MAKE_IT_ALL = False      # overwrites datahook (be careful!)
MAKE_IT = True      # overwrites datahook (be careful!)
PLOT_IT = True      # plots figure
SAVE_IT = True      # saves figures
###

def datahook(_file):
    if MAKE_IT_ALL:
        raise FileNotFoundError
    elif MAKE_IT and "kinematics" not in _file:
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

def get_kinematics(_files, _db):
    load_data = False
    kinematic_data = {}
    for ix, eachfile in enumerate(_files):
        thistype = os.path.basename(eachfile).split("_")[0]
        try:
            kinematic_data[thistype] = datahook(eachfile)
        except FileNotFoundError:
            print("[ERROR] File not found:", eachfile)
            load_data = True
    if load_data:
        group = _db.experiment("CANS").select()
        kinematics = Kinematics(_db)
        kinematic_data = kinematics.run_many(group, _VERBOSE=True)
        for ix, _data in enumerate(kinematic_data.values()):
            _data.to_csv(_files[ix], index=False, sep='\t', encoding='utf-8')
    return kinematic_data

def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)

    print("***\nProcessing data for Fig. 3...\n", flush=True)
    data_types = ["etho", "visit", "encounter"]

    ### Statistical analysis of behavioral discrete-valued time series
    filenames = [os.path.join(get_out(profile), each + "_segments.csv") for each in data_types]
    segments_data = {}
    stats = Statistics(db)
    kine_files = [os.path.join(get_out(profile), each + "_kinematics.csv") for each in data_types]
    kinematic_data = get_kinematics(kine_files, db)
    for ix, _file in enumerate(filenames):
        try:
            segments_data[data_types[ix]] = datahook(_file)
        except FileNotFoundError:
            segments_data[data_types[ix]] = stats.segments(kinematic_data[data_types[ix]])
            segments_data[data_types[ix]].to_csv(_file, index=False, sep='\t', encoding='utf-8')

    encounter_rate = stats.frequency(segments_data['encounter'], 1, 'session')
    visit_ratio = stats.visit_ratio(segments_data['encounter'], kinematic_data['encounter'], segments_data['visit'], kinematic_data['visit'])
    print("\n[DONE]\n***\n")


    ### Eventually plotting
    if PLOT_IT:
        figures = {}
        ### new mapping for plotting
        segments_data['visit']['mating'] = segments_data['visit']['mating'].map({1: True, 2: False})
        segments_data['visit']['metabolic'] = segments_data['visit']['metabolic'].map({1: '+', 2: '-', 3: '++'})
        encounter_rate['mating'] = encounter_rate['mating'].map({1: True, 2: False})
        encounter_rate['metabolic'] = encounter_rate['metabolic'].map({1: '+', 2: '-', 3: '++'})
        visit_ratio['mating'] = visit_ratio['mating'].map({1: True, 2: False})
        visit_ratio['metabolic'] = visit_ratio['metabolic'].map({1: '+', 2: '-', 3: '++'})

        # data for Fig. 3A
        a_data = segments_data['visit'].query("state == 1").drop_duplicates('session')[['mating', 'metabolic', 'session', 'total_length [min]']]
        a_data = a_data.rename(columns = {'total_length [min]': 'Total duration\nof yeast visits\n[min]', 'mating': 'Mated', 'metabolic': 'AA\npre-diet'})

        b_data = encounter_rate[['mating', 'metabolic', 'session', 'rate [1/min]']]
        b_data = b_data.rename(columns = {'rate [1/min]': 'Rate of yeast\nencounters\n[1/min]', 'mating': 'Mated', 'metabolic': 'AA\npre-diet'})

        c_data = visit_ratio.query("state == 1")[['mating', 'metabolic', 'session', 'ratio']]
        c_data = c_data.rename(columns = {'ratio': 'Ratio of\nof yeast visits\nper encounter', 'mating': 'Mated', 'metabolic': 'AA\npre-diet'})

        d_data = segments_data['visit'].query("state == 1").drop_duplicates('session')[['mating', 'metabolic', 'session', 'mean_length [min]']]
        d_data = d_data.rename(columns = {'mean_length [min]': 'Mean duration\nof yeast visits\n[min]', 'mating': 'Mated', 'metabolic': 'AA\npre-diet'})

        e_data = segments_data['visit'].query("state == 1 & mating == 1").drop_duplicates('session')[['num_segments', 'mean_length [min]', 'metabolic']]
        e_data = e_data.rename(columns = {'mean_length [min]': 'Mean duration\nof yeast visits\n[min]', 'num_segments': 'Number of yeast visits'})

        plot_data = {
                        'A':    a_data, # yeast visits (total durations)
                        'B':    b_data, # yeast encounter rate (#encounters/time spent outside)
                        'C':    c_data, # probability stopping (#encounters with visit/#encounters)
                        'D':    d_data, # yeast visits (avg durations)
                        'E':    e_data, # yeast visits (number vs avg duration)
        }
        #print(plot_data['A'])
        print("***\nPlotting data for Fig. 3...\n", flush=True)
        figures["fig_3"] = fig_3(plot_data)
        print("\n[DONE]\n***\n")
    #log.close()
    #log.show()

    #del kinematics
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
