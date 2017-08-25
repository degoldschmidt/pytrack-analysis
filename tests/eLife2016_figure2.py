import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prep
from pytrack_analysis.kinematics import Kinematics
from pytrack_analysis.statistics import Statistics
from pytrack_analysis.benchmark import multibench
from example_figures import fig_2
import logging

def get_fig_2(_data, _meta):
    f, ax = fig_2(_data, _meta)
    figs = {
                'fig_2': (f, ax),
            }
    return figs

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
    etho_filename = os.path.join(get_out(profile),"fig2_etho_data.csv")
    try:
        print("Found datahook for ethogram data in", etho_filename)
        etho_size = os.path.getsize(etho_filename)
        if np.log10(etho_size) > 8:
            print("Opening large csv file. Might take a while...")
            chunksize = 10 ** 5
            chunks = pd.read_csv(etho_filename, sep="\t", chunksize=chunksize)
            etho_data = pd.concat([chunk for chunk in chunks])
        else:
            etho_data = pd.read_csv(etho_filename, sep="\t")
    except FileNotFoundError:
        etho_data = kinematics.run_many(group, _VERBOSE=False)
        etho_data.to_csv(etho_filename, index=False, sep='\t', encoding='utf-8')

    ### Statistical analysis of ethogram sequences
    seq_filename = os.path.join(get_out(profile),"fig2_seq_data.csv")
    try:
        print("Found datahook for sequence data in", seq_filename)
        sequence_data = pd.read_csv(seq_filename, sep="\t")
    except FileNotFoundError:
        sequence_data = stats.sequence(etho_data)
        sequence_data = sequence_data.query("behavior == 4") ## only yeast micromovements
        sequence_data.to_csv(seq_filename, index=False, sep='\t', encoding='utf-8')


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
    test = multibench(SILENT=False)
    test(main)
    del test