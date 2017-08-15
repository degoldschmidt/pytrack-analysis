import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prep
from pytrack_analysis.kinematics import Kinematics
from pytrack_analysis.statistics import Statistics
from pytrack_analysis.benchmark import multibench
from example_figures import fig_1c, fig_1d, fig_1e_h
import logging

def stats_analysis(etho_data, _stats=[]):
    ### STEP 1: Get etho sequence data (lengths, total lengths, cumulative lengths)
    sequence_data = _stats.sequence(etho_data)
    sequence_data.loc[sequence_data['behavior']==4, 'behavior'] = 'Yeast'
    sequence_data.loc[sequence_data['behavior']==5, 'behavior'] = 'Sucrose'
    virgin_data = sequence_data.query('mating == 2')
    virgin_data = virgin_data.query("behavior == 'Yeast' or behavior == 'Sucrose'")
    virgin_data = virgin_data.drop_duplicates('total_length [min]')
    #print(len(virgin_data))
    mated_data = sequence_data.query('mating == 1')
    mated_data = mated_data.query("behavior == 'Yeast' or behavior == 'Sucrose'")
    mated_data = mated_data.drop_duplicates('total_length [min]')
    #print(len(mated_data))
    return [virgin_data, mated_data]



def fig_1cd(_data, _meta):
    ### PLOTTING
    ## Fig 1
    start = 56100#58085 50*180 =
    end = start+9000#65450#62577
    meta = _meta
    data = _data.loc[start:end]

    ## C
    f1c, a1c = fig_1c(data, meta)
    ## D
    data = _data.loc[start:end+370]
    f1d, a1d = fig_1d(data, meta)

    figs = {
                'fig_1C': (f1c, a1c),
                'fig_1D': (f1d, a1d),
            }
    return figs

def fig_1eg(_data, _meta):
    f, ax = fig_1e_h(_data, _meta)
    figs = {
                'fig_1E_G': (f, ax),
            }
    return figs

def main():
    DO_IT = "CDEG"
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)

    ### Example session "CANS_005" for Fig 1C,D
    figcd = {}
    if "CD" in DO_IT :
        this_one = "CANS_005"
        print("Process Fig. 1 C & D...", end="\t\t\t", flush=True)
        kinematics = Kinematics(db)
        kinematics.run(this_one, _ALL=True)
        figcd = fig_1cd(db.session(this_one).data, db.session(this_one))
        print("[DONE]")

    ### Fig. E-H
    figeg = {}
    if "EG" in DO_IT:
        print("Process Fig. 1 E & G...", flush=True)
        # selecting only AA+ rich and Canton S (obsolete)
        only_metab =  ["AA+ rich"]
        only_gene = ["Canton S"]
        group = db.experiment("CANS").select(genotype=only_gene, metabolic=only_metab)
        # initialize statistics object
        stats = Statistics(db)
        etho_filename = os.path.join(get_out(profile),"etho_data.csv")

        ### DATAHOOK IMPLEMENTATION
        if os.path.exists(etho_filename):
            if os.path.isfile(etho_filename):
                print("Found datahook for ethogram data in", etho_filename)
                etho_data = pd.read_csv(etho_filename, sep="\t")
            else:
                etho_data = kinematics.run_many(group, _VERBOSE=True)
                etho_data.to_csv(etho_filename, index=False, sep='\t', encoding='utf-8')
        else:
            etho_data = kinematics.run_many(group, _VERBOSE=True)
            etho_data.to_csv(etho_filename, index=False, sep='\t', encoding='utf-8')

        virgin_mated_data = stats_analysis(etho_data, _stats=stats)
        figeg = fig_1eg(virgin_mated_data, "Substrate")
        print("[DONE]")
    log.close()
    log.show()
    figfh = {}

    #del kinematics
    del db

    ### SAVE FIGURES TO FILE
    figures = {**figcd, **figeg, **figfh}
    pltdir = get_plot(profile)
    for k,v in figures.items():
        figtitle = k + '.pdf'
        v[0].savefig(os.path.join(pltdir, figtitle), dpi=300)#v[0].dpi)


if __name__ == '__main__':
    # runs as benchmark test
    test = multibench(SILENT=False)
    test(main)
    del test
