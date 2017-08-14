import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prep
from pytrack_analysis.kinematics import Kinematics
from pytrack_analysis.statistics import Statistics
from pytrack_analysis.benchmark import multibench
from example_figures import fig_1c, fig_1d
import logging

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set(color_codes=True)
#sns.set_style('ticks')

LOG_ME = False

def stats_analysis(db, _only=[]):
    ### Get data together
    if len(_only) == 0:
        _only = db.select()
    etho_data = {}
    for session in _only:
        etho_data[session.name] = session.data['etho']
    etho_data = pd.DataFrame(etho_data)

    stats = Statistics(db)

    ### STEP 1: Get etho sequence data (lengths, total lengths, cumulative lengths)
    sequence_data = stats.sequence(etho_data)
    print(len(sequence_data))
    virgin_data = sequence_data.query('mating == 2')
    virgin_data = virgin_data.query('behavior == 4 or behavior == 5')
    virgin_data = virgin_data.drop_duplicates('total_length [s]')
    print(len(virgin_data))
    mated_data = sequence_data.query('mating == 1')
    mated_data = mated_data.query('behavior == 4 or behavior == 5')
    mated_data = mated_data.drop_duplicates('total_length [s]')
    print(len(mated_data))


    ## plot testing
    f, axes = plt.subplots( 2, num="Fig. 1E")
    data = [virgin_data, mated_data]
    for ix,ax in enumerate(axes):
        ax = sns.boxplot(x="behavior", y="total_length [s]", data=data[ix], ax=ax)
        ax = sns.swarmplot(x="behavior", y="total_length [s]", data=data[ix], color='#3d3d3d', ax=ax)
        ax.set_xticklabels(["Yeast", "Sucrose"])
    plt.show()



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
                '1C': (f1c, a1c),
                '1D': (f1d, a1d),
            }
    plt.close("all")
    #plt.show()
    return figs

def main():
    DO_IT = "CD"
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
    if "EG" in DO_IT :
        print("Process Fig. 1 E & G...", flush=True)
        # selecting only AA+ rich and Canton S (obsolete)
        only_metab =  ["AA+ rich"]
        only_gene = ["Canton S"]
        group = db.experiment("CANS").select(genotype=only_gene, metabolic=only_metab)
        etho_data = kinematics.run_many(group, _VERBOSE=True)
        #stats_analysis(db, _only=group)
        print("[DONE]")
    log.close()
    #log.show()
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
    test = multibench(times=100)
    test(main)
    del test
