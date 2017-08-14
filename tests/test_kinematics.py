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

#### USED FOR PLOTTING
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

#### plotting
def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "ns"

def stats_analysis(etho_data, _stats=[]):
    #### USED FOR PLOTTING
    import seaborn as sns; sns.set(color_codes=True)
    sns.set_style('ticks')
    import scipy.stats as scistat

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


    ## plot testing
    f, axes = plt.subplots( 2, num="Fig. 1E/G", figsize=(3,3.5))
    print("Figsize [inches]: ", f.get_size_inches())
    data = [virgin_data, mated_data]
    substrate_colors = ['#ffc04c', '#4c8bff']  ##MATING COLORS #bc1a62","": "#1abc74"}
    title_label = ["Virgin", "Mated"]
    panel_label = ["E", "G"]
    ticks = [[0, 5, 1], [0,12,2]]
    tick_label = [ ["0", "1", "", "3", "", "5"], ["0", "2", "", "", "", "10", "12"]]
    lims = [[0,5], [0,12]]
    staty = [4.5, 9.5]
    for ix,ax in enumerate(axes):
        ### main data (box, swarm, median line)
        ax = sns.boxplot(x="behavior", y="total_length [min]", data=data[ix], order = ["Yeast", "Sucrose"], palette=substrate_colors, width=0.35, linewidth=0.0, boxprops=dict(lw=0.0), showfliers=False, ax=ax)
        ax = sns.swarmplot(x="behavior", y="total_length [min]", data=data[ix], order = ["Yeast", "Sucrose"], size=3, color='#666666', ax=ax)
        yeast_data = np.array(data[ix].query("behavior == 'Yeast'")["total_length [min]"])
        sucrose_data = np.array(data[ix].query("behavior == 'Sucrose'")["total_length [min]"])
        medians = [np.median(yeast_data), np.median(sucrose_data)]
        dx = 0.3
        for pos, median in enumerate(medians):
           ax.hlines(median, pos-dx, pos+dx, lw=1, zorder=10)

        ### stats annotation
        statistic, pvalue = scistat.ranksums(yeast_data, sucrose_data)
        y_max = np.max(np.concatenate((yeast_data, sucrose_data)))
        y_min = np.min(np.concatenate((yeast_data, sucrose_data)))
        y_max += abs(y_max - y_min)*0.05 ## move it up
        ax.annotate("", xy=(0, y_max), xycoords='data', xytext=(1, y_max), textcoords='data', arrowprops=dict(arrowstyle="-", fc='#000000', ec='#000000', lw=1,connectionstyle="bar,fraction=0.1"))
        ax.text(0.5, y_max + abs(y_max - y_min)*0.15, stars(pvalue), horizontalalignment='center', verticalalignment='center')

        print("pvalue:", pvalue)

        ### figure aesthetics
        ax.set_xlabel("") # remove xlabel
        ax.set_ylabel(title_label[ix]+"\n\nTotal duration\nof food micro-\nmovements [min]") # put a nice ylabel
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, x=-2, y=0.15) # rotates the xlabels by 30º
        #print(ax.get_xlim(), ax.get_ylim())
        #ax.set_aspect((ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]))
        sns.despine(ax=ax, bottom=True)
        ax.set_ylim(lims[ix])
        ax.set_yticks(np.arange(ticks[ix][0],ticks[ix][1]+1, ticks[ix][2]))
        #ax.set_yticklabels(tick_label[ix])
        ax.get_xaxis().set_tick_params(width=0) # no xticks markers

    plt.tight_layout()
    for ix,ax in enumerate(axes):
        ax.set_title(panel_label[ix], fontsize=16, fontweight='bold', loc='left', x=-.85, y= 1)
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
    DO_IT = "EG"
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

        stats_analysis(etho_data, _stats=stats)
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
