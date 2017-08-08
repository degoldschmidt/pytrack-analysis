import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prep
from pytrack_analysis.kinematics import Kinematics
from pytrack_analysis.statistics import Statistics
from pytrack_analysis.benchmark import multibench
from example_figures import fig_1c, fig_1d

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

DO_IT = ["Fig1"]

def kine_analysis(db, _experiment="CANS", _session="005", MULTI=False):
    # load session
    print(_experiment, _session)
    this_session = db.experiment(_experiment).session(_session)
    # load data
    raw_data, meta_data = this_session.load()
    #arena_env = {}

    ## STEP 1: NaN removal + interpolation + px-to-mm conversion
    clean_data = prep.interpolate(raw_data)
    clean_data = prep.to_mm(clean_data, meta_data.px2mm)

    ## STEP 2: Gaussian filtering
    window_len = 16 # = 0.32 s
    smoothed_data = prep.gaussian_filter(clean_data, _len=window_len, _sigma=window_len/10)

    ## STEP 3: regrouping data to body and head position
    body_pos, head_pos = smoothed_data[['body_x', 'body_y']], smoothed_data[['head_x', 'head_y']]

    ## STEP 4: Distance from patch
    kinematics = Kinematics(meta_data.dict)
    distance_patch = kinematics.distance_to_patch(head_pos, meta_data)

    ## STEP 5: Linear Speed
    head_speed = kinematics.linear_speed(head_pos, meta_data)
    window_len = 60 # = 1.2 s
    smooth_head_speed = prep.gaussian_filter(head_speed, _len=window_len, _sigma=window_len/10)
    window_len = 120 # = 1.2 s
    smoother_head = prep.gaussian_filter(smooth_head_speed, _len=window_len, _sigma=window_len/10)
    body_speed = kinematics.linear_speed(body_pos, meta_data)
    smooth_body_speed = prep.gaussian_filter(body_speed, _len=window_len, _sigma=window_len/10)
    speeds = pd.DataFrame({"head": smooth_head_speed["speed"], "body": smooth_body_speed["speed"], "smoother_head": smoother_head["speed"]})

    ## STEP 6: Angular Heading & Speed
    angular_heading = kinematics.head_angle(smoothed_data)
    angular_speed = kinematics.angular_speed(angular_heading, meta_data)

    ## STEP 7: Ethogram classification
    etho_dict = {   0: "resting",
                    1: "micromovement",
                    2: "walking",
                    3: "sharp turn",
                    4: "yeast micromovement",
                    5: "sucrose micromovement"}
    meta_data.dict["etho_class"] = etho_dict
    etho_vector, visits = kinematics.ethogram(speeds, angular_speed, distance_patch, meta_data)

    ## STEP 8: Food patch encounters (TODO)

    ## DESTROY object
    del kinematics

    ## data to add to db
    if not MULTI:
        this_session.add_data("head_pos", head_pos, descr="Head positions of fly in [mm].")
        this_session.add_data("body_pos", body_pos, descr="Body positions of fly in [mm].")
        this_session.add_data("distance_patches", distance_patch, descr="Distances between fly and individual patches in [mm].")
        this_session.add_data("head_speed", smooth_head_speed, descr="Gaussian-filtered (60 frames) linear speeds of head trajectory of fly in [mm/s].")
        this_session.add_data("body_speed", smooth_body_speed, descr="Gaussian-filtered (60 frames) linear speeds of body trajectory of fly in [mm/s].")
        this_session.add_data("smoother_head_speed", smoother_head, descr="Gaussian-filtered (120 frames) linear speeds of body trajectory of fly in [mm/s]. This is for classifying resting bouts.")
        this_session.add_data("angle", angular_heading, descr="Angular heading of fly in [o].")
        this_session.add_data("angular_speed", angular_speed, descr="Angular speed of fly in [o/s].")
        this_session.add_data("etho", etho_vector, descr="Ethogram classification. Dictionary is given to meta_data[\"etho_class\"].")
        this_session.add_data("visits", visits, descr="Food patch visits. 1: yeast, 2: sucrose.")
    else:
        this_session.add_data("etho", etho_vector, descr="Ethogram classification. Dictionary is given to meta_data[\"etho_class\"].")
        this_session.add_data("visits", visits, descr="Food patch visits. 1: yeast, 2: sucrose.")

def stats_analysis(db, _only=[]):
    ### Get data together
    if len(_only) == 0:
        _only = db.sessions()
    etho_data = {}
    for session in _only:
        etho_data[session.name] = session.data['etho']
    etho_data = pd.DataFrame(etho_data)

    stats = Statistics()

    ### STEP 1: Total durations of micromovements
    tot_dur_micromov = stats.total_durations_micromovements(etho_data)


def plotting(db, _experiment="CANS", _session="005"):
    ### PLOTTING
    ## Fig 1
    this_session = db.experiment(_experiment).session(_session)
    start = 56100#58085 50*180 =
    end = start+9000#65450#62577
    meta = this_session
    data = this_session.data.loc[start:end]

    ## C
    f1c, a1c = fig_1c(data, meta)
    ## D
    data = this_session.data.loc[start:end+370]
    f1d, a1d = fig_1d(data, meta)
    ## E
    ## F
    ## G
    ## H

    ## Fig 2


    figs = {
                '1C': (f1c, a1c),
                '1D': (f1d, a1d),
            }
    #plt.show()
    return figs


def main():
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)

    if "Fig1" in DO_IT:
        ### Example session "CANS_005" for Fig 1C,D
        kine_analysis(db)

        ### Fig. E-H
        only_metab =  ["AA+ rich"]
        only_gene = ["Canton S"]
        for session in db.sessions(genotype=only_gene, metabolic=only_metab):
            kine_analysis(db, _experiment=session.exp, _session=session.name, MULTI=True)
        num_mated, num_virgins = db.count(only_gene, ['Mated', 'Virgin'], only_metab)
        print( "Analyzed {1} mated {0} females and {2} virgin {0} females".format(db.last_select('Metabolic'), int(num_mated), int(num_virgins)) )
        stats_analysis(db, _only=db.sessions(genotype=only_gene, metabolic=only_metab))

    log.close()
    #log.show()
    figures = plotting(db)

    ### SAVE FIGURES TO FILE
    pltdir = get_plot(profile)
    for k,v in figures.items():
        figtitle = k + '.pdf'
        v[0].savefig(os.path.join(pltdir, figtitle), dpi=300)#v[0].dpi)


if __name__ == '__main__':
    # runs as benchmark test
    test = multibench()
    test(main)
    del test
