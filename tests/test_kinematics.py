import os
from pytrack_analysis.profile import *
from pytrack_analysis.database import *
from pytrack_analysis.logger import Logger
import pytrack_analysis.preprocessing as prep
from pytrack_analysis.kinematics import Kinematics, get_path
get_path("Kinematics log path:")

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

def analysis(db):
    # load session
    this_session = db.experiment("CANS").session("005")
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
    kinematics = Kinematics(smoothed_data, meta_data.dict)
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
    angles = pd.DataFrame({"heading": angular_heading["heading"], "speed": angular_speed["speed"]})

    ## STEP 7: Ethogram classification
    etho_dict = {   0: "resting",
                    1: "micromovement",
                    2: "walking",
                    3: "sharp turn",
                    4: "yeast micromovement",
                    5: "sucrose micromovement"}
    meta_data.dict["etho_class"] = etho_dict
    etho_vector, visits = kinematics.ethogram(speeds, angular_speed, distance_patch, meta_data)

    ## data to add to db
    this_session.add_data("head_pos", head_pos, descr="Head positions of fly in [mm].")
    this_session.add_data("distance_patches", distance_patch, descr="Distances between fly and individual patches in [mm].")
    this_session.add_data("speeds", speeds, descr="Linear speeds of body and head trajectory of fly in [mm/s].")
    this_session.add_data("angle", angular_heading, descr="Angular heading of fly in [o].")
    this_session.add_data("angl_speed", angular_speed, descr="Angular speed of fly in [o/s].")
    this_session.add_data("angles", angles, descr="Angular heading and speed of fly in [o] and in [o/s], respectively.")
    this_session.add_data("etho", etho_vector, descr="Ethogram classification.")
    this_session.add_data("visits", visits, descr="Food patch visits.")

def plotting(db):
    ### PLOTTING
    ## Fig 1
    this_session = db.experiment("CANS").session("005")
    start = 56100#58085 50*180 =
    end = start+9000#65450#62577
    meta = this_session
    data = this_session.data['distance_patches'].loc[start:end,['dist_patch_0']]
    fci, axci = fig_1c(data, meta, 0)
    data = this_session.data['speeds'].loc[start:end,['head', 'body']]
    fcii, axcii = fig_1c(data, meta, 1)
    data = this_session.data['angles'].loc[start:end, ['heading', 'speed'] ]
    fciii, axciii = fig_1c(data, meta, 2)
    data = this_session.data['etho'].loc[start:end, ['etho'] ]
    fciv, axciv = fig_1c(data, meta, 3)
    data = this_session.data['visits'].loc[start:end, ['visits'] ]
    fciv, axciv = fig_1c(data, meta, 4)
    data = this_session.data['head_pos'].loc[start:end,['head_x', 'head_y']]
    fd, axd = fig_1d(data, meta)

    fciname = './fci.pdf'
    #fci.savefig(fciname, dpi=fci.dpi)
    fdname = './fd.pdf'
    #fd.savefig(fdname, dpi=fd.dpi)

    plt.show()

def fig_test(data, meta):
    f, ax = plt.subplots( 1,
                            num="Test",
                            figsize=(4.5, 1.5),
                            dpi=300)
    a = np.array(data)[:,0]
    dy = 0.5
    x = np.arange(0,len(a))
    _lw = 0.1
    ax.vlines(x[a==1],-dy,dy, colors='#c97aaa', lw=_lw)
    ax.vlines(x[a==2],-dy,dy, colors='#5bd5ff', lw=_lw)
    ax.vlines(x[a==3],-dy,dy, colors='#04bf11', lw=_lw)
    ax.vlines(x[a==4],-dy,dy, colors='#f0e442', lw=_lw)
    ax.vlines(x[a==5],-dy,dy, colors='k', lw=_lw)
    #ax.imshow(a.T)
    ax.set_ylim([-dy,dy])
    return f, ax

def fig_1c_all(data, meta):
    pass

"""
PLOTTING FIG 1Ci-iii
"""
def fig_1c(data, meta, index):
    figlabels = {
                0: "i: Distance to Patch",
                1: "ii: Linear Speed",
                2: "iii: Angular Speed",
                3: "iv: Ethogram",
                4: "v: Food Patch Visits",
    }
    ylabels = {
                0: "Distance\nto patch\n[mm]",
                1: "Linear\nspeed\n[mm/s]",
                2: "Angular\nspeed\n[$^\circ$/s]",
                3: "Etho-\ngram",
                4: "Food\npatch\nvisits",
    }

    start = data.first_valid_index()
    #print(start)
    end = start+9000#65450#62577
    nsubs = [2,2,1,1,1]
    if index < 2:
        ## Ratios for grid
        splits = [0,1]
        end_at = [25,20]
        break_at = 6
        scale1 = 1
        scale2 = 5

        #for i in range(len(ratios_panel)):
        #    if i in splits:
        ylim  = [break_at, end_at[index]]
        #print(ylim)
        ylim2 = [0.0, break_at]
        ylimratio = (ylim[1]-ylim[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])/scale2
        ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])/scale1
        f, axes = plt.subplots( nsubs[index],
                                num="Fig. 1C"+figlabels[index],
                                sharex=True,
                                figsize=(4.5, 1.5),
                                dpi=300,
                                gridspec_kw={'height_ratios':[ylimratio, ylim2ratio]})
        axes[0].set_ylim(ylim)
        axes[1].set_ylim(ylim2)
    else:
        f, axes = plt.subplots( nsubs[index],
                                num="Fig. 1C"+figlabels[index],
                                sharex=True,
                                figsize=(4.5, 1.5),
                                dpi=300)

    if index == 0:
        axes[0].set_title("C", fontsize=16, fontweight='bold', loc='left', x=-0.3, y=1.05)
    elif index == 1:
        axes[0].set_title("C", fontsize=16, color='w', fontweight='bold', loc='left', x=-0.3, y=1.05)
    else:
        axes.set_title("C", fontsize=16, color='w', fontweight='bold', loc='left', x=-0.3, y=1.05)
    if index < 2:
        ### LABEL
        axes[1].set_ylabel(ylabels[index], fontsize=12)

        ### REMOVE SPINES
        # TOP
        axes[0].spines['top'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        # BOTTOM
        axes[0].spines['bottom'].set_visible(False)
        axes[1].spines['bottom'].set_visible(False)
        # RIGHT
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        # NO TOP TICKS
        axes[0].tick_params(labeltop='off')  # don't put tick labels at the top
        axes[0].set_xticks([])

        # I want major ticks to be every 5
        majors = np.arange(10, end_at[0]+1, 15)
        # I want minor ticks to be every 1
        minors = np.arange(10, end_at[0]+1, 5)
        # Specify tick label size
        axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)
        axes[0].tick_params(axis = 'both', which = 'minor', labelsize = 0)
        axes[0].set_yticks(majors)
        axes[0].set_yticks(minors, minor = True)

        # I want major ticks to be every 2
        majors = np.arange(0, break_at, 2)
        # I want minor ticks to be every 1
        minors = np.arange(0, break_at, scale1)
        # Specify tick label size
        axes[1].tick_params(axis = 'both', which = 'major', labelsize = 12)
        axes[1].tick_params(axis = 'both', which = 'minor', labelsize = 0)
        axes[1].set_yticks(majors)
        axes[1].set_yticks(minors, minor = True)
    elif index == 4:
        ### REMOVE SPINES
        # TOP
        axes.spines['top'].set_visible(False)
        # RIGHT
        axes.spines['right'].set_visible(False)
        axes.set_ylabel(ylabels[index], fontsize=12)
        axes.set_xlabel("Time [s]", fontsize=12)
    else:
        ### REMOVE SPINES
        # TOP
        axes.spines['top'].set_visible(False)
        # BOTTOM
        axes.spines['bottom'].set_visible(False)
        # RIGHT
        axes.spines['right'].set_visible(False)
        ### LABEL
        axes.set_ylabel(ylabels[index], fontsize=12)
        axes.set_xticks([])
    if index == 2:
        axes.set_yticks(np.arange(-400, 401, 200))


    # distance_to_patch
    #axes[0], axes[1] = brokenAxesDemo(5,25,1,5)
    lx1 = start
    lx2 = end
    if index == 0:
        axes[0].plot(data, 'k-', lw=1)
        axes[1].plot(data, 'k-', lw=1)
        axes[0].set_ylim([break_at, end_at[0]])
        axes[1].hlines(5, lx1, lx2, colors='#bbbbbb', linestyles='--', lw=1)
        axes[1].hlines(2.5, lx1, lx2, colors='#bbbbbb', linestyles='--', lw=1)
        axes[1].text(lx2+100, 5-0.5, "5 mm", color='#bbbbbb', fontsize=8)
        axes[1].text(lx2+100, 2.5-0.5, "2.5 mm", color='#bbbbbb', fontsize=8)
        axes[0].set_xlim([lx1,lx2])
        axes[1].set_xlim([lx1,lx2])
        axes[1].set_ylim([0,break_at])
    elif index == 1:
        axes[0].plot(data['head'], 'b-', lw=1)
        axes[0].plot(data['body'], 'k-', lw=1)
        axes[1].plot(data['head'], 'b-', lw=1)
        axes[1].plot(data['body'], 'k-', lw=1)
        axes[1].hlines(2., lx1, lx2, colors='#bbbbbb', linestyles='--', lw=1)
        axes[1].hlines(0.2, lx1, lx2, colors='#bbbbbb', linestyles='--', lw=1)
        axes[1].text(lx2+100, 2-0.4, "2 mm", color='#bbbbbb', fontsize=8)
        axes[1].text(lx2+100, 0.2-0.4, "0.2 mm", color='#bbbbbb', fontsize=8)
        lx1 = start
        lx2 = end
        axes[0].set_xlim([lx1,lx2])
        axes[1].set_xlim([lx1,lx2])
        axes[1].set_ylim([0,break_at])
    elif index == 2:
        axes.plot(data['speed'], 'k-', lw=1)
        #axes.plot(data['heading'], 'r-', lw=1)
        axes.hlines(125., lx1, lx2, colors='#bbbbbb', linestyles='--', lw=1)
        axes.hlines(-125, lx1, lx2, colors='#bbbbbb', linestyles='--', lw=1)
        axes.text(lx2+100, 125-40, "125 $^\circ$", color='#bbbbbb', fontsize=8)
        axes.text(lx2+100, -125-40, "-125 $^\circ$", color='#bbbbbb', fontsize=8)
        lx1 = start
        lx2 = end
        axes.set_xlim([lx1,lx2])
        axes.set_ylim([-400,400])
    elif index == 3:
        a = np.array(data)[:,0]
        dy = 0.5
        x = np.arange(lx1,lx2+1)
        _lw = 0.1
        axes.vlines(x[a==0],-dy,dy, colors='#ffffff', lw=_lw)
        axes.vlines(x[a==1],-dy,dy, colors='#c97aaa', lw=_lw)
        axes.vlines(x[a==2],-dy,dy, colors='#5bd5ff', lw=_lw)
        axes.vlines(x[a==3],-dy,dy, colors='#04bf11', lw=_lw)
        axes.vlines(x[a==4],-dy,dy, colors='#f0e442', lw=_lw)
        axes.vlines(x[a==5],-dy,dy, colors='k', lw=_lw)
        #axes.plot(data, 'k-', lw=_lw, zorder=0)
        axes.set_xlim([lx1,lx2])
        axes.set_ylim([-dy,dy])
        axes.spines['left'].set_visible(False)
        axes.set_yticks([])
    elif index == 4:
        a = np.array(data)[:,0]
        dy = 0.5
        x = np.arange(lx1,lx2+1)
        _lw = 0.1
        axes.vlines(x[a==1],-dy,dy, colors='#ffc04c', lw=_lw)
        axes.vlines(x[a==2],-dy,dy, colors='#4c8bff', lw=_lw)
        #axes.plot(data, 'k-', lw=_lw, zorder=0)
        axes.set_xlim([lx1,lx2])
        axes.set_ylim([-dy,dy])
        axes.spines['left'].set_visible(False)
        axes.set_yticks([])

    if index < 2:
        d = .005  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        b = 0.0225
        points = [0, 0.271-b, 0.715-b, 0.735-b]
        for dp in points:
            kwargs = dict(transform=axes[0].transAxes, color='#666666', clip_on=False, zorder=10, lw=1)
            axes[0].plot((dp - d, dp + d), (-2*d, 2*d), **kwargs)  # top-right diagonal (data)
            kwargs.update(transform=axes[1].transAxes)  # switch to the bottom axes
            axes[1].plot((dp - d, dp + d), (1 - 2*d, 1 + 2*d), **kwargs)  # bottom-right diagonal

        plt.tight_layout()
        axes[1].yaxis.set_label_coords(0.18, 0.45, transform=f.transFigure)
        plt.subplots_adjust(hspace=0.00)
    else:
        plt.tight_layout()
        if index == 3:
            axes.yaxis.set_label_coords(0.16, 0.42, transform=f.transFigure)

    if index == 0:
        for ax in axes:
            currpos = ax.get_position() # get the original position
            print(currpos)
    if index == 1:
        for ax in axes:
            currpos = ax.get_position() # get the original position
            print(currpos)
            #currpos.x0 += 0.1
            #ax.set_position(currpos) # set a new position
    if index > 1:
        currpos = axes.get_position() # get the original position
        print(currpos)

    return f, axes


"""
PLOTTING FIG 1D
"""
def fig_1d(data, meta):
    ### figure itself
    f = plt.figure("Fig. 1D Representative trajectory of a fly walking in the arena", figsize=(3.1, 3.1), dpi=300)
    ax = f.gca()
    ax.set_title("D", fontsize=16, fontweight='bold', loc='left', x=-0.05)
    # no axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    # visible range
    ax.set_xlim([-12, 22])
    ax.set_ylim([-20, 14])

    ### trajectory data
    subsampl=1 # subsampling, if needed
    x, y = np.array(data[data.columns[0]]), np.array(data[data.columns[1]])
    ax.plot(x[::subsampl], y[::subsampl], ls='-', lw=1, color="#888888")
    #ax.scatter(x[::subsampl], y[::subsampl], s=0.25, alpha=0.5)

    ### arena objects
    patch_color = {1: '#ffc04c', 2: '#4c8bff', 3: '#ffffff'}
    allowed = [0,2,3,4,5,6,12,13]
    zoom = False
    for i, patch in enumerate(meta.patches()):
        c = patch_color[patch["substrate"]]
        pos = (patch["position"][0], patch["position"][1]) # convert to tuple
        rad = patch["radius"]
        #plt.text(pos[0],pos[1], str(i))
        #ax.plot(pos[0], pos[1], "ro", markersize=2)
        ### plot only certain patches
        if zoom:
                ax.set_xlim([pos[0]-2.5, pos[0]+5])
                ax.set_ylim([pos[1]-2.5, pos[1]+5])
                circle = plt.Circle(pos, 2.5, edgecolor="#aaaaaa", fill=False, ls=(0,(4,4)), lw=2)
                circle.set_zorder(0)
                ax.add_artist(circle)
        if i in allowed:
            circle = plt.Circle(pos, rad, color=c, alpha=0.5)
            circle.set_zorder(0)
            ax.add_artist(circle)
        if i == 6:
            circle = plt.Circle(pos, 5., edgecolor="#aaaaaa", fill=False, ls=(0,(4,4)), lw=2)
            circle.set_zorder(0)
            ax.add_artist(circle)

    ### post adjustments & presentation
    ax.set_aspect('equal', 'datalim')
    return f, ax



if __name__ == '__main__':
    # filename of this script
    thisscript = os.path.basename(__file__).split('.')[0]
    profile = get_profile('Vero eLife 2016', 'degoldschmidt', script=thisscript)
    db = Database(get_db(profile)) # database from file
    log = Logger(profile, scriptname=thisscript)
    analysis(db)
    #db.show_data()
    log.close()
    #log.show()

    plotting(db)
