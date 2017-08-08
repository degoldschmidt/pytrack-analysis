import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

def fig_1c(data, meta):
    ## 5 subplots (3,3,3,2,2)
    f, axes = plt.subplots( 5,
                            num="Fig. 1C",
                            sharex=True,
                            figsize=(5.5, 5),
                            dpi=150,
                            gridspec_kw={'height_ratios':[3,3,3,1,1]})
    submeta = { "xlabel" : ["", "", "", "", "Time [s]"],
                "ylabel": [ "Distance\nto patch\n[mm]",
                            "Linear\nspeed\n[mm/s]",
                            "Angular\nspeed\n[$^\circ$/s]",
                            "Etho-\ngram",
                            "Food\npatch\nvisits"],
                "keep_spines": ["L", "L", "L", "", "B"],
                "keys": [['dist_patch_0'], ['head_speed', 'body_speed'],['angular_speed'],['etho'],['visits']],
                "types": ['line', 'line', 'line', 'discrete', 'discrete']
    }
    """
    These are the styles defined for each piece of data [TODO: this should be a class in the niceplot wrapper]
    """
    styles = {  "dist_patch_0":   {
                                    "c":    'k',
                                    "ls":   '-',
                                    "lw":   1,
                                    "z":    1,
                                },
                "head_speed":   {
                                    "c":    'b',
                                    "ls":   '-',
                                    "lw":   1,
                                    "z":    1,
                                },
                "body_speed":   {
                                    "c":    'k',
                                    "ls":   '-',
                                    "lw":   1,
                                    "z":    2,
                                },
                "angular_speed":   {
                                    "c":    'k',
                                    "ls":   '-',
                                    "lw":   1,
                                    "z":    2,
                                },
                "etho":         {
                                    "c":    ['#ffffff', '#c97aaa', '#5bd5ff', '#04bf11', '#f0e442', '#000000'],
                                    "lw":   0.1,
                                },
                "visits":       {
                                    "c":    ['#ffffff', '#ffc04c', '#4c8bff'],
                                    "lw":   0.1,
                                },
    }
    ### go through each ax
    datakeys = submeta['keys']
    types = submeta['types']
    lx = (data.first_valid_index(), data.first_valid_index()+9000)
    for ix,ax in enumerate(axes):
        ### labeling
        ax.set_xlabel(submeta["xlabel"][ix])
        ax.set_ylabel(submeta["ylabel"][ix])
        ### despining
        if "T" not in submeta["keep_spines"][ix]:
            ax.spines['top'].set_visible(False)
            ax.tick_params(labeltop='off')  # don't put tick labels at the top
        if "B" not in submeta["keep_spines"][ix]:
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticks([])
            ax.set_xticklabels([])
        if "L" not in submeta["keep_spines"][ix]:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        if "R" not in submeta["keep_spines"][ix]:
            ax.spines['right'].set_visible(False)

        if "B" in submeta["keep_spines"][ix]:
            ax.set_xticks(np.arange(lx[0], lx[1]+1, 50*60))
            ax.set_xticklabels(["0", "60", "120", "180"])

        ### data limits
        ax.set_xlim([lx[0],lx[1]])

        ### data plotting
        for key in datakeys[ix]:
            try:
                sty = styles[key]
                if types[ix] == 'line':
                    ax.plot(data[key], c=sty['c'], ls=sty['ls'], lw=sty['lw'], zorder=sty['z'])
                elif types[ix] == 'discrete':
                    a = np.array(data[key])
                    dy = 0.5
                    x = np.arange(lx[0],lx[1]+1)
                    for ic, col in enumerate(sty['c']):
                        ax.vlines(x[a==ic],-dy,dy, colors=col, lw=sty['lw'])
            except KeyError:
                print('You need to define a style dictionary for \'{:}\''.format(key))
        #ax.set_ylim([break_at, end_at[0]])

        ### annotation
        if ix == 0:
            ax.hlines(5, lx[0], lx[1], colors='#bbbbbb', linestyles='--', lw=1)
            ax.hlines(2.5, lx[0], lx[1], colors='#bbbbbb', linestyles='--', lw=1)
            ax.text(lx[1]+100, 5-0.5, "5 mm", color='#bbbbbb', fontsize=8)
            ax.text(lx[1]+100, 2.5-0.5, "2.5 mm", color='#bbbbbb', fontsize=8)
            ax.set_title("C", fontsize=16, fontweight='bold', loc='left', x=-0.3, y=1.05)
        if ix == 1:
            ax.hlines(2., lx[0], lx[1], colors='#bbbbbb', linestyles='--', lw=1)
            ax.hlines(0.2, lx[0], lx[1], colors='#bbbbbb', linestyles='--', lw=1)
            ax.text(lx[1]+100, 2-0.4, "2 mm", color='#bbbbbb', fontsize=8)
            ax.text(lx[1]+100, 0.2-0.4, "0.2 mm", color='#bbbbbb', fontsize=8)


        ax.yaxis.set_label_coords(-0.1, 0.5)

    plt.tight_layout()
    return f, axes

"""
PLOTTING FIG 1D
"""
def fig_1d(data, meta):
    ### figure itself
    f = plt.figure("Fig. 1D Representative trajectory of a fly walking in the arena", figsize=(5, 5), dpi=150)
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
    subsampl=3 # subsampling, if needed
    hx, hy = np.array(data['head_x']), np.array(data['head_y'])
    bx, by = np.array(data['body_x']), np.array(data['body_y'])
    hx, hy = hx[::subsampl], hy[::subsampl]
    bx, by = bx[::subsampl], by[::subsampl]
    dx, dy = hx-bx, hy-by
    etho = np.array(data['etho'])
    etho = etho[::subsampl]
    colors =  ['#ffffff', '#c97aaa', '#5bd5ff', '#04bf11', '#f0e442', '#000000']
    zs =  [1, 4, 3, 2, 10, 10]
    for ix, col in enumerate(colors):
        currbx, currby = bx.copy(), by.copy()
        currdx, currdy = dx.copy(), dy.copy()
        currbx[etho != ix] = np.nan
        currby[etho != ix] = np.nan
        currdx[etho != ix] = np.nan
        currdy[etho != ix] = np.nan
        ax.plot(currbx, currby, lw=1, color=col, zorder=zs[ix])
        ax.quiver(currbx, currby, currdx, currdy, units='xy', width=0.15,
                   scale=1, color=col, zorder=zs[ix])
    ax.plot(hx, hy, ls='-', lw=1.5, color="#888888", zorder=11)
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

""" ARCHIVE
PLOTTING FIG 1C
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
        axes[0].plot(data['head_speed'], 'b-', lw=1)
        axes[0].plot(data['body_speed'], 'k-', lw=1)
        axes[1].plot(data['head_speed'], 'b-', lw=1)
        axes[1].plot(data['body_speed'], 'k-', lw=1)
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
        axes.plot(data['angular_speed'], 'k-', lw=1)
        #axes.plot(data['angle'], 'r-', lw=1)
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
            #print(currpos)
    if index == 1:
        for ax in axes:
            currpos = ax.get_position() # get the original position
            #print(currpos)
            #currpos.x0 += 0.1
            #ax.set_position(currpos) # set a new position
    if index > 1:
        currpos = axes.get_position() # get the original position
        #print(currpos)

    return f, axes
"""
