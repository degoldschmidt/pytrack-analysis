# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
matplotlib.rcParams['font.weight'] = 'light'
import matplotlib.pyplot as plt
import sys

from plotting import swarmbox

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

def fig_1c(data, meta):
    ## 5 subplots (3,3,3,2,2)
    f, axes = plt.subplots( 5,
                            num="Fig. 1C",
                            sharex=True,
                            figsize=(4.5, 3.5), ##5.5,5
                            dpi=150,
                            gridspec_kw={'height_ratios':[2,2,2,1,1]})
    submeta = { "xlabel" : ["", "", "", "", "Time [s]"],
                "ylabel": [ "Distance\nto patch\n[mm]",
                            "Linear\nspeed\n[mm/s]",
                            "Angular\nspeed\n[\xb0/s]",
                            "Etho-\ngram",
                            "Food\npatch\nvisits"],
                "keep_spines": ["L", "L", "L", "", "B"],
                "keys": [['dist_patch_0'], ['head_speed', 'body_speed'],['angular_speed'],['ethogram'],['visits']],
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
                "ethogram":         {
                                    "c":    ['#ffffff', '#c97aaa', '#5bd5ff', '#04bf11', '#f0e442', '#000000'],
                                    "lw":   0.1,
                                },
                "visits":       {
                                    "c":    ['#ffffff', '#ffc04c', '#4c8bff'],
                                    "cs":    ['#ffffff', '#f6d18b', '#75a4fb'],
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
                    enc = np.array(data['encounters'])
                    dy = 0.5
                    x = np.arange(lx[0],lx[1]+1)
                    for ic, col in enumerate(sty['c']):
                        ax.vlines(x[a==ic],-dy,dy, colors=col, lw=sty['lw'])
                        if key == "visits":
                           #ax.plot(x, enc, "k-", alpha=0.5, lw=0.5) ### TODO
                           if ic > 0:
                              ax.vlines(x[enc==ic], 0.05+dy+np.zeros(x[enc==ic].shape), 0.1+dy+np.zeros(x[enc==ic].shape), colors=sty['cs'][ic], alpha=0.5, lw=sty['lw']) ### TODO
            except KeyError:
                print('You need to define a style dictionary for \'{:}\''.format(key))
        #ax.set_ylim([break_at, end_at[0]])

        ### annotation
        if ix == 0:
            ax.hlines(5, lx[0], lx[1], colors='#818181', linestyles='--', lw=1)
            ax.hlines(2.5, lx[0], lx[1], colors='#818181', linestyles='--', lw=1)
            ax.text(lx[1]+100, 5-0.5, "5 mm", color='#818181', fontsize=8)
            ax.text(lx[1]+100, 2.5-0.5, "2.5 mm", color='#818181', fontsize=8)
        if ix == 1:
            ax.hlines(2., lx[0], lx[1], colors='#818181', linestyles='--', lw=1)
            ax.hlines(0.2, lx[0], lx[1], colors='#818181', linestyles='--', lw=1)
            ax.text(lx[1]+100, 2-0.5, "2 mm", color='#818181', fontsize=8)
            ax.text(lx[1]+100, 0.2-1, "0.2 mm", color='#818181', fontsize=8)
        if ix == 2:
            ax.hlines(125., lx[0], lx[1], colors='#818181', linestyles='--', lw=1)
            ax.hlines(-125., lx[0], lx[1], colors='#818181', linestyles='--', lw=1)
            ax.text(lx[1]+100, 125.-25, "125 \xb0/s", color='#818181', fontsize=8)
            ax.text(lx[1]+100, -125.-25, "-125 \xb0/s", color='#818181', fontsize=8)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    plt.tight_layout(h_pad=-0.1,rect=(0,0,0.96,1.))
    #axes[0].set_title("C", fontsize=16, fontweight='bold', loc='left', x=-0.3, y=.8)
    plt.close("all")
    return f, axes

"""
PLOTTING FIG 1D
"""
def fig_1d(data, meta):
    ### figure itself
    f = plt.figure("Fig. 1D Representative trajectory of a fly walking in the arena", figsize=(3.5, 3.5), dpi=150) # 5,5
    ax = f.gca()
    #ax.set_title("D", fontsize=16, fontweight='bold', loc='left', x=-0.05)
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
    etho = np.array(data['ethogram'])
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
    allowed = [0,2,3,4,5,6,12,13,18]
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
        if i in allowed:
            circle = plt.Circle(pos, 5., edgecolor="#aaaaaa", fill=False, ls=(0,(4,4)), lw=1)
            circle.set_zorder(0)
            ax.add_artist(circle)

    ### post adjustments & presentation
    ax.set_aspect('equal', 'datalim')
    plt.close("all")
    return f, ax

def fig_1e_h(data, meta):
    """
    Fig. 1E-H
    """
    #### USED FOR PLOTTING
    import seaborn as sns; sns.set(color_codes=True)
    sns.set_style('ticks')
    import scipy.stats as scistat

    ## plot testing
    f, axes = plt.subplots( 2, 3, num="Fig. 1E/G", figsize=(8.,3), dpi=150, gridspec_kw={'width_ratios':[1,1.5,1.5]}) ##9,3.5 # ratio 1,1.5,1.5 # COLUMNS: 0=total durations, 1=histogram yeast, 2=histogram sucrose
    #print("Figsize [inches]: ", f.get_size_inches())
    substrate_colors = ['#ffc04c', '#4c8bff']
    title_label = ["Virgin", "Mated"]
    panel_label = ["E", "G", "F", "H"]
    movel_label = [-1.15,-0.4]
    ticks = [[0, 5, 1], [0,25,5]]
    tick_label = [ [" 0", " 1", "", " 3", "", "    5"], ["0", "5", "", "15", "", "25"]]
    lims = [[-0.5,5], [-1.2,25]] ### low = 0 - high/10
    staty = [4.5, 9.5]

    for ix,ax in enumerate(axes[:,0]):
      data_eg = data[ix].drop_duplicates("total_length [min]")
      ### main data (box, swarm, median line)
      ax = sns.boxplot(x="behavior", y="total_length [min]", data=data_eg, order = ["Yeast", "Sucrose"], palette=substrate_colors, saturation=1.0, width=0.35, linewidth=0.0, boxprops=dict(lw=0.0), showfliers=False, ax=ax)
      ax = sns.swarmplot(x="behavior", y="total_length [min]", data=data_eg, order = ["Yeast", "Sucrose"], size=2, color='#666666', ax=ax)
      yeast_data = np.array(data_eg.query("behavior == 'Yeast'")["total_length [min]"])
      sucrose_data = np.array(data_eg.query("behavior == 'Sucrose'")["total_length [min]"])
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
      ax.text(0.5, y_max + abs(y_max - y_min)*0.2, "p = {:.3g}".format(pvalue), horizontalalignment='center', verticalalignment='center', fontsize=8)
      #print("pvalue:", pvalue)

      ### figure aesthetics
      ax.set_xlabel("") # remove xlabel
      ax.set_ylabel(title_label[ix]+"\n\nTotal duration\nof food micro-\nmovements [min]") # put a nice ylabel
      ax.set_xticklabels(ax.get_xticklabels(), rotation=30, x=-2, y=0.15) # rotates the xlabels by 30º
      #print(ax.get_xlim(), ax.get_ylim())
      #ax.set_aspect((ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]))
      ax.set_ylim(lims[ix])
      ax.set_yticks(np.arange(ticks[ix][0],ticks[ix][1]+1, ticks[ix][2]))
      #ax.set_yticklabels(tick_label[ix])
      ax.get_xaxis().set_tick_params(color='#ffffff') # no xticks markers
      sns.despine(ax=ax, bottom=True, trim=True)

    for ix,ax in enumerate(axes[:,1:]):
      hist_data = [ np.array(data[ix].query("behavior == 'Yeast'")["length [s]"]), np.array(data[ix].query("behavior == 'Sucrose'")["length [s]"]) ]
      for jx,a in enumerate(ax):
         #weights = np.ones_like(hist_data[jx])/float(len(hist_data[jx]))
         #a.hist(hist_data[jx], align='left', rwidth=0.9, color=substrate_colors[jx])
         #print("max:", np.max(hist_data[jx]))
         _bins = np.arange(0,24.2,2.2) # [ 0, 2.2, 4.4, 6.6, 8.8, 11, 13.2, 15.4, 17.6, 19.8, 22 ]
         _, bins, patches = a.hist(np.clip(hist_data[jx], _bins[0], 20), bins=_bins, align='left', rwidth=0.9, color=substrate_colors[jx])
         #a = sns.distplot( np.clip(hist_data[jx], _bins[0], _bins[-1]) , bins=_bins, hist=True, color=substrate_colors[jx], ax=a)
         a.set_xlim([-1.5,22.5])
         #a.set_ylim([0.,1.])
         #a.set_yticklabels(["0", "0.5", "1"])
         sns.despine(ax=a, trim=True, offset=2)
         ### figure aesthetics
         a.set_xlabel("Micromovement duration [s]") # remove xlabel
         a.set_ylabel("Occurences,\nnormalized") # put a nice ylabel

    plt.tight_layout()
    axes[0,0].yaxis.set_label_coords(-0.35, 0.5)
    for colix in range(axes.shape[1]-1):
      for ix,ax in enumerate(axes[:,colix]):
         pass#ax.set_title(panel_label[ix+2*colix], fontsize=16, fontweight='bold', loc='left', x=movel_label[colix])
    plt.close("all")
    return f, ax

def fig_2(_data, _meta):
    """
    Fig. 2
    ##MATING COLORS
    #dd1c77	(221,28,119)
    #f4b7d2	(244,183,210)
    #2ca25f	(44,162,95)
    #99d8b3	(153,216,179)
    #e5f9e9	(229,249,233)
    """
    #### USED FOR PLOTTING
    import seaborn as sns; sns.set(color_codes=True)
    sns.set_style('ticks')
    import scipy.stats as scistat

    ## data unpacking
    etho_data = _data['D']
    sequence_data = _data['C']
    ## time series limits (120 mins = 50[frames/s]*60[secs/min]*120 = 360000 frames)
    lx = (0, len(etho_data.index))

    ### sort the etho_data by ascending sum of yeast micromovements and then split them by "metabolic"
    etho_sum = {col: np.sum(etho_data[col]==4) for col in etho_data.columns} # creates a dictionary with column name as key and sum of YEAST micromovements as value
    etho_sum = [entry[0] for entry in sorted(etho_sum.items(), key=lambda x: x[1])] # create list with tuples sorted by values (sum of Y micromov), then take only sessions
    indices = [_meta.session(entry).condition-1 for entry in etho_sum] # list of indices from Condition = mating and metabolic states (sorted by sessions as above) [0: ]
    split_etho_sum = [[entry for i, entry in enumerate(etho_sum) if indices[i]==ix] for ix in range(5)] # splits session names into the five conditions (sorted)
    nethos = [len(entry) for entry in split_etho_sum] # lengths of each split (how many ethos per condition)
    max_nethos = max(nethos)

    colors = ['#dd1c77', '#f9c6dd', '#2ca25f', '#99d8b3', '#b9eec3']
    conds = {"mating": [2,2,1,1,1], "metabolic": [3,2,3,1,2]}

    f, axes = plt.subplots( 3, 5, num="Fig. 2C-E", figsize=(8.,5.), sharey=False, gridspec_kw={'height_ratios':[1,1.2,1]})

    for (row, col), ax in np.ndenumerate(axes):
        max_dur = 90.
        div = 30.
        """
        This is the yeast micromovent histograms (C)
        """
        if row == 0:

            ## data selecting
            currdata = sequence_data.drop_duplicates("session")
            currdata = currdata.query("mating == "+str(conds["mating"][col]))
            currdata = currdata.query("metabolic == "+str(conds["metabolic"][col]))
            ## data plotting
            ax.set_ylim([-2.,max_dur + 2.]) # this is needed for swarmplot to work!!!
            ax = sns.boxplot(x="state", y="total_length [min]", data=currdata, color=colors[col], saturation=1.0, width=0.35, linewidth=0.0, boxprops=dict(lw=0.0), showfliers=False, ax=ax)
            ax = sns.swarmplot(x="state", y="total_length [min]", data=currdata, size=2, color='#666666', ax=ax)
            currdata = np.array(currdata["total_length [min]"])
            median = np.median(currdata)
            dx = 0.3
            ax.hlines(median, -dx, dx, lw=1, zorder=10)

            ## figure aesthetics
            ax.set_yticks(np.arange(0, max_dur+1, div))
            if col == 0:
                ax.set_ylabel("Total duration\nof yeast micro-\nmovements [min]")
                sns.despine(ax=ax, bottom=True, trim=True)
            else:
                sns.despine(ax=ax, bottom=True, left=True, trim=True)
                ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        """
        This is the ethogram stacks
        """
        if row == 1:
            smpl=100
            if col == 0:
                ax.set_ylabel("Ethogram index")
            for ieth, etho in enumerate(split_etho_sum[col]): # go through all entries per column
                a = np.array(etho_data.loc[:, etho])[::smpl] # data to np.array
                x = np.arange(lx[0],lx[1],smpl)
                for ic, color in enumerate(['#ffffff', '#c97aaa', '#5bd5ff', '#04bf11', '#f0e442', '#000000']):
                    ax.vlines(x[a==ic],ieth,ieth+1, colors=color, lw=0.1)
            ax.set_ylim([0, max_nethos])
            ax.set_yticks([nethos[col], nethos[col]+1, nethos[col]])
            ax.get_yaxis().set_tick_params(length=2, pad=0.0)
            ax.set_xticks([lx[0], lx[1]/2, lx[1]])
            ax.set_xticklabels(["0", "60", "120"])
            sns.despine(ax=ax, left=True, trim=True)

        """
        This is the cumulative duration plot
        """
        if row == 2:
            currdata = sequence_data
            currdata = currdata.query("mating == "+str(conds["mating"][col]))
            currdata = currdata.query("metabolic == "+str(conds["metabolic"][col]))
            #print(currdata[["frame_index", "cumulative_length [min]"]].head(10))

            ## data plotting
            ax = cum_plot(currdata, time="frame_index", unit="session", value="cumulative_length [min]", color=colors[col], upper=len(etho_data.index), ax=ax)

            ## figure aesthetics
            ax.set_xlabel("Time [min]")
            ax.set_xticks([lx[0], lx[1]/2, lx[1]])
            ax.set_xticklabels(["0", "60", "120"])
            ax.set_ylim([-2.,max_dur + 2.])
            ax.set_yticks(np.arange(0, max_dur+1, div))
            if col == 0:
                ax.set_ylabel("Cumulative duration\nof yeast micro-\nmovements [min]")
                sns.despine(ax=ax, offset=2, trim=True)
            else:
                ax.get_yaxis().set_visible(False)
                sns.despine(ax=ax, offset=2, left=True, trim=True)




    plt.tight_layout(w_pad=-0.05)
    plt.close("all")
    return (f, axes)

def fig_3(data):
      import seaborn as sns; sns.set(color_codes=True)
      import matplotlib.font_manager as fm
      import matplotlib.cm as cm
      import matplotlib.colors as colors

      sns.set_style('ticks')
      f, axes = plt.subplots(2,3, figsize=(8,6), dpi=300)
      print("Figure size:", f.get_size_inches())
      swarmcolors = ['#dd1c77', '#f9c6dd', '#2ca25f', '#99d8b3', '#b9eec3']
      axes[0, 0] = swarmbox.swarmbox(x=['Mated', 'AA\npre-diet'], y='Total duration\nof yeast visits\n[min]', data=data['A'], colors=swarmcolors, ax=axes[0, 0], order=[[False, True], ['++', '+', '-']])
      axes[0, 1] = swarmbox.swarmbox(x=['Mated', 'AA\npre-diet'], y='Rate of yeast\nencounters\n[1/min]', data=data['B'], colors=swarmcolors, ax=axes[0, 1], order=[[False, True], ['++', '+', '-']])
      axes[0, 2] = swarmbox.swarmbox(x=['Mated', 'AA\npre-diet'], y='Rate of yeast\nencounters\n[1/min]', data=data['B'], colors=swarmcolors, ax=axes[0, 2], order=[[False, True], ['++', '+', '-']])
      #axes[0, 2] = swarmbox.swarmbox(x=['Mated', 'AA\npre-diet'], y='Probability of\nstopping at a\nyeast patch', data=data['C'], colors=colors, ax=axes[0, 2], order=[[False, True], ['++', '+', '-']])
      axes[1, 0] = swarmbox.swarmbox(x=['Mated', 'AA\npre-diet'], y='Mean duration\nof yeast visits\n[min]', data=data['D'], colors=swarmcolors, ax=axes[1, 0], order=[[False, True], ['++', '+', '-']])
      axes[1, 1] = plt.subplot2grid((2, 3), (1, 1), colspan=2)
      x = 'Number of yeast visits'
      y = 'Mean duration\nof yeast visits\n[min]'
      axes[1, 1].plot(data['E'].query("metabolic == '++'")[x], data['E'].query("metabolic == '++'")[y], '.', markersize=10, color=swarmcolors[2], zorder=3, label='AA++')
      axes[1, 1].plot(data['E'].query("metabolic == '+'")[x], data['E'].query("metabolic == '+'")[y], '.', markersize=10, color=swarmcolors[3], zorder=3, label='AA+')
      axes[1, 1].plot(data['E'].query("metabolic == '-'")[x], data['E'].query("metabolic == '-'")[y], '.', markersize=10, color=swarmcolors[4], zorder=3, label='AA-')
      fontfile = "C:\\Windows\\Fonts\\Quicksand-Regular.ttf"
      if sys.platform == "darwin":
          fontfile = "/Users/degoldschmidt/Library/Fonts/Quicksand-Regular.ttf"
      textprop = fm.FontProperties(fname=fontfile)
      axes[1, 1].set_xlim([0.,420])
      axes[1, 1].set_ylim([-0.3,3.])
      sns.despine(ax=axes[1, 1], trim=True, offset=2.5)
      isolines = [5, 20, 35, 50, 65, 80, 95]
      lines_col = cm.get_cmap("Reds", len(isolines)+1)
      listcolors = [colors.rgb2hex(lines_col(i)[:3]) for i in range(lines_col.N)]

      xline = np.arange(1, axes[1, 1].get_xlim()[1], axes[1, 1].get_xlim()[1]/1000)
      for ix, iso in enumerate(isolines):
          axes[1, 1].plot(xline, iso/xline, '-', color=listcolors[ix+1])
      axes[1, 1].set_xlim([0.,420])
      axes[1, 1].set_ylim([-0.3,3.])
      axes[1, 1].set_xlabel(x, fontproperties=textprop, fontsize=12)
      axes[1, 1].set_ylabel(y, fontproperties=textprop, fontsize=12)
      axes[1, 1].legend()
      for xtic in axes[1, 1].get_xticklabels():
          xtic.set_fontproperties(textprop)
      for ytic in axes[1, 1].get_yticklabels():
          ytic.set_fontproperties(textprop)
      axes[1, 1].set_position([0.45, 0.125, 0.48, 0.312]) #0.42
      axes[1, 2].axis("off")
      return (f, axes)

def cum_plot(data, time=None, unit=None, value=None, color=None, estimator=np.median, upper=360000, ax=None):
    ## step 0: reduce data
    df = data[[time, unit, value]]
    ## step 1: define array of frames
    new_x = np.arange(upper) ### TODO
    med_y = np.zeros(len(new_x))

    ## step 2: groupby unit -> list
    gp = df.groupby(unit)
    new_y = []
    # groups() returns a dict with unit:indices as k:v pair
    for unit, indices  in gp.groups.items():
        selected_unit = df.loc[indices]
        x0 = np.array(selected_unit[time])
        y0 = np.array(selected_unit[value])
        ## step 3: interpolate to fit frame array -> list of functions
        from scipy import interpolate
        ## add bounds
        x0b = np.append(0, x0)
        x0b = np.append(x0b, int(len(new_x)))
        y0b = np.append(0.0, y0)
        y0b = np.append(y0b, y0[-1])
        f = interpolate.interp1d(x0b, y0b, kind='zero')    # interpolation function
        new_y.append(f(new_x))
        ## step 4: plot interpolated signals over frames with gray solid lines
        ax.plot(new_y[-1], c="#666666", alpha=0.5, lw=1)
    ## step 5: plot median of signals along frames axis with given color
    median = estimator(np.array(new_y), axis=0, overwrite_input=True)
    ax.plot(median, c=color, lw=2)
    return ax
