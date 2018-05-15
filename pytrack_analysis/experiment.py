import os
import os.path as op
from datetime import datetime
from pytrack_analysis.cli import colorprint, flprint, prn, query_yn, bc
from pytrack_analysis.yamlio import read_yaml, write_yaml
import itertools

""" ### v0.1
Namespace constants
"""
SUFFIX_EXP = 'pytrack_exp'

""" ### v0.1
Returns setup from video file
"""
def parse_setup(video):
    setup = video.split('.')[0].split('_')[0]
    return setup

""" ### v0.1
Returns timestamp from video file
"""
def parse_time(video):
    timestampstr = video.split('.')[0][-19:]
    dtstamp = datetime.strptime(timestampstr, "%Y-%m-%dT%H_%M_%S")
    return dtstamp, timestampstr[:-3]

""" ### v0.1
Adds new video(s) and returns dict of experiment details
"""
def add(videos, _dict):
    listtimes = []
    new_videos = []
    outdict = _dict.copy()
    for video in videos:
        if video.name not in _dict['Videos']:
            new_videos.append(video)
        videotime, _ = parse_time(video.name)
        listtimes.append(videotime)
        listtimes = sorted(listtimes)
    outdict['Start date'] = datetime.strftime(listtimes[0], '%Y-%m-%d')
    outdict['End date'] = datetime.strftime(listtimes[-1], '%Y-%m-%d')
    outdict['Number of videos'] = len(videos)
    outdict['Videos'] = [video.name for video in videos]

    for video in new_videos:
        outdict['Conditions'][video.name] = set_conditions(video, variables=outdict['Variables'])

    uniques = []
    for video in videos:
        conds = outdict['Conditions'][video.name]
        condslist = [v for k, v in conds.items()]
        for i in range(4):
            fly_cond = [el[i] for el in condslist]
            if fly_cond not in uniques:
                uniques.append(fly_cond)
    outdict['Number of conditions'] = len(uniques)
    show(outdict)

    outfile = '{}_{}.yaml'.format(SUFFIX_EXP, outdict['ID'])
    if query_yn('Confirm and save experiment yaml file {}?'.format(outfile), default='yes'):
        write_yaml(op.join(video.dir, outfile), outdict)
        return outdict
    else:
        if query_yn('Want to restart adding video from registry?'.format(outfile), default='no'):
            return add(videos, _dict)
        else:
            return _dict

""" ### v0.1
Removes new video(s) and returns dict of experiment details
"""
def remove(videos, _dict):
    listtimes = []
    new_videos = []
    outdict = _dict.copy()
    for video in _dict['Videos']:
        if video not in [vid.name for vid in videos]:
            new_videos.append(video)
    for video in videos:
        videotime, _ = parse_time(video.name)
        listtimes.append(videotime)
        listtimes = sorted(listtimes)
    outdict['Start date'] = datetime.strftime(listtimes[0], '%Y-%m-%d')
    outdict['End date'] = datetime.strftime(listtimes[-1], '%Y-%m-%d')
    outdict['Number of videos'] = len(videos)
    outdict['Videos'] = [video.name for video in videos]

    for video in new_videos:
        print(video)
        del outdict['Conditions'][video]

    uniques = []
    for video in videos:
        conds = outdict['Conditions'][video.name]
        condslist = [v for k, v in conds.items()]
        for i in range(4):
            fly_cond = [el[i] for el in condslist]
            if fly_cond not in uniques:
                uniques.append(fly_cond)
    outdict['Number of conditions'] = len(uniques)
    show(outdict)

    outfile = SUFFIX_EXP+outdict['ID']+'.yaml'
    if query_yn('Confirm and save experiment yaml file {}?'.format(outfile), default='yes'):
        write_yaml(op.join(video.dir, outfile), outdict)
        return outdict
    else:
        if query_yn('Want to restart removing video from registry?'.format(outfile), default='no'):
            return remove(videos, _dict)
        else:
            return _dict

""" ### v0.1
Returns dict of experiment details
"""
def register(videos):
    outdict = {}
    listtimes = []
    for video in videos:
        videotime, _ = parse_time(video.name)
        listtimes.append(videotime)
        listtimes = sorted(listtimes)
    outdict['Title'] = input('Title of the experiment: ')
    ID = ''
    while len(ID) != 4:
        ID = input('Four-letter ID: ')
    outdict['ID'] = ID.upper()
    outdict['Start date'] = datetime.strftime(listtimes[0], '%Y-%m-%d')
    outdict['End date'] = datetime.strftime(listtimes[-1], '%Y-%m-%d')
    outdict['Number of videos'] = len(videos)
    outdict['Videos'] = [video.name for video in videos]

    outdict['Constants'] = set_constants(outdict['Title'])
    outdict['Variables'], outdict['all_conditions'] = set_variables(outdict['Title'])
    outdict['Conditions'] = {}
    for video in videos:
        outdict['Conditions'][video.name] = set_conditions(video, variables=outdict['Variables'], shortcuts=outdict['all_conditions'])

    uniques = []
    for video in videos:
        conds = outdict['Conditions'][video.name]
        condslist = [v for k, v in conds.items()]
        for i in range(4):
            fly_cond = [el[i] for el in condslist]
            if fly_cond not in uniques:
                uniques.append(fly_cond)
    outdict['Number of conditions'] = len(uniques)
    show(outdict)

    outfile = SUFFIX_EXP+outdict['ID']+'.yaml'
    if query_yn('Confirm and save experiment yaml file {}?'.format(outfile), default='yes'):
        write_yaml(op.join(video.dir, 'pytrack_res', outfile), outdict)
        return outdict
    else:
        return register(videos)

""" ### v0.1
Sets conditions of video
"""
def set_conditions(video, variables=None, shortcuts=None):
    writing = True
    condition_dict = {}
    k = None
    print('\nEnter'+bc.OKBLUE+' conditions '+bc.ENDC+'for'+bc.OKBLUE+' {}'.format(video.name)+bc.ENDC)
    if shortcuts is None:
        for k in variables.keys():
            lv = len(v)
            for i in range(4-lv):   ### if less then 4 conditions are given, last condition will be repeated
                v.append(v[-1])
            for each in v:
                if each not in variables[k]:
                    print('Warning: {} not found in possible values for {}.'. format(each, k))
                    return set_conditions(video, variables=variables)
    else:
        v = input('Enter shortcut index (multiple values are separated by whitespace): ')
        v = v.split(' ')
        for i, el in enumerate(v):
            if el.isdigit():
                index = int(el)
                if index >= 0 and index < len(shortcuts):
                    v[i] = shortcuts[index]
                else:
                    print('Warning: index {} out of range for {}.'. format(index, shortcuts))
                    return set_conditions(video, variables=variables, shortcuts=shortcuts)
        for k in variables.keys():
            condition_dict[k] = [el[k] for el in v]
        k = None
    return condition_dict

""" ### v0.1
Sets constants of experiment
"""
def set_constants(exptitle):
    writing = True
    constants_dict = {}
    k = None
    print('\nEnter'+bc.OKBLUE+' constants '+bc.ENDC+'for'+bc.OKBLUE+' {}'.format(exptitle)+bc.ENDC)
    while writing:
        if k == None:
            k = input('\nPlease type constants key ("enter to quit"): ')
        if k == '':
            writing = False
        else:
            v = input('Value for key'+bc.BOLD+' {}'.format(k)+bc.ENDC+': '.format(k))
            constants_dict[k] = v
            k = None
    return constants_dict

""" ### v0.1
Sets variables of experiment
"""
def set_variables(exptitle):
    writing = True
    variables_dict = {}
    k = None
    print('\nEnter'+bc.OKBLUE+' variables '+bc.ENDC+'for'+bc.OKBLUE+' {}'.format(exptitle)+bc.ENDC)
    while writing:
        if k == None:
            k = input('\nPlease type variables key ("enter to quit"): ')
        if k == '':
            writing = False
        else:
            v = input('Possible values for key'+bc.BOLD+' {} '.format(k)+bc.ENDC+'(multiple values are separated by whitespace): '.format(k))
            v = v.split(' ')
            variables_dict[k] = v
            k = None
    print('All possible conditions:')
    elms = []
    keys = []
    for k, v in variables_dict.items():
        keys.append(k)
        elms.append(v)
    all_conds = list(itertools.product(*elms))
    all_conds_out = []
    for i, cond in enumerate(all_conds):
        print('{}:\t{}'.format(i, cond))
        all_conds_out.append({k: el for k, el in zip(keys, cond)})
    print(all_conds_out)
    return variables_dict, all_conds_out

def show(_dict):
    print(bc.BOLD+'\nExperiment summary:\n'+bc.ENDC)
    for k, v in _dict.items():
        if type(v) is dict:
            print("{}:".format(k))
            for k2, v2 in v.items():
                print("\t{}:\t{}".format(k2, v2))
        else:
            print("{}:\t{}".format(k, v))
    print()
