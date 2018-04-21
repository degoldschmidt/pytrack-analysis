import os
import numpy as np
import pandas as pd

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.posttracking import frameskips, get_displacements, mistracks, get_head_tail, get_corrected_flips
from pytrack_analysis.viz import plot_along, plot_fly, plot_interval, plot_overlay, plot_ts

def main():
    experiment = 'DIFF'
    user = 'degoldschmidt'
    profile = get_profile(experiment, user)

    basedir = profile.set_folder('/Users/degoldschmidt/Desktop/tracking_test_data')

    ### Define raw data structure
    colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
    #colunits = ['Datetime', 's',            's',        'px',       'px',       'rad',      'px',       'px']
    raw_data = VideoRawData(experiment, basedir)
    ### go through all session
    for i_session, video in enumerate(raw_data.videos):
        ###

        ### arena + food spots
        video.load_arena()
        ### trajectory data
        video.load_data()
        video.data.reindex(colnames)
        #video.data.center_to_arena(video.arenas)
        ### fly/experiment metadata
        #for fly_idx, fly_data in enumerate(raw_data.get_data()):

        ###
        video.unload_data()
    del profile
    """
        mistrk_list = []
        ### for each arena
        for i_arena, each_df in enumerate(raw_data.get_data()):
            ### compute head and tail positions
            each_df['head_x'], each_df['head_y'], each_df['tail_x'], each_df['tail_y'] = get_head_tail(each_df, x='body_x', y='body_y', angle='angle', major='major')
            ### compute frame-to-frame displacements
            arena = raw_data.arenas[i_arena]
            each_df['displacement'], each_df['dx'], each_df['dy'], each_df['mov_angle'], each_df['align'], each_df['acc'] = get_displacements(each_df, x='body_x', y='body_y', angle='angle')
            ### detect mistracked frames
            each_df, mistr = mistracks(each_df, i_arena, dr='displacement', major='major', thresholds=(4*8.543, 5*8.543))
            mistrk_list.append(len(mistr))

            file_id = 4 * (i_session) + i_arena
            _file = os.path.join(folders['processed'],'pixeldiff','{}_{:03d}.csv'.format(experiment, file_id))
            ### flips START-----
            df = pd.read_csv(_file, index_col='frame')
            each_df['headpx'], each_df['tailpx'] = df['headpx'], df['tailpx']
            each_df = get_corrected_flips(each_df)
        ### scale trajectories to mm
        #print(raw_data.get_data(0).head(3))
        scale = 8.543
        raw_data.set_scale('fix_scale', scale, unit='mm')
        raw_data.flip_y()
        print(mistrk_list)
        #print(raw_data.get_data(0).head(3))
        #plot_traj(raw_data, scale, time=(raw_data.first_frame, raw_data.last_frame), only='tail')
        for i_arena, each_df in enumerate(raw_data.get_data()):
            file_id = 4 * i_session + i_arena
            _file = os.path.join(folders['processed'], 'post_tracking','{}_{:03d}.csv'.format(experiment, file_id))
            out_df = each_df[['datetime', 'elapsed_time', 'frame_dt', 'body_x', 'body_y', 'head_x', 'head_y', 'tail_x', 'tail_y', 'angle', 'major', 'minor', 'flipped']]
            out_df.to_csv(_file, index_label='frame')
            meta_dict = {}
            arena = raw_data.arenas[i_arena]

            #### meta_dict save
            import yaml
            import io
            with open(os.path.join(folders['manual'],'conditions.yaml'), 'r') as stream:
                try:
                    conds = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            meta_dict['arena'] = {'x': float(arena.x), 'y': float(arena.y), 'layout': conds['arena_layout'], 'name': arena.name, 'outer': float(arena.outer), 'radius': float(arena.r), 'scale': arena.pxmm}
            meta_dict['condition'] = raw_data.condition[i_arena]
            meta_dict['datafile'] = _file
            meta_dict['datetime'] = raw_data.timestamp
            meta_dict['flags'] = {'mistracked_frames': mistrk_list[i_arena]}
            spots = arena.spots
            meta_dict['food_spots'] = [{'x': float(each.rx), 'y': float(each.ry), 'r': 1.5, 'substr': each.substrate} for each in spots]
            meta_dict['fly'] = {'genotype': conds['genotype'], 'mating': conds['mating'], 'metabolic': raw_data.condition[i_arena], 'n_per_arena': conds['num_flies'], 'sex': conds['sex']}
            meta_dict['setup'] = {'light': conds['light'], 'humidity': conds['humidity'], 'name': conds['setup'], 'room': 'behavior room', 'temperature': '25C'}
            meta_dict['video'] = {'dir': folders['videos'], 'file': raw_data.video_file, 'first_frame': int(raw_data.first_frame), 'last_frame': int(raw_data.last_frame), 'nframes': len(each_df.index), 'start_time': raw_data.starttime}
            _yaml = _file[:-4]+'.yaml'
            with io.open(_yaml, 'w', encoding='utf8') as f:
                yaml.dump(meta_dict, f, default_flow_style=False, allow_unicode=True)
    """

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
