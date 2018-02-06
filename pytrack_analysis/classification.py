"""
Classification class: loads kinematics data and metadata >> processes and returns classification data
"""
class Classification(Node):

    def __init__(self):
        pass

    def classify_behavior(self, _kinedata, _meta):
        ## 1) smoothed head: 2 mm/s speed threshold walking/nonwalking
        ## 2) body speed, angular speed: sharp turn
        ## 3) gaussian filtered smooth head (120 frames): < 0.2 mm/s
        ## 4) rest of frames >> micromovement
        """
        {       0: "resting",
                1: "micromovement",
                2: "walking",
                3: "sharp turn",
                4: "yeast micromovement",
                5: "sucrose micromovement"}
        """
        head_pos = _kinedata[['head_x', 'head_y']]
        speed = np.array(_kinedata["smooth_head_speed"])
        bspeed = np.array(_kinedata["smooth_body_speed"])
        smoother = np.array(_kinedata["smoother_head_speed"])
        turn = np.array(_kinedata["angular_speed"])
        all_spots = ['distance_patch_'+str(ix) for ix in range(12)]
        aps = np.array(_kinedata[all_spots]).T
        amin = np.amin(aps, axis=0) # all patches minimum distance
        imin = np.argmin(aps, axis=0) # number of patch with minimum distance

        ethogram = np.zeros(speed.shape, dtype=np.int) - 1 ## non-walking/-classified
        ethogram[speed > 2] = 2      ## walking
        ethogram[speed > 20] = 6      ## jumps or mistracking

        mask = (ethogram == 2) & (bspeed < 4) & (np.abs(turn) >= 125.)
        ethogram[mask] = 3           ## sharp turn

        ethogram[smoother <= 0.2] = 0 # new resting

        ethogram[ethogram == -1] = 1 # new micromovement

        ethogram = self.two_pixel_rule(ethogram, head_pos, join=[1])

        visits = np.zeros(ethogram.shape)
        encounters = np.zeros(ethogram.shape)
        encounter_index = np.zeros(ethogram.shape, dtype=np.int) - 1

        substrate_dict = {'10% yeast':1, '20 mM sucrose': 2}
        substrates = np.array([substrate_dict[each_spot['substrate']] for each_spot in _meta['food_spots']])
        visit_mask = (amin <= 2.5) & (ethogram == 1)    # distance < 2.5 mm and Micromovement
        visits[visit_mask] = substrates[imin[visit_mask]]
        encounters[amin <= 3] = substrates[imin[amin <= 3]]
        encounter_index[amin <= 3] = imin[amin <= 3]

        for i in range(1, amin.shape[0]):
            if encounter_index[i-1] >= 0:
                if visits[i-1] > 0:
                    if aps[encounter_index[i-1], i] <= 5. and visits[i] == 0:
                        visits[i] = visits[i-1]
                if encounters[i-1] > 0:
                    if aps[encounter_index[i-1], i] <= 5. and encounters[i] == 0:
                        encounters[i] = encounters[i-1]
                        encounter_index[i] = encounter_index[i-1]
        visits = self.two_pixel_rule(visits, head_pos, join=[1,2])
        encounters = self.two_pixel_rule(encounters, head_pos, join=[1,2])

        mask_yeast = (ethogram == 1) & (visits == 1) & (amin <= 2.5)    # yeast
        mask_sucrose = (ethogram == 1) & (visits == 2) & (amin <= 2.5)  # sucrose
        ethogram[mask_yeast] = 4     ## yeast micromovement
        ethogram[mask_sucrose] = 5   ## sucrose micromovement

        return  ethogram, visits, encounters, encounter_index

    def rle(self, inarray):
            """ run length encoding. Partial credit to R rle function.
                Multi datatype arrays catered for including non Numpy
                returns: tuple (runlengths, startpositions, values) """
            ia = np.array(inarray)                  # force numpy
            n = len(ia)
            if n == 0:
                return (None, None, None)
            else:
                y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
                i = np.append(np.where(y), n - 1)   # must include last element posi
                z = np.diff(np.append(-1, i))       # run lengths
                p = np.cumsum(np.append(0, z))[:-1] # positions
                return(z, p, ia[i])


    def two_pixel_rule(self, _dts, _pos, join=[]):
        _pos = np.array(_pos)
        for j in join:
            segm_len, segm_pos, segm_val = self.rle(_dts) #lengths, pos, behavior_class = self.rle(_X[col])
            for (length, start, val) in zip(segm_len, segm_pos, segm_val):
                if start == 0 or start+length == len(_dts):
                    continue
                if val not in join and _dts[start-1] == j and _dts[start+length] == j:
                    dist_vector = _pos[start:start+length,:] - _pos[start,:].transpose()
                    lens_vector = np.linalg.norm(dist_vector, axis=1)
                    if np.all(lens_vector <= 2*0.15539): ## length <= 2 * 0.15539
                        _dts[start:start+length] = j
        if len(join) == 0:
            print("Give values to join segments.")
            return _dts
        else:
            return _dts
