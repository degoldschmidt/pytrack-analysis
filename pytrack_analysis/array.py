import numpy as np

def rle(inarray, dt=None):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray, dtype=np.int32)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        if dt is None:
            return(z, p, ia[i]) # simply return array runlengths
        else:
            try:
                dt = np.array(dt)   # force numpy
                l = np.zeros(z.shape) ## real time durations
                for j,_ in enumerate(p[:-1]):
                    l[j] = np.sum(dt[p[j]:p[j+1]])
                return(z, p, ia[i], l) # return array runlengths & real time durations
            except TypeError:
                print('Your array is invalid')
