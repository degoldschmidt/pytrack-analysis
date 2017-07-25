from timeit import default_timer as timer
import numpy as np

class benchmark(object):

    def __init__(self, msg, fmt="%0.9g"):
        """
        Creates benchmark test object with message msg and format fmt

        args:
        * msg [str] : message for the print-out (e.g.: "Performed test XXX in")
        kwargs:
        * fmt [str] : string formatter for time (default: decimal precision to ns)
        return:
        * None
        """
        self.msg = msg
        self.fmt = fmt


    def __enter__(self):
        """ Starts default timer """
        self.start = timer()
        return self

    def __exit__(self, *args):
        """ Stops timer and prints message and times in given format """
        t = timer() - self.start
        if len(self.msg) > 0:
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t


class multibench(object):
    """
    Creates multiple (times) benchmark tests object with message msg and format fmt

    kwargs:
    * times [int] : number of times
    * msg [str] : message for the print-out (e.g.: "Performed test XXX in"); note that _SILENT may overwrite this option
    * fmt [str] : string formatter for time (default: decimal precision to ns)
    * SILENT [bool] : option for silenced tests (no individual printouts)
    """
    def __init__(self, times=1, msg="", fmt="%0.9g", SILENT=True):
        print("Starting benchmark:")
        if not SILENT:
            self.msg = msg
        else:
            self.msg = ""
            self.t = np.zeros(times)

    """
    Calling the object performs the tests for given function f

    attr:
    * f [function] : function to be tested
    """
    def __call__(self, f):
        self.f = f
        for i,thistime in enumerate(self.t):
            with benchmark(self.msg) as result:
                self.f()
            self.t[i] = result.time

    """
    Print-out when object is destroyed
    """
    def __del__(self):
        print("Test {:} for {:} repititions. Total time: {:} s. Avg: {:} s. Max: {:} s.".format(self.f.__name__, len(self.t), np.sum(self.t), np.mean(self.t), np.max(self.t)))
