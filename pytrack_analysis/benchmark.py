""" Classes for performing benchmark tests. """
# -*- coding: utf-8 -*-
# author: degoldschmidt
# date: 25/8/2017
# -*- reviewed
# -*- in use
from timeit import default_timer as timer
import numpy as np
import os, sys

__all__ = ["Benchmark", "Multibench"]

class Benchmark(object):
    """ Class used for performing a single benchmark test of a given function and measuring time performance """

    def __init__(self, msg, fmt="%0.9g"):
        """
        Creates benchmark test object with message msg and format fmt

        Args
        ----
        msg : string
            Message for the print-out (e.g.: "Performed test XXX in")

        Kwargs
        ------
        fmt : string
            String formatter for printing out the time (default: decimal precision to ns)
        """
        self.msg = msg
        self.fmt = fmt


    def __enter__(self):
        """
        Starts default timer

        Returns
        -------
        itself : object
            For whatever reason...
        """
        self.start = timer()
        return self

    def __exit__(self, *args):
        """
        Stops timer and prints message and times in given format

        Args
        ----
        args :
            Variable number of arguments, but not really needed?!
        """
        t = timer() - self.start
        if len(self.msg) > 0:
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t


class Multibench(object):
    """ Class used for performing multiple benchmark test of a given function and measuring time performance """

    def __init__(self, msg, times=1, fmt="%0.9g", SILENT=True):
        """
        Creates an object that when called, performs benchmark test for variable number of repititions

        Args
        ----
        msg : string
            Message for the print-out (e.g.: "Performed test XXX in")
            NOTE: _SILENT overwrites this option.

        Kwargs
        ------
        times : integer
            Number of times to repeat benchmark test (default: 1)
        fmt : string
            String formatter for time (default: decimal precision to ns)
        SILENT : boolean
            Option for silenced tests using ``sys.stdout = open(os.devnull, "w")`` (default: True)
        """
        print("*** RUNNING BENCHMARK ***")
        print("#times: {}\nSTDOUT silenced: {}\n***".format(times, SILENT))
        self.t = np.zeros(times)
        self.stdout = sys.stdout
        self.silenced = SILENT
        if not SILENT:
            self.msg = msg
        else:
            self.msg = ""
            sys.stdout = open(os.devnull, "w")

    def __call__(self, f):
        """
        Calling the object performs the tests for given function f

        Args:
        f : callable
            Function to be tested
        """
        self.f = f
        for i,thistime in enumerate(self.t):
            t_end = "\t" if self.silenced else "\n"
            print("#{}".format(i+1), end=t_end, file=self.stdout, flush=True)
            with benchmark(self.msg) as result:
                self.f()
            self.t[i] = result.time
            print("= {} s".format(result.time), file=self.stdout)
        sys.stdout = self.stdout

    def __del__(self):
        """
        Print-out when object is destroyed
        """
        print("Test {:} for {:} repititions. Total time: {:} s. Avg: {:} s. Max: {:} s.".format(self.f.__name__, len(self.t), np.sum(self.t), np.mean(self.t), np.max(self.t)), file=sys.stdout)
