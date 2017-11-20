import csv
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox, filedialog
from tkinter import ttk
from datetime import datetime as dt
import os
import codecs, json, yaml
import time
from string import Template

class DeltaTemplate(Template):
    delimiter = "%"

def get_created(filepath):
    try:
        return time.ctime(os.path.getctime(filepath))
    except OSError:
        return os.path.getctime(filepath)

def get_modified(filepath):
    try:
        return time.ctime(os.path.getmtime(filepath))
    except OSError:
        return os.path.getmtime(filepath)

def now():
    return dt.now()

def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:06.3f}'.format(seconds + tdelta.microseconds/1000000)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)

def yaml_dump(_title, _filedir):
    outdict = {}
    fulltitle = _title+'.yaml'
    outdict[fulltitle] = {}
    outdict['directory'] = _filedir
    sessions = outdict[fulltitle]
    avis = [eachvidfile for eachvidfile in os.listdir(_filedir) if "avi" in eachvidfile]
    for j, eachavi in enumerate(avis):
        print(eachavi)
        datafiles = [eachdatafile for eachdatafile in os.listdir(_filedir) if "fly" in eachdatafile]
        for i, eachcsv in enumerate(datafiles):
            print('\t'+eachcsv)
            index = "{:04d}".format(i+j*4)
            sessions[_title+'_'+index] = {}
            session = sessions[_title+'_'+index]
            session['video'] = eachavi
            session['raw_data'] = eachcsv
            session['valid_data'] = _title+'_'+index+'.csv'
            session['meta_data'] = _title+'_'+index+'.json'

    ts_dict = {}
    for session in sessions:
        ts_dict[session] = {}
        ts_dict[session]["created"] = get_created(os.path.join(_filedir, session+'.csv'))
        ts_dict[session]["modified"] = get_modified(os.path.join(_filedir, session+'.csv'))

    with open(os.path.join(_filedir,_title)+'.yaml', 'w+') as outfile:
        outfile.write("# This is a database yaml file created for the pyTrack-analysis framework. " + now().strftime("%y-%m-%d %H:%M:%S") + "\n")
        outfile.write("---\n")
        outfile.write("# file structure\n")
        yaml.safe_dump(outdict, outfile, default_flow_style=False, allow_unicode=True)
        outfile.write("---\n")
        outfile.write("# timestamps\n")
        yaml.safe_dump(ts_dict, outfile, default_flow_style=False, allow_unicode=True)
        outfile.write("...")

if __name__ == "__main__":
    startdt = now()
    if os.name == 'nt':
        _filedir = "E:/Dennis/Google Drive/PhD Project/Tracking Analysis/tracking test/data/"
    else:
        _filedir = "/Users/degoldschmidt/Google Drive/PhD Project/Tracking Analysis/tracking test/data/" #open_file()

    ### enter database name
    name = "TEST" #input("Please enter database name: ")

    yaml_dump(name, _filedir)
    #filestruct, timestamps = load_yaml(os.path.join(_filedir, name)+".yaml")
    #print(timestamps)
    #json_dump(_filedir)
    #data_dump(_filedir)
    print("Done. Runtime:", strfdelta(now() - startdt, "%H:%M:%S"))
