import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import numpy as np
import librosa
import datetime
import os
import torch
import pickle
import multiprocessing

def tsprint(s):
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + s)


def get_lines(filename):
    with open(filename, "r") as f:
        try:
            return f.readlines()
        except:
            tsprint("ERROR: cannot read lines of .osu file.")
    
   
def parse_difficulty(lines):

    difficulty = [-1,-1,-1,-1,-1,-1]

    for line in lines:
        #difficulty
        if line.startswith("HPDrainRate"): difficulty[0] = float(line.split(":", 1)[1])
        elif line.startswith("CircleSize"): difficulty[1] = float(line.split(":", 1)[1])
        elif line.startswith("OverallDifficulty"): difficulty[2] = float(line.split(":", 1)[1])
        elif line.startswith("ApproachRate"): difficulty[3] = float(line.split(":", 1)[1])
        elif line.startswith("SliderMultiplier"): difficulty[4] = float(line.split(":", 1)[1])
        elif line.startswith("SliderTickRate"): difficulty[5] = float(line.split(":", 1)[1])
        elif not (line.startswith("[Difficulty]")): break

    #check if all the difficulty stats are there
    for val in difficulty:
        if val == -1:
            tsprint("ERROR: Not a valid osu! map due to insufficient stats.")
            return -1


    return difficulty

def frange(start,stop,step):
    istop = int((stop-start) // step)
    return [start + i * step for i in range(istop)]
    
def getCurveType(curveType):
    if(curveType == 'B'):
        return 0b0000
    elif(curveType == 'C'):
        return 0b0001
    elif(curveType == 'L'):
        return 0b0010
    elif(curveType == 'P'):
        return 0b0100

def getCurvePts(curvePts):
    return [[int(x) for x in pt.split(':')] for pt in curvePts]

def getOutput(filename):
    # Open the file
    f = open(filename, 'r')

    # Go to the file position where HitObjects start
    while("HitObject" not in f.readline()):
        pass

    # format the objects as two ndarrays, one for all attribs except sliderpts, and one for only sliderpts
    target = []
    sliderpts = []

    ######################
    # Hit Circles (bit index 0): x,y,time,type,hitSound
    # Sliders (bit index 1): x,y,time,type,hitSound,curveType|curvePoints,slides,length
    # Spinners (bit index 3): x,y,time,type,hitSound,endTime
    # (hit sample ommitted from ends + edge sounds and sets)
    ######################

    for line in f:
        objData = line.split(',')
        if(int(objData[3]) & 0b00000010):
            # Slider data
            sliderData = objData[5].split('|')
            curveType = getCurveType(sliderData[0])
            sliderpts.append(getCurvePts(sliderData[1:]))

            objData = [int(x) for x in objData[:4]] + [curveType, int(objData[6]), float(objData[7][:-1]), 0]
        elif(int(objData[3]) & 0b00001000):
            # Spinner Data
            objData = [int(x) for x in objData[:4]] + [getCurveType('B'), 0, 0, int(objData[5])]
        else:
            #Hit Circle Data
            objData = [int(x) for x in objData[:4]] + [getCurveType('B'), 0, 0, 0] # Add dummy data for non-slider attribs
        target.append(objData)
    f.close()
    return target, sliderpts

def formatOutput(filename):
    # Get the output
    target, sliderpts = getOutput(filename)
    pos = 0
    newTarget = [] 
    if len(target) == 0 or len(target[0]) == 0:
        tsprint("ERROR: Insufficient data for hitpoints.")
        return
    for t in range(0, target[-1][2] + 1, 10):
        if(pos < len(target) and target[pos][2] == t):
            newTarget.append(target[pos])
            newTarget[-1][2] = 1
            pos += 1
        else:
            newTarget.append([0,0,0,0,0,0,0,0])


    maxlen = max([len(x) for x in sliderpts]) if len(sliderpts) > 0 else 0
    for i in range(len(sliderpts)):
        while(len(sliderpts[i]) < maxlen):
            sliderpts[i].append([0,0])

    newTarget = torch.tensor(np.stack(tuple(newTarget), axis=0)).to_sparse_csr() 
    if len(sliderpts) > 0:  
        sliderpts = torch.tensor(np.stack(tuple(sliderpts), axis=0)).to_sparse()
    else: 
        sliderpts = torch.tensor([]).to_sparse()
    
    return newTarget, sliderpts


def process_file(file):
    if file.endswith(".osu"):
        tsprint("Starting file processing for " + file)
        out = formatOutput(folder + file)
        pickle.dump(out, open("pickles/" + file[:-4] + ".pkl", "wb"))
        tsprint("Finished file processing for " + file)

def get_timings_from_pkl(filename):
    count = 0
    while os.path.isfile(filename + "_" + str(count) + ".pkl"): 
        count += 1
    count -= 1
 
    with open(filename + "_" + str(count) + ".pkl", "rb") as f:
          pkl = pickle.load(f)

    
    indices = pkl[0].indices.numpy()
    values = pkl[0].values.numpy()

    max_length = 0
    for index in indices: max_length = (index[0] if index[0] > max_length else max_length)
    
    timings = [0] * (max_length + 1)
    
    for index in indices: timings[index[0]] = 1
        
    return timings

if __name__ == "__main__":
    folder = 'maps/'
    dirs = os.listdir(folder)
    
    pool = multiprocessing.Pool(4)
    pool.map(process_file, dirs)
    pool.close()
    pool.join()
############################
# TO LOAD THE DATA BACK format: (target, sliderpts) #
# newOut = pickle.load(open("125_1.pkl", "rb"))
# print(newOut)
############################
