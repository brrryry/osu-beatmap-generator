import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import numpy as np
import librosa
import datetime
import os
from tensorflow import sparse
import pickle
import multiprocessing

def tsprint(s):
    '''Prints a timestamped string.'''
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + s)
    
   
def parse_difficulty(lines):
    '''Parses the difficulty stats from an .osu file. Takes in the lines of the file as input.'''
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
            tsprint("ERROR: Not a valid osu! map due to insufficient stats.") #if not, return -1
            return -1


    return difficulty

    
def getCurveType(curveType):
    '''Returns the curve type as a bitfield. Takes in the curve type as input.'''
    if(curveType == 'B'):
        return 0b0001
    elif(curveType == 'C'):
        return 0b0010
    elif(curveType == 'L'):
        return 0b0100
    elif(curveType == 'P'):
        return 0b1000

def getCurveType_inverse(curveType):
    '''Returns the curve type as a string. Takes in the curve type as input.'''
    if(curveType == 0b0001):
        return 'B'
    elif(curveType == 0b0010):
        return 'C'
    elif(curveType == 0b0100):
        return 'L'
    elif(curveType == 0b1000):
        return 'P'

def getCurvePts(curvePts):
    '''Returns the curve points as a list of lists. Takes in the curve points as input.'''
    return [[int(x) for x in pt.split(':')] for pt in curvePts]


def getOutput(filename):
    '''Returns the output of the .osu file as two ndarrays (hitobjects and slider data respectively). 
       Takes in the filename as input.'''
    
    # Open the file
    f = open(filename, 'r')

    # Go to the file position where HitObjects start
    while("HitObject" not in f.readline()):
        pass

    # format the objects as two ndarrays, one for all attribs except sliderpts, and one for only sliderpts
    target = [] 
    sliderpts = []

    ######################
    # Notes for the format:
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
            objData = [int(x) for x in objData[:4]] + [getCurveType('L'), 0, 0, int(objData[5])]
        else:
            #Hit Circle Data
            objData = [int(x) for x in objData[:4]] + [getCurveType('L'), 0, 0, 0] # Add dummy data for non-slider attribs
        target.append(objData)
    f.close()
    return target, sliderpts

def formatOutput(filename):
    '''Formats the output of the .osu file. Takes in the filename as input.'''
    # Get the output
    target, sliderpts = getOutput(filename)
    pos = 0
    newTarget = [] 
    if len(target) == 0 or len(target[0]) == 0:
        #No HitObjects
        tsprint("ERROR: Insufficient data for hitpoints.")
        return
    for t in range(0, target[-1][2] + 1):
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

    print(newTarget)
    #If sliderpoints exist, return them
    if len(sliderpts) > 0: return sparse.from_dense(np.stack(tuple(newTarget), axis=0)), sparse.from_dense(np.stack(tuple(sliderpts), axis=0))
    #Otherwise, just return an empty ndarray
    return sparse.from_dense(np.stack(tuple(newTarget), axis=0)), sparse.from_dense([])


def get_timings_from_pkl(filename):
    '''Returns the timings of a osu map from a .pkl file. Takes in the filename as input.'''
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

def process_file(file):
    if file.endswith(".osu"):
        tsprint("Starting file processing for " + file)
        out = formatOutput(folder + file)
        pickle.dump(out, open("pickles/" + file[:-4] + ".pkl", "wb"))
        tsprint("Finished file processing for " + file)


def convert_array_to_osu(array, filename):
    '''Converts an array to an osu file. Takes in the array and filename as input.'''
    with open(filename, 'w') as f:
        f.write("osu file format v14\n\n[General]\nAudioFilename: audio.mp3\nAudioLeadIn: 0\nPreviewTime: -1\nCountdown: 0\nSampleSet: Normal\nStackLeniency: 0.7\nMode: 0\nLetterboxInBreaks: 0\nWidescreenStoryboard: 0\n\n[Editor]\nDistanceSpacing: 1.0\nBeatDivisor: 4\nGridSize: 16\nTimelineZoom: 1.0\n\n[Metadata]\nTitle: osu!stream\nTitleUnicode: osu!stream\nArtist: Team Nekokan\nArtistUnicode: Team Nekokan\nCreator: peppy\nVersion: Easy\nSource: osu!stream\nTags: dnb drumandbass drum bass\nBeatmapID: 0\nBeatmapSetID: -1\n\n[Difficulty]\nHPDrainRate: 6\nCircleSize: 4\nOverallDifficulty: 4\nApproachRate: 9\nSliderMultiplier: 1.4\nSliderTickRate: 1\n\n[Events]\n//Background and Video events\n//Break Periods\n//Storyboard Layer 0 (Background)\n//Storyboard Layer 1 (Fail)\n//Storyboard Layer 2 (Pass)\n//Storyboard Layer 3 (Foreground)\n//Storyboard Sound Samples\n\n[TimingPoints]\n//offset, milliseconds per beat, meter, sample set, sample index, volume, inherited, kiai\n0,600,4,1,0,100,0,0\n\n[HitObjects]\n")
        for i in range(len(array)):
            if array[i] != 0:
                f.write(str(i) + ",192,192,0,1,0,0:0:0:0:\n")
        f.close()

if __name__ == "__main__":
    folder = 'maps/'
    dirs = os.listdir(folder)
    
    pool = multiprocessing.Pool(4)
    pool.map(process_file, dirs)
    pool.close()
    pool.join()