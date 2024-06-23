import requests
import zipfile
import io
import os
from concurrent.futures import ThreadPoolExecutor
import random
import traceback
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import librosa
import datetime
import pickle
import multiprocessing

extract_path_maps = "maps/"
extract_path_audio = "audio/"
extract_path_pickles = "pickles/"


##################
# Helper Functions
##################
def tsprint(s):
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + s)

# get file count in directory
def get_file_count(directory):
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count


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

def remove_osu_map_files(id):
    try:
        count = 0
        while os.path.isfile(extract_path_maps + id + "_" + str(count) + ".osu"):
            os.remove(extract_path_maps + id + "_" + str(count) + ".osu")
            count += 1
    except:
        tsprint("Error removing osu map with ID " + id)
    try:
        count = 0
        while os.path.isfile(extract_path_pickles + id + "_" + str(count) + ".pkl"):
            os.remove(extract_path_pickles + id + "_" + str(count) + ".pkl")
            count += 1
    except:
        tsprint("Error removing pickle with ID " + id)
    try:
        if os.path.isfile(extract_path_audio + id + ".mp3"): os.remove(extract_path_audio + id + ".mp3")
    except:
        tsprint("Error removing audio with ID " + id)

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

            try:
                objData = [int(x) for x in objData[:4]] + [curveType, int(objData[6]), float(objData[7][:-1]), 0]
            except:
                tsprint("ERROR: Insufficient data for slider.")
                return [], []
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
        remove_osu_map_files(filename.split("/")[-1].split("_")[0])
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
        out = formatOutput(extract_path_maps + file)
        pickle.dump(out, open(extract_path_pickles + file[:-4] + ".pkl", "wb"))
        tsprint("Finished file processing for " + file)


######################
# bruh
######################


def downloadMap(id):
    try:
        try:
            response = requests.get("https://beatconnect.io/b/" + id)
            if response.status_code != 200:
                tsprint("Error downloading map with ID " + id)
                return
        except:
            tsprint("Error downloading map with ID " + id)
            print(str(e))
        
        # Unzip the response in memory
        zip_file = zipfile.ZipFile(io.BytesIO(response.content)) 
        count = 0

        for name in zip_file.namelist():
            if name.endswith(".osu"):
                zip_file.extract(name, extract_path_maps)
                try:
                    os.rename(extract_path_maps + name, extract_path_maps + id  + "_" + str(count) + ".osu")
                    process_file(id + "_" + str(count) + ".osu")
                except Exception as e:
                    tsprint("Error renaming file with ID " + id)
                    traceback.print_exc()
                    os.remove(extract_path_maps + name)
                    return

                #crop the osu file to only have [HitObjects] section
                with open(extract_path_maps + id + "_" + str(count) + ".osu", "r") as f:
                    try:
                        lines = f.readlines()
                        if parse_difficulty(lines) == -1: continue
                    except:
                        tsprint("Error reading file with ID " + id)
                        traceback.print_exc()
                        os.remove(extract_path_maps + id + "_" + str(count) + ".osu")
                        return
                    writing = False
                    with open(extract_path_maps + id + "_" + str(count) + ".osu", "w") as f2:
                        for i, line in enumerate(lines):
                            if line.startswith("[HitObjects]"): writing = True
                            elif line.startswith("[Difficulty]"): writing = True
                            elif line.startswith("[Editor]"): writing = False
                            elif line.startswith("[Events]"): writing = False
                            elif line.startswith("[TimingPoints]"): writing = True
                            elif line.startswith("[Colours]"): writing = False
                            elif line.startswith("[Metadata]"): writing = False

                            if writing:
                                f2.write(line)

                count += 1
            elif name.endswith(".mp3"):
                zip_file.extract(name, extract_path_audio)
                try:
                    os.rename(extract_path_audio + name, extract_path_audio + id + ".mp3")
                except:
                    tsprint("Error renaming file with ID " + id)
                    traceback.print_exc()
                    os.remove(extract_path_audio + name)
                    print(str(e))

        tsprint("Downloaded map with ID " + id)

    except:
        tsprint("Error downloading map with ID " + id)
        return

def fetch_data(num_maps=1000):
    tsprint(f'Starting download of {num_maps} maps!')
    # Download maps in parallel
    while True:
        if get_file_count(extract_path_maps) > num_maps:
            break
        with ThreadPoolExecutor() as executor:
            # Submit downloadMap function for each map ID

            # Generate 10000 random numbers from 500 to 1000000
            random_numbers = [random.randint(1, 1000000) for _ in range(25)]

            futures = [executor.submit(downloadMap, str(i)) for i in random_numbers]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()

    tsprint(f'Downloaded {num_maps} maps!')


if __name__ == "__main__":
    fetch_data(1000)



