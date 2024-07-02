###########################
# This script will collect osu! maps from an osu! API (beatconnect.io) and extract them.
# The maps are then processed and saved in a pickle file. (pytorch tensors)
# The pickle files are used to train the model.
# The directories to each section (audio, maps, pickles) can be edited in the config file. 
###########################


# Python library imports
import requests
import zipfile
import io
import os
import random
import datetime
import torch
import numpy as np
import librosa
import pickle

# Our imports
import config #config file


extract_path_maps = config.map_path
extract_path_audio = config.audio_path
extract_path_pickles = config.pickle_path

##################
# Helper Functions
##################
def tsprint(s):
    """
    Prints a string with a timestamp in front of it.
    """
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + s)

def parse_difficulty(lines: [str]) -> list:
    """
    Parses the difficulty of an osu! map.
    @param lines: The lines of the osu! map file.
    """
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

    return difficulty

def remove_osu_map_files(id):
    """
    Given an map id, removes all the files related to that map.
    """
    try:
        count = 0 #use a counter since there may be multiple .osu files.
        while os.path.isfile(extract_path_maps + id + "_" + str(count) + ".osu"):
            os.remove(extract_path_maps + id + "_" + str(count) + ".osu")
            count += 1
    except:
        tsprint("Error removing osu map with ID " + id)
    try:
        count = 0 #use a counter since there may be multiple .pkl files.
        while os.path.isfile(extract_path_pickles + id + "_" + str(count) + ".pkl"):
            os.remove(extract_path_pickles + id + "_" + str(count) + ".pkl")
            count += 1
    except:
        tsprint("Error removing pickle with ID " + id)
    try:
        if os.path.isfile(extract_path_audio + id + ".mp3"): os.remove(extract_path_audio + id + ".mp3")
    except:
        tsprint("Error removing audio with ID " + id)

def curve_letter_to_bin(letter: str) -> int:
    """
    Converts a curve letter to its binary component
    """
    if(curveType == 'B'):
        return 0b0000
    elif(curveType == 'C'):
        return 0b0001
    elif(curveType == 'L'):
        return 0b0010
    elif(curveType == 'P'):
        return 0b0100

def get_curve_points(curvePts):
    """
    Processes curve points.
    """
    return [[int(x) for x in pt.split(':')] for pt in curvePts]


##################
# Main Functions
##################

def getOutput():
    """
    Given a file, parse it and return each of its hitobjects/sliders in a list format.
    """
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
    """
    Given a filename, get the output using getOutput() and format it in centisecond intervals.
    """
    # Get the output
    target, sliderpts = getOutput(filename)
    pos = 0

    # If there is no data, return nothing
    if not target:
        return None

    # Create the output
    output = []
    for i in range(len(target)):
        # Get the current object
        obj = target[i]

        # If the object is a slider
        if obj[3] & 0b00000010:
            # Get the slider points
            pts = sliderpts[pos]
            pos += 1

            # Add the slider points to the object
            obj = obj + pts

        # Add the object to the output
        output.append(obj)

    return output

def downloadMap(id):
    """
    Given a map id, try to download it from the osu! API.
    NOTE: This function will only extract the RELEVANT data in an .osu file (HitObjects, Difficulty, TimingPoints).
    """
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
    
##################
# Main
##################
def collect_data(num_maps=1000):
    """
    Collects num_maps maps from the osu! API and processes them.
    """
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
    collect_data(1000)
