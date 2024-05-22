import requests
import zipfile
import io
import os
from concurrent.futures import ThreadPoolExecutor
import random
import datetime

extract_path_maps = "E:/osu-beatmap-generator-data/maps/"
extract_path_audio = "E:/osu-beatmap-generator-data/audio/"

def tsprint(s):
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + s)

# get file count in directory
def get_file_count(directory):
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count

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
                except Exception as e:
                    tsprint("Error renaming file with ID " + id)
                    print(str(e))
                    return

                #crop the osu file to only have [HitObjects] section
                with open(extract_path_maps + id + "_" + str(count) + ".osu", "r") as f:
                    try:
                        lines = f.readlines()
                    except:
                        tsprint("Error reading file with ID " + id)
                        print(str(e))
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
                    print(str(e))

        tsprint("Downloaded map with ID " + id)
    except:
        tsprint("Error downloading map with ID " + id)
        return

if __name__ == "__main__":
    tsprint("Starting download of 10000 maps")
    # Download maps in parallel
    while True:
        if get_file_count(extract_path_maps) > 10000:
            break
        with ThreadPoolExecutor() as executor:
            # Submit downloadMap function for each map ID

            # Generate 10000 random numbers from 500 to 1000000
            random_numbers = [random.randint(1, 1000000) for _ in range(1000)]

            futures = [executor.submit(downloadMap, str(i)) for i in random_numbers]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()

    tsprint("Downloaded 10000 maps!")



