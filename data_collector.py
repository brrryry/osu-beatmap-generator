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
import datetime
import numpy as np
import logging
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import osu  
from osu import Client

# Our imports
import config #config file

extract_path_maps = config.map_path
extract_path_audio = config.audio_path
extract_path_pickles = config.pickle_path

# define osu API client for searching
client = Client.from_credentials(
    config.osu_api_client_id, 
    config.osu_api_client_secret, 
    config.osu_api_redirect_uri
)
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename = f"logs/data_collector_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

NUM_MAPS = 1500

# Event used to signal all threads to stop gracefully
stop_event = threading.Event()
processed_lock = threading.Lock()
processed_maps_file = "processed_maps.txt"
processed_map_ids = set()

# Load processed maps
if os.path.exists(processed_maps_file):
    with open(processed_maps_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    processed_map_ids.add(int(line))
                except ValueError:
                    pass

# Also populate from existing files to be safe
try:
    if os.path.exists(extract_path_maps):
        for f in os.listdir(extract_path_maps):
            if f.endswith(".osu"):
                try:
                    processed_map_ids.add(int(f.split('_')[0]))
                except ValueError:
                    pass
except Exception:
    pass

try:
    if os.path.exists(extract_path_audio):
        for f in os.listdir(extract_path_audio):
            if f.endswith(".mp3") or f.endswith(".ogg"):
                try:
                    processed_map_ids.add(int(os.path.splitext(f)[0]))
                except ValueError:
                    pass
except Exception:
    pass

def mark_map_processed(map_id):
    try:
        mid = int(map_id)
    except ValueError:
        return
    with processed_lock:
        if mid not in processed_map_ids:
            processed_map_ids.add(mid)
            with open(processed_maps_file, 'a') as f:
                f.write(f"{mid}\n")

def signal_handler(sig, frame):
    """
    Handles SIGINT (Ctrl+C) and SIGTERM. Sets stop_event to stop all threads gracefully.
    """
    logger.info("Termination signal received. Setting stop flag...")
    print("\nStopping data collection... Waiting for current download threads to exit.")
    stop_event.set()

##################
# Helper Functions
##################
def get_curve_type(letter: str) -> int:
    """
    Converts a curve letter to its binary component
    """
    if(letter == 'B'):
        return 0b0001
    elif(letter == 'C'):
        return 0b0010
    elif(letter == 'L'):
        return 0b0100
    elif(letter == 'P'):
        return 0b1000


def get_curve_points(curvePts):
    """
    Processes curve points.
    """
    return [[int(x) for x in pt.split(':')] for pt in curvePts]

def remove_osu_map(map_id):
    # remove all files associated with the map_id
    for file in os.listdir(extract_path_maps):
        if file.startswith(map_id):
            os.remove(extract_path_maps + file)
    for file in os.listdir(extract_path_audio):
        if file.startswith(map_id):
            os.remove(extract_path_audio + file)
    logger.info(f"Removed map {map_id}")


##################
# Main Functions
##################

def fetch_map(map_id, difficulty_threshold = 5.0):
    """
    Given a map id, pull it from nerinyan.
    Get all maps that satisfy the difficulty threshold.
    Additionally, get the source MP3 of the map.
    
    @param map_id: The id of the map to fetch.
    @param difficulty_threshold: The minimum difficulty of the map.
    """
    if stop_event.is_set():
        return

    map_id_str = str(map_id)
    logger.info(f"Attempting to fetch map {map_id_str}!")

    if os.path.exists(os.path.join(extract_path_maps, f"{map_id_str}_0.osu")):
        logger.info(f"Map {map_id_str} is already in the folder!")
        mark_map_processed(map_id)
        return

    url = os.path.join(config.api_link, map_id_str)
    max_retries = 5
    backoff = 2.0
    response = None
    
    for attempt in range(max_retries):
        if stop_event.is_set():
            return
        try:
            # Respect rate limit with a small delay before requesting
            if stop_event.wait(1.0):
                return
            
            response = requests.get(url, timeout=15)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_time = float(retry_after) if retry_after else backoff
                logger.warning(f"Rate limited (429) on map {map_id_str}. Retrying in {sleep_time} seconds (attempt {attempt+1}/{max_retries})...")
                if stop_event.wait(sleep_time):
                    return
                backoff *= 2.0
                continue
            
            if response.status_code == 404:
                logger.warning(f"Map {map_id_str} not found on server (404). Skipping.")
                mark_map_processed(map_id)
                return
                
            response.raise_for_status()
            logger.info(f"Successfully fetched map {map_id_str}!")
            break
        except Exception as e:
            if stop_event.is_set():
                return
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch map {map_id_str} after {max_retries} attempts - {str(e)}")
                if response is not None and response.status_code in [400, 401, 403, 404]:
                    mark_map_processed(map_id)
                return
            logger.warning(f"Error fetching map {map_id_str}: {str(e)}. Retrying in {backoff} seconds...")
            if stop_event.wait(backoff):
                return
            backoff *= 2.0

    if stop_event.is_set() or response is None:
        return

    # extract the map in memory
    try:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    except Exception as e:
        logger.error(f"Failed to extract map {map_id_str} - {str(e)}")
        mark_map_processed(map_id)
        return
    
    osu_files_kept = 0
    audio_extracted = False
    extracted_audio_files = []

    for file in zip_file.namelist():
        if stop_event.is_set():
            return

        if file.endswith(".osu"):
            zip_file.extract(file, extract_path_maps)
            try:
                with open(os.path.join(extract_path_maps, file), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                difficulty = 0.0
                for line in content.split('\n'):
                    if line.startswith("OverallDifficulty:"):
                        difficulty = float(line.split(":")[1].strip())
                        break
                
                if difficulty >= difficulty_threshold:
                    new_name = f"{map_id_str}_{osu_files_kept}.osu"
                    target_osu_path = os.path.join(extract_path_maps, new_name)
                    if os.path.exists(target_osu_path):
                        os.remove(target_osu_path)
                    os.rename(os.path.join(extract_path_maps, file), target_osu_path)
                    osu_files_kept += 1
                else:
                    os.remove(os.path.join(extract_path_maps, file))
            except Exception as e:
                logger.error(f"Failed to process .osu file {file} - {str(e)}")
                if os.path.exists(os.path.join(extract_path_maps, file)):
                    try:
                        os.remove(os.path.join(extract_path_maps, file))
                    except Exception:
                        pass

        elif file.endswith(".mp3") or file.endswith(".ogg"):
            zip_file.extract(file, extract_path_audio)
            ext = ".mp3" if file.endswith(".mp3") else ".ogg"
            temp_extracted_path = os.path.join(extract_path_audio, file)
            target_path = os.path.join(extract_path_audio, f"{map_id_str}{ext}")
            
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(temp_extracted_path, target_path)
                audio_extracted = True
                extracted_audio_files.append(target_path)
            except Exception as e:
                logger.error(f"Failed to rename audio file {file} to {map_id_str}{ext} - {str(e)}")
                if os.path.exists(temp_extracted_path):
                    try:
                        os.remove(temp_extracted_path)
                    except Exception:
                        pass

    # if there is no audio file or no osu files met difficulty threshold, discard
    if (osu_files_kept == 0 or not audio_extracted) and not stop_event.is_set():
        logger.info(f"Discarding map {map_id_str} (osu_files_kept={osu_files_kept}, audio_extracted={audio_extracted})")
        # remove any extracted osu files
        for i in range(osu_files_kept):
            p = os.path.join(extract_path_maps, f"{map_id_str}_{i}.osu")
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        # remove audio files
        for p in extracted_audio_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    else:
        logger.info(f"Successfully processed map {map_id_str} with {osu_files_kept} difficulties!")

    # Mark as processed in all cases so we don't try it again
    mark_map_processed(map_id)


# Function to fetch maps!
def fetch_maps(num_maps = NUM_MAPS, difficulty_threshold = 5.0):
    page = 1
    while not stop_event.is_set():
        try:
            file_count = os.listdir(extract_path_audio)
        except FileNotFoundError:
            file_count = []
        if len(file_count) >= num_maps:
            break

        # fetch maps in batches
        filter = osu.util.BeatmapsetSearchFilter()
        filter.set_mode(osu.GameModeInt.STANDARD)
        filter.set_status(osu.BeatmapsetSearchStatus.RANKED)  
        filter.set_sort(osu.BeatmapsetSearchSort.PLAYS)

        logger.info(f'Fetched page {page} of maps...')
        print(f"Fetching page {page} of maps... Current audio files: {len(file_count)}/{num_maps}")

        try:
            beatmapsearchresult = client.search_beatmapsets(filters=filter, page=page)
        except Exception as e:
            logger.error(f"Failed to search beatmapsets on page {page}: {e}")
            print(f"Error fetching map page {page}: {e}. Retrying in 5 seconds...")
            if stop_event.wait(5.0):
                break
            continue

        map_ids = [beatmapset.id for beatmapset in beatmapsearchresult.beatmapsets]

        # remove any map ids that have already been processed
        map_ids = [map_id for map_id in map_ids if int(map_id) not in processed_map_ids]

        if not map_ids:
            page += 1
            continue

        if stop_event.is_set():
            break

        # use concurrent threads to fetch maps
        with ThreadPoolExecutor(max_workers=2) as executor:
            # call fetcher function on each map id concurrently
            futures = [executor.submit(fetch_map, map_id, difficulty_threshold) for map_id in map_ids]

            try:
                # wait for all threads to finish, checking stop_event periodically
                for future in futures:
                    while not stop_event.is_set():
                        try:
                            future.result(timeout=0.1)
                            break
                        except FutureTimeoutError:
                            continue
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received inside fetch_maps context, stopping...")
                stop_event.set()
        
        if stop_event.is_set():
            break
            
        page += 1
    
    if stop_event.is_set():
        logger.info("Data collection was interrupted and stopped.")
        print("\nData collection was interrupted and stopped.")
    else:
        logger.info(f'Successfully fetched {num_maps} maps!')
        print(f"\nSuccessfully fetched {num_maps} maps!")
        

if __name__ == "__main__":
    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (graceful shutdown)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal only works in main thread (e.g. might fail if run in notebook/subthread)
        pass

    logger.info("Starting the data collection process!")
    # create folders
    if not os.path.exists(extract_path_maps):
        os.makedirs(extract_path_maps)
    if not os.path.exists(extract_path_audio):
        os.makedirs(extract_path_audio)
    if not os.path.exists(extract_path_pickles):
        os.makedirs(extract_path_pickles)

    try:
        # fetch maps
        fetch_maps()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main thread. Stopping gracefully...")
        print("\nKeyboardInterrupt caught. Stopping gracefully...")
        stop_event.set()

    

