import os
import zipfile
import argparse

def main():
    parser = argparse.ArgumentParser(description="Package osu! beatmap folder into an .osz archive")
    parser.add_argument("--map_id", type=str, default="917915", help="Mapset ID to package")
    parser.add_argument("--maps_dir", type=str, default="data/maps", help="Maps directory path")
    parser.add_argument("--audio_dir", type=str, default="data/audio", help="Audio directory path")
    parser.add_argument("--output_name", type=str, default=None, help="Optional output .osz name")
    args = parser.parse_args()
    
    map_id_str = args.map_id
    maps_dir = args.maps_dir
    audio_dir = args.audio_dir
    
    # We find the matching test osu map to determine the audio filename
    osu_test_path = os.path.join(maps_dir, f"{map_id_str}_test.osu")
    
    # Check what audio source we have (supports mp3 or ogg)
    audio_src = None
    for ext in [".mp3", ".ogg"]:
        p = os.path.join(audio_dir, f"{map_id_str}{ext}")
        if os.path.exists(p):
            audio_src = p
            break
            
    if audio_src is None:
        print(f"Error: audio source for map {map_id_str} not found in {audio_dir}!")
        return
        
    audio_filename = "audio.mp3" # default fallback
    if os.path.exists(osu_test_path):
        with open(osu_test_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith("AudioFilename:"):
                    audio_filename = line.split(":", 1)[1].strip()
                    print(f"Detected audio filename in map: '{audio_filename}'")
                    break

    # Get the title and artist from the osu file to name the archive nicely
    archive_name = f"{map_id_str}.osz"
    if os.path.exists(osu_test_path):
        title = "Unknown_Title"
        artist = "Unknown_Artist"
        with open(osu_test_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith("Title:"):
                    title = line.split(":", 1)[1].strip().replace(" ", "_")
                elif line.startswith("Artist:"):
                    artist = line.split(":", 1)[1].strip().replace(" ", "_")
        archive_name = f"{map_id_str}_{artist}_-_{title}.osz"
        
    if args.output_name is not None:
        output_osz = args.output_name
    else:
        output_osz = archive_name

    # Find all map files starting with map_id
    osu_files = [f for f in os.listdir(maps_dir) if f.startswith(map_id_str) and f.endswith(".osu")]
    print(f"Map files to include: {osu_files}")

    try:
        with zipfile.ZipFile(output_osz, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add audio file
            print(f"Adding audio: {audio_src} -> {audio_filename}")
            zip_file.write(audio_src, audio_filename)
                
            # Add all .osu files
            for osu in osu_files:
                osu_path = os.path.join(maps_dir, osu)
                print(f"Adding map: {osu_path} -> {osu}")
                zip_file.write(osu_path, osu)
                
        print(f"Success! .osz file created at: {os.path.abspath(output_osz)}")
    except Exception as e:
        print(f"Failed to create .osz archive: {e}")

if __name__ == "__main__":
    main()
