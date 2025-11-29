import whisper
import os
import warnings

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Filter out annoying warnings
warnings.filterwarnings("ignore")

def extract_transcript(video_path):
    print(f"Loading Whisper Model... (This might take a moment)")
    # We use the 'base' model. It's accurate but fast enough for laptops.
    model = whisper.load_model("tiny")
    
    print(f"Transcribing {video_path}...")
    # This does the magic:
    result = model.transcribe(video_path)
    
    # The result contains a list of 'segments' (chunks of text)
    segments = result['segments']
    
    print("\n--- TRANSCRIPTION COMPLETE ---\n")
    
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        # Print format: [00:05.00 -> 00:09.00]: Hello world
        print(f"[{start_time:.2f}s -> {end_time:.2f}s]: {text.strip()}")

    return segments

if __name__ == "__main__":
    # Point to your video file
    video_file = "data/test_video.mp4"
    
    if os.path.exists(video_file):
        extract_transcript(video_file)
    else:
        print(f"Error: Could not find file at {video_file}")
        print("Please make sure you put 'test_video.mp4' inside the 'data' folder.")