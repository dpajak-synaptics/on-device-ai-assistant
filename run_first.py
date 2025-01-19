import os
import time
import requests
import numpy as np
import soundfile as sf
from tqdm import tqdm
from src.agent import Agent
from src.audio_manager import AudioManager
from src.speech_to_text import SpeechToText
from src.text_to_speech import text_to_speech
# from src.llm import llm

DEMO_DIR = os.path.dirname(__file__)
SAMPLING_RATE = 16000
CHUNK_SIZE = 512
LOOKBACK_CHUNKS = 5
MARKER_LENGTH = 6
MAX_SPEECH_SECS = 15
VAD_THRESHOLD = 0.2
VAD_MIN_SILENCE_DURATION_MS = 300

ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[36m"
ANSI_RESET = "\033[0m"

QA_FILES = [f'{DEMO_DIR}/data/mars.json', f'{DEMO_DIR}/data/dishwasher.json']

JFK_URL = "https://github.com/ggerganov/whisper.cpp/raw/refs/heads/master/samples/jfk.wav"
JFK_FILE = "jfk.wav"

if __name__ == "__main__":
    print(f"\n\n{ANSI_CYAN}Synaptics On-Device Voice Assistant Setup{ANSI_RESET}")
    print('-----------------------------------------')
    print('This script preloads all needed models and pre-generates text-to-speech.\n')
    print('This may take a few minutes.\n')

    print(f"{ANSI_YELLOW}Testing USB audio device...{ANSI_RESET}")
    audio = AudioManager()

    print(f"{ANSI_YELLOW}Initialising Agent...{ANSI_RESET}")
    agent = Agent()

    print(f"{ANSI_YELLOW}Initialising Speech To Text...{ANSI_RESET}")
    speech_to_text = SpeechToText()

    # Download the test file if it does not exist
    if not os.path.exists(JFK_FILE):
        print(f"Downloading {JFK_FILE}...")
        r = requests.get(JFK_URL)
        with open(JFK_FILE, 'wb') as f:
            f.write(r.content)

    # Read jfk.wav and transcribe it
    jfkAudio, sr = sf.read(JFK_FILE)
    # If the sample rate is not 16000, you may want to resample here
    if sr != SAMPLING_RATE:
        print(f"Warning: {JFK_FILE} has sample rate {sr}. It should be {SAMPLING_RATE}.")

    # Calculate duration in seconds
    duration_seconds = len(jfkAudio) / sr
    print(f"Audio Duration: {duration_seconds:.2f} seconds")

    # Number of transcription runs for averaging
    num_runs = 10
    total_time = 0.0

    print(f"{ANSI_YELLOW}Benchmarking Speech To Text...{ANSI_RESET}")
    for i in tqdm(range(num_runs), desc="Transcribing"):
        start = time.time()
        result = speech_to_text.transcribe(jfkAudio)
        print(result)
        end = time.time()
        elapsed_ms = (end - start) * 1000  # Convert to milliseconds
        total_time += elapsed_ms

    average_ms_per_transcription = total_time / num_runs
    ms_per_second = average_ms_per_transcription / duration_seconds

    print(f"\n{ANSI_YELLOW}Benchmark Results:{ANSI_RESET}")
    print(f"Average Time per Transcription: {average_ms_per_transcription:.2f} ms")
    print(f"Time to Transcribe 1 Second of Audio: {ms_per_second:.2f} ms/sec")

    # Pre-generate some TTS
    print(f"{ANSI_YELLOW}Pre-generating Text To Speech...{ANSI_RESET}")
    for qa_file in QA_FILES:
        agent.change_agent(qa_file)
        print(qa_file)
        for pair in tqdm(agent.qa_pairs, desc=f"Processing {qa_file}"):
            _ = text_to_speech(pair['answer'], agent.voiceModel, agent.voiceJson)
