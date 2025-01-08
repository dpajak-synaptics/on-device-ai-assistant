import os
import time
import numpy as np
from tqdm import tqdm
from src.agent import Agent
from src.audio_manager import AudioManager
from src.speech_to_text import SpeechToText
from src.text_to_speech import text_to_speech
#from src.llm import llm

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

if __name__ == "__main__":
    print(f"\n\n{ANSI_CYAN}Synaptics On-Device Voice Assistant Setup{ANSI_RESET}")
    print('-----------------------------------------')
    print('This script preloads all necessary AI models and pre-generates text-to-speech in order to improve assistant performance.\n\nThis can take several minutes.\n\n')

    # Initialization
    print(f"{ANSI_YELLOW}Testing USB audio device connected...{ANSI_RESET}")
    audio = AudioManager()
    print(f"{ANSI_YELLOW}Initialising Agent...{ANSI_RESET}")
    agent = Agent()
    print(f"{ANSI_YELLOW}Initialising Speech To Text...{ANSI_RESET}")
    speech_to_text = SpeechToText()
    
    for i in range(1, 4):
        agent.handle_query(agent.qa_pairs[0]['question'])

    fakeAudio = np.random.rand(SAMPLING_RATE * 5).astype(np.float32) * 2 - 1

    for i in range(1, 4): 
        start = time.time()
        speech_to_text.transcribe(fakeAudio)
        end = time.time()
        print(f"{ANSI_YELLOW}Speech To Text (5s audio): {(end-start)*1000:.2f} ms{ANSI_RESET}")

    print(f"{ANSI_YELLOW}Pre-generating Text To Speech...{ANSI_RESET}")
    for qa_file in QA_FILES:
        agent.change_agent(qa_file)
        print(qa_file)
        for pair in tqdm(agent.qa_pairs):
            wav = text_to_speech(pair['answer'],agent.voiceModel, agent.voiceJson)
