import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from queue import Queue
from src.agent import Agent
from src.audio_manager import AudioManager
from silero_vad import load_silero_vad, VADIterator
from sklearn.metrics.pairwise import cosine_similarity
from src.speech_to_text import SpeechToText
from src.generate_embedding import generate_embedding
from src.text_to_speech import text_to_speech

DEMO_DIR = os.path.dirname(__file__)
SAMPLING_RATE = 16000
CHUNK_SIZE = 512
LOOKBACK_CHUNKS = 5
MARKER_LENGTH = 6
MAX_SPEECH_SECS = 15
VAD_THRESHOLD = 0.2
VAD_MIN_SILENCE_DURATION_MS = 300

def initialize_vad():
    """Initialize and return the Voice Activity Detection model and iterator."""
    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
    )
    return vad_iterator


if __name__ == "__main__":
    print("Synaptics On-Device Voice Assistant Demo")

    # Initialization

    vad_iterator = initialize_vad()
    agent = Agent()
    audio = AudioManager()

    audio.play(f"{DEMO_DIR}/sound/welcome.wav")

    # Live audio processing loop
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)
    recording = False
    speech_to_text = SpeechToText()
    arecord_process = None

    print("Press Ctrl+C to quit.")

    try:
        while True:
            audio.start_arecord(CHUNK_SIZE)  # Start arecord
            for chunk in audio.read(CHUNK_SIZE):
                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True

                    if "end" in speech_dict and recording:
                        recording = False
                        audio.stop_arecord()  # Stop recording

                        # Transcribe speech to text
                        query = speech_to_text.transcribe(speech)
                        print(f"Transcribed Query: {query}")

                        # Handle query with the agent
                        response = agent.handle_query(query)
                        print(f"Agent Response: {response}")

                        # Convert text to speech and play the answer
                        text = text_to_speech(response)
                        audio.play(text)

                        speech *= 0.0
                        break  # Exit the chunk loop to restart arecord

                elif recording and (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                    recording = False
                    audio.stop_arecord()  # Stop recording

                    # Transcribe speech to text
                    query = speech_to_text.transcribe(speech)
                    print(f"Transcribed Query: {query}")

                    # Handle query with the agent
                    response = agent.handle_query(query)
                    print(f"Agent Response: {response}")

                    # Convert text to speech and play the answer
                    text = text_to_speech(response)
                    audio.play(text)

                    vad_iterator.triggered = False
                    vad_iterator.temp_end = 0
                    vad_iterator.current_sample = 0
                    break  # Exit the chunk loop to restart arecord

            gc.collect()

    except KeyboardInterrupt:
        print("Exiting... Goodbye!")
        audio.stop_arecord()  # Ensure arecord is stopped
