import os
import gc
import time
import numpy as np
from src.agent import Agent
from src.audio_manager import AudioManager
from silero_vad import load_silero_vad, VADIterator
from sklearn.metrics.pairwise import cosine_similarity
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
ANSI_RESET = "\033[0m"

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
    agent = Agent()
    accuracy = 0

    # print("Testing agent...")
    # for question, expected_answer in zip(agent.valQuestions, agent.valAnswers):

    #     response = agent.handle_query(question)
    #     if (response['answer'] == expected_answer) and (response['confidence'] > agent.threshold):
    #         accuracy += response['confidence']
    #     else:
    #         accuracy -= response['confidence']
    #         print(f"\nTest question: {question}")
    #         print(f"Test answer: {response['answer']}")
    #         print(f"Similarity: {response['confidence']}")
    #         # print(f"Expected answer: {expected_answer}")

    # print(f"Mean similarity: {accuracy / len(agent.valQuestions) * 100:.2f}%") 

    vad_iterator = initialize_vad()
    audio = AudioManager()
    audio.play(f"{DEMO_DIR}/sound/welcome.wav")

    # Live audio processing loop
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)
    recording = False
    speech_to_text = SpeechToText()
    arecord_process = None

    # Array of strings to ask for more details
    detail_prompts = [
        "Please provide me a little more detail.",
        "Sorry, I don't know the answer."
    ]
    prompt_index = 0

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

                    if recording and ("end" in speech_dict or (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS):
                        recording = False
                        audio.stop_arecord()  # Stop recording

                        # Transcribe speech to text
                        start_transcribe = time.time()
                        query = speech_to_text.transcribe(speech)
                        end_transcribe = time.time()
                        transcribe_time = (end_transcribe - start_transcribe) * 1000  # Convert to milliseconds
                        print(f"{ANSI_YELLOW}Transcription Time: {transcribe_time:.2f} ms{ANSI_RESET}")
                        print(f"Transcribed Query: {query}")

                        length = len(query)
                        print(f"Length of response: {length}")

                        # If response answer is empty, break
                        if length == 0:
                            print("Voice detected but no text output.")
                            speech *= 0.0
                            break

                        # Handle query with the agent
                        start_response = time.time()
                        response = agent.handle_query(query)
                        end_response = time.time()
                        response_time = (end_response - start_response) * 1000  # Convert to milliseconds
                        print(f"{ANSI_YELLOW}Response Generation Time: {response_time:.2f} ms{ANSI_RESET}")
                            
                        print(f"Agent Response: {response}")

                        if response['confidence'] > agent.threshold:
                            # Convert text to speech and play the answer
                            text = text_to_speech(response['answer'], agent.voiceModel, agent.voiceJson)
                            audio.play(text)
                            speech *= 0.0
                            prompt_index = 0  # Reset prompt index
                            break  # Exit the chunk loop to restart arecord

                        else:
                            # not confident enough, continue recording
                            text = text_to_speech(detail_prompts[prompt_index], agent.voiceModel, agent.voiceJson)
                            audio.play(text)
                            audio.start_arecord(CHUNK_SIZE)
                            recording = True
                            prompt_index= prompt_index+1
                            if prompt_index >= len(detail_prompts):
                                speech *= 0.0
                                prompt_index = 0  # Reset prompt index
                                break  # Exit the chunk loop to restart arecord


            gc.collect()

    except KeyboardInterrupt:
        print("Exiting... Goodbye!")
        audio.stop_arecord()  # Ensure arecord is stopped
