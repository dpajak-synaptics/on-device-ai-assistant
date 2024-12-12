import os
import subprocess
import json
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from queue import Queue
from silero_vad import load_silero_vad, VADIterator
from sklearn.metrics.pairwise import cosine_similarity
from src.speech_to_text import SpeechToText
from src.generate_embedding import generate_embedding
from src.text_to_speech import text_to_speech
from src.utils import get_usb_audio_device, file_checksum

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

def read_audio_from_arecord(device, sample_rate=16000, chunk_size=512):
    """Read audio data from ALSA using arecord."""
    command = f"arecord -D {device} -f S16_LE -r {sample_rate} -c 2"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=chunk_size, shell=True
    )

    while True:
        data = process.stdout.read(chunk_size * 4)
        if not data:
            break
        yield np.frombuffer(data, dtype=np.int16)[::2].astype(np.float32) / 32768.0

def load_manual_data():
    """Load QA data from a JSON file and cache it as a TSV file."""
    filename = f'{DEMO_DIR}/data/question_answers.json'
    with open(filename, 'r') as file:
        manual_data = json.load(file)

    qa_df = pd.DataFrame(manual_data)
    qa_df.to_csv(f'{DEMO_DIR}/cached/metadata.tsv', index=False, header=True, sep='\t')

    return manual_data

def load_embeddings(qa_pairs):
    """Load or generate embeddings for QA pairs."""
    filename = f'{DEMO_DIR}/cached/vectors-{file_checksum(json.dumps(qa_pairs, sort_keys=True))}.tsv'

    if os.path.exists(filename):
        embeddings = pd.read_csv(filename, sep='\t', header=None).values
        return embeddings

    print("Generating question embeddings...")
    questions = [pair["question"] for pair in qa_pairs]
    question_embeddings = np.array([generate_embedding(q) for q in tqdm(questions)])

    embedding_df = pd.DataFrame(question_embeddings)
    embedding_df.to_csv(filename, sep='\t', index=False, header=False)

    return question_embeddings

def answer_query(query, qa_pairs, question_embeddings):
    """Find the best answer to a query using cosine similarity."""
    query_embedding = generate_embedding(query)

    if query_embedding is None:
        return "Sorry, I couldn't generate an embedding for your query."

    similarities = cosine_similarity([query_embedding], question_embeddings).flatten()
    threshold = max(0.3, similarities.mean() - similarities.std())

    if similarities.max() < threshold:
        return "Sorry, I don't know the answer to that question."

    best_match_idx = np.argmax(similarities)
    return qa_pairs[best_match_idx]["answer"]

def run_command_tokens(answer):
    """Replace tokens in the answer with results of corresponding commands."""
    with open(f'{DEMO_DIR}/data/tools.json') as f:
        token_command_list = json.load(f)

    for item in token_command_list:
        token = item.get("token")
        command = item.get("command")
        if token and command and token in answer:
            response = os.popen(command).read().strip()
            if response:
                answer = answer.replace(token, response)
    return answer

def handle_end_of_recording(speech, marker, qa_pairs, question_embeddings, audio_device):
    """Process the end of a recording: transcribe, answer, and speak the response."""
    if len(marker) != MARKER_LENGTH:
        raise ValueError("Unexpected marker length.")

    text = speechToText.transcribe(speech)
    query = text

    raw_answer = answer_query(query, qa_pairs, question_embeddings)
    answer = run_command_tokens(raw_answer)

    text_to_speech(answer, audio_device)
    gc.collect()

if __name__ == "__main__":
    print("Synaptics On-Device Voice Assistant Demo")

    # Initialization
    AUDIO_DEVICE = get_usb_audio_device()
    speechToText = SpeechToText()
    vad_iterator = initialize_vad()

    qa_pairs = load_manual_data()
    question_embeddings = load_embeddings(qa_pairs)

    os.system(f'aplay -D {AUDIO_DEVICE} {DEMO_DIR}/sound/Welcome.wav')

    # Live audio processing loop
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)
    recording = False

    print("Press Ctrl+C to quit live captions.")

    try:
        for chunk in read_audio_from_arecord(AUDIO_DEVICE, SAMPLING_RATE, CHUNK_SIZE):
            speech = np.concatenate((speech, chunk))
            if not recording:
                speech = speech[-lookback_size:]

            speech_dict = vad_iterator(chunk)
            if speech_dict:
                if "start" in speech_dict and not recording:
                    recording = True

                if "end" in speech_dict and recording:
                    recording = False
                    handle_end_of_recording(speech, "<STOP>", qa_pairs, question_embeddings, AUDIO_DEVICE)
                    speech *= 0.0

            elif recording and (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                recording = False
                handle_end_of_recording(speech, "<SNIP>", qa_pairs, question_embeddings, AUDIO_DEVICE)
                vad_iterator.triggered = False
                vad_iterator.temp_end = 0
                vad_iterator.current_sample = 0

    except KeyboardInterrupt:
        print("Exiting... Goodbye!")