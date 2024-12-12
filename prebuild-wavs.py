import os
import json
import subprocess
import time

# Constants
SAMPLE_RATE = 16000
play_audio = 'aplay'  # Adjust as needed based on your setup
text_to_speech_model_path = "./models/piper/en_US-lessac-medium.onnx"

def text_to_speech(answer, filename):
    """Generate a WAV file from text using the TTS model."""
    text_to_speech_command = f'echo "{answer}" | ./models/piper/piper --model {text_to_speech_model_path} --output_file {filename}'
    os.system(text_to_speech_command)

def generate_wav_files():
    """Load the dataset and generate WAV files for all answers."""
    # Load dataset
    filename = './data/question_answers.json'
    with open(filename, 'r') as file:
        manual_data = json.load(file)

    qa_pairs = manual_data

    # Create the cached directory if it does not exist
    cached_dir = './cached'
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)

    # Generate WAV files for each answer
    for pair in qa_pairs:
        answer = pair.get("answer")
        if answer:
            # Create a checksum based on the answer length and content
            checksum = sum(bytearray(answer.encode()))
            wav_filename = f'{cached_dir}/answer-{len(answer)}-{checksum}.wav'

            # Check if the WAV file already exists
            if not os.path.exists(wav_filename):
                print(f'Generating WAV file for answer: "{answer}"')
                text_to_speech(answer, wav_filename)
                print(f'Saved: {wav_filename}')
            else:
                print(f'WAV file already exists: {wav_filename}')

if __name__ == '__main__':
    generate_wav_files()
