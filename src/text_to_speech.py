import os
from huggingface_hub import hf_hub_download
from src.utils import file_checksum
import time

DEMO_DIR = os.path.dirname(__file__)

def download_model_files():
    """Download required model files from Hugging Face Hub."""

    # Download files using huggingface_hub with default cache directory
    onnx_model_path = hf_hub_download(
        repo_id="rhasspy/piper-voices", 
        filename="en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    )
    json_model_path = hf_hub_download(
        repo_id="rhasspy/piper-voices", 
        filename="en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    )

    return onnx_model_path, json_model_path

def text_to_speech(answer, AUDIO_DEVICE):
    """Convert text to speech and play the audio."""
    checksum = file_checksum(answer)
    cache_dir = f"{DEMO_DIR}/../cached"
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"answer-{checksum}.wav")

    # Check if the audio file is already cached
    if os.path.exists(filename):
        print(f"Playing cached audio file: {filename}")
        os.system(f"aplay -D {AUDIO_DEVICE} {filename}")
        return

    # Ensure models are downloaded
    VOICE_MODEL_ONNX_FILE, _ = download_model_files()

    # Generate audio using the Piper model
    print(f"Generating audio for: {answer}")
    text_to_speech_command = (
        f"echo \"{answer}\" | {DEMO_DIR}/../models/piper/piper --model {VOICE_MODEL_ONNX_FILE} --output_file {filename}"
    )
    os.system(text_to_speech_command)
  
    # Play the generated audio file
    os.system(f"aplay -D {AUDIO_DEVICE} {filename}")