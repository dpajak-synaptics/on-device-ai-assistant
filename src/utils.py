import os
import subprocess
import hashlib
import json

def file_checksum(content: str, hash_length: int = 16) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:hash_length]

def wait_for_audio():


    
    print('Waiting for audio device...')
    while True:
        process=os.popen("aplay -l | grep USB\ Audio && sleep 0.5")
        output=process.read()
        process.close()
        if 'USB Audio' in output:
            print(output)
            break
    return

def get_usb_audio_device():
    """
    Finds the audio device ID for a USB Audio device using `aplay -l`.
    
    Returns:
        str: The audio device identifier (e.g., 'hw:1,0'), or None if no USB Audio device is found.
    """
    wait_for_audio()

    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        for line in lines:
            if "USB Audio" in line:
                # Extract card and device numbers
                card_line = line.split()
                card_index = card_line[1][:-1]  # Removes trailing colon
                device_name = f"plughw:{card_index},0"
                print(f"Found audio device: {device_name}")
                return device_name  
    except subprocess.CalledProcessError as e:
        print(f"Error running `aplay -l`: {e}")
        return None
    return None