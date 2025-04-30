import subprocess
import os
import numpy as np

class AudioManager:
    def __init__(self, device=None, sample_rate=16000):
        self._device = device or self._get_usb_audio_device()
        self._sample_rate = sample_rate
        self.arecord_process = None

    @property
    def device(self):
        """Get the current audio device."""
        return self._device

    @device.setter
    def device(self, new_device):
        """Set a new audio device."""
        self._device = new_device

    @property
    def sample_rate(self):
        """Get the current sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate):
        """Set a new sample rate."""
        self._sample_rate = new_sample_rate

    def play(self, filename):
        """Play the audio file using the specified audio device."""
        if not self._device:
            raise RuntimeError("Audio device not set.")
        
        device = "plugplay" if getattr(self, "_astra_version", "") == "1.6.0" else self._device
        subprocess.run(["aplay", "-D", device, filename], check=True)


    def start_arecord(self, chunk_size=512):
        """Start the arecord subprocess."""
        if self.arecord_process:
            self.stop_arecord()
        command = f"arecord -D {self._device} -f S16_LE -r {self._sample_rate} -c 2"
        self.arecord_process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=chunk_size, shell=True
        )

    def stop_arecord(self):
        """Stop the arecord subprocess."""
        if self.arecord_process:
            self.arecord_process.terminate()
            self.arecord_process.wait()
            self.arecord_process = None

    def read(self, chunk_size=512):
        """Read audio data from the arecord subprocess."""
        if not self.arecord_process:
            raise RuntimeError("arecord process not running.")

        while True:
            data = self.arecord_process.stdout.read(chunk_size * 4)
            if not data:
                break
            yield np.frombuffer(data, dtype=np.int16)[::2].astype(np.float32) / 32768.0

    def wait_for_audio(self):
        """Wait until a USB audio device is available."""
        print('Waiting for audio device...')
        while True:
            process = os.popen("aplay -l | grep USB\\ Audio && sleep 0.5")
            output = process.read()
            process.close()
            if 'USB Audio' in output:
                print(output)
                break

    def _get_astra_version(self):
        """Return Astra SDK version if available."""
        try:
            with open("/etc/astra_version", "r") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _create_asoundrc_for_sdk_1_6(self):
        """Write ~/.asoundrc configuration for Astra SDK v1.6 using detected USB Audio device."""
        try:
            pcm_output = subprocess.check_output("cat /proc/asound/pcm", shell=True, text=True)
            for line in pcm_output.splitlines():
                if "usb" in line.lower() and "audio" in line.lower():
                    hw_id = line.split(":")[0].strip()
                    card, dev = [str(int(x)) for x in hw_id.split("-")]
                    name = line.split(":")[1].strip().split()[0]
                    asoundrc_content = f"""
pcm.plugplay {{
    type plug
    slave {{
        pcm "hw:{card},{dev}"
        period_size 1024
        buffer_size 2048
    }}
}}
"""
                    with open(os.path.expanduser("~/.asoundrc"), "w") as f:
                        f.write(asoundrc_content)
                    print(f"~/.asoundrc created using USB Audio card {card}, device {dev} ({name})")
                    return
            print("No USB audio device found in /proc/asound/pcm")
        except Exception as e:
            print(f"Failed to create ~/.asoundrc: {e}")

    def _get_usb_audio_device(self):
        """Finds the audio device ID for a USB Audio device using `aplay -l`."""
        self.wait_for_audio()
        self._astra_version = self._get_astra_version()

        if self._astra_version == "1.6.0":
            self._create_asoundrc_for_sdk_1_6()
            print("Using 'plugplay' device for Astra SDK v1.6.0")
            return "plugplay"

        try:
            result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            for line in lines:
                if "USB Audio" in line:
                    card_line = line.split()
                    card_index = card_line[1][:-1]  # Removes trailing colon
                    device_name = f"plughw:{card_index},0"
                    # print(f"Found audio device: {device_name}")
                    return device_name
        except subprocess.CalledProcessError as e:
            print(f"Error running `aplay -l`: {e}")
            return None

        return "default"

