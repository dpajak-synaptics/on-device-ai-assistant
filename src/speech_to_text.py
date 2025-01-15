from huggingface_hub import hf_hub_download
import onnxruntime
import numpy as np
import time
import os
from tokenizers import Tokenizer


DEMO_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DEMO_DIR, '../models/')

class SpeechToText:
    def __init__(self, model_name="base", rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        
        self.rate = rate
        self.model_name = model_name

        # Load model weights
        self.preprocess, self.encode, self.uncached_decode, self.cached_decode = self._load_weights_from_hf_hub(model_name)

        # Initialize ONNX runtime sessions
        self.preprocess_session = onnxruntime.InferenceSession(self.preprocess)
        self.encode_session = onnxruntime.InferenceSession(self.encode)
        self.uncached_decode_session = onnxruntime.InferenceSession(self.uncached_decode)
        self.cached_decode_session = onnxruntime.InferenceSession(self.cached_decode)

        # Load tokenizer
        self.tokenizer = self._load_tokenizer_from_hf_hub()

        # Metrics
        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0

        # Warmup
        self.transcribe(np.zeros(int(rate), dtype=np.float32))

    def _load_weights_from_hf_hub(self, model_name):
        repo = "UsefulSensors/moonshine"
        model_name = model_name.split("/")[-1]
        return (
            hf_hub_download(repo, f"{x}.onnx", subfolder=f"onnx/{model_name}", local_dir=MODELS_DIR)
            for x in ("preprocess", "encode", "uncached_decode", "cached_decode")
        )

    def _load_tokenizer_from_hf_hub(self):
        repo = "UsefulSensors/moonshine-base"
        tokenizer_file = hf_hub_download(repo, "tokenizer.json", local_dir=MODELS_DIR)
        return Tokenizer.from_file(tokenizer_file)

    def _generate(self, audio, max_len=None):
        """Generate tokens from audio input."""
        if max_len is None:
            max_len = int((audio.shape[-1] / 16_000) * 6)  # max 6 tokens per second of audio

        preprocessed = self.preprocess_session.run([], {"args_0": audio})[0]
        seq_len = [preprocessed.shape[-2]]

        context = self.encode_session.run([], {"args_0": preprocessed, "args_1": seq_len})[0]
        inputs = [[1]]
        seq_len = [1]

        tokens = [1]
        logits, *cache = self.uncached_decode_session.run(
            [], {"args_0": inputs, "args_1": context, "args_2": seq_len}
        )

        for _ in range(max_len):
            next_token = logits.squeeze().argmax()
            tokens.append(next_token)
            if next_token == 2:
                break

            seq_len[0] += 1
            inputs = [[next_token]]
            logits, *cache = self.cached_decode_session.run(
                [],
                {
                    "args_0": inputs,
                    "args_1": context,
                    "args_2": seq_len,
                    **{f"args_{i+3}": x for i, x in enumerate(cache)},
                },
            )
        return tokens

    def transcribe(self, speech):
        """Transcribes the given speech audio to text."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self._generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch([tokens])[0]

        self.inference_secs += time.time() - start_time
        return text
