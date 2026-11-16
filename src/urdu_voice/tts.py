"""Local Urdu text-to-speech with Meta's MMS-TTS model (runs on CPU, no internet after the first download)."""
import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

MODEL_NAME = "facebook/mms-tts-urd-script_arabic"


class UrduVoice:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = VitsModel.from_pretrained(MODEL_NAME)
        self.sample_rate = self.model.config.sampling_rate  # 16000

    def say(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] < 3:  # nothing MMS can say (empty, or only characters it drops): it would crash
            return np.zeros(self.sample_rate // 4, dtype=np.float32)
        torch.manual_seed(0)  # the model adds a little randomness to timing; a fixed seed gives the same audio every time
        with torch.no_grad():
            return self.model(**inputs).waveform[0].numpy()