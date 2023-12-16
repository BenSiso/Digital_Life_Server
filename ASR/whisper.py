import logging
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperService:
    def __init__(self, 
                 model_id_or_path: str = None,
                 device: str = 'cuda'):
        logging.info('Initializing ASR Service (Whisper)...')

        torch_dtype = torch.float16 if "cuda" in device else torch.float32

        # Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: 
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id_or_path, 
            torch_dtype=torch_dtype, 
            # low_cpu_mem_usage=True, 
            use_safetensors=False
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id_or_path)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

    def infer(self, wav_path: str):
        stime = time.time()
        result = self.pipe(wav_path)
        etime = time.time()
        logging.info(f"ASR Result: {result}. time used {etime - stime:.2f}.")
        return result['text']


if __name__ == '__main__':
    model_id_or_path = r"G:\models\whisper-large-v3"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    service = WhisperService(model_id_or_path, device)

    wav_path = r"C:\Users\AgainstEntropy\Documents\录音\录音_mix.wav"
    result = service.infer(wav_path)
    print(result)