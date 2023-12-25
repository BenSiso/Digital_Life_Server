import logging
import time

import torch


class WhisperService:
    def __init__(self, 
                 model_path: str = None,
                 flash_attn: bool = False,
                 device: str = 'cuda'):
        logging.info('Initializing ASR Service (Whisper)...')
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        if flash_attn:
            try:
                import flash_attn
            except ImportError:
                raise ImportError("Please install flash_attn using `pip install flash-attn --no-build-isolation`")

        torch_dtype = torch.float16 if "cuda" in device else torch.float32

        # Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: 
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            use_flash_attention_2=flash_attn,
            # low_cpu_mem_usage=True, 
            use_safetensors=False
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_path)

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

    def infer(self, wav_path: str) -> str:
        stime = time.time()
        result = self.pipe(wav_path)
        etime = time.time()
        logging.info(f"ASR Result: {result}. time used {etime - stime:.2f}.")
        return result['text']


class FasterWhisperService:
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'cuda',
                 int8: bool = False):
        logging.info('Initializing ASR Service (Faster Whisper)...')
        from faster_whisper import WhisperModel

        if device == "cuda":
            compute_type = "int8_float16" if int8 else "float16"

        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def infer(self, wav_path: str) -> str:
        stime = time.time()
        segments, _ = self.model.transcribe(wav_path)
        etime = time.time()
        segments = list(segments)
        logging.info(f"ASR Result:\n")
        for seg in segments:
            logging.info("[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text))
        logging.info(f"time used {etime - stime:.2f}.")

        result_text = " ".join([seg.text for seg in segments])
        return result_text


if __name__ == '__main__':
    model_path = "/home/paperspace/models/faster-whisper-large-v3"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    service = FasterWhisperService(model_path, device)

    wav_path = "/home/paperspace/Digital_Life_Server/ASR/mix.wav"
    result = service.infer(wav_path)
    print(result)