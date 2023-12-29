import logging
import os
import time
import torch

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

import soundfile
import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class xTTService:
    def __init__(self, 
                 model_dir: str = None,
                 device: str = 'cuda',
                 reference_audio: str = None):
        """
        初始化 xTTS 服务

        Args:
            model_dir (str): 模型路径
            device (str):
        """
        logging.info('Initializing xTTS Service')

        config = XttsConfig()
        config.load_json(f"{model_dir}/config.json")
        self.config = config

        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        model.to(device)
        self.model = model

        if reference_audio is None:
            reference_audio = f"{model_dir}/samples/en_sample.wav"
        self.conditions = model.get_conditioning_latents(audio_path=[reference_audio])

    def read(self, text: str, lang: str) -> np.ndarray:
        """
        读取文本并生成音频

        Args:
            text (str): 要转换为音频的文本

        Returns:
            np.ndarray: 生成的音频信号
        """
        assert lang in self.config.languages, f"Language {lang} not supported."
        out = self.model.inference(
            text,
            lang,
            *self.conditions,
        )
        return out['wav']

    def read_save(self, 
                  text: str, 
                  lang: str,
                  filename: str, 
                  sr: int = 24000):
        """
        读取文本并生成音频，并保存到文件

        Args:
            text (str): 要转换为音频的文本
            filename (str): 保存音频的文件名
            sr (int): 采样率 (default: 24000)
        """
        stime = time.time()
        wav = self.read(text, lang)
        soundfile.write(filename, wav, sr)
        logging.info('Xtts Synth Done, time used %.2f' % (time.time() - stime))


if __name__ == '__main__':
    model_dir = "/mnt/g/models/XTTS-v2"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    tts = xTTService(model_dir, device)

    text = "It took me quite a long time to develop a voice and now that I have it I am not going to be silent."
    wav_save_path = "./xtts_test.wav"
    print('Synthesizing...')
    tts.read_save(text, 'en', wav_save_path)
    print('Done! Find result in %s' % os.path.abspath(wav_save_path))
