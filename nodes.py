import torch
import random
import librosa
import zipfile
import torchaudio
import numpy as np
import os,sys
import folder_paths
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, now_dir)
input_dir = folder_paths.get_input_directory()
output_dir = os.path.join(folder_paths.get_output_directory(),"cosyvoice_dubb")
pretrained_models = os.path.join(now_dir,"pretrained_models")

from modelscope import snapshot_download

import ffmpeg
import audiosegment
from srt import parse as SrtPare
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
inference_mode_list = [
    'v1/预训练音色 (SFT)',
    'v1/3s极速复刻 (Zero-shot)',
    'v1/跨语种复刻 (Cross-lingual)',
    'v1/自然语言控制 (Instruct)',
    'v2/零样本语音克隆 (Zero-shot)',
    'v2/跨语种语音克隆 (Cross-lingual)',
    'v2/指令控制合成 (Instruct)',
]

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
prompt_sr, target_sr = 16000, 24000 # CosyVoice2 sample rate is 24k
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(prompt_sr * 0.2))], dim=1)
    return speech

def speed_change(input_audio, speed, sr):
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")
    raw_audio = input_audio.astype(np.int16).tobytes()
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)
    output_stream = input_stream.filter('atempo', speed)
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )
    processed_audio = np.frombuffer(out, np.int16)
    return processed_audio

class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"
    CATEGORY = "AIFSH_CosyVoice"
    def encode(self,text):
        return (text, )

from time import time as ttime
class CosyVoiceNode:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None
        self.cosyvoice2 = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("TEXT",),
                "speed":("FLOAT", {"default": 1.0}),
                "inference_mode":(inference_mode_list, {"default": 'v1/预训练音色 (SFT)'}),
                "sft_dropdown":(sft_spk_list, {"default":"中文女"}),
                "seed":("INT", {"default": 42})
            },
            "optional":{
                "prompt_text":("TEXT",),
                "prompt_wav": ("AUDIO",),
                "instruct_text":("TEXT",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "AIFSH_CosyVoice"

    def generate(self,tts_text,speed,inference_mode,sft_dropdown,seed,
                 prompt_text=None,prompt_wav=None,instruct_text=None):
        t0 = ttime()
        
        version, mode_name = inference_mode.split('/', 1)
        current_sr = target_sr

        # Model loading
        if version == 'v2':
            if self.cosyvoice2 is None:
                print("Loading CosyVoice2 model...")
                self.cosyvoice2 = CosyVoice2('iic/CosyVoice2-0.5B', fp16=torch.cuda.is_available())
            current_sr = self.cosyvoice2.sample_rate
            self.model_dir = 'iic/CosyVoice2-0.5B'
            if self.cosyvoice is not None:
                print("Unloading CosyVoice v1 model.")
                self.cosyvoice = None
                del self.cosyvoice
        else: # version == 'v1'
            model_id = None
            if '自然语言控制' in mode_name:
                model_id = "iic/CosyVoice-300M-Instruct"
            elif '3s极速复刻' in mode_name or '跨语种复刻' in mode_name:
                model_id = "iic/CosyVoice-300M"
            elif '预训练音色' in mode_name:
                model_id = "iic/CosyVoice-300M-SFT"
            
            if model_id:
                model_dir = os.path.join(pretrained_models, model_id.split('/')[1])
                if self.model_dir != model_dir:
                    print(f"Loading CosyVoice v1 model: {model_id}")
                    snapshot_download(model_id=model_id, local_dir=model_dir)
                    self.model_dir = model_dir
                    self.cosyvoice = CosyVoice(self.model_dir, fp16=torch.cuda.is_available())
                    current_sr = self.cosyvoice.sample_rate
                    if self.cosyvoice2 is not None:
                        print("Unloading CosyVoice2 model.")
                        self.cosyvoice2 = None
                        del self.cosyvoice2
                current_sr = self.cosyvoice.sample_rate
            else:
                 raise ValueError(f"Unknown v1 mode: {mode_name}")

        # Prepare prompt audio
        prompt_speech_16k = None
        if prompt_wav:
            waveform = prompt_wav['waveform'].squeeze(0)
            source_sr = prompt_wav['sample_rate']
            speech = waveform.mean(dim=0,keepdim=True)
            if source_sr != prompt_sr:
                speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
            prompt_speech_16k = postprocess(speech)

        # Inference logic
        set_all_random_seed(seed)
        output = None

        if version == 'v1':
            print(f'get v1 inference request: {mode_name}')
            if '预训练音色' in mode_name:
                output = self.cosyvoice.inference_sft(tts_text, sft_dropdown, stream=False)
            elif '3s极速复刻' in mode_name:
                assert prompt_speech_16k is not None, "V1 3s极速复刻 (Zero-shot) mode requires prompt_wav."
                assert prompt_text, "V1 3s极速复刻 (Zero-shot) mode requires prompt_text."
                output = self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False)
            elif '跨语种复刻' in mode_name:
                assert prompt_speech_16k is not None, "V1 跨语种复刻 (Cross-lingual) mode requires prompt_wav."
                output = self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False)
            elif '自然语言控制' in mode_name:
                assert instruct_text, "V1 自然语言控制 (Instruct) mode requires instruct_text."
                output = self.cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=False)
        
        elif version == 'v2':
            print(f'get v2 inference request: {mode_name}')
            
            if '零样本语音克隆' in mode_name:
                assert prompt_speech_16k is not None, f"V2 mode '{mode_name}' requires prompt_wav."
                assert prompt_text, "V2 零样本语音克隆 (Zero-shot) mode requires prompt_text."
                output = self.cosyvoice2.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False)
            elif '跨语种语音克隆' in mode_name:
                assert prompt_speech_16k is not None, f"V2 mode '{mode_name}' requires prompt_wav."
                output = self.cosyvoice2.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False)
            elif '指令控制合成' in mode_name:
                assert prompt_speech_16k is not None, f"V2 mode '{mode_name}' requires prompt_wav."
                assert instruct_text, "V2 指令控制合成 (Instruct) mode requires instruct_text."
                output = self.cosyvoice2.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=False)

        if output is None:
            raise ValueError(f"Could not generate audio for mode: {inference_mode}")

        # Post-processing
        output_list = []
        for out_dict in output:
            tts_speech = out_dict['tts_speech']
            if isinstance(tts_speech, torch.Tensor):
                output_numpy = tts_speech.squeeze(0).numpy() * 32768
            else: # CosyVoice2 may return numpy array
                output_numpy = tts_speech * 32768

            output_numpy = output_numpy.astype(np.int16)
            if speed != 1.0:
                output_numpy = speed_change(output_numpy,speed,current_sr)
            output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
        
        t1 = ttime()
        print("cost time \t %.3f" % (t1-t0))
        audio = {"waveform": torch.cat(output_list,dim=1).unsqueeze(0),"sample_rate":current_sr}
        return (audio,)

class CosyVoiceDubbingNode:
    def __init__(self):
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_srt":("SRT",),
                "prompt_wav": ("AUDIO",),
                "language":(["<|zh|>","<|en|>","<|jp|>","<|yue|>","<|ko|>"],),
                "if_single":("BOOLEAN",{
                    "default": True
                }),
                "seed":("INT",{
                    "default": 42
                })
            },
            "optional":{
                "prompt_srt":("SRT",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "AIFSH_CosyVoice"

    def generate(self,tts_srt,prompt_wav,language,if_single,seed,prompt_srt=None):
        model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
        snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
        set_all_random_seed(seed)
        if self.cosyvoice is None:
            self.cosyvoice = CosyVoice(model_dir)
        
        with open(tts_srt, 'r', encoding="utf-8") as file:
            text_file_content = file.read()
        text_subtitles = list(SrtPare(text_file_content))

        if prompt_srt:
            with open(prompt_srt, 'r', encoding="utf-8") as file:
                prompt_file_content = file.read()
            prompt_subtitles = list(SrtPare(prompt_file_content))

        waveform = prompt_wav['waveform'].squeeze(0)
        source_sr = prompt_wav['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        speech_numpy = speech.squeeze(0).numpy() * 32768
        speech_numpy = speech_numpy.astype(np.int16)
        audio_seg = audiosegment.from_numpy_array(speech_numpy,prompt_sr)
        assert audio_seg.duration_seconds > 3, "prompt wav should be > 3s"
        new_audio_seg = audiosegment.silent(0,self.cosyvoice.sample_rate)
        for i,text_sub in enumerate(text_subtitles):
            start_time = text_sub.start.total_seconds() * 1000
            end_time = text_sub.end.total_seconds() * 1000
            if i == 0:
                new_audio_seg += audio_seg[:start_time]
            
            if if_single:
                curr_tts_text = language + text_sub.content
            else:
                curr_tts_text = language + text_sub.content[1:]
                speaker_id = text_sub.content[0]
            
            prompt_wav_seg = audio_seg[start_time:end_time]
            if prompt_srt:
                prompt_text_list = [prompt_subtitles[i].content]
            while prompt_wav_seg.duration_seconds < 30:
                for j in range(i+1,len(text_subtitles)):
                    j_start = text_subtitles[j].start.total_seconds() * 1000
                    j_end = text_subtitles[j].end.total_seconds() * 1000
                    if if_single:
                        prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                        if prompt_srt:
                            prompt_text_list.append(prompt_subtitles[j].content)
                    else:
                        if text_subtitles[j].content[0] == speaker_id:
                            prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                            if prompt_srt:
                                prompt_text_list.append(prompt_subtitles[j].content)
                for j in range(0,i):
                    j_start = text_subtitles[j].start.total_seconds() * 1000
                    j_end = text_subtitles[j].end.total_seconds() * 1000
                    if if_single:
                        prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                        if prompt_srt:
                            prompt_text_list.append(prompt_subtitles[j].content)
                    else:
                        if text_subtitles[j].content[0] == speaker_id:
                            prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                            if prompt_srt:
                                prompt_text_list.append(prompt_subtitles[j].content)

                if prompt_wav_seg.duration_seconds > 3:
                    break
            print(f"prompt_wav {prompt_wav_seg.duration_seconds}s")
            prompt_wav_seg.export(os.path.join(output_dir,f"{i}_prompt.wav"),format="wav")
            prompt_wav_seg_numpy = prompt_wav_seg.to_numpy_array() / 32768
            prompt_speech_16k = postprocess(torch.Tensor(prompt_wav_seg_numpy).unsqueeze(0))
            if prompt_srt:
                prompt_text = ','.join(prompt_text_list)
                print(f"prompt_text:{prompt_text}")
                curr_output = self.cosyvoice.inference_zero_shot(curr_tts_text,prompt_text,prompt_speech_16k, stream=False)
            else:
                curr_output = self.cosyvoice.inference_cross_lingual(curr_tts_text, prompt_speech_16k, stream=False)
            
            curr_output_numpy = curr_output['tts_speech'].squeeze(0).numpy() * 32768
            curr_output_numpy = curr_output_numpy.astype(np.int16)
            text_audio = audiosegment.from_numpy_array(curr_output_numpy,self.cosyvoice.sample_rate)
            text_audio_dur_time = text_audio.duration_seconds * 1000

            if i < len(text_subtitles) - 1:
                nxt_start = text_subtitles[i+1].start.total_seconds() * 1000
                dur_time =  nxt_start - start_time
            else:
                org_dur_time = audio_seg.duration_seconds * 1000
                dur_time = org_dur_time - start_time
            
            ratio = text_audio_dur_time / dur_time

            if text_audio_dur_time > dur_time:
                tmp_numpy = speed_change(curr_output_numpy,ratio,self.cosyvoice.sample_rate)
                tmp_audio = audiosegment.from_numpy_array(tmp_numpy,self.cosyvoice.sample_rate)
                tmp_audio += audiosegment.silent(dur_time - tmp_audio.duration_seconds*1000,self.cosyvoice.sample_rate)
            else:
                tmp_audio = text_audio + audiosegment.silent(dur_time - text_audio_dur_time,self.cosyvoice.sample_rate)
          
            new_audio_seg += tmp_audio

            if i == len(text_subtitles) - 1:
                new_audio_seg += audio_seg[end_time:]

        output_numpy = new_audio_seg.to_numpy_array() / 32768
        audio = {"waveform": torch.stack([torch.Tensor(output_numpy).unsqueeze(0)]),"sample_rate":self.cosyvoice.sample_rate}
        return (audio,)

class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_CosyVoice"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)
