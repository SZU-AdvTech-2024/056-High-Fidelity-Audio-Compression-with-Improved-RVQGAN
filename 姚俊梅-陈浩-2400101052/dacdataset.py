import logging
import torch
import librosa
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from audiotools.core import AudioSignal

class DACDataset(Dataset):
    def __init__(
        self, 
        filelist: str, 
        sample_rate: int = 44100,
        duration: float = 0.38,
        padding_mode: str = 'constant',
        audio_channel: int = 2,  # 支持指定音频通道
    ):
        super().__init__()

        filelist = Path(filelist)

        self.files = [
            Path(line.strip())  # 直接将相对路径转换为Path对象
            for line in filelist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("._")
        ]
        self.sample_rate = sample_rate
        self.slice_samples = int(sample_rate * duration)
        self.padding_mode = padding_mode
        self.audio_channel = audio_channel  # 保存音频通道数

    def __len__(self):
        return len(self.files)

    def get_item(self, idx):
        file = self.files[idx]
        try:
            audio, _ = librosa.load(file, sr=self.sample_rate, mono=(self.audio_channel == 1))
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            return None

        # 对音频进行截取或填充
        if self.audio_channel == 2 and audio.ndim == 1:
            # 单通道扩展为双通道（复制一份）
            audio = np.stack([audio, audio], axis=0)
        if self.slice_samples is not None:
            if audio.shape[1] > self.slice_samples:
                start = np.random.randint(0, audio.shape[1] - self.slice_samples)
                audio = audio[:,start : start + self.slice_samples]
            elif audio.shape[1] < self.slice_samples:
                padding = self.slice_samples - audio.shape[1]
                audio = np.pad(audio, (0, padding), mode=self.padding_mode)

        # 处理音频归一化
        if len(audio) == 0:
            logging.error(f"Empty audio after processing: {file}")
            return None

        max_value = np.abs(audio).max()
        if max_value > 1.0:
            audio = audio / max_value

        # 调整通道数
        if self.audio_channel == 2 and audio.ndim == 1:
            # 单通道扩展为双通道（复制一份）
            audio = np.stack([audio, audio], axis=0)
        elif self.audio_channel == 1 and audio.ndim == 2:
            # 如果加载的是双通道但只需要单通道，取平均值变为单通道
            audio = np.mean(audio, axis=0)

        # 转换为 PyTorch 张量
        audio = torch.from_numpy(audio)
        if audio.ndim==1:
            audio = audio.unsqueeze(0)#[channel,length]
        
        return {
            "audio": audio,
            "filename": str(file)
        }
        
    def __getitem__(self, idx):
        try:
            
            item = self.get_item(idx)
            if item is None:                
                print(f"None in {idx}")

                raise ValueError(f"Invalid item at index {idx}")
            return item
        except Exception as e:
            logging.error(f"Error loading item at index {idx}: {e}")
            return None

class DACtrainDataset(Dataset):
    def __init__(
        self, 
        filelist: str, 
        sample_rate: int = 44100,
        duration: float = 0.38,
        padding_mode: str = 'constant',
        audio_channel: int = 2,  # 支持指定音频通道
        batchsize:int = 16,
    ):
        super().__init__()

        filelist = Path(filelist)

        self.files = [
            Path(line.strip())  # 直接将相对路径转换为Path对象
            for line in filelist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("._")
        ]
        self.sample_rate = sample_rate
        self.slice_samples = int(sample_rate * duration)
        self.padding_mode = padding_mode
        self.audio_channel = audio_channel  # 保存音频通道数
        self.num_segments=batchsize

    def __len__(self):
        return len(self.files)

    def get_item(self, idx):
        file = self.files[idx]
        

        try:
            audio, sr = librosa.load(file, sr=self.sample_rate, mono=(self.audio_channel == 1))
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            return None


        num_segments = self.num_segments
        if audio.ndim == 1:
    # 获取音频长度
            length = len(audio)
            
            # 如果长度小于所需的 slice_samples，则进行填充
            if length < self.slice_samples:
                # 计算需要填充的长度
                padding_length = self.slice_samples - length
                # 用 0 进行填充，使音频达到所需长度
                audio = np.pad(audio, (0, padding_length), mode='constant')
                length = self.slice_samples
            
            # 将一维音频堆叠成二维
            audio = np.stack([audio, audio], axis=0)

        else:
            # 如果是二维音频，获取最后一个通道的长度
            length = len(audio[-1])
            if length < self.slice_samples:
                padding_length = self.slice_samples - length
                # 对最后一个通道进行填充
                audio[-1] = np.pad(audio[-1], (0, padding_length), mode='constant')
                length = self.slice_samples

        segment_duration = length // num_segments  # 每份的长度
        extract_duration = self.slice_samples  # 0.38秒的样本数

        # 存储抽取的音频片段
        extracted_segments = []

        for i in range(num_segments):
            start_idx = i * segment_duration #开始
            end_idx = start_idx + segment_duration # 结束
            
            # 计算抽取起始索引，确保不超出边界
           

    
    # 随机选择一个抽取起始索引
            if(segment_duration-extract_duration-1<=0):
                random_offset = 0
            else:
                random_offset = np.random.randint(0, segment_duration-extract_duration-1)#起始偏移量
            extract_start_idx = start_idx + random_offset#起始
            
            # 抽取音频片段并存储
            segment = audio[:,extract_start_idx:extract_start_idx + extract_duration]
            if len(segment[-1])!=extract_duration:
                segment0 = np.pad(segment[0], (0, extract_duration-len(segment[-1])), mode='constant')
                segment1 = np.pad(segment[1], (0, extract_duration-len(segment[-1])), mode='constant')
                segment=np.stack([segment0, segment1], axis=0)
            if self.audio_channel == 1 and segment.ndim == 2:
            # 如果加载的是双通道但只需要单通道，取平均值变为单通道
                segment = np.mean(segment, axis=0)            
            if len(segment) == 0:
                logging.error(f"Empty audio after processing: {file}")
                return None
            max_value = np.abs(segment).max()
            if max_value > 1.0:
                segment = segment / max_value

            segment = torch.from_numpy(segment)
            if segment.ndim==1:
                segment = segment.unsqueeze(0)#[channel,length]
            if len(segment[-1])!=self.slice_samples:
                print("find!")
            extracted_segments.append(segment)
        
        
        return {
            "audio": extracted_segments,
            "filename": str(file),
            "idx":idx
        }

    def __getitem__(self, idx):
        try:
            
            item = self.get_item(idx)
            if item is None:                
                print(f"None in {idx}")

                raise ValueError(f"Invalid item at index {idx}")
            return item
        except Exception as e:
            logging.error(f"Error loading item at index {idx}: {e}")
            return None


def traincollate(batch):
    audio=[]
    for sample in batch:
        if sample == None:
            continue
        for clip in sample['audio']:
            audio.append(clip)
    audio=torch.stack(audio)
    return {
        'audio':audio
    }
