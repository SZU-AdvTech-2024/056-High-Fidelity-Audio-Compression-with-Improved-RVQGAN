import audiotools as at
import numpy as np
import torch
import tqdm
from audiotools.core import AudioSignal
from mss import MSS
from model.demucs import Demucs

def xxx(model,audio,teacher,device):
     
     signal = AudioSignal(audio, sample_rate=44100)

            # 直接使用整个信号，不进行维度检查
     pre_audio = model.model_dac.preprocess(signal.audio_data, sample_rate=44100)
     pre_audio = pre_audio.to(device)

            # 使用整个音频进行推理
     semetic_real = teacher.encode(pre_audio.to('cpu'))
     semetic_real = semetic_real.to(device)
     return pre_audio,semetic_real

def main(
    file: str,
    n_samples: int = 1024,
    device: str = "cuda",
):
    with open(file, 'r') as f:
        files = [line.strip() for line in f][:n_samples]
        
    signals = [
        at.AudioSignal.salient_excerpt(f, loudness_cutoff=-20, duration=1.0)
        for f in files
    ]

    with torch.no_grad():
        teacher = Demucs(audio_channels=2)
        teacher = teacher.to('cpu')
        teacher.load_state_dict(torch.load("/home/yuechengl/mss/demucs-e07c671f.th"))
        model = MSS(mode='dac')
        checkpoint_path = "/home/yuechengl/mss/multigpucheckpoints/dac/best_mel_loss_0_7051_step_154000.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        model=model.to(device)
        model.eval()

        codes = []
        for x in tqdm.tqdm(signals):
            x = x.to(device)
            r,s = xxx(model=model,audio=x.audio_data,teacher=teacher,device='cuda')
            o = model(r, s)
            codes.append(o["codes"].cpu())

        codes = torch.cat(codes, dim=-1)
        entropy = []

        for i in range(codes.shape[1]):
            codes_ = codes[0, i, :]
            counts = torch.bincount(codes_)
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))

        pct = sum(entropy) / (10 * len(entropy))
        print(f"Entropy for each codebook: {entropy}")
        print(f"Effective percentage: {pct * 100}%")


if __name__ == "__main__":
        main(file='/home/yuechengl/mss/filelist/music/music_val_filelist.txt')