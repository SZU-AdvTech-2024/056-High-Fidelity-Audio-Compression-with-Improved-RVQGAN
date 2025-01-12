import torch
from pathlib import Path
from audiotools.core import AudioSignal
from torch.utils.data import DataLoader
from dacdataset import DACDataset
from mss import MSS
from tqdm import tqdm
from model.demucs import Demucs

# 初始化教师模型
teacher = Demucs(audio_channels=2)
teacher = teacher.to('cpu')
teacher.load_state_dict(torch.load("/home/yuechengl/mss/demucs-e07c671f.th"))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = MSS(mode="distill").to(device)
model.eval()  # 设置模型为评估模式

# 加载保存的模型权重
checkpoint_path = "/home/ch/descript-audio-codec/ckpt/generator_best_mel_loss_0_9739_step_834000.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

# 准备推理数据集
test_dataset = DACDataset(filelist="/home/yuechengl/mss/inf.txt", sample_rate=44100, duration=16)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

# 创建一个文件夹来保存生成的音频
output_dir = Path("inference_results/abc")
output_dir.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
        if batch is None or 'audio' not in batch:
            continue  # 跳过无效的批次

        try:
            audio = batch.get('audio')  # 使用 get 方法安全地获取 audio
            if audio is None or audio.size(0) == 0:
                continue
            signal = AudioSignal(audio, sample_rate=44100)

            # 直接使用整个信号，不进行维度检查
            pre_audio = model.model_dac.preprocess(signal.audio_data, sample_rate=44100)
            pre_audio = pre_audio.to(device)

            # 使用整个音频进行推理
            semetic_real = teacher.encode(pre_audio.to('cpu'))
            semetic_real = semetic_real.to(device)

            # 确保 audio 是有效的
            audio = audio.to(device)

            # 调用生成器进行推理
            out = model(pre_audio, semetic_real)

            # 处理输出的音频
            if out is not None and isinstance(out, dict) and "audio" in out:
                generated_audio = out["audio"]
                if generated_audio is not None and generated_audio.size(0) > 0:
                    generated_audio = generated_audio.cpu()
                    recons = AudioSignal(generated_audio, sample_rate=44100)

                    # 保存推理生成的音频
                    output_path_recon = output_dir / f"reconstructed_audio_{batch_idx}.wav"
                    recons.write(output_path_recon)
                    print(f"Saved reconstructed audio: {output_path_recon}")

            else:
                print(f"No audio output for batch {batch_idx}")

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
