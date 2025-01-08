import time
from model.demucs import Demucs
import torch
from dacdataset import DACDataset
from torch.utils.data import DataLoader

teacher=Demucs(audio_channels=2)
teacher = teacher.to('cpu')
teacher.load_state_dict(torch.load("/home/yuechengl/mss/demucs-e07c671f.th"))
start_time = time.time()
train_dataset = DACDataset(filelist="/home/yuechengl/mss/filelist/music/music_train_filelist.txt", sample_rate=44100, duration=0.38)
train_loader = DataLoader(train_dataset, batch_size=15, num_workers=6)
for i, audio_real in enumerate(train_loader, ):
        if audio_real is None:
            continue  # 跳过无效的批次

        output = {}

        
            
        # 确保形状一致
        

        start_time = time.time()
        with torch.no_grad():
            semetic_real = teacher.encode(audio_real['audio'].to('cpu'))
        end_time = time.time()
        print(f"运行时间: {end_time - start_time} 秒")

# 你的代码块
for i in range(1000000):
    pass

end_time = time.time()

print(f"运行时间: {end_time - start_time} 秒")