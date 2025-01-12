import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from audiotools import util
from audiotools import ml
from audiotools.core import AudioSignal
from audiotools.ml.decorators import Tracker
from mss import MSS
from model.demucs import Demucs
from dac.model.discriminator import Discriminator as disc
import re
from dac.nn.loss import (
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
    GANLoss,
    L1Loss,
)
from dacdataset import DACDataset,DACtrainDataset,traincollate
from torch.utils.tensorboard import SummaryWriter
import time

# 初始化TensorBoard

# 学习率调度器
def ExponentialLR(optimizer, gamma: float = 0.999996):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

# 初始化加速器
accel = ml.Accelerator(False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print("Available CUDA devices:", torch.cuda.device_count())
#print(torch.cuda.current_device())
mode = "dac"
# 初始化模型
model = MSS(mode = mode).to(device)
model = torch.nn.DataParallel(model)
model = model.cuda()
writer = SummaryWriter(log_dir=f"./runs/"+f"{mode}/"+time.strftime('%y-%m-%d_%H.%M', time.localtime()))

discriminator = disc().to(device)
discriminator = torch.nn.DataParallel(discriminator)
discriminator=discriminator.cuda()
#设置语义老师
if mode != 'dac':
    teacher=Demucs(audio_channels=2)
    teacher = teacher.to('cpu')
    teacher.load_state_dict(torch.load("/home/yuechengl/mss/demucs-e07c671f.th"))
# 准备模型
#model = accel.prepare_model(model)
#discriminator = accel.prepare_model(discriminator)

# 优化器和调度器
optimizer_g = torch.optim.AdamW(params=model.parameters(), lr=0.0001, betas=[0.8, 0.99])
scheduler_g = ExponentialLR(optimizer_g)

optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=0.0001, betas=[0.8, 0.99])
scheduler_d = ExponentialLR(optimizer_d)

# 定义损失函数
waveform_loss = L1Loss().to(device)
stft_loss = MultiScaleSTFTLoss().to(device)
mel_loss = MelSpectrogramLoss().to(device)
gan_loss = GANLoss(discriminator).to(device)

# 加载数据集
train_dataset = DACtrainDataset(filelist="/home/ch/descript-audio-codec/filelist/speech_train_DAC_meger.txt", sample_rate=44100, duration=0.38, batchsize=4)
train_loader = DataLoader(train_dataset, batch_size=4, num_workers=6,collate_fn=traincollate)

val_dataset = DACtrainDataset(filelist="/home/ch/descript-audio-codec/filelist/speech_val_DAC.txt", sample_rate=44100, duration=5,batchsize=1)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2, collate_fn=traincollate,shuffle=False)

# 初始化 Tracker
tracker = Tracker()

# 保存验证损失和最优模型信息a
metadata = {
    "val_loss": [],
    "best_val_loss": float('inf'),  # 用于跟踪最好的验证损失
}
# 定义超参数
batch_size = 72
epochs = 100000
noise_dim = 100
audio_dim = 16000  # 例如1秒的音频采样率为16kHz
accumulation_steps = 5  # 梯度累积步数
validation_steps = 8000  # 每8000步进行一次评估
if mode == 'distill':
    lambdas = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
        "vq/distill_loss":10
    }
else:
    lambdas = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0
        }

class TrainingState:
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                 mel_loss, stft_loss, waveform_loss, gan_loss, val_data, tracker):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.mel_loss = mel_loss
        self.stft_loss = stft_loss
        self.waveform_loss = waveform_loss
        self.gan_loss = gan_loss
        self.val_data = val_data
        self.tracker = tracker

# 创建训练状态对象
state = TrainingState(
    generator=model,
    discriminator=discriminator,
    optimizer_g=optimizer_g,
    optimizer_d=optimizer_d,
    scheduler_g=scheduler_g,
    scheduler_d=scheduler_d,
    mel_loss=mel_loss,
    stft_loss=stft_loss,
    waveform_loss=waveform_loss,
    gan_loss=gan_loss,
    val_data=val_dataset,
    tracker=tracker
)

def load_checkpoint(state, load_path="./checkpoints/"+f'{mode}'):
    load_path = Path(load_path)
    
    step_pattern = re.compile(r"step_(\d+)\.pth")

    def get_step(filename):
        match = step_pattern.search(filename.name)
        return int(match.group(1)) if match else -1  # 若无匹配到则返回 -1
    
    generator_checkpoints = sorted(load_path.glob("generator_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
    discriminator_checkpoints = sorted(load_path.glob("discriminator_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)

    generator_checkpoint = generator_checkpoints[0] if generator_checkpoints else None
    discriminator_checkpoint = discriminator_checkpoints[0] if discriminator_checkpoints else None

    generator_extra = sorted(load_path.glob("generator_extra_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)[0]
    discriminator_extra = sorted(load_path.glob("discriminator_extra_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)[0]


    # 找到最新的生成器检查点
    
    
    # 加载生成器的状态字典
    state_dict = torch.load(generator_checkpoint, map_location=device)
    state.generator.module.load_state_dict(state_dict)

    # 加载生成器的其他状态：优化器、调度器、元数据等
    extra = torch.load(generator_extra, map_location=device)
    state.optimizer_g.load_state_dict(extra["optimizer.pth"])
    state.scheduler_g.load_state_dict(extra["scheduler.pth"])
    state.tracker.load_state_dict(extra["tracker.pth"])  # 加载 tracker 的状态
    metadata.update(extra["metadata.pth"])  # 更新元数据
    #  # 恢复保存的步数
    try:
        step = extra["step.pth"]
    except Exception as e:
        step = 240000

    print(f"已加载生成器检查点：{generator_checkpoint} 和步数：{step}")
    
    # 同样的方法加载判别器

    
    state_dict = torch.load(discriminator_checkpoint, map_location=device)
    state.discriminator.module.load_state_dict(state_dict)
    
    extra = torch.load(discriminator_extra, map_location=device)
    state.optimizer_d.load_state_dict(extra["optimizer.pth"])
    state.scheduler_d.load_state_dict(extra["scheduler.pth"])

    print(f"已加载判别器检查点：{discriminator_checkpoint}")
    
    return step

def validate(state, val_dataloader, accel, writer, epoch):
    state.generator.eval()
    aggregated_output = {
        "loss": 0.0,
        "mel/loss": 0.0,
        "stft/loss": 0.0,
        "waveform/loss": 0.0,
    }
    total_batches = len(val_dataloader)
    filelists = []  # 用于收集文件名

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch is None:
                continue  # 跳过无效的批次

            batch = util.prepare_batch(batch, accel.device)
              # 调整为1维（或其他合适形状）
            signal = AudioSignal(batch['audio'], sample_rate=44100)
            if signal.audio_data.dim() == 2:
                signal.audio_data = signal.audio_data.unsqueeze(1)
            pre_audio = state.generator.module.model_dac.preprocess(signal.audio_data, sample_rate=44100)
            originlength = signal.audio_data.shape[-1]
            
            if mode !="dac":
                semetic_real = teacher.encode(pre_audio.to('cpu'))
                semetic_real = semetic_real.to(accel.device)
            else:
                semetic_real = None

            out = state.generator(pre_audio, semetic_real,length=originlength)
            new_length = out["audio"].size(-1)
            pad_size = originlength - new_length
            if pad_size > 0:
            # (前面的填充, 后面的填充)，只填充在时间维度
                signal.audio_data=signal.audio_data[...,:new_length]
            
            # 确保所有张量在同一设备上
            recons = AudioSignal(out["audio"], signal.sample_rate)
            recons.audio_data = recons.audio_data.to(accel.device)

            # 计算此批次的损失
            batch_output = {
                "mel/loss": state.mel_loss(recons, signal),
                "stft/loss": state.stft_loss(recons, signal),
                "waveform/loss": state.waveform_loss(recons, signal)
            }
            batch_output["loss"] = sum(batch_output.values())

            # 聚合所有批次的损失
            for key in aggregated_output:
                aggregated_output[key] += batch_output[key]

            state.tracker.track("val", total_batches)

            # 保存音频到 TensorBoard
            if batch_idx < 5:  # 例如，只保存前5个批次的音频
    # 分别保存左声道和右声道
                original_audio = signal.audio_data.squeeze(0)
                reconstructed_audio = recons.audio_data.squeeze(0)
                
                # 分别保存左声道和右声道
                writer.add_audio(f'batch{batch_idx}/Original_Left', original_audio[0], step//accumulation_steps, sample_rate=44100)
                writer.add_audio(f'batch{batch_idx}/Reconstructed_Left', reconstructed_audio[0], step//accumulation_steps, sample_rate=44100)

                writer.add_audio(f'batch{batch_idx}/Original_Right', original_audio[1], step//accumulation_steps, sample_rate=44100)
                
                writer.add_audio(f'batch{batch_idx}/Reconstructed_Right', reconstructed_audio[1], step//accumulation_steps, sample_rate=44100)

            # 收集文件名
            if 'filename' in batch:
                filelists.extend(batch['filename'])
            elif 'filenames' in batch:
                filelists.extend(batch['filenames'])
            # 根据您的数据结构，添加更多键

    # 计算平均损失，转移到以便于日志记录
    for key in aggregated_output:
        aggregated_output[key] = (aggregated_output[key] / total_batches).item()  # 计算平均值
        writer.add_scalar(f'Validation/{key}', aggregated_output[key], step)  # 写入TensorBoard

    writer.add_scalar('Validation/Mel_Loss', aggregated_output["mel/loss"], step)

    metadata["val_loss"].append(aggregated_output)
    metadata.setdefault("val_filenames", []).extend(filelists)  # 保存文件名

    # 返回前，确保输出格式正确
    aggregated_output = {key: {"value": value} for key, value in aggregated_output.items()}

    # 检查并保存最优模型
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    state.generator.train()
    return aggregated_output, filelists  # 返回损失和文件名

def checkpoint(state, filelists, step,realstep, save_path="./checkpoints/"+f"{mode}"):
    step_pattern = re.compile(r"step_(\d+)\.pth")

    def get_step(filename):
        match = step_pattern.search(filename.name)
        return int(match.group(1)) if match else -1  # 若无匹配到则返回 -1
    tags = ["latest"]
    save_path = Path(save_path)

    # 检查并创建保存路径
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_path.absolute()}")

    # 提取最新的验证损失中的mel/loss作为比较值
    current_mel_loss = metadata["val_loss"][-1]["mel/loss"]

    # 确保current_mel_loss是浮点数
    if isinstance(current_mel_loss, dict) and "value" in current_mel_loss:
        current_mel_loss = current_mel_loss["value"]

    # 格式化验证损失，保留4位小数，并替换小数点为下划线
    formatted_mel_loss = f"{current_mel_loss:.4f}".replace('.', '_')

    if metadata["best_val_loss"] > current_mel_loss:
        print(f"发现新的最佳生成器模型...")
        tags.append("best")
        metadata["best_val_loss"] = current_mel_loss

    for tag in tags:
        # 使用验证损失和step来命名文件，确保文件名唯一
        generator_filename = f"generator_{tag}_mel_loss_{formatted_mel_loss}_step_{step}.pth"
        discriminator_filename = f"discriminator_{tag}_mel_loss_{formatted_mel_loss}_step_{step}.pth"

        # Move models to CPU before saving
        generator_state_dict = accel.unwrap(state.generator).state_dict()
        discriminator_state_dict = accel.unwrap(state.discriminator).state_dict()

        generator_extra = {
            "optimizer.pth": dict(state.optimizer_g.state_dict().items()) ,
            "scheduler.pth": dict(state.scheduler_g.state_dict().items()) ,
            "tracker.pth": state.tracker.state_dict(),  # Ensure tracker state is on CPU
            "metadata.pth": metadata,
            "val_filenames": filelists,  # 包含文件名
            "step.pth": realstep, 
        }

        torch.save(generator_state_dict, save_path / generator_filename)
        torch.save(generator_extra, save_path / f"generator_extra_{tag}_mel_loss_{formatted_mel_loss}_step_{step}.pth")

        discriminator_extra = {
            "optimizer.pth": dict(state.optimizer_d.state_dict().items()),
            "scheduler.pth": dict(state.scheduler_d.state_dict().items()),
            "val_filenames": filelists  # 可选：包含文件名
        }

        torch.save(discriminator_state_dict, save_path / discriminator_filename)
        torch.save(discriminator_extra, save_path / f"discriminator_extra_{tag}_mel_loss_{formatted_mel_loss}_step_{step}.pth")

        if tag == "latest":
            generator_latest_files = sorted(save_path.glob("generator_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
            discriminator_latest_files = sorted(save_path.glob("discriminator_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
            generator_extra_latest_files = sorted(save_path.glob("generator_extra_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
            discriminator_extra_latest_files = sorted(save_path.glob("discriminator_extra_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)

            for old_file in generator_latest_files[10:]:  # 删除多余的文件，保留最新的10个
                old_file.unlink()
                print(f"已删除旧的generator latest文件: {old_file.name}")
            for old_file in discriminator_latest_files[10:]:
                old_file.unlink()
                print(f"已删除旧的discriminator latest文件: {old_file.name}")
            for old_file in generator_extra_latest_files[10:]:
                old_file.unlink()
                print(f"已删除旧的generator extra latest文件: {old_file.name}")
            for old_file in discriminator_extra_latest_files[10:]:
                old_file.unlink()
                print(f"已删除旧的discriminator extra latest文件: {old_file.name}")

        # 删除旧的best文件，仅保留最新的一个
        if tag == "best":
            generator_best_files = sorted(save_path.glob("generator_best_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
            discriminator_best_files = sorted(save_path.glob("discriminator_best_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
            generator_extra_best_files = sorted(save_path.glob("generator_extra_best_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
            discriminator_extra_best_files = sorted(save_path.glob("discriminator_extra_best_mel_loss_*_step_*.pth"), key=get_step, reverse=True)

            for old_file in generator_best_files[1:]:  # 删除除最新的best文件
                old_file.unlink()
                print(f"已删除旧的generator best文件: {old_file.name}")
            for old_file in discriminator_best_files[1:]:
                old_file.unlink()
                print(f"已删除旧的discriminator best文件: {old_file.name}")
            for old_file in generator_extra_best_files[1:]:
                old_file.unlink()
                print(f"已删除旧的generator extra best文件: {old_file.name}")
            for old_file in discriminator_extra_best_files[1:]:
                old_file.unlink()
                print(f"已删除旧的discriminator extra best文件: {old_file.name}")

    print(f"已保存检查点: {generator_filename} 和 {discriminator_filename}")



# 开始训练
try:
    step = load_checkpoint(state=state)  # 全局步数计数器
except Exception as e:
    print("未找到检查点 重新开始训练")
    step = 0

Step = step/accumulation_steps


for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    discriminator.train()
    epoch_output = []  # 保存每个batch的输出

    for i, audio_real in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
        if audio_real is None:
            continue  # 跳过无效的批次

        audio_real = util.prepare_batch(audio_real, accel.device)
        output = {}

        with torch.no_grad():
            signal = AudioSignal(audio_real['audio'], sample_rate=44100)
            
        # 确保形状一致
        if signal.audio_data.dim() == 2:
            signal.audio_data = signal.audio_data.unsqueeze(1)

        with accel.autocast():
            pre_audio = model.module.model_dac.preprocess(signal.audio_data, sample_rate=44100)
            originlength = signal.audio_data.shape[-1]

            if mode !='dac':

                with torch.no_grad():
                    semetic_real = teacher.encode(pre_audio.to('cpu'))
                semetic_real = semetic_real.to(accel.device)
            else:
                semetic_real=None
                
            out = model(pre_audio, semetic_real, length=originlength)
            new_length = out["audio"].size(-1)
            pad_size = originlength - new_length
            if pad_size > 0:
            # (前面的填充, 后面的填充)，只填充在时间维度
                signal.audio_data=signal.audio_data[...,:new_length]
            with torch.no_grad():
                recons = AudioSignal(out['audio'], sample_rate=44100)

            # 确保 recons 和 signal 的数据类型和设备一致
            recons.audio_data = recons.audio_data.to(signal.audio_data.dtype).to(signal.audio_data.device)

            commitment_loss = out["vq/commitment_loss"]
            codebook_loss = out["vq/codebook_loss"]
            if mode == 'distill':
                distill_loss = out["distill_loss"].mean()

        # 判别器损失计算
        with accel.autocast():
            output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)

            
        # 判别器的反向传播和梯度累积
        optimizer_d.zero_grad()
        accel.backward(output["adv/disc_loss"])
        output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)

        # 更新判别器的参数
        if (i + 1) % accumulation_steps == 0:  # 每 accumulation_steps 个batch更新一次
            optimizer_d.step()
            scheduler_d.step()

        # 生成器的前向传播和损失计算
        with accel.autocast():
            output["stft/loss"] = stft_loss(recons, signal).mean()
            output["mel/loss"] = mel_loss(recons, signal).mean()
            output["waveform/loss"] = waveform_loss(recons, signal).mean()
            output["adv/gen_loss"], output["adv/feat_loss"] = gan_loss.generator_loss(recons, signal)
            output["vq/commitment_loss"] = commitment_loss.mean()
            output["vq/codebook_loss"] = codebook_loss.mean()
            if mode=='distill':
                output["vq/distill_loss"] = distill_loss
            output["loss"] = sum(lambdas.get(k, 0) * output[k] for k in output if k in lambdas).mean()

        # 生成器的反向传播和梯度累积
        optimizer_g.zero_grad()
        accel.backward(output["loss"])
        output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)

        # 更新生成器的参数
        if (i + 1) % accumulation_steps == 0:  # 每 accumulation_steps 个batch更新一次
            optimizer_g.step()
            scheduler_g.step()

        accel.update()  # 更新状态

        # 记录学习率和批次大小
        output["other/learning_rate"] = optimizer_g.param_groups[0]["lr"]
        output["other/batch_size"] = signal.batch_size * accel.world_size

        # 将当前batch的输出添加到epoch_output
        epoch_output.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output.items()})
        state.tracker.track("train", 250000, completed=state.tracker.step)
        state.tracker.log("train", "value", history=False)

        # TensorBoard 记录
        if (i + 1) %accumulation_steps==0:
            current_step = step//accumulation_steps  # 使用全局步数作为current_step
            writer.add_scalar('Train/Mel_Loss', output["mel/loss"], current_step)
            writer.add_scalar('Train/STFT_Loss', output["stft/loss"], current_step)
            writer.add_scalar('Train/Waveform_Loss', output["waveform/loss"], current_step)
            if mode=='distill':
                writer.add_scalar('Train/distill_Loss',output["vq/distill_loss"],current_step)
            writer.add_scalar('Train/Loss', output["loss"], current_step)
            writer.add_scalar('Train/Learning_Rate', output["other/learning_rate"], current_step)

        # 每3000步进行一次评估和保存检查点
        step += 1
        if step % validation_steps == 0:
            print(f"Step {step}: 开始验证...")
            try:
                val_summary, val_filelists = validate(state, val_dataloader, accel, writer, epoch)
                checkpoint(state, val_filelists, step//accumulation_steps,realstep=step)
                print(f"Step {step}: 验证完成，保存检查点。")
                print(f"Validation Summary: {val_summary}")
            except Exception as e:
                print(f"Step {step}: 评估或保存检查点时发生错误: {e}")

    # 汇总每个epoch的输出
    epoch_summary = {k: sum(d[k] for d in epoch_output) / len(epoch_output) for k in epoch_output[0]}
    writer.add_scalar('Train/Epoch_Loss', epoch_summary["loss"], epoch)
    print(f"Training Summary: {epoch_summary}")

    # 每个epoch结束后清理缓存
    torch.cuda.empty_cache()

# ** Step 5: 训练结束后进行一次最终验证和保存检查点 **
# 确保所有步数都被评估和保存
if step % validation_steps >= 1:
    print(f"Step {step}: 进行最终验证...")
    try:
        val_summary, val_filelists = validate(state, val_dataloader, accel, writer, epoch)
        checkpoint(state, val_filelists, step)
        print(f"Step {step}: 最终验证完成，保存检查点。")
        print(f"Final Validation Summary: {val_summary}")
    except Exception as e:
        print(f"Step {step}: 评估或保存检查点时发生错误: {e}")

# 训练结束后关闭TensorBoard writer
writer.close()


