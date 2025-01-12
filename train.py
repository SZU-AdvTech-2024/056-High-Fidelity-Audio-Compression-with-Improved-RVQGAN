import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
from audiotools import util
from audiotools import ml
from audiotools.core import AudioSignal
from audiotools.ml.decorators import Tracker
from mss import MSS
from dac.model.discriminator import Discriminator as disc
from dac.nn.loss import (
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
    GANLoss,
    L1Loss,
)
from dacdataset import DACDataset
from torch.utils.tensorboard import SummaryWriter

def d_axis_distill_loss(feature, target_feature):
    n = min(feature.size(1), target_feature.size(1))
    distill_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
    return distill_loss

def t_axis_distill_loss(feature, target_feature, lambda_sim=1):
    n = min(feature.size(1), target_feature.size(1))
    l1_loss = torch.functional.l1_loss(feature[:, :n], target_feature[:, :n], reduction='mean')
    sim_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=-1))).mean()
    distill_loss = l1_loss + lambda_sim * sim_loss
    return distill_loss 

writer = SummaryWriter(log_dir="runs/dac_experiment")

# 学习率调度器
def ExponentialLR(optimizer, gamma: float = 0.999996):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

# 初始化加速器
accel = ml.Accelerator(False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
model = MSS().to(device)
discriminator = disc().to(device)

# 准备模型
model = accel.prepare_model(model)
discriminator = accel.prepare_model(discriminator)

# 优化器和调度器
optimizer_g = torch.optim.AdamW(params=model.parameters(), lr=0.0001, betas=[0.8, 0.99])
scheduler_g = ExponentialLR(optimizer_g)

optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=0.0001, betas=[0.8, 0.99])
scheduler_d = ExponentialLR(optimizer_d)

# 定义损失函数
waveform_loss = L1Loss()
stft_loss = MultiScaleSTFTLoss()
mel_loss = MelSpectrogramLoss()
gan_loss = GANLoss(discriminator)

# 加载数据集
train_dataset = DACDataset(filelist="/home/ch/descript-audio-codec/filelist/filelist_fma_mtg_not_exit_low.txt", sample_rate=44100, duration=0.38)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8)

val_dataset = DACDataset(filelist="/home/ch/descript-audio-codec/filelist/filelist_fma_mtg_not_exit_low.txt", sample_rate=44100, duration=0.38)
val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False)

# 初始化 Tracker
tracker = Tracker()

# 保存验证损失和最优模型信息
metadata = {
    "val_loss": [],
    "best_val_loss": float('inf'),  # 用于跟踪最好的验证损失
}

# 定义超参数
batch_size = 72
epochs = 100
noise_dim = 100
audio_dim = 16000  # 例如1秒的音频采样率为16kHz
lambdas = {
    "mel/loss": 100.0,
    "adv/feat_loss": 2.0,
    "adv/gen_loss": 1.0,
    "vq/commitment_loss": 0.25,
    "vq/codebook_loss": 1.0,
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

def train(model, discriminator, train_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d,
          gan_loss, stft_loss, mel_loss, waveform_loss, lambdas, accel, epochs):
    model.train()
    discriminator.train()

    for epoch in range(epochs):
        epoch_output = []  # 用于保存每个 batch 的输出
        for i, audio_real in enumerate(train_loader):
            output = {}
            audio = audio_real['audio']
            with torch.no_grad():
                signal = AudioSignal(audio, sample_rate=44100, device=device)

            with accel.autocast():
                out = model(audio)
                with torch.no_grad():
                    recons = AudioSignal(out['audio'], sample_rate=44100, device=device)
                commitment_loss = out["vq/commitment_loss"]
                codebook_loss = out["vq/codebook_loss"]

            with accel.autocast():
                output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)

            optimizer_d.zero_grad()
            accel.backward(output["adv/disc_loss"])
            accel.scaler.unscale_(optimizer_d)
            output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
            accel.step(optimizer_d)
            scheduler_d.step()

            with accel.autocast():

                # 在计算 mel_loss 之前检查 recons 和 signal 的形状及内容
                # print(f"Reconstructed audio shape: {recons.audio_data.shape}")
                # print(f"Original signal shape: {signal.audio_data.shape}")
                # print(f"Sample rate: {recons.sample_rate}")
                # 检查音频数据是否在 [-1, 1] 范围内
                # print(f"Reconstructed audio min/max: {recons.audio_data.min()}/{recons.audio_data.max()}")
                # print(f"Original signal min/max: {signal.audio_data.min()}/{signal.audio_data.max()}")

                # 计算 mel_loss
                output["stft/loss"] = stft_loss(recons, signal).mean()
                output["mel/loss"] = mel_loss(recons, signal).mean()
                output["waveform/loss"] = waveform_loss(recons, signal).mean()
                (
                    output["adv/gen_loss"],
                    output["adv/feat_loss"],
                ) = gan_loss.generator_loss(recons, signal)
                output["vq/commitment_loss"] = commitment_loss.mean()
                output["vq/codebook_loss"] = codebook_loss.mean()
                output["loss"] = sum(lambdas.get(k, 0) * output[k] for k in output if k in lambdas).mean()

            optimizer_g.zero_grad()
            accel.backward(output["loss"])
            accel.scaler.unscale_(optimizer_g)
            output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            accel.step(optimizer_g)
            scheduler_g.step()
            accel.update()

            output["other/learning_rate"] = optimizer_g.param_groups[0]["lr"]
            output["other/batch_size"] = signal.batch_size * accel.world_size

            epoch_output.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in output.items()})
            state.tracker.track("train", 250000, completed=state.tracker.step)
            state.tracker.log("train", "value", history=False)

            writer.add_scalar('Train/Loss', output["loss"].item(), epoch*len(train_loader) + i)
            writer.add_scalar('Train/Learning_Rate', optimizer_g.param_groups[0]["lr"], epoch * len(train_loader) + i)

        # 汇总每个 epoch 的输出
        epoch_summary = {k: sum(d[k] for d in epoch_output) / len(epoch_output) for k in epoch_output[0]}

        writer.add_scalar('Train/Epoch_Loss', epoch_summary["loss"], epoch)
        return epoch_summary

def validate(state, val_dataloader, accel, writer, epoch):
    state.generator.eval()
    aggregated_output = {
        "loss": 0.0,
        "mel/loss": 0.0,
        "stft/loss": 0.0,
        "waveform/loss": 0.0,
    }
    total_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in val_dataloader:
            batch = util.prepare_batch(batch, accel.device)
            signal = AudioSignal(batch['audio'], sample_rate=44100)
            out = state.generator(signal.audio_data, signal.sample_rate)
            recons = AudioSignal(out["audio"], signal.sample_rate)

            # Calculate losses for this batch
            batch_output = {
                "mel/loss": state.mel_loss(recons, signal),
                "stft/loss": state.stft_loss(recons, signal),
                "waveform/loss": state.waveform_loss(recons, signal)
            }
            batch_output["loss"] = sum(batch_output.values())

            # Aggregate losses over all batches
            for key in aggregated_output:
                aggregated_output[key] += batch_output[key].item()

            state.tracker.track("val", total_batches)
    
    # Average the loss over all batches
    for key in aggregated_output:
        aggregated_output[key] /= total_batches
        writer.add_scalar(f'Validation/{key}', aggregated_output[key], epoch)

    metadata["val_loss"].append(aggregated_output)

    # ** Ensure values are properly formatted **
    for key, value in aggregated_output.items():
        aggregated_output[key] = {"value": value}

    # Log validation results
    # state.tracker.log("val", aggregated_output, history=False)

    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()

    return aggregated_output


def checkpoint(state, save_path="checkpoints"):
    tags = ["latest"]
    save_path = Path(save_path)

    # 检查并创建保存路径
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_path.absolute()}")
    # print(f"Saving to {str(Path('.').absolute())}")

    # 提取最新的验证损失中的 mel/loss 作为比较值
    current_mel_loss = metadata["val_loss"][-1]["mel/loss"]

    # 确保 current_mel_loss 是浮点数
    if isinstance(current_mel_loss, dict) and "value" in current_mel_loss:
        current_mel_loss = current_mel_loss["value"]

    if metadata["best_val_loss"] > current_mel_loss:
        print(f"New best generator found...")
        tags.append("best")
        metadata["best_val_loss"] = current_mel_loss

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata
        }

        torch.save(accel.unwrap(state.generator).state_dict(), os.path.join(save_path, f"generator_{tag}.pth"))
        torch.save(generator_extra, os.path.join(save_path, f"generator_extra_{tag}.pth"))

        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        torch.save(accel.unwrap(state.discriminator).state_dict(), os.path.join(save_path, f"discriminator_{tag}.pth"))
        torch.save(discriminator_extra, os.path.join(save_path, f"discriminator_extra_{tag}.pth"))

# 开始训练
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_summary = train(model, discriminator, train_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                          gan_loss, stft_loss, mel_loss, waveform_loss, lambdas, accel, epochs)
    val_summary = validate(state, val_dataloader, accel, writer, epoch)
    checkpoint(state)
    print(f"Training Summary: {train_summary}")
    print(f"Validation Summary: {val_summary}")

    # 每个 epoch 结束后清理缓存
    torch.cuda.empty_cache()
