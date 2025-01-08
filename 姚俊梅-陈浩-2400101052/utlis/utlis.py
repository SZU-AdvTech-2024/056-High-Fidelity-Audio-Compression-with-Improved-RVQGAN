def acccheckpoint(state, filelists, step,realstep, save_path=savepath):
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
        filename = f"{tag}_mel_loss_{formatted_mel_loss}_step_{step}.pth"
        extraname = f"extra_{tag}_mel_loss_{formatted_mel_loss}_step_{step}.pth"
        extra= {
            "epoch": epoch,
            "step": realstep,
            'metadata':metadata,
            'val_filelists':val_filelists
        }
        if(accelerator.is_main_process):
            torch.save(extra,save_path / extraname)
            accelerator.save_state(save_path / filename)
        # Move models to CPU before saving
        if(accelerator.is_main_process):
            if tag == "latest":
                latest_files = sorted(save_path.glob("latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
                
                for old_file in latest_files[5:]:  # 删除多余的文件，保留最新的10个
                    shutil.rmtree(old_file)
                    print(f"已删除旧的latest文件: {old_file.name}")
                latest_files = sorted(save_path.glob("extra_latest_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
                for old_file in latest_files[5:]:  # 删除多余的文件，保留最新的10个
                    os.remove(old_file)
                    print(f"已删除旧的latest文件: {old_file.name}")

            # 删除旧的best文件，仅保留最新的一个
            if tag == "best":
                best_files = sorted(save_path.glob("best_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
                for old_file in best_files[1:]:  # 删除除最新的best文件
                    shutil.rmtree(old_file)
                    print(f"已删除旧的best文件: {old_file.name}")
                best_files = sorted(save_path.glob("extra_best_mel_loss_*_step_*.pth"), key=get_step, reverse=True)
                for old_file in best_files[1:]:  # 删除除最新的best文件
                    os.remove(old_file)
                    print(f"已删除旧的best文件: {old_file.name}")
                
    accelerator.wait_for_everyone()
    print(f"已保存检查点: {filename} ")