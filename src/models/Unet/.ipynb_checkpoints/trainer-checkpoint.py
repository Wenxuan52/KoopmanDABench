import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import random

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 数据移到设备
        inputs = inputs.to(device)  # [batch_size, H, W]
        targets = targets.to(device)  # [batch_size, prediction_steps, H, W]
        
        # 如果输入是3维，添加通道维度
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)  # [batch_size, 1, H, W]
        
        # 前向传播
        optimizer.zero_grad()
        
        # 自回归预测多个时间步
        prediction_steps = targets.shape[1]
        predictions = []
        current_input = inputs
        
        for step in range(prediction_steps):
            pred = model(current_input)  # [batch_size, 1, H, W]
            predictions.append(pred)
            current_input = pred  # 使用预测作为下一步输入
        
        # 堆叠预测结果
        predictions = torch.stack(predictions, dim=1)  # [batch_size, prediction_steps, 1, H, W]
        predictions = predictions.squeeze(2)  # [batch_size, prediction_steps, H, W]
        
        # 计算重构损失
        loss = criterion(predictions, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印进度
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 如果输入是3维，添加通道维度
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            
            # 自回归预测
            prediction_steps = targets.shape[1]
            predictions = []
            current_input = inputs
            
            for step in range(prediction_steps):
                pred = model(current_input)
                predictions.append(pred)
                current_input = pred
            
            predictions = torch.stack(predictions, dim=1)
            predictions = predictions.squeeze(2)
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, train_losses, val_losses, save_dir, is_best=False):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # 保存最新检查点
    torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
    
    # 保存最佳模型
    if is_best:
        torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
        print(f'保存最佳模型，验证损失: {loss:.6f}')

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    print(f'加载检查点，epoch: {checkpoint["epoch"]}, loss: {checkpoint["loss"]:.6f}')
    return checkpoint['epoch'], train_losses, val_losses

def plot_losses(train_losses, val_losses, save_dir):
    """绘制损失曲线"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss', color='blue')
    plt.plot(val_losses, label='validation loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('validation loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='train loss', color='blue')
    plt.plot(val_losses, label='validation loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('validation loss (log-scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    # plt.show()

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, 
                weight_decay=1e-5, save_dir='./checkpoints', early_stopping_patience=20,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """完整训练过程"""
    
    # 设置设备
    model = model.to(device)
    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.8
    )
    
    # 损失函数 - MSE重构损失
    criterion = nn.MSELoss()
    
    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    print(f"开始训练，共 {num_epochs} 个epoch")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        print(f'Epoch {epoch}/{num_epochs}:')
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step()
        
        # 检查是否是最佳模型
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # 保存检查点
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, 
                       train_losses, val_losses, save_dir, is_best)
        
        # 打印epoch结果
        epoch_time = time.time() - epoch_start_time
        print(f'  训练损失: {train_loss:.6f}')
        print(f'  验证损失: {val_loss:.6f}')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'  时间: {epoch_time:.2f}s')
        print(f'  最佳验证损失: {best_val_loss:.6f}')
        print('-' * 60)
        
        # 早停检查
        if early_stopping_counter >= early_stopping_patience:
            print(f'验证损失连续 {early_stopping_patience} 个epoch没有改善，提前停止训练')
            break
    
    total_time = time.time() - start_time
    print(f'训练完成！总时间: {total_time:.2f}s')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    
    # 绘制损失曲线
    plot_losses(train_losses, val_losses, save_dir)
    
    return train_losses, val_losses

def predict_sequence(model, input_frame, prediction_steps, device):
    """预测序列"""
    model.eval()
    
    with torch.no_grad():
        # 确保输入格式正确
        if input_frame.dim() == 2:  # [H, W]
            input_frame = input_frame.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif input_frame.dim() == 3:  # [1, H, W] 或 [batch, H, W]
            if input_frame.shape[0] == 1:
                input_frame = input_frame.unsqueeze(0)  # [1, 1, H, W]
            else:
                input_frame = input_frame.unsqueeze(1)  # [batch, 1, H, W]
        
        input_frame = input_frame.to(device)
        
        # 自回归预测
        predictions = []
        current_input = input_frame
        
        for step in range(prediction_steps):
            pred = model(current_input)
            predictions.append(pred.cpu())
            current_input = pred
        
        # 堆叠结果
        predictions = torch.stack(predictions, dim=1)  # [batch, steps, 1, H, W]
        predictions = predictions.squeeze(2)  # [batch, steps, H, W]
        
        return predictions


def set_seed(seed=42):
    """
    设置所有相关库的随机数种子，确保实验可重复
    
    Args:
        seed (int): 随机数种子，默认为42
    """
    # Python 内置random模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 确保CuDNN的确定性行为（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量（某些操作系统级别的随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)


# 使用示例
if __name__ == "__main__":
    SEED=42
    set_seed(SEED)

    from channel_K_residual_all import channel_UNET
    from K_residual_all import UNET
    from residual_con import UNET_InputResidual
    from no_residual import UNET_NoResidual
    from utils import create_single_frame_loaders
    
    # 创建数据加载器
    train_loader, val_loader = create_single_frame_loaders(
        data_path='./data/kolmogorov',
        dataset_type='kolmogorov',
        prediction_steps=5,
        batch_size=64,
        normalize=True,
        train_ratio=0.8,
    )
    
    # 创建模型
    model = channel_UNET()
    
    # 开始训练
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-5,
        save_dir='./results/channel/',
        early_stopping_patience=20
    )
    

    plot_losses(train_losses, val_losses, save_dir='./results/channel/')
    # 示例：使用训练好的模型进行预测
    # 加载最佳模型
    # checkpoint = torch.load('./checkpoints/best_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # 获取一个样本进行预测
    # sample_input, _ = next(iter(val_loader))
    # predictions = predict_sequence(model, sample_input[0], prediction_steps=5, device='cuda')
    # print(f"预测结果形状: {predictions.shape}")  # [1, 5, H, W]