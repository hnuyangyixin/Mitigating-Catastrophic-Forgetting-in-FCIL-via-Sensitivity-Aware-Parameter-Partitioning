import random
import numpy as np
import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import DataLoader, Dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prediction(model, testloader, device):
    """全局评估 - 使用完整测试集，不限制预测标签范围"""
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in testloader:
            # 数据处理
            inputs, labels = (batch[1], batch[2]) if len(batch) == 3 else (batch[0], batch[1])
            inputs, labels = inputs.to(device), labels.to(device)

            # 模型预测
            _, logits = model(inputs)
            
            # 获取预测结果（所有可能的标签）
            _, preds = torch.max(logits, 1)

            # 计算准确率
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total * 100 if total > 0 else 0


def prediction_task(model, testloader, device, task_classes):
    """评估模型在特定任务上的准确率 - 仅使用当前任务数据"""
    model.to(device)
    model.eval()
    correct, total = 0, 0

    # 创建任务特定测试集
    indices = []
    dataset = testloader.dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample[2] if len(sample) == 3 else sample[1]
        label = label.item() if torch.is_tensor(label) else int(label)
        if label in task_classes:
            indices.append(idx)
    
    if not indices:
        print("????????")
        return 0.0

    # 创建子集和DataLoader
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    task_loader = DataLoader(subset, batch_size=testloader.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in task_loader:
            # 处理不同格式的数据集
            if len(batch) == 3:  # (idx, inputs, labels)
                inputs, labels = batch[1], batch[2]
            else:  # (inputs, labels)
                inputs, labels = batch[0], batch[1]
            
            inputs, labels = inputs.to(device), labels.to(device)
            _, logits = model(inputs)
            
            # 获取预测结果（所有可能的标签）
            _, preds = torch.max(logits, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total * 100 if total > 0 else 0


# ==================== PackNet 相关工具函数 ====================

def get_prunable_params(model):
    """
    获取模型中需要按客户端划分的参数名列表
    包括：卷积层权重、BN层的weight和bias
    不包括：分类器fc层（按类别划分）
    """
    prunable_params = []
    for name, param in model.named_parameters():
        # 卷积层权重
        if 'conv' in name and 'weight' in name:
            prunable_params.append(name)
        # BN层的可学习参数（weight和bias）
        elif 'bn' in name and ('weight' in name or 'bias' in name):
            prunable_params.append(name)
    return prunable_params


def create_client_base_masks(model, num_clients):
    """
    为每个客户端创建基础参数掩码
    将模型参数空间均匀划分给各客户端（不重叠）
    
    Returns:
        client_base_masks: dict[client_id] -> dict[param_name] -> mask tensor (all on CPU)
    """
    client_base_masks = {i: {} for i in range(num_clients)}
    prunable_params = get_prunable_params(model)
    
    for name, param in model.named_parameters():
        # 确保在CPU上操作
        param_cpu = param.detach().cpu()
        
        if name in prunable_params:
            # 获取参数的总元素数
            total_elements = param_cpu.numel()
            # 每个客户端分配的元素数
            elements_per_client = total_elements // num_clients
            
            # 创建随机排列的索引（在CPU上）
            indices = torch.randperm(total_elements)
            
            for client_id in range(num_clients):
                # 为每个客户端分配不重叠的参数子集
                start_idx = client_id * elements_per_client
                if client_id == num_clients - 1:
                    # 最后一个客户端获取剩余所有参数
                    end_idx = total_elements
                else:
                    end_idx = start_idx + elements_per_client
                
                # 创建掩码（在CPU上）
                mask = torch.zeros(total_elements, dtype=torch.bool)
                mask[indices[start_idx:end_idx]] = True
                mask = mask.reshape(param_cpu.shape)
                
                client_base_masks[client_id][name] = mask.cpu()
        else:
            # 非剪枝参数（如BN层）每个客户端都可以使用
            for client_id in range(num_clients):
                client_base_masks[client_id][name] = torch.ones(param_cpu.shape, dtype=torch.bool).cpu()
    
    return client_base_masks


def compute_importance(model, dataloader, device, available_mask):
    """
    计算参数重要性（基于权重绝对值）
    只在可用参数范围内计算
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器（用于可能的梯度重要性计算，当前版本使用权重绝对值）
        device: 设备
        available_mask: 当前可用参数的掩码
    
    Returns:
        importance: dict[param_name] -> importance tensor (on device)
    """
    importance = {}
    prunable_params = get_prunable_params(model)
    
    model.to(device)
    
    for name, param in model.named_parameters():
        if name in prunable_params:
            # 使用权重绝对值作为重要性
            imp = param.data.abs().clone()
            # 只考虑可用参数的重要性
            if name in available_mask:
                avail_mask = available_mask[name].float().to(device)
                imp = imp * avail_mask
            importance[name] = imp
    
    return importance


def prune_and_create_task_mask(model, importance, available_mask, prune_ratio, device):
    """
    基于重要性进行剪枝，创建任务专属掩码
    
    Args:
        model: 模型
        importance: 参数重要性
        available_mask: 当前可用参数的掩码
        prune_ratio: 剪枝比例（保留的参数比例）
        device: 设备
    
    Returns:
        task_mask: 当前任务保留的参数掩码 (CPU)
        new_available_mask: 剪枝后剩余可用参数的掩码 (CPU)
    """
    task_mask = {}
    new_available_mask = {}
    prunable_params = get_prunable_params(model)
    
    for name, param in model.named_parameters():
        if name in prunable_params and name in importance:
            # 确保available_mask在正确的设备上
            avail_mask = available_mask[name].to(device)
            imp = importance[name].to(device)
            
            # 获取可用参数的重要性值
            available_importance = imp[avail_mask]
            
            if available_importance.numel() > 0:
                # 计算保留参数的数量
                num_keep = max(1, int(available_importance.numel() * prune_ratio))
                
                # 获取重要性阈值
                if num_keep >= available_importance.numel():
                    threshold = 0
                else:
                    threshold = torch.kthvalue(available_importance.flatten(), 
                                              available_importance.numel() - num_keep + 1)[0]
                
                # 创建任务掩码：在可用参数中保留重要的
                task_mask_tensor = (imp >= threshold) & avail_mask
                task_mask[name] = task_mask_tensor.cpu()
                
                # 更新可用掩码：移除已分配给当前任务的参数
                new_available_mask[name] = (avail_mask & (~task_mask_tensor)).cpu()
            else:
                task_mask[name] = torch.zeros_like(param, dtype=torch.bool).cpu()
                new_available_mask[name] = available_mask[name].cpu()
        else:
            # 非剪枝参数
            if name in available_mask:
                task_mask[name] = available_mask[name].cpu()
                new_available_mask[name] = available_mask[name].cpu()
            else:
                task_mask[name] = torch.ones(param.shape, dtype=torch.bool).cpu()
                new_available_mask[name] = torch.ones(param.shape, dtype=torch.bool).cpu()
    
    return task_mask, new_available_mask


def apply_mask_to_model(model, mask, device):
    """
    将掩码应用到模型参数上（将非掩码参数置零）
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name].float().to(device)


def apply_mask_to_gradient(model, mask, device):
    """
    将掩码应用到梯度上（只更新掩码内的参数）
    """
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask:
            param.grad *= mask[name].float().to(device)


def freeze_masked_params(model, frozen_mask, device):
    """
    冻结已分配给之前任务的参数（将其梯度置零）
    
    Args:
        model: 模型
        frozen_mask: 需要冻结的参数掩码（True表示冻结）
        device: 设备
    """
    for name, param in model.named_parameters():
        if param.grad is not None and name in frozen_mask:
            # 冻结掩码为True的参数
            freeze_mask = frozen_mask[name].float().to(device)
            param.grad *= (1 - freeze_mask)


def combine_masks(mask1, mask2):
    """
    合并两个掩码（逻辑或）
    返回CPU上的掩码
    """
    combined = {}
    for name in mask1:
        if name in mask2:
            combined[name] = (mask1[name].cpu() | mask2[name].cpu())
        else:
            combined[name] = mask1[name].cpu()
    for name in mask2:
        if name not in combined:
            combined[name] = mask2[name].cpu()
    return combined


class TaskIDPredictor(nn.Module):
    """
    任务ID预测器
    使用特征提取器的输出预测样本属于哪个任务
    """
    def __init__(self, feature_dim=512, num_tasks=5):
        super(TaskIDPredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_tasks)
        )
    
    def forward(self, features):
        return self.classifier(features)


def train_task_id_predictor(predictor, model, dataloaders_per_task, device, epochs=20, lr=0.001):
    """
    训练任务ID预测器
    
    Args:
        predictor: TaskIDPredictor实例
        model: 特征提取模型
        dataloaders_per_task: dict[task_id] -> dataloader
        device: 设备
        epochs: 训练轮数
        lr: 学习率
    """
    predictor.to(device)
    model.to(device)
    model.eval()  # 特征提取器不更新
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        predictor.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 遍历每个任务的数据
        for task_id, dataloader in dataloaders_per_task.items():
            for batch in dataloader:
                if len(batch) == 3:
                    inputs, labels = batch[1], batch[2]
                else:
                    inputs, labels = batch[0], batch[1]
                
                inputs = inputs.to(device)
                task_labels = torch.full((inputs.size(0),), task_id, dtype=torch.long, device=device)
                
                # 提取特征
                with torch.no_grad():
                    features, _ = model(inputs)
                
                # 预测任务ID
                optimizer.zero_grad()
                task_logits = predictor(features)
                loss = criterion(task_logits, task_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, preds = torch.max(task_logits, 1)
                total_correct += (preds == task_labels).sum().item()
                total_samples += inputs.size(0)
        
        if (epoch + 1) % 5 == 0:
            acc = total_correct / total_samples * 100
            print(f'Task ID Predictor - Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Acc: {acc:.2f}%')
    
    return predictor


def predict_task_id(predictor, model, inputs, device):
    """
    预测输入样本的任务ID
    """
    predictor.eval()
    model.eval()
    
    with torch.no_grad():
        inputs = inputs.to(device)
        features, _ = model(inputs)
        task_logits = predictor(features)
        _, task_ids = torch.max(task_logits, 1)
    
    return task_ids