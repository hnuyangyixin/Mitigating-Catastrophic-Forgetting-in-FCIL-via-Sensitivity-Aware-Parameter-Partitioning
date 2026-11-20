import random
import numpy as np
import torch
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
#fedbabu fedetf    用这个场景在sota方法下训练一下看看
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
            # print("-----------------------")
            # print("predicted:")
            # print(preds)
            # print("labels:")
            # print(labels)
            # print("-----------------------")
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total * 100 if total > 0 else 0


# import random
# import numpy as np
# import torch
# from torch import tensor
# from torch.utils.data import DataLoader, Dataset


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True




# def prediction(model, testloader, device):
#     """全局评估 - 使用完整测试集，不限制预测标签范围"""
#     model.to(device)
#     model.eval()
#     correct, total = 0, 0

#     with torch.no_grad():
#         for batch in testloader:
#             # 数据处理
#             inputs, labels = (batch[1], batch[2]) if len(batch) == 3 else (batch[0], batch[1])
#             inputs, labels = inputs.to(device), labels.to(device)

#             # 模型预测
#             _, logits = model(inputs)
            
#             # 获取预测结果（所有可能的标签）
#             _, preds = torch.max(logits, 1)

#             # 计算准确率
#             total += labels.size(0)
#             correct += (preds == labels).sum().item()

#     return correct / total * 100 if total > 0 else 0

# def prediction_task(model, testloader, device, task_classes):
#     """评估模型在特定任务上的准确率 - 仅使用当前任务数据"""
#     model.to(device)
#     model.eval()
#     correct, total = 0, 0

#     # 创建任务特定测试集
#     indices = []
#     dataset = testloader.dataset
#     for idx in range(len(dataset)):
#         sample = dataset[idx]
#         label = sample[2] if len(sample) == 3 else sample[1]
#         label = label.item() if torch.is_tensor(label) else int(label)
#         if label in task_classes:
#             indices.append(idx)
    
#     if not indices:
#         print("????????")
#         return 0.0
# #fedbabu fedetf    用这个场景在sota方法下训练一下看看
#     # 创建子集和DataLoader
#     from torch.utils.data import Subset
#     subset = Subset(dataset, indices)
#     task_loader = DataLoader(subset, batch_size=testloader.batch_size, shuffle=False)

#     with torch.no_grad():
#         for batch in task_loader:
#             # 处理不同格式的数据集
#             if len(batch) == 3:  # (idx, inputs, labels)
#                 inputs, labels = batch[1], batch[2]
#             else:  # (inputs, labels)
#                 inputs, labels = batch[0], batch[1]
            
#             inputs, labels = inputs.to(device), labels.to(device)
#             _, logits = model(inputs)
            
#             # 获取预测结果（所有可能的标签）
#             _, preds = torch.max(logits, 1)
#             # print("-----------------------")
#             # print("predicted:")
#             # print(preds)
#             # print("labels:")
#             # print(labels)
#             # print("-----------------------")
#             total += labels.size(0)
#             correct += (preds == labels).sum().item()

#     return correct / total * 100 if total > 0 else 0