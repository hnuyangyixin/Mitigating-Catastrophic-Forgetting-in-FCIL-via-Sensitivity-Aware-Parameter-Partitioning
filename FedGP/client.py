import copy
import torch
import torch.nn as nn
from ResNet import resnet18_cbam  # 使用ResNet作为分类模型
import random
import time
from copy import deepcopy
import numpy as np
import pandas
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from DataframeToDataset import ListtoDataset
from utils import prediction


class Client(object):
    def __init__(self, dataset, lr, device, task_num, id, epoch_local, subset, batch_size, mask, num_classes):
        self.id = id
        self.task_num = task_num
        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.device = device
        self.dataset = dataset
        self.mask = mask
        self.num_classes = num_classes  # 添加总类别数

        # 添加标签映射字典
        self.label_mapping = {}
        self.reverse_mapping = {}

        self.test_loader = []
        self.train_dataset = subset
        self.train_data = None

        # 使用单一分类模型替代多个VAE模型
        self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.global_model = None  # 接收的全局模型

        # 梯度约束相关变量
        self.w_glob_er = None
        self.grad_dims = None

    def recive_global_models(self, global_model, w_glob_er=None, grad_dims=None):
        """
        接收全局模型和梯度约束信息
        
        Args:
            global_model: 聚合后的全局模型
            w_glob_er: 全局梯度约束向量（可选）
            grad_dims: 梯度维度信息（可选）
        """
        # 直接接收全局模型
        self.model.load_state_dict(global_model.state_dict())
        self.w_glob_er = w_glob_er
        self.grad_dims = grad_dims

    def get_data(self, task_id):
        # 修复：只在任务切换时更新数据（每5轮切换一次任务）
        if task_id % 5 == 0:
            self.task_id = task_id // 5
            self.current_class = self.mask[self.task_id]
            self.trainset = self.train_dataset[self.task_id]
            if self.task_id == 0:
                self.first_task = self.current_class

        print(
            f'{self.id} client，task {self.task_id} (round {task_id}) contains {self.current_class} classes, there are {len(self.current_class)} classes')

        # 创建标签映射
        self.label_mapping = {cls: idx for idx, cls in enumerate(self.current_class)}
        self.reverse_mapping = {idx: cls for idx, cls in enumerate(self.current_class)}

        traindata, testdata = random_split(self.trainset,
                                           [int(len(self.trainset) * 0.8),
                                            len(self.trainset) - int(len(self.trainset) * 0.8)])
        testdata = deepcopy(testdata)
        testloader = DataLoader(testdata, shuffle=True, batch_size=self.batch_size)

        # 修复：只在新任务开始时添加test_loader，避免重复添加
        if task_id % 5 == 0:
            self.test_loader.append(testloader)
        
        self.train_data = traindata

    def train(self, task_id):
        """
        本地训练
        
        Args:
            task_id: 全局轮次ID (0-24)
            
        Returns:
            model_state: 训练后的模型状态字典
            client_grad: 客户端梯度（可选，用于梯度聚合）
        """
        self.get_data(task_id)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        # 使用分类模型和交叉熵损失
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 计算当前任务ID（用于日志）
        current_task = task_id // 5
        print(f'{self.id} client is training on task {current_task} (round {task_id})')

        for epoch in range(self.epoch_local):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                _, logits = self.model(inputs)  # 使用完整的logits（100维）
                
                # 直接使用完整的交叉熵损失，不限制为当前任务类别
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        print(f'{self.id} client finish training')
        self.final_state = copy.deepcopy(self.model.state_dict())
        
        # 修复：使用任务ID而不是全局轮次ID来评估
        self.evaluate_local_models(current_task)
        
        # 修复：返回模型状态字典和梯度（当前简化版本梯度返回None）
        return self.final_state, None

    def evaluate_local_models(self, task_index):
        """
        评估本地模型在指定任务上的表现
        
        Args:
            task_index: 任务索引 (0-4)，不是全局轮次ID
        """
        # 加载最终训练好的模型
        self.model.load_state_dict(self.final_state)
        self.model.eval()
        correct, total = 0, 0
        
        # 安全检查：确保task_index在有效范围内
        if task_index >= len(self.test_loader):
            print(f'Warning: task_index {task_index} out of range, test_loader length is {len(self.test_loader)}')
            return 0.0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader[task_index]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, logits = self.model(inputs)
                
                # 不再使用mask，直接使用完整的logits进行评估
                _, pred_idx = torch.max(logits, 1)
                predicted_labels = pred_idx
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        acc = correct / total * 100 if total > 0 else 0.0
        print(f'Client {self.id}\'s local model on task {task_index} acc is {acc:.2f}%')
        return acc



# # 计算通信开销

# import copy
# import torch
# import torch.nn as nn
# from ResNet import resnet18_cbam  # 使用ResNet作为分类模型
# import random
# import time
# from copy import deepcopy
# import numpy as np
# import pandas
# from torch.utils.data import random_split, DataLoader
# from tqdm import tqdm
# from DataframeToDataset import ListtoDataset
# from utils import prediction


# class Client(object):
#     def __init__(self, dataset, lr, device, task_num, id, epoch_local, subset, batch_size, mask, num_classes):
#         self.id = id
#         self.task_num = task_num
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.dataset = dataset
#         self.mask = mask
#         self.num_classes = num_classes  # 添加总类别数

#         # 添加标签映射字典
#         self.label_mapping = {}
#         self.reverse_mapping = {}

#         self.test_loader = []
#         self.train_dataset = subset
#         self.train_data = None

#         # 使用单一分类模型替代多个VAE模型
#         self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.global_model = None  # 接收的全局模型

#         # 梯度约束相关变量
#         self.w_glob_er = None
#         self.grad_dims = None

#         # 用于统计每轮训练的样本数（计算FLOPs用）
#         self.num_train_samples_this_round = 0

#     def recive_global_models(self, global_model, w_glob_er=None, grad_dims=None):
#         """
#         接收全局模型和梯度约束信息
        
#         Args:
#             global_model: 聚合后的全局模型
#             w_glob_er: 全局梯度约束向量（可选）
#             grad_dims: 梯度维度信息（可选）
#         """
#         # 直接接收全局模型
#         self.model.load_state_dict(global_model.state_dict())
#         self.w_glob_er = w_glob_er
#         self.grad_dims = grad_dims

#     def get_data(self, task_id):
#         # 修复：只在任务切换时更新数据（每5轮切换一次任务）
#         if task_id % 5 == 0:
#             self.task_id = task_id // 5
#             self.current_class = self.mask[self.task_id]
#             self.trainset = self.train_dataset[self.task_id]
#             if self.task_id == 0:
#                 self.first_task = self.current_class

#         print(
#             f'{self.id} client，task {self.task_id} (round {task_id}) contains {self.current_class} classes, there are {len(self.current_class)} classes')

#         # 创建标签映射
#         self.label_mapping = {cls: idx for idx, cls in enumerate(self.current_class)}
#         self.reverse_mapping = {idx: cls for idx, cls in enumerate(self.current_class)}

#         traindata, testdata = random_split(self.trainset,
#                                            [int(len(self.trainset) * 0.8),
#                                             len(self.trainset) - int(len(self.trainset) * 0.8)])
#         testdata = deepcopy(testdata)
#         testloader = DataLoader(testdata, shuffle=True, batch_size=self.batch_size)

#         # 修复：只在新任务开始时添加test_loader，避免重复添加
#         if task_id % 5 == 0:
#             self.test_loader.append(testloader)
        
#         self.train_data = traindata

#     def train(self, task_id):
#         """
#         本地训练
        
#         Args:
#             task_id: 全局轮次ID (0-24)
            
#         Returns:
#             model_state: 训练后的模型状态字典
#             client_grad: 客户端梯度（可选，用于梯度聚合）
#         """
#         self.get_data(task_id)
#         self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

#         # 记录本轮训练样本数
#         self.num_train_samples_this_round = len(self.train_data)

#         # 使用分类模型和交叉熵损失
#         self.model.to(self.device)
#         self.model.train()
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()

#         # 计算当前任务ID（用于日志）
#         current_task = task_id // 5
#         print(f'{self.id} client is training on task {current_task} (round {task_id})')

#         for epoch in range(self.epoch_local):
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 optimizer.zero_grad()
#                 _, logits = self.model(inputs)  # 使用完整的logits（100维）
                
#                 # 直接使用完整的交叉熵损失，不限制为当前任务类别
#                 loss = criterion(logits, labels)
#                 loss.backward()
#                 optimizer.step()

#         print(f'{self.id} client finish training')
#         self.final_state = copy.deepcopy(self.model.state_dict())
        
#         # 修复：使用任务ID而不是全局轮次ID来评估
#         self.evaluate_local_models(current_task)
        
#         # 修复：返回模型状态字典和梯度（当前简化版本梯度返回None）
#         return self.final_state, None

#     def evaluate_local_models(self, task_index):
#         """
#         评估本地模型在指定任务上的表现
        
#         Args:
#             task_index: 任务索引 (0-4)，不是全局轮次ID
#         """
#         # 加载最终训练好的模型
#         self.model.load_state_dict(self.final_state)
#         self.model.eval()
#         correct, total = 0, 0
        
#         # 安全检查：确保task_index在有效范围内
#         if task_index >= len(self.test_loader):
#             print(f'Warning: task_index {task_index} out of range, test_loader length is {len(self.test_loader)}')
#             return 0.0
        
#         with torch.no_grad():
#             for inputs, labels in self.test_loader[task_index]:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 _, logits = self.model(inputs)
                
#                 # 不再使用mask，直接使用完整的logits进行评估
#                 _, pred_idx = torch.max(logits, 1)
#                 predicted_labels = pred_idx
#                 total += labels.size(0)
#                 correct += (predicted_labels == labels).sum().item()

#         acc = correct / total * 100 if total > 0 else 0.0
#         print(f'Client {self.id}\'s local model on task {task_index} acc is {acc:.2f}%')
#         return acc