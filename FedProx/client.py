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

    def recive_global_models(self, global_model):
        # 直接接收全局模型
        self.model.load_state_dict(global_model.state_dict())

    def get_data(self, task_id):
        if task_id % 5 == 0:
            self.task_id = task_id // 5
            self.current_class = self.mask[self.task_id]
            self.trainset = self.train_dataset[self.task_id]
            if self.task_id == 0:
                self.first_task = self.current_class

        print(
            f'{self.id} client，{task_id} task contains {self.current_class} classes, there are {len(self.current_class)} classes')

        # 创建标签映射
        self.label_mapping = {cls: idx for idx, cls in enumerate(self.current_class)}
        self.reverse_mapping = {idx: cls for idx, cls in enumerate(self.current_class)}

        traindata, testdata = random_split(self.trainset,
                                           [int(len(self.trainset) * 0.8),
                                            len(self.trainset) - int(len(self.trainset) * 0.8)])
        testdata = deepcopy(testdata)
        testloader = DataLoader(testdata, shuffle=True, batch_size=self.batch_size)

        self.test_loader.append(testloader)
        self.train_data = traindata

    def train(self, task_id, mu=0.01):
        self.get_data(task_id)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        # 使用分类模型和交叉熵损失
        self.model.to(self.device)
        self.model.train()
        
        # FedProx: 保存全局模型参数用于计算proximal term
        # 必须在优化开始前保存，并detach防止梯度传播
        global_weight_collector = [param.clone().detach() for param in self.model.parameters()]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        print(f'{self.id} client is training on No. {task_id} task with FedProx (mu={mu})')

        for epoch in range(self.epoch_local):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                _, logits = self.model(inputs)  # 使用完整的logits（100维）
                
                # 原始损失
                loss = criterion(logits, labels)

                # FedProx: 添加Proximal Term
                # loss += (mu / 2) * ||w - w^t||^2
                if mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_weight_collector):
                        proximal_term += (local_param - global_param).norm(2)**2
                    loss += (mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

        print(f'{self.id} client finish training')
        self.final_state = copy.deepcopy(self.model.state_dict())
        self.evaluate_local_models(task_id)

    # def evaluate_local_models(self, index):
    #     index = index // 5
    #     acc = prediction(self.model, self.test_loader[index], self.device)
    #     print(f'Client {self.id}\'s local model on {index} task acc is {acc}')

    def evaluate_local_models(self, index):
        # 加载最终训练好的模型
        self.model.load_state_dict(self.final_state)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader[index]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, logits = self.model(inputs)
                
                # 不再使用mask，直接使用完整的logits进行评估
                _, pred_idx = torch.max(logits, 1)
                predicted_labels = pred_idx
                # print("-----------------------")
                # print("predicted:")
                # print(predicted_labels)
                # print("labels:")
                # print(labels)
                # print("-----------------------")
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        acc = correct / total * 100
        print(f'Client {self.id}\'s local model on {index} task acc is {acc:.2f}%')
        return acc



# # 计算开销统计

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

#         # 用于统计每轮训练的样本数（计算FLOPs用）
#         self.num_train_samples_this_round = 0

#     def recive_global_models(self, global_model):
#         # 直接接收全局模型
#         self.model.load_state_dict(global_model.state_dict())

#     def get_data(self, task_id):
#         if task_id % 5 == 0:
#             self.task_id = task_id // 5
#             self.current_class = self.mask[self.task_id]
#             self.trainset = self.train_dataset[self.task_id]
#             if self.task_id == 0:
#                 self.first_task = self.current_class

#         print(
#             f'{self.id} client，{task_id} task contains {self.current_class} classes, there are {len(self.current_class)} classes')

#         # 创建标签映射
#         self.label_mapping = {cls: idx for idx, cls in enumerate(self.current_class)}
#         self.reverse_mapping = {idx: cls for idx, cls in enumerate(self.current_class)}

#         traindata, testdata = random_split(self.trainset,
#                                            [int(len(self.trainset) * 0.8),
#                                             len(self.trainset) - int(len(self.trainset) * 0.8)])
#         testdata = deepcopy(testdata)
#         testloader = DataLoader(testdata, shuffle=True, batch_size=self.batch_size)

#         self.test_loader.append(testloader)
#         self.train_data = traindata

#     def train(self, task_id, mu=0.01):
#         self.get_data(task_id)
#         self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

#         # 记录本轮训练样本数
#         self.num_train_samples_this_round = len(self.train_data)

#         # 使用分类模型和交叉熵损失
#         self.model.to(self.device)
#         self.model.train()
        
#         # FedProx: 保存全局模型参数用于计算proximal term
#         # 必须在优化开始前保存，并detach防止梯度传播
#         global_weight_collector = [param.clone().detach() for param in self.model.parameters()]

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()

#         print(f'{self.id} client is training on No. {task_id} task with FedProx (mu={mu})')

#         for epoch in range(self.epoch_local):
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 optimizer.zero_grad()
#                 _, logits = self.model(inputs)  # 使用完整的logits（100维）
                
#                 # 原始损失
#                 loss = criterion(logits, labels)

#                 # FedProx: 添加Proximal Term
#                 # loss += (mu / 2) * ||w - w^t||^2
#                 if mu > 0:
#                     proximal_term = 0.0
#                     for local_param, global_param in zip(self.model.parameters(), global_weight_collector):
#                         proximal_term += (local_param - global_param).norm(2)**2
#                     loss += (mu / 2) * proximal_term

#                 loss.backward()
#                 optimizer.step()

#         print(f'{self.id} client finish training')
#         self.final_state = copy.deepcopy(self.model.state_dict())
#         self.evaluate_local_models(task_id)

#     # def evaluate_local_models(self, index):
#     #     index = index // 5
#     #     acc = prediction(self.model, self.test_loader[index], self.device)
#     #     print(f'Client {self.id}\'s local model on {index} task acc is {acc}')

#     def evaluate_local_models(self, index):
#         # 加载最终训练好的模型
#         self.model.load_state_dict(self.final_state)
#         self.model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for inputs, labels in self.test_loader[index]:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 _, logits = self.model(inputs)
                
#                 # 不再使用mask，直接使用完整的logits进行评估
#                 _, pred_idx = torch.max(logits, 1)
#                 predicted_labels = pred_idx
#                 # print("-----------------------")
#                 # print("predicted:")
#                 # print(predicted_labels)
#                 # print("labels:")
#                 # print(labels)
#                 # print("-----------------------")
#                 total += labels.size(0)
#                 correct += (predicted_labels == labels).sum().item()

#         acc = correct / total * 100
#         print(f'Client {self.id}\'s local model on {index} task acc is {acc:.2f}%')
#         return acc