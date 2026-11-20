import copy
import torch
import torch.nn as nn
from ResNet import resnet18_cbam
from torch.utils.data import random_split, DataLoader
from copy import deepcopy
import math
import gc


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
        self.num_classes = num_classes
        self.label_mapping = {}
        self.reverse_mapping = {}
        
        # 修改：不再累积所有test_loader，改为字典存储，按需加载
        self.test_data_indices = {}  # 存储测试数据的索引而非DataLoader
        self.train_dataset = subset
        self.train_data = None
        self.trainset = None  # 当前任务的训练集引用

        # 使用单一分类模型
        self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.classifier_grad = None
        self.initial_state = None
        self.projected_grad = None
        self.final_state = None

    def recive_global_models(self, global_model):
        # 优化：只复制state_dict，不复制整个模型
        self.model.load_state_dict(global_model.state_dict())
        self.global_model.load_state_dict(global_model.state_dict())
        self.global_model.to(self.device)

    def receive_projected_grad(self, projected_grad):
        self.projected_grad = projected_grad

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

        traindata, testdata = random_split(
            self.trainset,
            [int(len(self.trainset) * 0.8), len(self.trainset) - int(len(self.trainset) * 0.8)]
        )
        
        # 修改：不再累积test_loader，而是存储当前任务的测试数据
        # 只保留当前任务的测试数据用于本地评估
        self.current_test_data = deepcopy(testdata)
        self.train_data = traindata

    def _get_test_loader(self, task_idx=None):
        """按需创建测试DataLoader"""
        if task_idx is None:
            # 使用当前任务的测试数据
            return DataLoader(self.current_test_data, shuffle=False, batch_size=self.batch_size)
        return None

    def calculate_regularization(self):
        local_w = self.model.fc2.weight
        global_w = self.global_model.fc2.weight

        reg = 0.0

        for class_i in self.current_class:
            local_i = local_w[class_i]
            global_i = global_w[class_i]

            # cosine similarity
            sim_pos = torch.nn.functional.cosine_similarity(local_i, global_i, dim=0)

            # positive logit
            pos_logit = torch.exp(sim_pos)

            # all negative logits
            neg_logits = []
            for class_k in range(self.num_classes):
                if class_k == class_i:
                    continue
                sim_neg = torch.nn.functional.cosine_similarity(local_i, global_w[class_k], dim=0)
                neg_logits.append(torch.exp(sim_neg))
            
            neg_sum = torch.stack(neg_logits).sum()

            # InfoNCE loss
            class_loss = -torch.log(pos_logit / neg_sum)

            reg = reg + class_loss

        return reg

    def calculate_regularization_only_denominator(self):
        # 1. 获取权重
        local_w = self.model.fc2.weight      # shape: [100, feature_dim]
        global_w = self.global_model.fc2.weight # shape: [100, feature_dim]
        
        local_w_norm = torch.nn.functional.normalize(local_w, p=2, dim=1)
        global_w_norm = torch.nn.functional.normalize(global_w, p=2, dim=1)

        current_indices = torch.tensor(self.current_class, device=self.device)
        current_local_vectors = local_w_norm[current_indices] 

        sim_matrix = torch.matmul(current_local_vectors, global_w_norm.T)
        exp_sim_matrix = torch.exp(sim_matrix) # shape: [len(current), 100]
        row_sums = exp_sim_matrix.sum(dim=1) # shape: [len(current)]
        self_exp_sim = exp_sim_matrix.gather(1, current_indices.unsqueeze(1)).squeeze(1)
        neg_sum = row_sums - self_exp_sim
        class_losses = torch.log(neg_sum + 1e-8)
        reg = class_losses.sum()

        return reg
    
    def calculate_regularization_only_numerator(self):
        local_w = self.model.fc2.weight
        global_w = self.global_model.fc2.weight

        reg = 0.0

        for class_i in self.current_class:
            local_i = local_w[class_i]
            global_i = global_w[class_i]

            # positive cosine similarity
            sim_pos = torch.nn.functional.cosine_similarity(
                local_i, global_i, dim=0
            )

            # 只保留分子的部分： - sim_pos
            class_loss = -sim_pos

            reg = reg + class_loss

        return reg

    def phase2_train(self, task_id):
        """使用投影梯度进行正则化训练"""
        # 获取数据并初始化 train_loader
        self.get_data(task_id)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.model.to(self.device)

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epoch_local):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                mapped_labels = torch.tensor([self.label_mapping[int(l)] for l in labels]).to(self.device)

                optimizer.zero_grad()

                feat, logits = self.model(inputs)

                task_logits = logits[:, list(self.current_class)]
                loss = criterion(task_logits, mapped_labels)
                denominator_loss = self.calculate_regularization_only_denominator()
                loss = loss + 0.1 * denominator_loss
                loss.backward()

                # 手动将缺席类的分类器梯度置零
                with torch.no_grad():
                    for k in range(self.num_classes):
                        if k not in self.current_class:
                            if self.model.fc2.weight.grad is not None:
                                self.model.fc2.weight.grad[k] = 0

                optimizer.step()

        print(f'{self.id} client finish training')

        self.final_state = copy.deepcopy(self.model.state_dict())
        self.evaluate_local_models(task_id)
        
        # 清理内存
        self._cleanup_after_training()

    def _cleanup_after_training(self):
        """训练后清理内存"""
        # 删除train_loader释放内存
        if hasattr(self, 'train_loader'):
            del self.train_loader
        
        # 清理训练数据引用
        if hasattr(self, 'train_data') and self.train_data is not None:
            del self.train_data
            self.train_data = None
        
        # 清理当前测试数据（评估完成后）
        if hasattr(self, 'current_test_data') and self.current_test_data is not None:
            del self.current_test_data
            self.current_test_data = None
            
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()

    def evaluate_local_models(self, index):
        # 加载最终训练好的模型
        self.model.load_state_dict(self.final_state)
        self.model.eval()
        correct, total = 0, 0
        
        # 使用当前任务的测试数据
        test_loader = self._get_test_loader()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, logits = self.model(inputs)
                
                # 不再使用mask，直接使用完整的logits进行评估
                _, pred_idx = torch.max(logits, 1)
                predicted_labels = pred_idx
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        acc = correct / total * 100
        print(f'Client {self.id}\'s local model on {index} task acc is {acc:.2f}%')
        
        # 清理test_loader
        del test_loader
        
        return acc









# # 计算通信开销


# import copy
# import torch
# import torch.nn as nn
# from ResNet import resnet18_cbam
# from torch.utils.data import random_split, DataLoader
# from copy import deepcopy
# import math
# import gc
 
 
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
#         self.num_classes = num_classes
#         self.label_mapping = {}
#         self.reverse_mapping = {}
        
#         # 修改：不再累积所有test_loader，改为字典存储，按需加载
#         self.test_data_indices = {}  # 存储测试数据的索引而非DataLoader
#         self.train_dataset = subset
#         self.train_data = None
#         self.trainset = None  # 当前任务的训练集引用
 
#         # 使用单一分类模型
#         self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.classifier_grad = None
#         self.initial_state = None
#         self.projected_grad = None
#         self.final_state = None
 
#         # 用于统计每轮训练的样本数（计算FLOPs用）
#         self.num_train_samples_this_round = 0
 
#     def recive_global_models(self, global_model):
#         # 优化：只复制state_dict，不复制整个模型
#         self.model.load_state_dict(global_model.state_dict())
#         self.global_model.load_state_dict(global_model.state_dict())
#         self.global_model.to(self.device)
 
#     def receive_projected_grad(self, projected_grad):
#         self.projected_grad = projected_grad
 
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
 
#         traindata, testdata = random_split(
#             self.trainset,
#             [int(len(self.trainset) * 0.8), len(self.trainset) - int(len(self.trainset) * 0.8)]
#         )
        
#         # 修改：不再累积test_loader，而是存储当前任务的测试数据
#         # 只保留当前任务的测试数据用于本地评估
#         self.current_test_data = deepcopy(testdata)
#         self.train_data = traindata
 
#     def _get_test_loader(self, task_idx=None):
#         """按需创建测试DataLoader"""
#         if task_idx is None:
#             # 使用当前任务的测试数据
#             return DataLoader(self.current_test_data, shuffle=False, batch_size=self.batch_size)
#         return None
 
#     def calculate_regularization(self):
#         local_w = self.model.fc2.weight
#         global_w = self.global_model.fc2.weight
 
#         reg = 0.0
 
#         for class_i in self.current_class:
#             local_i = local_w[class_i]
#             global_i = global_w[class_i]
 
#             # cosine similarity
#             sim_pos = torch.nn.functional.cosine_similarity(local_i, global_i, dim=0)
 
#             # positive logit
#             pos_logit = torch.exp(sim_pos)
 
#             # all negative logits
#             neg_logits = []
#             for class_k in range(self.num_classes):
#                 if class_k == class_i:
#                     continue
#                 sim_neg = torch.nn.functional.cosine_similarity(local_i, global_w[class_k], dim=0)
#                 neg_logits.append(torch.exp(sim_neg))
            
#             neg_sum = torch.stack(neg_logits).sum()
 
#             # InfoNCE loss
#             class_loss = -torch.log(pos_logit / neg_sum)
 
#             reg = reg + class_loss
 
#         return reg
 
#     def calculate_regularization_only_denominator(self):
#         # 1. 获取权重
#         local_w = self.model.fc2.weight      # shape: [100, feature_dim]
#         global_w = self.global_model.fc2.weight # shape: [100, feature_dim]
        
#         local_w_norm = torch.nn.functional.normalize(local_w, p=2, dim=1)
#         global_w_norm = torch.nn.functional.normalize(global_w, p=2, dim=1)
 
#         current_indices = torch.tensor(self.current_class, device=self.device)
#         current_local_vectors = local_w_norm[current_indices] 
 
#         sim_matrix = torch.matmul(current_local_vectors, global_w_norm.T)
#         exp_sim_matrix = torch.exp(sim_matrix) # shape: [len(current), 100]
#         row_sums = exp_sim_matrix.sum(dim=1) # shape: [len(current)]
#         self_exp_sim = exp_sim_matrix.gather(1, current_indices.unsqueeze(1)).squeeze(1)
#         neg_sum = row_sums - self_exp_sim
#         class_losses = torch.log(neg_sum + 1e-8)
#         reg = class_losses.sum()
 
#         return reg
    
#     def calculate_regularization_only_numerator(self):
#         local_w = self.model.fc2.weight
#         global_w = self.global_model.fc2.weight
 
#         reg = 0.0
 
#         for class_i in self.current_class:
#             local_i = local_w[class_i]
#             global_i = global_w[class_i]
 
#             # positive cosine similarity
#             sim_pos = torch.nn.functional.cosine_similarity(
#                 local_i, global_i, dim=0
#             )
 
#             # 只保留分子的部分： - sim_pos
#             class_loss = -sim_pos
 
#             reg = reg + class_loss
 
#         return reg
 
#     def phase2_train(self, task_id):
#         """使用投影梯度进行正则化训练"""
#         # 获取数据并初始化 train_loader
#         self.get_data(task_id)
 
#         self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
#         self.model.to(self.device)
 
#         # 记录本轮训练样本数
#         self.num_train_samples_this_round = len(self.train_data)
 
#         self.model.train()
 
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()
 
#         for epoch in range(self.epoch_local):
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 mapped_labels = torch.tensor([self.label_mapping[int(l)] for l in labels]).to(self.device)
 
#                 optimizer.zero_grad()
 
#                 feat, logits = self.model(inputs)
 
#                 task_logits = logits[:, list(self.current_class)]
#                 loss = criterion(task_logits, mapped_labels)
#                 denominator_loss = self.calculate_regularization_only_denominator()
#                 loss = loss + 0.1 * denominator_loss
#                 loss.backward()
 
#                 # 手动将缺席类的分类器梯度置零
#                 with torch.no_grad():
#                     for k in range(self.num_classes):
#                         if k not in self.current_class:
#                             if self.model.fc2.weight.grad is not None:
#                                 self.model.fc2.weight.grad[k] = 0
 
#                 optimizer.step()
 
#         print(f'{self.id} client finish training')
 
#         self.final_state = copy.deepcopy(self.model.state_dict())
#         self.evaluate_local_models(task_id)
        
#         # 清理内存
#         self._cleanup_after_training()
 
#     def _cleanup_after_training(self):
#         """训练后清理内存"""
#         # 删除train_loader释放内存
#         if hasattr(self, 'train_loader'):
#             del self.train_loader
        
#         # 清理训练数据引用
#         if hasattr(self, 'train_data') and self.train_data is not None:
#             del self.train_data
#             self.train_data = None
        
#         # 清理当前测试数据（评估完成后）
#         if hasattr(self, 'current_test_data') and self.current_test_data is not None:
#             del self.current_test_data
#             self.current_test_data = None
            
#         # 清理GPU缓存
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         # 强制垃圾回收
#         gc.collect()
 
#     def evaluate_local_models(self, index):
#         # 加载最终训练好的模型
#         self.model.load_state_dict(self.final_state)
#         self.model.eval()
#         correct, total = 0, 0
        
#         # 使用当前任务的测试数据
#         test_loader = self._get_test_loader()
        
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 _, logits = self.model(inputs)
                
#                 # 不再使用mask，直接使用完整的logits进行评估
#                 _, pred_idx = torch.max(logits, 1)
#                 predicted_labels = pred_idx
#                 total += labels.size(0)
#                 correct += (predicted_labels == labels).sum().item()
 
#         acc = correct / total * 100
#         print(f'Client {self.id}\'s local model on {index} task acc is {acc:.2f}%')
        
#         # 清理test_loader
#         del test_loader
        
#         return acc