import copy
import torch
import torch.nn as nn
from ResNet import resnet18_cbam
from copy import deepcopy
from torch.utils.data import random_split, DataLoader
from utils import (
    get_prunable_params, compute_importance,
    prune_and_create_task_mask, combine_masks
)


class Client(object):
    def __init__(self, dataset, lr, device, task_num, id, epoch_local, subset, batch_size, 
                 mask, num_classes, client_base_mask, prune_ratio=0.5, retrain_epochs=5):
        self.id = id
        self.task_num = task_num
        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.device = device
        self.dataset = dataset
        self.mask = mask  # 类别掩码
        self.num_classes = num_classes

        # PackNet相关
        self.client_base_mask = client_base_mask  # 客户端的固定参数空间（不重叠）
        self.prune_ratio = prune_ratio
        self.retrain_epochs = retrain_epochs
        
        # 任务掩码管理（在客户端自己的参数空间内再按任务划分）
        self.task_masks = {}  # task_id -> mask
        self.frozen_mask = None  # 已冻结的参数（之前任务）
        self.available_mask = None  # 当前可用的参数
        
        # GPU缓存
        self.client_base_mask_gpu = None
        self.available_mask_gpu = None
        self.current_task_mask_gpu = None
        
        # 已学习的类别
        self.learned_classes = set()

        self.test_loader = []
        self.train_dataset = subset
        self.train_data = None

        self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.prunable_params = None

    def _init_prunable_params(self):
        if self.prunable_params is None:
            self.prunable_params = set(get_prunable_params(self.model))

    def _move_masks_to_gpu(self):
        """将所有需要的掩码移到GPU（只执行一次或更新时）"""
        if self.client_base_mask_gpu is None:
            self.client_base_mask_gpu = {
                k: v.float().to(self.device) 
                for k, v in self.client_base_mask.items()
            }
        
        if self.available_mask is not None:
            self.available_mask_gpu = {
                k: v.float().to(self.device) 
                for k, v in self.available_mask.items()
            }

    def initialize_masks(self):
        """初始化：可用掩码 = 客户端基础掩码"""
        self.available_mask = {k: v.clone() for k, v in self.client_base_mask.items()}
        self.frozen_mask = {k: torch.zeros_like(v) for k, v in self.client_base_mask.items()}

    def recive_global_models(self, global_model):
        """接收全局模型 - 直接加载，因为参数不重叠所以不会冲突"""
        self.model.load_state_dict(global_model.state_dict())

    def get_data(self, task_id):
        if task_id % 5 == 0:
            self.task_id = task_id // 5
            self.current_class = self.mask[self.task_id]
            self.trainset = self.train_dataset[self.task_id]
            if self.task_id == 0:
                self.initialize_masks()

        print(f'Client {self.id}, Task {self.task_id}: classes {self.current_class}')

        traindata, testdata = random_split(self.trainset,
                                           [int(len(self.trainset) * 0.8),
                                            len(self.trainset) - int(len(self.trainset) * 0.8)])
        testdata = deepcopy(testdata)
        self.test_loader.append(DataLoader(testdata, shuffle=True, batch_size=self.batch_size))
        self.train_data = traindata

    def train(self, task_id):
        """PackNet训练流程"""
        self.get_data(task_id)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        current_task = task_id // 5
        
        # 初始化
        self._init_prunable_params()
        self._move_masks_to_gpu()
        
        # ============ 阶段1：训练 ============
        print(f'Client {self.id} - Task {current_task}: Phase 1 - Training')
        self._train_phase(self.epoch_local, use_available_mask=True)

        # ============ 阶段2：剪枝 ============
        print(f'Client {self.id} - Task {current_task}: Phase 2 - Pruning')
        importance = compute_importance(self.model, self.train_loader, self.device, self.available_mask)
        task_mask, new_available_mask = prune_and_create_task_mask(
            self.model, importance, self.available_mask, self.prune_ratio, self.device
        )
        
        # 保存任务掩码，更新可用掩码
        self.task_masks[current_task] = task_mask
        self.available_mask = new_available_mask
        self.frozen_mask = combine_masks(self.frozen_mask, task_mask)
        
        # 更新GPU缓存
        self.current_task_mask_gpu = {k: v.float().to(self.device) for k, v in task_mask.items()}
        self._move_masks_to_gpu()

        # ============ 阶段3：重训练 ============
        print(f'Client {self.id} - Task {current_task}: Phase 3 - Retraining')
        self._train_phase(self.retrain_epochs, use_available_mask=False)
        
        # 更新已学习类别
        self.learned_classes.update(self.current_class)
        
        print(f'Client {self.id} finished task {current_task}, learned classes: {sorted(self.learned_classes)}')
        self.final_state = copy.deepcopy(self.model.state_dict())
        self._print_mask_statistics(current_task)
        self.evaluate_local_models(current_task)

    def _train_phase(self, epochs, use_available_mask=True):
        """训练阶段"""
        self.model.to(self.device)
        self.model.train()
        
        lr = self.learning_rate if use_available_mask else self.learning_rate * 0.1
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                
                optimizer.zero_grad()
                _, logits = self.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                
                # 应用梯度掩码
                self._apply_gradient_mask(use_available_mask)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}')

    def _apply_gradient_mask(self, use_available_mask=True):
        """
        应用梯度掩码
        - 特征提取器（conv + BN）：只更新客户端自己的参数空间
        - 分类器：只更新当前任务的类别
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            if name in self.prunable_params:
                # 特征提取器 + BN层：限制在客户端自己的参数空间内
                if use_available_mask:
                    # 训练阶段：可用参数
                    if name in self.available_mask_gpu:
                        param.grad *= self.available_mask_gpu[name]
                    else:
                        param.grad.zero_()
                else:
                    # 重训练阶段：当前任务的参数
                    if name in self.current_task_mask_gpu:
                        param.grad *= self.current_task_mask_gpu[name]
                    else:
                        param.grad.zero_()
            
            elif 'fc' in name and 'weight' in name:
                # 分类器权重：只更新当前类别
                if len(param.grad.shape) == 2:
                    num_out = param.grad.shape[0]
                    class_mask = torch.zeros(num_out, dtype=torch.float32, device=self.device)
                    for cls in self.current_class:
                        if cls < num_out:
                            class_mask[cls] = 1.0
                    param.grad *= class_mask.unsqueeze(1)
            
            elif 'fc' in name and 'bias' in name:
                # 分类器偏置：只更新当前类别
                num_out = param.grad.shape[0]
                class_mask = torch.zeros(num_out, dtype=torch.float32, device=self.device)
                for cls in self.current_class:
                    if cls < num_out:
                        class_mask[cls] = 1.0
                param.grad *= class_mask

    def _print_mask_statistics(self, task_id):
        total_params = 0
        base_params = 0
        task_params = 0
        available_params = 0
        
        for name in self.prunable_params:
            param = dict(self.model.named_parameters())[name]
            total_params += param.numel()
            if name in self.client_base_mask:
                base_params += self.client_base_mask[name].sum().item()
            if task_id in self.task_masks and name in self.task_masks[task_id]:
                task_params += self.task_masks[task_id][name].sum().item()
            if name in self.available_mask:
                available_params += self.available_mask[name].sum().item()
        
        print(f'Client {self.id} - Task {task_id} Statistics:')
        print(f'  Total params: {total_params:,}')
        print(f'  Client space: {base_params:,} ({base_params/total_params*100:.2f}%)')
        print(f'  Task {task_id} used: {task_params:,} ({task_params/total_params*100:.2f}%)')
        print(f'  Remaining: {available_params:,} ({available_params/total_params*100:.2f}%)')

    def evaluate_local_models(self, index):
        self.model.load_state_dict(self.final_state)
        self.model.to(self.device)
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader[index]:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                _, logits = self.model(inputs)
                _, preds = torch.max(logits, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = correct / total * 100 if total > 0 else 0
        print(f'Client {self.id} local model on task {index}: {acc:.2f}%')
        return acc

    def get_all_task_masks(self):
        return self.task_masks




# # 计算通信开销


# import copy
# import torch
# import torch.nn as nn
# from ResNet import resnet18_cbam
# from copy import deepcopy
# from torch.utils.data import random_split, DataLoader
# from utils import (
#     get_prunable_params, compute_importance,
#     prune_and_create_task_mask, combine_masks
# )


# class Client(object):
#     def __init__(self, dataset, lr, device, task_num, id, epoch_local, subset, batch_size, 
#                  mask, num_classes, client_base_mask, prune_ratio=0.5, retrain_epochs=5):
#         self.id = id
#         self.task_num = task_num
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.dataset = dataset
#         self.mask = mask  # 类别掩码
#         self.num_classes = num_classes

#         # PackNet相关
#         self.client_base_mask = client_base_mask  # 客户端的固定参数空间（不重叠）
#         self.prune_ratio = prune_ratio
#         self.retrain_epochs = retrain_epochs
        
#         # 任务掩码管理（在客户端自己的参数空间内再按任务划分）
#         self.task_masks = {}  # task_id -> mask
#         self.frozen_mask = None  # 已冻结的参数（之前任务）
#         self.available_mask = None  # 当前可用的参数
        
#         # GPU缓存
#         self.client_base_mask_gpu = None
#         self.available_mask_gpu = None
#         self.current_task_mask_gpu = None
        
#         # 已学习的类别
#         self.learned_classes = set()

#         self.test_loader = []
#         self.train_dataset = subset
#         self.train_data = None

#         self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.prunable_params = None

#         # 用于统计每轮训练的样本数（计算FLOPs用）
#         self.num_train_samples_this_round = 0

#     def _init_prunable_params(self):
#         if self.prunable_params is None:
#             self.prunable_params = set(get_prunable_params(self.model))

#     def _move_masks_to_gpu(self):
#         """将所有需要的掩码移到GPU（只执行一次或更新时）"""
#         if self.client_base_mask_gpu is None:
#             self.client_base_mask_gpu = {
#                 k: v.float().to(self.device) 
#                 for k, v in self.client_base_mask.items()
#             }
        
#         if self.available_mask is not None:
#             self.available_mask_gpu = {
#                 k: v.float().to(self.device) 
#                 for k, v in self.available_mask.items()
#             }

#     def initialize_masks(self):
#         """初始化：可用掩码 = 客户端基础掩码"""
#         self.available_mask = {k: v.clone() for k, v in self.client_base_mask.items()}
#         self.frozen_mask = {k: torch.zeros_like(v) for k, v in self.client_base_mask.items()}

#     def recive_global_models(self, global_model):
#         """接收全局模型 - 直接加载，因为参数不重叠所以不会冲突"""
#         self.model.load_state_dict(global_model.state_dict())

#     def get_data(self, task_id):
#         if task_id % 5 == 0:
#             self.task_id = task_id // 5
#             self.current_class = self.mask[self.task_id]
#             self.trainset = self.train_dataset[self.task_id]
#             if self.task_id == 0:
#                 self.initialize_masks()

#         print(f'Client {self.id}, Task {self.task_id}: classes {self.current_class}')

#         traindata, testdata = random_split(self.trainset,
#                                            [int(len(self.trainset) * 0.8),
#                                             len(self.trainset) - int(len(self.trainset) * 0.8)])
#         testdata = deepcopy(testdata)
#         self.test_loader.append(DataLoader(testdata, shuffle=True, batch_size=self.batch_size))
#         self.train_data = traindata

#     def train(self, task_id):
#         """PackNet训练流程"""
#         self.get_data(task_id)
#         self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
#         current_task = task_id // 5

#         # 记录本轮训练样本数
#         self.num_train_samples_this_round = len(self.train_data)
        
#         # 初始化
#         self._init_prunable_params()
#         self._move_masks_to_gpu()
        
#         # ============ 阶段1：训练 ============
#         print(f'Client {self.id} - Task {current_task}: Phase 1 - Training')
#         self._train_phase(self.epoch_local, use_available_mask=True)

#         # ============ 阶段2：剪枝 ============
#         print(f'Client {self.id} - Task {current_task}: Phase 2 - Pruning')
#         importance = compute_importance(self.model, self.train_loader, self.device, self.available_mask)
#         task_mask, new_available_mask = prune_and_create_task_mask(
#             self.model, importance, self.available_mask, self.prune_ratio, self.device
#         )
        
#         # 保存任务掩码，更新可用掩码
#         self.task_masks[current_task] = task_mask
#         self.available_mask = new_available_mask
#         self.frozen_mask = combine_masks(self.frozen_mask, task_mask)
        
#         # 更新GPU缓存
#         self.current_task_mask_gpu = {k: v.float().to(self.device) for k, v in task_mask.items()}
#         self._move_masks_to_gpu()

#         # ============ 阶段3：重训练 ============
#         print(f'Client {self.id} - Task {current_task}: Phase 3 - Retraining')
#         self._train_phase(self.retrain_epochs, use_available_mask=False)
        
#         # 更新已学习类别
#         self.learned_classes.update(self.current_class)
        
#         print(f'Client {self.id} finished task {current_task}, learned classes: {sorted(self.learned_classes)}')
#         self.final_state = copy.deepcopy(self.model.state_dict())
#         self._print_mask_statistics(current_task)
#         self.evaluate_local_models(current_task)

#     def _train_phase(self, epochs, use_available_mask=True):
#         """训练阶段"""
#         self.model.to(self.device)
#         self.model.train()
        
#         lr = self.learning_rate if use_available_mask else self.learning_rate * 0.1
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
#         criterion = nn.CrossEntropyLoss()

#         for epoch in range(epochs):
#             epoch_loss = 0
#             num_batches = 0
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 optimizer.zero_grad()
#                 _, logits = self.model(inputs)
#                 loss = criterion(logits, labels)
#                 loss.backward()
                
#                 # 应用梯度掩码
#                 self._apply_gradient_mask(use_available_mask)
                
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
#                 num_batches += 1
            
#             if (epoch + 1) % 5 == 0:
#                 print(f'  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}')

#     def _apply_gradient_mask(self, use_available_mask=True):
#         """
#         应用梯度掩码
#         - 特征提取器（conv + BN）：只更新客户端自己的参数空间
#         - 分类器：只更新当前任务的类别
#         """
#         for name, param in self.model.named_parameters():
#             if param.grad is None:
#                 continue
            
#             if name in self.prunable_params:
#                 # 特征提取器 + BN层：限制在客户端自己的参数空间内
#                 if use_available_mask:
#                     # 训练阶段：可用参数
#                     if name in self.available_mask_gpu:
#                         param.grad *= self.available_mask_gpu[name]
#                     else:
#                         param.grad.zero_()
#                 else:
#                     # 重训练阶段：当前任务的参数
#                     if name in self.current_task_mask_gpu:
#                         param.grad *= self.current_task_mask_gpu[name]
#                     else:
#                         param.grad.zero_()
            
#             elif 'fc' in name and 'weight' in name:
#                 # 分类器权重：只更新当前类别
#                 if len(param.grad.shape) == 2:
#                     num_out = param.grad.shape[0]
#                     class_mask = torch.zeros(num_out, dtype=torch.float32, device=self.device)
#                     for cls in self.current_class:
#                         if cls < num_out:
#                             class_mask[cls] = 1.0
#                     param.grad *= class_mask.unsqueeze(1)
            
#             elif 'fc' in name and 'bias' in name:
#                 # 分类器偏置：只更新当前类别
#                 num_out = param.grad.shape[0]
#                 class_mask = torch.zeros(num_out, dtype=torch.float32, device=self.device)
#                 for cls in self.current_class:
#                     if cls < num_out:
#                         class_mask[cls] = 1.0
#                 param.grad *= class_mask

#     def _print_mask_statistics(self, task_id):
#         total_params = 0
#         base_params = 0
#         task_params = 0
#         available_params = 0
        
#         for name in self.prunable_params:
#             param = dict(self.model.named_parameters())[name]
#             total_params += param.numel()
#             if name in self.client_base_mask:
#                 base_params += self.client_base_mask[name].sum().item()
#             if task_id in self.task_masks and name in self.task_masks[task_id]:
#                 task_params += self.task_masks[task_id][name].sum().item()
#             if name in self.available_mask:
#                 available_params += self.available_mask[name].sum().item()
        
#         print(f'Client {self.id} - Task {task_id} Statistics:')
#         print(f'  Total params: {total_params:,}')
#         print(f'  Client space: {base_params:,} ({base_params/total_params*100:.2f}%)')
#         print(f'  Task {task_id} used: {task_params:,} ({task_params/total_params*100:.2f}%)')
#         print(f'  Remaining: {available_params:,} ({available_params/total_params*100:.2f}%)')

#     def evaluate_local_models(self, index):
#         self.model.load_state_dict(self.final_state)
#         self.model.to(self.device)
#         self.model.eval()
#         correct, total = 0, 0
        
#         with torch.no_grad():
#             for inputs, labels in self.test_loader[index]:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 _, logits = self.model(inputs)
#                 _, preds = torch.max(logits, 1)
#                 total += labels.size(0)
#                 correct += (preds == labels).sum().item()

#         acc = correct / total * 100 if total > 0 else 0
#         print(f'Client {self.id} local model on task {index}: {acc:.2f}%')
#         return acc

#     def get_all_task_masks(self):
#         return self.task_masks