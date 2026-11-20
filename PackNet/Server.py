# import copy
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# import os

# from client import Client
# from utils import (
#     prediction, prediction_task, create_client_base_masks,
#     TaskIDPredictor, train_task_id_predictor, get_prunable_params
# )
# from ResNet import resnet18_cbam


# class Server(object):
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path,
#                  prune_ratio=0.5, retrain_epochs=5, task_id_epochs=20):
#         self.client_num = client_num
#         self.task_num = task_num
#         self.data = subset
#         self.mask = mask
#         self.dataset = dataset
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.num_classes = class_num
#         self.test_loader = test_dataset
#         self.save_path = save_path

#         # PackNet参数
#         self.prune_ratio = prune_ratio
#         self.retrain_epochs = retrain_epochs
#         self.task_id_epochs = task_id_epochs

#         # 全局模型
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        
#         # 为每个客户端创建不重叠的参数掩码
#         self.client_base_masks = create_client_base_masks(self.global_model, self.client_num)
        
#         # 预缓存
#         self._prunable_params = set(get_prunable_params(self.global_model))
#         self._client_masks_float = {
#             cid: {k: v.float() for k, v in masks.items()}
#             for cid, masks in self.client_base_masks.items()
#         }
        
#         # 任务ID预测器
#         self.task_id_predictors = {}
#         self.client_task_dataloaders = {i: {} for i in range(self.client_num)}
        
#         self.clients = []
#         os.makedirs(self.save_path, exist_ok=True)

#     def init_clients(self):
#         print("="*60)
#         print("Initializing clients with non-overlapping parameter partitions")
#         print("="*60)
        
#         for i in range(self.client_num):
#             client = Client(
#                 self.dataset, self.learning_rate, self.device,
#                 self.task_num, i, self.epoch_local, self.data[i],
#                 self.batch_size, self.mask[i], self.num_classes,
#                 self.client_base_masks[i],
#                 self.prune_ratio, self.retrain_epochs
#             )
#             self.clients.append(client)
#             self.task_id_predictors[i] = TaskIDPredictor(feature_dim=512, num_tasks=self.task_num)
        
#         self._print_partition_statistics()
#         print("Initialization complete\n")

#     def _print_partition_statistics(self):
#         total_params = sum(p.numel() for n, p in self.global_model.named_parameters() if n in self._prunable_params)
        
#         for client_id in range(self.client_num):
#             client_params = sum(
#                 self._client_masks_float[client_id][name].sum().item()
#                 for name in self._prunable_params
#                 if name in self._client_masks_float[client_id]
#             )
#             print(f"Client {client_id}: {int(client_params):,} params ({client_params/total_params*100:.1f}%)")

#     def merge_client_models(self, client_models):
#         """
#         合并客户端模型 - 直接拼接，不是聚合！
#         全部在GPU上执行提高效率
#         """
#         # 确保全局模型在GPU上
#         self.global_model.to(self.device)
#         global_state = self.global_model.state_dict()
#         merged_state = {}
        
#         # 收集所有客户端的state_dict（保持在GPU上）
#         client_states = [m.state_dict() for m in client_models]
        
#         # 确保掩码在GPU上（只做一次）
#         if not hasattr(self, '_client_masks_gpu'):
#             self._client_masks_gpu = {
#                 cid: {k: v.to(self.device) for k, v in masks.items()}
#                 for cid, masks in self._client_masks_float.items()
#             }
        
#         for name in global_state.keys():
#             if name in self._prunable_params:
#                 # 特征提取器 + BN可学习参数：直接拼接
#                 merged = torch.zeros_like(global_state[name], dtype=torch.float32, device=self.device)
                
#                 for cid in range(self.client_num):
#                     mask = self._client_masks_gpu[cid].get(name)
#                     if mask is not None:
#                         client_param = client_states[cid][name].to(self.device).float()
#                         merged += client_param * mask
                
#                 merged_state[name] = merged
            
#             elif 'fc' in name and 'weight' in name:
#                 # 分类器权重：按类别拼接
#                 merged = global_state[name].clone().float()
#                 num_out = merged.shape[0]
                
#                 for cid, client in enumerate(self.clients):
#                     client_param = client_states[cid][name].to(self.device).float()
#                     for cls in client.learned_classes:
#                         if cls < num_out:
#                             merged[cls] = client_param[cls]
                
#                 merged_state[name] = merged
            
#             elif 'fc' in name and 'bias' in name:
#                 # 分类器偏置：按类别拼接
#                 merged = global_state[name].clone().float()
#                 num_out = merged.shape[0]
                
#                 for cid, client in enumerate(self.clients):
#                     client_param = client_states[cid][name].to(self.device).float()
#                     for cls in client.learned_classes:
#                         if cls < num_out:
#                             merged[cls] = client_param[cls]
                
#                 merged_state[name] = merged
            
#             elif 'running_mean' in name or 'running_var' in name:
#                 # BN的统计量：也按客户端拼接
#                 bn_weight_name = name.replace('running_mean', 'weight').replace('running_var', 'weight')
                
#                 merged = torch.zeros_like(global_state[name], dtype=torch.float32, device=self.device)
                
#                 for cid in range(self.client_num):
#                     mask = self._client_masks_gpu[cid].get(bn_weight_name)
#                     if mask is not None:
#                         client_param = client_states[cid][name].to(self.device).float()
#                         merged += client_param * mask
                
#                 merged_state[name] = merged
            
#             elif 'num_batches_tracked' in name:
#                 # BN追踪数：取最大值
#                 merged_state[name] = max(client_states[cid][name] for cid in range(self.client_num))
            
#             else:
#                 # 其他参数
#                 merged_state[name] = client_states[0][name]
        
#         self.global_model.load_state_dict(merged_state)
#         return self.global_model

#     def train_clients(self):
#         all_eval_results = [] # 用于存储最后一次评估的结果——————————————————————
#         for i in range(self.task_num * 5):
#             print(f"\n{'='*60}")
#             print(f"Round {i}, Task {i // 5}")
#             print(f"{'='*60}")

#             # 分发全局模型
#             for client in self.clients:
#                 client.recive_global_models(self.global_model)

#             # 客户端本地训练
#             client_models = []
#             for client in self.clients:
#                 client.train(i)
#                 client_models.append(client.model)
                
#                 # 保存数据加载器
#                 current_task = i // 5
#                 if current_task not in self.client_task_dataloaders[client.id]:
#                     self.client_task_dataloaders[client.id][current_task] = client.train_loader

#             # 合并模型（拼接，非聚合）
#             print("\nMerging client models (concatenation, not aggregation)...")
#             self.global_model = self.merge_client_models(client_models)

#             # 每个任务结束后评估
#             if (i + 1) % 5 == 0:
#                 current_task = i // 5
#                 print(f"\nTraining Task ID Predictors...")
#                 self._train_all_task_id_predictors(current_task)
#                 # self.model_global_eval(i)
#                 all_eval_results.append(self.model_global_eval(i))#————————————————————————————

#         # 保存最终模型
#         final_path = os.path.join(self.save_path, f'{self.dataset}_packnet_final.pth')
#         torch.save(self.global_model.state_dict(), final_path)
#         print(f"\nSaved final model to {final_path}")
#         print("Training complete!")
#         return all_eval_results # 返回最终结果给 main.py————————————————————

#     def _train_all_task_id_predictors(self, current_task):
#         for client in self.clients:
#             dataloaders = {
#                 tid: self.client_task_dataloaders[client.id][tid]
#                 for tid in range(current_task + 1)
#                 if tid in self.client_task_dataloaders[client.id]
#             }
#             if dataloaders:
#                 self.task_id_predictors[client.id] = train_task_id_predictor(
#                     self.task_id_predictors[client.id],
#                     self.global_model, dataloaders, self.device,
#                     epochs=self.task_id_epochs, lr=0.001
#                 )

#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         acc_all = prediction(self.global_model, self.test_loader, self.device)
#         print(f'\n{self.dataset}: Round {taskid}, Overall accuracy: {acc_all:.2f}%')

#         task_index = taskid // 5
#         for t in range(task_index + 1):
#             all_classes = set()
#             for client_mask in self.mask:
#                 all_classes.update(client_mask[t])
#             acc_task = prediction_task(self.global_model, self.test_loader, self.device, list(all_classes))
#             print(f'{self.dataset}: Task {t} accuracy: {acc_task:.2f}%')
#             results[f'task_{t}'] = acc_task#——————————————————————————————

#         # 保存checkpoint
#         ckpt_path = os.path.join(self.save_path, f'{self.dataset}_packnet_task_{task_index}.pth')
#         torch.save(self.global_model.state_dict(), ckpt_path)
#         return results # 返回结果字典——————————————————————————————




# 3 turns

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os

from client import Client
from utils import (
    prediction, prediction_task, create_client_base_masks,
    TaskIDPredictor, train_task_id_predictor, get_prunable_params
)
from ResNet import resnet18_cbam


class Server(object):
    def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
                 local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path,
                 prune_ratio=0.5, retrain_epochs=5, task_id_epochs=20):
        self.client_num = client_num
        self.task_num = task_num
        self.data = subset
        self.mask = mask
        self.dataset = dataset
        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.device = device
        self.num_classes = class_num
        self.test_loader = test_dataset
        self.save_path = save_path

        # PackNet参数
        self.prune_ratio = prune_ratio
        self.retrain_epochs = retrain_epochs
        self.task_id_epochs = task_id_epochs

        # 全局模型
        self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        
        # 为每个客户端创建不重叠的参数掩码
        self.client_base_masks = create_client_base_masks(self.global_model, self.client_num)
        
        # 预缓存
        self._prunable_params = set(get_prunable_params(self.global_model))
        self._client_masks_float = {
            cid: {k: v.float() for k, v in masks.items()}
            for cid, masks in self.client_base_masks.items()
        }
        
        # 任务ID预测器
        self.task_id_predictors = {}
        self.client_task_dataloaders = {i: {} for i in range(self.client_num)}
        
        self.clients = []
        os.makedirs(self.save_path, exist_ok=True)

    def init_clients(self):
        print("="*60)
        print("Initializing clients with non-overlapping parameter partitions")
        print("="*60)
        
        for i in range(self.client_num):
            client = Client(
                self.dataset, self.learning_rate, self.device,
                self.task_num, i, self.epoch_local, self.data[i],
                self.batch_size, self.mask[i], self.num_classes,
                self.client_base_masks[i],
                self.prune_ratio, self.retrain_epochs
            )
            self.clients.append(client)
            self.task_id_predictors[i] = TaskIDPredictor(feature_dim=512, num_tasks=self.task_num)
        
        self._print_partition_statistics()
        print("Initialization complete\n")

    def _print_partition_statistics(self):
        total_params = sum(p.numel() for n, p in self.global_model.named_parameters() if n in self._prunable_params)
        
        for client_id in range(self.client_num):
            client_params = sum(
                self._client_masks_float[client_id][name].sum().item()
                for name in self._prunable_params
                if name in self._client_masks_float[client_id]
            )
            print(f"Client {client_id}: {int(client_params):,} params ({client_params/total_params*100:.1f}%)")

    def merge_client_models(self, client_models):
        """
        合并客户端模型 - 直接拼接，不是聚合！
        全部在GPU上执行提高效率
        """
        # 确保全局模型在GPU上
        self.global_model.to(self.device)
        global_state = self.global_model.state_dict()
        merged_state = {}
        
        # 收集所有客户端的state_dict（保持在GPU上）
        client_states = [m.state_dict() for m in client_models]
        
        # 确保掩码在GPU上（只做一次）
        if not hasattr(self, '_client_masks_gpu'):
            self._client_masks_gpu = {
                cid: {k: v.to(self.device) for k, v in masks.items()}
                for cid, masks in self._client_masks_float.items()
            }
        
        for name in global_state.keys():
            if name in self._prunable_params:
                # 特征提取器 + BN可学习参数：直接拼接
                merged = torch.zeros_like(global_state[name], dtype=torch.float32, device=self.device)
                
                for cid in range(self.client_num):
                    mask = self._client_masks_gpu[cid].get(name)
                    if mask is not None:
                        client_param = client_states[cid][name].to(self.device).float()
                        merged += client_param * mask
                
                merged_state[name] = merged
            
            elif 'fc' in name and 'weight' in name:
                # 分类器权重：按类别拼接
                merged = global_state[name].clone().float()
                num_out = merged.shape[0]
                
                for cid, client in enumerate(self.clients):
                    client_param = client_states[cid][name].to(self.device).float()
                    for cls in client.learned_classes:
                        if cls < num_out:
                            merged[cls] = client_param[cls]
                
                merged_state[name] = merged
            
            elif 'fc' in name and 'bias' in name:
                # 分类器偏置：按类别拼接
                merged = global_state[name].clone().float()
                num_out = merged.shape[0]
                
                for cid, client in enumerate(self.clients):
                    client_param = client_states[cid][name].to(self.device).float()
                    for cls in client.learned_classes:
                        if cls < num_out:
                            merged[cls] = client_param[cls]
                
                merged_state[name] = merged
            
            elif 'running_mean' in name or 'running_var' in name:
                # BN的统计量：也按客户端拼接
                bn_weight_name = name.replace('running_mean', 'weight').replace('running_var', 'weight')
                
                merged = torch.zeros_like(global_state[name], dtype=torch.float32, device=self.device)
                
                for cid in range(self.client_num):
                    mask = self._client_masks_gpu[cid].get(bn_weight_name)
                    if mask is not None:
                        client_param = client_states[cid][name].to(self.device).float()
                        merged += client_param * mask
                
                merged_state[name] = merged
            
            elif 'num_batches_tracked' in name:
                # BN追踪数：取最大值
                merged_state[name] = max(client_states[cid][name] for cid in range(self.client_num))
            
            else:
                # 其他参数
                merged_state[name] = client_states[0][name]
        
        self.global_model.load_state_dict(merged_state)
        return self.global_model

    def train_clients(self):
        all_eval_results = [] # 用于存储每次评估的结果——————————————————————
        for i in range(self.task_num * 5):
            print(f"\n{'='*60}")
            print(f"Round {i}, Task {i // 5}")
            print(f"{'='*60}")

            # 分发全局模型
            for client in self.clients:
                client.recive_global_models(self.global_model)

            # 客户端本地训练
            client_models = []
            for client in self.clients:
                client.train(i)
                client_models.append(client.model)
                
                # 保存数据加载器
                current_task = i // 5
                if current_task not in self.client_task_dataloaders[client.id]:
                    self.client_task_dataloaders[client.id][current_task] = client.train_loader

            # 合并模型（拼接，非聚合）
            print("\nMerging client models (concatenation, not aggregation)...")
            self.global_model = self.merge_client_models(client_models)

            # 每个任务结束后评估
            if (i + 1) % 5 == 0:
                current_task = i // 5
                print(f"\nTraining Task ID Predictors...")
                self._train_all_task_id_predictors(current_task)
                # self.model_global_eval(i)
                final_results = self.model_global_eval(i)#————————————————————————————
                all_eval_results.append(final_results)#————————————————————————————

        # 保存最终模型
        final_path = os.path.join(self.save_path, f'{self.dataset}_packnet_final.pth')
        torch.save(self.global_model.state_dict(), final_path)
        print(f"\nSaved final model to {final_path}")
        print("Training complete!")
        return all_eval_results # 返回所有评估结果给 main.py————————————————————

    def _train_all_task_id_predictors(self, current_task):
        for client in self.clients:
            dataloaders = {
                tid: self.client_task_dataloaders[client.id][tid]
                for tid in range(current_task + 1)
                if tid in self.client_task_dataloaders[client.id]
            }
            if dataloaders:
                self.task_id_predictors[client.id] = train_task_id_predictor(
                    self.task_id_predictors[client.id],
                    self.global_model, dataloaders, self.device,
                    epochs=self.task_id_epochs, lr=0.001
                )

    def model_global_eval(self, taskid):
        results = {} # 创建字典存储结果——————————————————————————————————
        results['round'] = taskid # 记录轮次——————————————————————————————
        acc_all = prediction(self.global_model, self.test_loader, self.device)
        print(f'\n{self.dataset}: Round {taskid}, Overall accuracy: {acc_all:.2f}%')
        results['all_tasks'] = acc_all # 存储总精度——————————————————————————

        task_index = taskid // 5
        for t in range(task_index + 1):
            all_classes = set()
            for client_mask in self.mask:
                all_classes.update(client_mask[t])
            acc_task = prediction_task(self.global_model, self.test_loader, self.device, list(all_classes))
            print(f'{self.dataset}: Task {t} accuracy: {acc_task:.2f}%')
            results[f'task_{t}'] = acc_task#——————————————————————————————

        # 保存checkpoint
        ckpt_path = os.path.join(self.save_path, f'{self.dataset}_packnet_task_{task_index}.pth')
        torch.save(self.global_model.state_dict(), ckpt_path)
        return results # 返回结果字典——————————————————————————————



# # 计算通信开销



# import copy
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# import os
 
# from client import Client
# from utils import (
#     prediction, prediction_task, create_client_base_masks,
#     TaskIDPredictor, train_task_id_predictor, get_prunable_params
# )
# from ResNet import resnet18_cbam
 
 
# class Server(object):
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path,
#                  prune_ratio=0.5, retrain_epochs=5, task_id_epochs=20):
#         self.client_num = client_num
#         self.task_num = task_num
#         self.data = subset
#         self.mask = mask
#         self.dataset = dataset
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.num_classes = class_num
#         self.test_loader = test_dataset
#         self.save_path = save_path
 
#         # PackNet参数
#         self.prune_ratio = prune_ratio
#         self.retrain_epochs = retrain_epochs
#         self.task_id_epochs = task_id_epochs
 
#         # 全局模型
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        
#         # 为每个客户端创建不重叠的参数掩码
#         self.client_base_masks = create_client_base_masks(self.global_model, self.client_num)
        
#         # 预缓存
#         self._prunable_params = set(get_prunable_params(self.global_model))
#         self._client_masks_float = {
#             cid: {k: v.float() for k, v in masks.items()}
#             for cid, masks in self.client_base_masks.items()
#         }
        
#         # 任务ID预测器
#         self.task_id_predictors = {}
#         self.client_task_dataloaders = {i: {} for i in range(self.client_num)}
        
#         self.clients = []
#         os.makedirs(self.save_path, exist_ok=True)
 
#         # ========== 计算和通信开销统计 ==========
#         self.total_comm_bytes = 0       # 累计通信字节数
#         self.total_train_flops = 0      # 累计训练FLOPs
#         self._forward_flops = None      # 单次前向FLOPs（延迟计算）
 
#     def _get_model_bytes(self):
#         """计算模型参数+缓冲区的总字节数"""
#         total_bytes = 0
#         for param in self.global_model.parameters():
#             total_bytes += param.nelement() * param.element_size()
#         for buf in self.global_model.buffers():
#             total_bytes += buf.nelement() * buf.element_size()
#         return total_bytes
 
#     def _estimate_forward_flops(self):
#         """使用hook估算单个样本前向传播的FLOPs"""
#         if self._forward_flops is not None:
#             return self._forward_flops
 
#         total_flops = [0]
 
#         def hook_fn(module, inp, out):
#             if isinstance(module, nn.Conv2d):
#                 out_h, out_w = out.shape[2], out.shape[3]
#                 total_flops[0] += 2 * module.out_channels * out_h * out_w * \
#                     (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
#             elif isinstance(module, nn.Linear):
#                 total_flops[0] += 2 * module.in_features * module.out_features
#             elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
#                 total_flops[0] += 2 * out.numel()
 
#         hooks = []
#         for m in self.global_model.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
#                 hooks.append(m.register_forward_hook(hook_fn))
 
#         if self.dataset == 'TinyImageNet':
#             img_size = 64
#         elif self.dataset == 'MNIST':
#             img_size = 28
#         else:
#             img_size = 32
 
#         dummy = torch.randn(1, 3, img_size, img_size)
#         self.global_model.eval()
#         with torch.no_grad():
#             self.global_model(dummy)
#         self.global_model.train()
 
#         for h in hooks:
#             h.remove()
 
#         self._forward_flops = total_flops[0]
#         return self._forward_flops
 
#     def _format_bytes(self, num_bytes):
#         if num_bytes >= 1e9:
#             return f"{num_bytes / 1e9:.2f} GB"
#         elif num_bytes >= 1e6:
#             return f"{num_bytes / 1e6:.2f} MB"
#         elif num_bytes >= 1e3:
#             return f"{num_bytes / 1e3:.2f} KB"
#         return f"{num_bytes} B"
 
#     def _format_flops(self, flops):
#         if flops >= 1e15:
#             return f"{flops / 1e15:.2f} PFLOPs"
#         elif flops >= 1e12:
#             return f"{flops / 1e12:.2f} TFLOPs"
#         elif flops >= 1e9:
#             return f"{flops / 1e9:.2f} GFLOPs"
#         elif flops >= 1e6:
#             return f"{flops / 1e6:.2f} MFLOPs"
#         return f"{flops:.0f} FLOPs"
 
#     def init_clients(self):
#         print("="*60)
#         print("Initializing clients with non-overlapping parameter partitions")
#         print("="*60)
        
#         for i in range(self.client_num):
#             client = Client(
#                 self.dataset, self.learning_rate, self.device,
#                 self.task_num, i, self.epoch_local, self.data[i],
#                 self.batch_size, self.mask[i], self.num_classes,
#                 self.client_base_masks[i],
#                 self.prune_ratio, self.retrain_epochs
#             )
#             self.clients.append(client)
#             self.task_id_predictors[i] = TaskIDPredictor(feature_dim=512, num_tasks=self.task_num)
        
#         self._print_partition_statistics()
#         print("Initialization complete\n")
 
#     def _print_partition_statistics(self):
#         total_params = sum(p.numel() for n, p in self.global_model.named_parameters() if n in self._prunable_params)
        
#         for client_id in range(self.client_num):
#             client_params = sum(
#                 self._client_masks_float[client_id][name].sum().item()
#                 for name in self._prunable_params
#                 if name in self._client_masks_float[client_id]
#             )
#             print(f"Client {client_id}: {int(client_params):,} params ({client_params/total_params*100:.1f}%)")
 
#     def merge_client_models(self, client_models):
#         """
#         合并客户端模型 - 直接拼接，不是聚合！
#         全部在GPU上执行提高效率
#         """
#         # 确保全局模型在GPU上
#         self.global_model.to(self.device)
#         global_state = self.global_model.state_dict()
#         merged_state = {}
        
#         # 收集所有客户端的state_dict（保持在GPU上）
#         client_states = [m.state_dict() for m in client_models]
        
#         # 确保掩码在GPU上（只做一次）
#         if not hasattr(self, '_client_masks_gpu'):
#             self._client_masks_gpu = {
#                 cid: {k: v.to(self.device) for k, v in masks.items()}
#                 for cid, masks in self._client_masks_float.items()
#             }
        
#         for name in global_state.keys():
#             if name in self._prunable_params:
#                 # 特征提取器 + BN可学习参数：直接拼接
#                 merged = torch.zeros_like(global_state[name], dtype=torch.float32, device=self.device)
                
#                 for cid in range(self.client_num):
#                     mask = self._client_masks_gpu[cid].get(name)
#                     if mask is not None:
#                         client_param = client_states[cid][name].to(self.device).float()
#                         merged += client_param * mask
                
#                 merged_state[name] = merged
            
#             elif 'fc' in name and 'weight' in name:
#                 # 分类器权重：按类别拼接
#                 merged = global_state[name].clone().float()
#                 num_out = merged.shape[0]
                
#                 for cid, client in enumerate(self.clients):
#                     client_param = client_states[cid][name].to(self.device).float()
#                     for cls in client.learned_classes:
#                         if cls < num_out:
#                             merged[cls] = client_param[cls]
                
#                 merged_state[name] = merged
            
#             elif 'fc' in name and 'bias' in name:
#                 # 分类器偏置：按类别拼接
#                 merged = global_state[name].clone().float()
#                 num_out = merged.shape[0]
                
#                 for cid, client in enumerate(self.clients):
#                     client_param = client_states[cid][name].to(self.device).float()
#                     for cls in client.learned_classes:
#                         if cls < num_out:
#                             merged[cls] = client_param[cls]
                
#                 merged_state[name] = merged
            
#             elif 'running_mean' in name or 'running_var' in name:
#                 # BN的统计量：也按客户端拼接
#                 bn_weight_name = name.replace('running_mean', 'weight').replace('running_var', 'weight')
                
#                 merged = torch.zeros_like(global_state[name], dtype=torch.float32, device=self.device)
                
#                 for cid in range(self.client_num):
#                     mask = self._client_masks_gpu[cid].get(bn_weight_name)
#                     if mask is not None:
#                         client_param = client_states[cid][name].to(self.device).float()
#                         merged += client_param * mask
                
#                 merged_state[name] = merged
            
#             elif 'num_batches_tracked' in name:
#                 # BN追踪数：取最大值
#                 merged_state[name] = max(client_states[cid][name] for cid in range(self.client_num))
            
#             else:
#                 # 其他参数
#                 merged_state[name] = client_states[0][name]
        
#         self.global_model.load_state_dict(merged_state)
#         return self.global_model
 
#     def train_clients(self):
#         final_results = {} # 用于存储最后一次评估的结果——————————————————————
 
#         # ========== 计算开销预估 ==========
#         model_bytes = self._get_model_bytes()
#         forward_flops_per_sample = self._estimate_forward_flops()
#         train_flops_per_sample = 2 * forward_flops_per_sample  # 前向 + 反向
#         print(f"\n{'='*50}")
#         print(f"[Overhead] Model size: {self._format_bytes(model_bytes)}")
#         print(f"[Overhead] Forward FLOPs per sample: {self._format_flops(forward_flops_per_sample)}")
#         print(f"{'='*50}\n")
 
#         for i in range(self.task_num * 5):
#             print(f"\n{'='*60}")
#             print(f"Round {i}, Task {i // 5}")
#             print(f"{'='*60}")
 
#             # 分发全局模型
#             for client in self.clients:
#                 client.recive_global_models(self.global_model)
 
#             # 下行通信量
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # 客户端本地训练
#             client_models = []
#             round_flops = 0
#             for client in self.clients:
#                 client.train(i)
#                 client_models.append(client.model)
                
#                 # 保存数据加载器
#                 current_task = i // 5
#                 if current_task not in self.client_task_dataloaders[client.id]:
#                     self.client_task_dataloaders[client.id][current_task] = client.train_loader
 
#                 # PackNet FLOPs: 阶段1(epoch_local) + 阶段3(retrain_epochs)
#                 total_train_epochs = self.epoch_local + self.retrain_epochs
#                 client_flops = client.num_train_samples_this_round * total_train_epochs * train_flops_per_sample
#                 round_flops += client_flops
 
#             self.total_train_flops += round_flops
 
#             # 上行通信量
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # 合并模型（拼接，非聚合）
#             print("\nMerging client models (concatenation, not aggregation)...")
#             self.global_model = self.merge_client_models(client_models)
 
#             # 每个任务结束后评估
#             if (i + 1) % 5 == 0:
#                 current_task = i // 5
#                 print(f"\nTraining Task ID Predictors...")
#                 self._train_all_task_id_predictors(current_task)
#                 # self.model_global_eval(i)
#                 final_results = self.model_global_eval(i)#————————————————————————————
 
#         # 保存最终模型
#         final_path = os.path.join(self.save_path, f'{self.dataset}_packnet_final.pth')
#         torch.save(self.global_model.state_dict(), final_path)
#         print(f"\nSaved final model to {final_path}")
 
#         # ========== 打印开销统计 ==========
#         print(f"\n{'='*50}")
#         print(f"[Overhead Summary]")
#         print(f"  Total Training FLOPs:        {self._format_flops(self.total_train_flops)}")
#         print(f"  Total Communication Bytes:   {self._format_bytes(self.total_comm_bytes)}")
#         print(f"  Model Size (per transfer):   {self._format_bytes(model_bytes)}")
#         total_rounds = self.task_num * 5
#         print(f"  Total Rounds:                {total_rounds}")
#         print(f"  Comm per Round:              {self._format_bytes(2 * self.client_num * model_bytes)}")
#         print(f"{'='*50}\n")
 
#         print("Training complete!")
#         return final_results # 返回最终结果给 main.py————————————————————
 
#     def _train_all_task_id_predictors(self, current_task):
#         for client in self.clients:
#             dataloaders = {
#                 tid: self.client_task_dataloaders[client.id][tid]
#                 for tid in range(current_task + 1)
#                 if tid in self.client_task_dataloaders[client.id]
#             }
#             if dataloaders:
#                 self.task_id_predictors[client.id] = train_task_id_predictor(
#                     self.task_id_predictors[client.id],
#                     self.global_model, dataloaders, self.device,
#                     epochs=self.task_id_epochs, lr=0.001
#                 )
 
#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         acc_all = prediction(self.global_model, self.test_loader, self.device)
#         print(f'\n{self.dataset}: Round {taskid}, Overall accuracy: {acc_all:.2f}%')
 
#         task_index = taskid // 5
#         for t in range(task_index + 1):
#             all_classes = set()
#             for client_mask in self.mask:
#                 all_classes.update(client_mask[t])
#             acc_task = prediction_task(self.global_model, self.test_loader, self.device, list(all_classes))
#             print(f'{self.dataset}: Task {t} accuracy: {acc_task:.2f}%')
#             results[f'task_{t}'] = acc_task#——————————————————————————————
 
#         # 保存checkpoint
#         ckpt_path = os.path.join(self.save_path, f'{self.dataset}_packnet_task_{task_index}.pth')
#         torch.save(self.global_model.state_dict(), ckpt_path)
#         return results # 返回结果字典——————————————————————————————