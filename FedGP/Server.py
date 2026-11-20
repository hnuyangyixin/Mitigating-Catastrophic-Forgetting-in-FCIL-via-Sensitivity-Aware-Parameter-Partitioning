# import copy
# import random

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# from client import Client
# from utils import prediction, prediction_task
# import os

# from ResNet import resnet18_cbam  # 使用ResNet作为分类模型


# class Server(object):
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method, local_method,
#                 FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
#         self.client_num = client_num
#         self.task_num = task_num

#         self.feature_extractor = FE

#         self.data = subset
#         self.mask = mask
#         self.dataset = dataset
#         self.test_dataset = test_dataset

#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size

#         self.global_method = global_method
#         self.local_method = local_method

#         self.pseudo_samples = pseudo_samples
#         self.device = device
#         self.num_classes = class_num

#         self.clients = []
#         self.alpha = alpha
#         self.global_models = {}
#         self.encoder_layer_sizes = [512, 256]
#         self.laten_size = 128
#         self.decoder_layer_sizes = [256, 512]

#         self.test_loader = test_dataset
#         self.save_path = save_path

#         # 全局分类模型
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.clients = []

#         # 添加梯度约束相关变量
#         self.w_glob_er = None  # 全局梯度约束向量
#         self.grad_dims = None  # 梯度维度信息
#         self.buffer_size = 1000  # 梯度约束缓冲区大小

#     def init_clients(self):
#         print("Init clients...")
#         for i in range(self.client_num):
#             self.clients.append(Client(
#                 self.dataset, self.learning_rate, self.device,
#                 self.task_num, i, self.epoch_local, self.data[i],
#                 self.batch_size, self.mask[i], self.num_classes
#             ))
#         print("Complete")

#     def FedAvg(self, models):
#         """
#         标准FedAvg实现
        
#         Args:
#             models: 客户端模型状态字典的列表
            
#         Returns:
#             聚合后的全局模型
#         """
#         # 过滤掉None值
#         valid_models = [m for m in models if m is not None]
        
#         if len(valid_models) == 0:
#             print("Warning: No valid models to aggregate!")
#             return self.global_model
        
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             # 直接从状态字典中获取参数
#             param_list = [model[k] for model in valid_models]
#             # 确保参数的数据类型为浮点型
#             param_list = [param.float() for param in param_list]
#             # 堆叠参数并求平均值
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model
    
#     def FedSum(self, w):
#         """聚合梯度向量（FedSum）"""
#         # 过滤掉None值
#         valid_w = [grad for grad in w if grad is not None]
        
#         if not valid_w:
#             return None
        
#         w_avg = copy.deepcopy(valid_w[0])
#         for i in range(1, len(valid_w)):
#             w_avg = torch.add(w_avg, valid_w[i])
#         w_avg = torch.div(w_avg, len(valid_w))
#         return w_avg

#     def compute_global_grad_constraint(self):
#         """计算全局梯度约束向量（模拟实现）"""
#         # 实际实现应根据论文使用代表性数据计算
#         return torch.randn(np.sum(self.grad_dims)) * 0.1 if self.grad_dims else None

#     def train_clients(self):
#         all_eval_results = [] # 用于存储最后一次评估的结果——————————————————————
#         # 初始化梯度维度信息
#         if self.grad_dims is None:
#             dummy_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#             self.grad_dims = []
#             for param in dummy_model.parameters():
#                 self.grad_dims.append(param.data.numel())
        
#         # 确保保存路径存在
#         os.makedirs(self.save_path, exist_ok=True)
        
#         for i in range(self.task_num * 5):
#             current_task = i // 5
#             print(f"--------global round {i}, task number {current_task}-----------")
            
#             # 计算全局梯度约束
#             self.w_glob_er = self.compute_global_grad_constraint()
            
#             # 分发全局模型和梯度约束
#             for client in self.clients:
#                 client.recive_global_models(
#                     global_model=self.global_model,
#                     w_glob_er=self.w_glob_er,
#                     grad_dims=self.grad_dims
#                 )

#             # 客户端本地训练
#             client_models = []
#             client_grads = []  # 收集客户端梯度
#             for client in self.clients:
#                 model_state, client_grad = client.train(i)
#                 client_models.append(model_state)
#                 if client_grad is not None:
#                     client_grads.append(client_grad)

#             # 服务器聚合
#             print("使用FedAvg进行模型聚合")
#             self.global_model = self.FedAvg(client_models)
            
#             # 聚合梯度约束
#             if client_grads:
#                 print("使用FedSum进行梯度聚合")
#                 self.w_glob_er = self.FedSum(client_grads)

#             # 评估全局模型 - 每完成一个任务（5轮）进行评估
#             if (i + 1) % 5 == 0:
#                 # self.model_global_eval(i)
#                 all_eval_results.append(self.model_global_eval(i))#————————————————————————————

#         # 训练结束后保存最终模型
#         final_model_path = os.path.join(self.save_path, 'final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"Saved final global model at {final_model_path}")
        
#         print("All process complete")
#         return all_eval_results # 返回最终结果给 main.py————————————————————

#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         """
#         全局模型评估
        
#         Args:
#             taskid: 全局轮次ID (0-24)
#         """
#         # 整体测试集准确率 - 使用完整测试集
#         acc_all = prediction(
#             self.global_model,
#             self.test_loader,
#             self.device
#         )
#         print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        

#         # 对每个任务进行评估
#         task_index = taskid // 5  # 计算当前完成的任务索引
#         for t in range(task_index + 1):
#             # 获取任务t的类别列表
#             all_task_classes = set()
#             for client_mask in self.mask:
#                 all_task_classes.update(client_mask[t])
#             task_classes = list(all_task_classes)
#             acc_task = self.evaluate_task(t, task_classes)
#             print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
#             results[f'task_{t}'] = acc_task#——————————————————————————————

#         # 保存当前任务结束后的全局模型
#         model_path = os.path.join(self.save_path, f'{self.dataset}_global_model_task_{task_index}.pth')
#         torch.save(self.global_model.state_dict(), model_path)
#         print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
#         return results # 返回结果字典——————————————————————————————

#     def evaluate_task(self, task_id, task_classes):
#         """评估模型在特定任务上的表现 - 使用完整标签空间"""
#         return prediction_task(
#             self.global_model,
#             self.test_loader,  # 使用完整测试集
#             self.device,
#             task_classes
#         )



# 3 turns

import copy
import random
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
 
from client import Client
from utils import prediction, prediction_task
import os
 
from ResNet import resnet18_cbam  # 使用ResNet作为分类模型
 
 
class Server(object):
    def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method, local_method,
                FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
        self.client_num = client_num
        self.task_num = task_num
 
        self.feature_extractor = FE
 
        self.data = subset
        self.mask = mask
        self.dataset = dataset
        self.test_dataset = test_dataset
 
        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size
 
        self.global_method = global_method
        self.local_method = local_method
 
        self.pseudo_samples = pseudo_samples
        self.device = device
        self.num_classes = class_num
 
        self.clients = []
        self.alpha = alpha
        self.global_models = {}
        self.encoder_layer_sizes = [512, 256]
        self.laten_size = 128
        self.decoder_layer_sizes = [256, 512]
 
        self.test_loader = test_dataset
        self.save_path = save_path
 
        # 全局分类模型
        self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.clients = []
 
        # 添加梯度约束相关变量
        self.w_glob_er = None  # 全局梯度约束向量
        self.grad_dims = None  # 梯度维度信息
        self.buffer_size = 1000  # 梯度约束缓冲区大小
 
    def init_clients(self):
        print("Init clients...")
        for i in range(self.client_num):
            self.clients.append(Client(
                self.dataset, self.learning_rate, self.device,
                self.task_num, i, self.epoch_local, self.data[i],
                self.batch_size, self.mask[i], self.num_classes
            ))
        print("Complete")
 
    def FedAvg(self, models):
        """
        标准FedAvg实现
        
        Args:
            models: 客户端模型状态字典的列表
            
        Returns:
            聚合后的全局模型
        """
        # 过滤掉None值
        valid_models = [m for m in models if m is not None]
        
        if len(valid_models) == 0:
            print("Warning: No valid models to aggregate!")
            return self.global_model
        
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            # 直接从状态字典中获取参数
            param_list = [model[k] for model in valid_models]
            # 确保参数的数据类型为浮点型
            param_list = [param.float() for param in param_list]
            # 堆叠参数并求平均值
            global_dict[k] = torch.stack(param_list, 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model
    
    def FedSum(self, w):
        """聚合梯度向量（FedSum）"""
        # 过滤掉None值
        valid_w = [grad for grad in w if grad is not None]
        
        if not valid_w:
            return None
        
        w_avg = copy.deepcopy(valid_w[0])
        for i in range(1, len(valid_w)):
            w_avg = torch.add(w_avg, valid_w[i])
        w_avg = torch.div(w_avg, len(valid_w))
        return w_avg
 
    def compute_global_grad_constraint(self):
        """计算全局梯度约束向量（模拟实现）"""
        # 实际实现应根据论文使用代表性数据计算
        return torch.randn(np.sum(self.grad_dims)) * 0.1 if self.grad_dims else None
 
    def train_clients(self):
        all_eval_results = [] # 用于存储每次评估的结果——————————————————————
        # 初始化梯度维度信息
        if self.grad_dims is None:
            dummy_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
            self.grad_dims = []
            for param in dummy_model.parameters():
                self.grad_dims.append(param.data.numel())
        
        # 确保保存路径存在
        os.makedirs(self.save_path, exist_ok=True)
        
        for i in range(self.task_num * 5):
            current_task = i // 5
            print(f"--------global round {i}, task number {current_task}-----------")
            
            # 计算全局梯度约束
            self.w_glob_er = self.compute_global_grad_constraint()
            
            # 分发全局模型和梯度约束
            for client in self.clients:
                client.recive_global_models(
                    global_model=self.global_model,
                    w_glob_er=self.w_glob_er,
                    grad_dims=self.grad_dims
                )
 
            # 客户端本地训练
            client_models = []
            client_grads = []  # 收集客户端梯度
            for client in self.clients:
                model_state, client_grad = client.train(i)
                client_models.append(model_state)
                if client_grad is not None:
                    client_grads.append(client_grad)
 
            # 服务器聚合
            print("使用FedAvg进行模型聚合")
            self.global_model = self.FedAvg(client_models)
            
            # 聚合梯度约束
            if client_grads:
                print("使用FedSum进行梯度聚合")
                self.w_glob_er = self.FedSum(client_grads)
 
            # 评估全局模型 - 每完成一个任务（5轮）进行评估
            if (i + 1) % 5 == 0:
                # self.model_global_eval(i)
                final_results = self.model_global_eval(i)#————————————————————————————
                all_eval_results.append(final_results)#————————————————————————————
 
        # 训练结束后保存最终模型
        final_model_path = os.path.join(self.save_path, 'final_global_model.pth')
        torch.save(self.global_model.state_dict(), final_model_path)
        print(f"Saved final global model at {final_model_path}")
        
        print("All process complete")
        return all_eval_results # 返回所有评估结果给 main.py————————————————————
 
    def model_global_eval(self, taskid):
        results = {} # 创建字典存储结果——————————————————————————————————
        results['round'] = taskid # 记录轮次——————————————————————————————
        """
        全局模型评估
        
        Args:
            taskid: 全局轮次ID (0-24)
        """
        # 整体测试集准确率 - 使用完整测试集
        acc_all = prediction(
            self.global_model,
            self.test_loader,
            self.device
        )
        print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        results['all_tasks'] = acc_all # 存储总精度——————————————————————————
        
 
        # 对每个任务进行评估
        task_index = taskid // 5  # 计算当前完成的任务索引
        for t in range(task_index + 1):
            # 获取任务t的类别列表
            all_task_classes = set()
            for client_mask in self.mask:
                all_task_classes.update(client_mask[t])
            task_classes = list(all_task_classes)
            acc_task = self.evaluate_task(t, task_classes)
            print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
            results[f'task_{t}'] = acc_task#——————————————————————————————
 
        # 保存当前任务结束后的全局模型
        model_path = os.path.join(self.save_path, f'{self.dataset}_global_model_task_{task_index}.pth')
        torch.save(self.global_model.state_dict(), model_path)
        print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
        return results # 返回结果字典——————————————————————————————
 
    def evaluate_task(self, task_id, task_classes):
        """评估模型在特定任务上的表现 - 使用完整标签空间"""
        return prediction_task(
            self.global_model,
            self.test_loader,  # 使用完整测试集
            self.device,
            task_classes
        )


# # 计算通信开销

# import copy
# import random
 
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
 
# from client import Client
# from utils import prediction, prediction_task
# import os
 
# from ResNet import resnet18_cbam  # 使用ResNet作为分类模型
 
 
# class Server(object):
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method, local_method,
#                 FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
#         self.client_num = client_num
#         self.task_num = task_num
 
#         self.feature_extractor = FE
 
#         self.data = subset
#         self.mask = mask
#         self.dataset = dataset
#         self.test_dataset = test_dataset
 
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
 
#         self.global_method = global_method
#         self.local_method = local_method
 
#         self.pseudo_samples = pseudo_samples
#         self.device = device
#         self.num_classes = class_num
 
#         self.clients = []
#         self.alpha = alpha
#         self.global_models = {}
#         self.encoder_layer_sizes = [512, 256]
#         self.laten_size = 128
#         self.decoder_layer_sizes = [256, 512]
 
#         self.test_loader = test_dataset
#         self.save_path = save_path
 
#         # 全局分类模型
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.clients = []
 
#         # 添加梯度约束相关变量
#         self.w_glob_er = None  # 全局梯度约束向量
#         self.grad_dims = None  # 梯度维度信息
#         self.buffer_size = 1000  # 梯度约束缓冲区大小
 
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
 
#     def _get_tensor_bytes(self, tensor):
#         """计算单个tensor的字节数"""
#         if tensor is None:
#             return 0
#         return tensor.nelement() * tensor.element_size()
 
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
#         """将字节数格式化为可读字符串"""
#         if num_bytes >= 1e9:
#             return f"{num_bytes / 1e9:.2f} GB"
#         elif num_bytes >= 1e6:
#             return f"{num_bytes / 1e6:.2f} MB"
#         elif num_bytes >= 1e3:
#             return f"{num_bytes / 1e3:.2f} KB"
#         return f"{num_bytes} B"
 
#     def _format_flops(self, flops):
#         """将FLOPs格式化为可读字符串"""
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
#         print("Init clients...")
#         for i in range(self.client_num):
#             self.clients.append(Client(
#                 self.dataset, self.learning_rate, self.device,
#                 self.task_num, i, self.epoch_local, self.data[i],
#                 self.batch_size, self.mask[i], self.num_classes
#             ))
#         print("Complete")
 
#     def FedAvg(self, models):
#         """
#         标准FedAvg实现
        
#         Args:
#             models: 客户端模型状态字典的列表
            
#         Returns:
#             聚合后的全局模型
#         """
#         # 过滤掉None值
#         valid_models = [m for m in models if m is not None]
        
#         if len(valid_models) == 0:
#             print("Warning: No valid models to aggregate!")
#             return self.global_model
        
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             # 直接从状态字典中获取参数
#             param_list = [model[k] for model in valid_models]
#             # 确保参数的数据类型为浮点型
#             param_list = [param.float() for param in param_list]
#             # 堆叠参数并求平均值
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model
    
#     def FedSum(self, w):
#         """聚合梯度向量（FedSum）"""
#         # 过滤掉None值
#         valid_w = [grad for grad in w if grad is not None]
        
#         if not valid_w:
#             return None
        
#         w_avg = copy.deepcopy(valid_w[0])
#         for i in range(1, len(valid_w)):
#             w_avg = torch.add(w_avg, valid_w[i])
#         w_avg = torch.div(w_avg, len(valid_w))
#         return w_avg
 
#     def compute_global_grad_constraint(self):
#         """计算全局梯度约束向量（模拟实现）"""
#         # 实际实现应根据论文使用代表性数据计算
#         return torch.randn(np.sum(self.grad_dims)) * 0.1 if self.grad_dims else None
 
#     def train_clients(self):
#         final_results = {} # 用于存储最后一次评估的结果——————————————————————
#         # 初始化梯度维度信息
#         if self.grad_dims is None:
#             dummy_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#             self.grad_dims = []
#             for param in dummy_model.parameters():
#                 self.grad_dims.append(param.data.numel())
        
#         # 确保保存路径存在
#         os.makedirs(self.save_path, exist_ok=True)
 
#         # ========== 计算开销预估 ==========
#         model_bytes = self._get_model_bytes()
#         forward_flops_per_sample = self._estimate_forward_flops()
#         train_flops_per_sample = 2 * forward_flops_per_sample  # 前向 + 反向
#         print(f"\n{'='*50}")
#         print(f"[Overhead] Model size: {self._format_bytes(model_bytes)}")
#         print(f"[Overhead] Forward FLOPs per sample: {self._format_flops(forward_flops_per_sample)}")
#         print(f"{'='*50}\n")
        
#         for i in range(self.task_num * 5):
#             current_task = i // 5
#             print(f"--------global round {i}, task number {current_task}-----------")
            
#             # 计算全局梯度约束
#             self.w_glob_er = self.compute_global_grad_constraint()
            
#             # 分发全局模型和梯度约束
#             for client in self.clients:
#                 client.recive_global_models(
#                     global_model=self.global_model,
#                     w_glob_er=self.w_glob_er,
#                     grad_dims=self.grad_dims
#                 )
 
#             # 下行通信量：server -> 每个client
#             # 1) 全局模型
#             self.total_comm_bytes += self.client_num * model_bytes
#             # 2) w_glob_er 梯度约束向量
#             if self.w_glob_er is not None:
#                 self.total_comm_bytes += self.client_num * self._get_tensor_bytes(self.w_glob_er)
 
#             # 客户端本地训练
#             client_models = []
#             client_grads = []  # 收集客户端梯度
#             round_flops = 0
#             for client in self.clients:
#                 model_state, client_grad = client.train(i)
#                 client_models.append(model_state)
#                 if client_grad is not None:
#                     client_grads.append(client_grad)
#                 # 累计本轮FLOPs
#                 client_flops = client.num_train_samples_this_round * self.epoch_local * train_flops_per_sample
#                 round_flops += client_flops
 
#             self.total_train_flops += round_flops
 
#             # 上行通信量：每个client -> server
#             # 1) 模型状态字典
#             self.total_comm_bytes += self.client_num * model_bytes
#             # 2) 客户端梯度（如果有）
#             for grad in client_grads:
#                 self.total_comm_bytes += self._get_tensor_bytes(grad)
 
#             # 服务器聚合
#             print("使用FedAvg进行模型聚合")
#             self.global_model = self.FedAvg(client_models)
            
#             # 聚合梯度约束
#             if client_grads:
#                 print("使用FedSum进行梯度聚合")
#                 self.w_glob_er = self.FedSum(client_grads)
 
#             # 评估全局模型 - 每完成一个任务（5轮）进行评估
#             if (i + 1) % 5 == 0:
#                 # self.model_global_eval(i)
#                 final_results = self.model_global_eval(i)#————————————————————————————
 
#         # 训练结束后保存最终模型
#         final_model_path = os.path.join(self.save_path, 'final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"Saved final global model at {final_model_path}")
 
#         # ========== 打印开销统计 ==========
#         print(f"\n{'='*50}")
#         print(f"[Overhead Summary]")
#         print(f"  Total Training FLOPs:        {self._format_flops(self.total_train_flops)}")
#         print(f"  Total Communication Bytes:   {self._format_bytes(self.total_comm_bytes)}")
#         print(f"  Model Size (per transfer):   {self._format_bytes(model_bytes)}")
#         total_rounds = self.task_num * 5
#         print(f"  Total Rounds:                {total_rounds}")
#         print(f"{'='*50}\n")
        
#         print("All process complete")
#         return final_results # 返回最终结果给 main.py————————————————————
 
#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         """
#         全局模型评估
        
#         Args:
#             taskid: 全局轮次ID (0-24)
#         """
#         # 整体测试集准确率 - 使用完整测试集
#         acc_all = prediction(
#             self.global_model,
#             self.test_loader,
#             self.device
#         )
#         print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        
 
#         # 对每个任务进行评估
#         task_index = taskid // 5  # 计算当前完成的任务索引
#         for t in range(task_index + 1):
#             # 获取任务t的类别列表
#             all_task_classes = set()
#             for client_mask in self.mask:
#                 all_task_classes.update(client_mask[t])
#             task_classes = list(all_task_classes)
#             acc_task = self.evaluate_task(t, task_classes)
#             print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
#             results[f'task_{t}'] = acc_task#——————————————————————————————
 
#         # 保存当前任务结束后的全局模型
#         model_path = os.path.join(self.save_path, f'{self.dataset}_global_model_task_{task_index}.pth')
#         torch.save(self.global_model.state_dict(), model_path)
#         print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
#         return results # 返回结果字典——————————————————————————————
 
#     def evaluate_task(self, task_id, task_classes):
#         """评估模型在特定任务上的表现 - 使用完整标签空间"""
#         return prediction_task(
#             self.global_model,
#             self.test_loader,  # 使用完整测试集
#             self.device,
#             task_classes
#         )