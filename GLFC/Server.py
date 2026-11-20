# import copy
# import random
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from client import Client
# from ResNet import resnet18_cbam
# from utils import prediction, prediction_task
# from ProxyServer import proxyServer

# class Server(object):
    
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path,
#                  memory_size, encode_model, args):
#         self.client_num = client_num
#         self.task_num = task_num
#         self.feature_extractor = FE
#         self.data = subset
#         self.mask = mask
#         self.dataset = dataset
#         self.test_loader = test_dataset
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.num_classes = class_num
#         self.save_path = save_path
        
#         # GLFC specific
#         self.memory_size = memory_size
#         self.encode_model = encode_model
#         self.args = args
#         self.clients = []
        
#         # Initialize Global Model
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        
#         # Initialize Proxy Server
#         self.proxy_server = proxyServer(
#             device=self.device,
#             learning_rate=self.learning_rate,
#             numclass=self.num_classes,
#             feature_extractor=None, # ResNet has integrated FE
#             encode_model=self.encode_model,
#             test_transform=None # Can add transform if needed
#         )

#         self.init_clients()
#         os.makedirs(self.save_path, exist_ok=True)

#     def init_clients(self):
#         print(f"{self.dataset}: Init clients...")
#         for i in range(self.client_num):
#             self.clients.append(Client(
#                 self.dataset, self.learning_rate, self.device,
#                 self.task_num, i, self.epoch_local, self.data[i],
#                 self.batch_size, self.mask[i], self.num_classes,
#                 self.memory_size, self.encode_model
#             ))
#         print(f"{self.dataset}: Complete")

#     def FedAvg(self, models):
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             param_list = [model.state_dict()[k].float() for model in models]
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model

#     def train_clients(self):
#         all_eval_results = [] # 用于存储最后一次评估的结果——————————————————————
#         old_client_0 = []
        
#         for task_idx in range(self.task_num):
#             print(f"\n{self.dataset}: -------- Global Task {task_idx} -----------")
            
#             # 模拟GLFC中的global round循环 (这里简化为每个task训练几轮)
#             # 在原始GLFC中，global round是外部循环，task是内部状态。
#             # 这里适配你的代码结构：外层循环控制任务。
#             # 我们假设每个任务进行多次联邦聚合（例如5轮）
            
#             rounds_per_task = 5 # 可以根据需要调整
            
#             for round_idx in range(rounds_per_task):
#                 print(f"{self.dataset}:   Round {round_idx} in Task {task_idx}")
                
#                 # 1. 下发模型
#                 for client in self.clients:
#                     client.recive_global_models(self.global_model)
                
#                 # 2. 收集客户端梯度和模型
#                 client_models = []
#                 pool_grad = []
                
#                 # 随机选择客户端 (全选或部分)
#                 # clients_index = random.sample(range(self.client_num), int(self.client_num * 0.5))
#                 clients_index = range(self.client_num) 

#                 for i in clients_index:
#                     client = self.clients[i]
                    
#                     # 客户端训练
#                     # 注意：GLFC中client需要知道自己是不是“旧”客户端(old_client_0)
#                     # 你的mask逻辑已经隐含了类别分配，这里简化传递
#                     # 0: old classes only (not strictly used here due to mask logic), 1: new classes
#                     # 你的setup是每个client在每个task有新数据
                    
#                     local_model_state, proto_grad = client.train_glfc(task_idx, self.global_model)
#                     client_models.append(client.model)
                    
#                     if proto_grad is not None:
#                         pool_grad.extend(proto_grad)

#                 # 3. 联邦聚合
#                 self.global_model = self.FedAvg(client_models)
#                 self.proxy_server.model = copy.deepcopy(self.global_model)

#                 # 4. Proxy Server处理 (重构数据)
#                 if len(pool_grad) > 0:
#                     print(f"{self.dataset}:     Proxy Server: Reconstructing with {len(pool_grad)} gradients...")
#                     self.proxy_server.dataloader(pool_grad)
#                     # GLFC通常用重构数据来monitor或select best model，这里我们让它更新monitor loader即可
            
#             # 任务结束，更新样本库
#             print(f"{self.dataset}:   Updating exemplar sets for all clients...")
#             for client in self.clients:
#                 client.update_new_set(task_idx)

#             final_round_of_task = (task_idx + 1) * rounds_per_task - 1
#             # 评估
#             # self.model_global_eval(task_idx)
#             all_eval_results.append(self.model_global_eval(i))#————————————————————————————

#         final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
#         return all_eval_results # 返回最终结果给 main.py————————————————————

#     def model_global_eval(self, global_round):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         self.global_model.eval()
        
#         # 1. 计算所有任务的总准确率
#         acc_all = prediction(self.global_model, self.test_loader, self.device)
#         print(f'{self.dataset}: Round {global_round} finishes, global model accuracy (all tasks): {acc_all:.2f}%')

#         # 2. 逐个计算每个 Task 的准确率
#         # 即使任务还没开始训练，也可以评估（准确率应该很低），这样可以看到从0增长的过程
#         for t in range(self.task_num):
#             # 收集该任务包含的所有类别
#             task_classes = set()
#             for i in range(self.client_num):
#                 # self.mask[i] 是该客户端的任务列表
#                 if t < len(self.mask[i]):
#                     task_classes.update(self.mask[i][t])
            
#             task_classes_list = list(task_classes)
            
#             if len(task_classes_list) > 0:
#                 acc_task = prediction_task(self.global_model, self.test_loader, self.device, task_classes_list)
#                 print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
#                 results[f'task_{t}'] = acc_task#——————————————————————————————
#             else:
#                 print(f'{self.dataset}: Global model on task {t} test accuracy: N/A')

#         self.global_model.train()
#         return results # 返回结果字典——————————————————————————————




# 3 turns

import copy
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from client import Client
from ResNet import resnet18_cbam
from utils import prediction, prediction_task
from ProxyServer import proxyServer

class Server(object):
    
    def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
                 local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path,
                 memory_size, encode_model, args):
        self.client_num = client_num
        self.task_num = task_num
        self.feature_extractor = FE
        self.data = subset
        self.mask = mask
        self.dataset = dataset
        self.test_loader = test_dataset
        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.device = device
        self.num_classes = class_num
        self.save_path = save_path
        
        # GLFC specific
        self.memory_size = memory_size
        self.encode_model = encode_model
        self.args = args
        self.clients = []
        
        # Initialize Global Model
        self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        
        # Initialize Proxy Server
        self.proxy_server = proxyServer(
            device=self.device,
            learning_rate=self.learning_rate,
            numclass=self.num_classes,
            feature_extractor=None, # ResNet has integrated FE
            encode_model=self.encode_model,
            test_transform=None # Can add transform if needed
        )

        self.init_clients()
        os.makedirs(self.save_path, exist_ok=True)

    def init_clients(self):
        print(f"{self.dataset}: Init clients...")
        for i in range(self.client_num):
            self.clients.append(Client(
                self.dataset, self.learning_rate, self.device,
                self.task_num, i, self.epoch_local, self.data[i],
                self.batch_size, self.mask[i], self.num_classes,
                self.memory_size, self.encode_model
            ))
        print(f"{self.dataset}: Complete")

    def FedAvg(self, models):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            param_list = [model.state_dict()[k].float() for model in models]
            global_dict[k] = torch.stack(param_list, 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model

    def train_clients(self):
        all_eval_results = [] # 用于存储每次评估的结果——————————————————————
        old_client_0 = []
        
        for task_idx in range(self.task_num):
            print(f"\n{self.dataset}: -------- Global Task {task_idx} -----------")
            
            # 模拟GLFC中的global round循环 (这里简化为每个task训练几轮)
            # 在原始GLFC中，global round是外部循环，task是内部状态。
            # 这里适配你的代码结构：外层循环控制任务。
            # 我们假设每个任务进行多次联邦聚合（例如5轮）
            
            rounds_per_task = 5 # 可以根据需要调整
            
            for round_idx in range(rounds_per_task):
                print(f"{self.dataset}:   Round {round_idx} in Task {task_idx}")
                
                # 1. 下发模型
                for client in self.clients:
                    client.recive_global_models(self.global_model)
                
                # 2. 收集客户端梯度和模型
                client_models = []
                pool_grad = []
                
                # 随机选择客户端 (全选或部分)
                # clients_index = random.sample(range(self.client_num), int(self.client_num * 0.5))
                clients_index = range(self.client_num) 

                for i in clients_index:
                    client = self.clients[i]
                    
                    # 客户端训练
                    # 注意：GLFC中client需要知道自己是不是“旧”客户端(old_client_0)
                    # 你的mask逻辑已经隐含了类别分配，这里简化传递
                    # 0: old classes only (not strictly used here due to mask logic), 1: new classes
                    # 你的setup是每个client在每个task有新数据
                    
                    local_model_state, proto_grad = client.train_glfc(task_idx, self.global_model)
                    client_models.append(client.model)
                    
                    if proto_grad is not None:
                        pool_grad.extend(proto_grad)

                # 3. 联邦聚合
                self.global_model = self.FedAvg(client_models)
                self.proxy_server.model = copy.deepcopy(self.global_model)

                # 4. Proxy Server处理 (重构数据)
                if len(pool_grad) > 0:
                    print(f"{self.dataset}:     Proxy Server: Reconstructing with {len(pool_grad)} gradients...")
                    self.proxy_server.dataloader(pool_grad)
                    # GLFC通常用重构数据来monitor或select best model，这里我们让它更新monitor loader即可
            
            # 任务结束，更新样本库
            print(f"{self.dataset}:   Updating exemplar sets for all clients...")
            for client in self.clients:
                client.update_new_set(task_idx)

            final_round_of_task = (task_idx + 1) * rounds_per_task - 1
            # 评估
            # self.model_global_eval(task_idx)
            final_results = self.model_global_eval(i)#————————————————————————————
            all_eval_results.append(final_results)#————————————————————————————

        final_model_path = os.path.join(self.save_path, f'{self.dataset}_final_global_model.pth')
        torch.save(self.global_model.state_dict(), final_model_path)
        return all_eval_results # 返回所有评估结果给 main.py————————————————————

    def model_global_eval(self, global_round):
        results = {} # 创建字典存储结果——————————————————————————————————
        results['round'] = taskid
        results['round'] = global_round # 记录轮次——————————————————————————————
        self.global_model.eval()
        
        # 1. 计算所有任务的总准确率
        acc_all = prediction(self.global_model, self.test_loader, self.device)
        print(f'{self.dataset}: Round {global_round} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        results['all_tasks'] = acc_all # 存储总精度——————————————————————————

        # 2. 逐个计算每个 Task 的准确率
        # 即使任务还没开始训练，也可以评估（准确率应该很低），这样可以看到从0增长的过程
        for t in range(self.task_num):
            # 收集该任务包含的所有类别
            task_classes = set()
            for i in range(self.client_num):
                # self.mask[i] 是该客户端的任务列表
                if t < len(self.mask[i]):
                    task_classes.update(self.mask[i][t])
            
            task_classes_list = list(task_classes)
            
            if len(task_classes_list) > 0:
                acc_task = prediction_task(self.global_model, self.test_loader, self.device, task_classes_list)
                print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
                results[f'task_{t}'] = acc_task#——————————————————————————————
            else:
                print(f'{self.dataset}: Global model on task {t} test accuracy: N/A')

        self.global_model.train()
        return results # 返回结果字典——————————————————————————————




# # 计算通信开销

# import copy
# import random
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from client import Client
# from ResNet import resnet18_cbam
# from utils import prediction, prediction_task
# from ProxyServer import proxyServer
 
# class Server(object):
    
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path,
#                  memory_size, encode_model, args):
#         self.client_num = client_num
#         self.task_num = task_num
#         self.feature_extractor = FE
#         self.data = subset
#         self.mask = mask
#         self.dataset = dataset
#         self.test_loader = test_dataset
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.num_classes = class_num
#         self.save_path = save_path
        
#         # GLFC specific
#         self.memory_size = memory_size
#         self.encode_model = encode_model
#         self.args = args
#         self.clients = []
        
#         # Initialize Global Model
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        
#         # Initialize Proxy Server
#         self.proxy_server = proxyServer(
#             device=self.device,
#             learning_rate=self.learning_rate,
#             numclass=self.num_classes,
#             feature_extractor=None, # ResNet has integrated FE
#             encode_model=self.encode_model,
#             test_transform=None # Can add transform if needed
#         )
 
#         self.init_clients()
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
 
#     def _get_proto_grad_bytes(self, pool_grad):
#         """计算proto_grad列表的总字节数"""
#         total_bytes = 0
#         for grad_list in pool_grad:
#             # grad_list 是一个 list of tensors (每个参数的梯度)
#             if isinstance(grad_list, (list, tuple)):
#                 for g in grad_list:
#                     if isinstance(g, torch.Tensor):
#                         total_bytes += g.nelement() * g.element_size()
#             elif isinstance(grad_list, torch.Tensor):
#                 total_bytes += grad_list.nelement() * grad_list.element_size()
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
#         print(f"{self.dataset}: Init clients...")
#         for i in range(self.client_num):
#             self.clients.append(Client(
#                 self.dataset, self.learning_rate, self.device,
#                 self.task_num, i, self.epoch_local, self.data[i],
#                 self.batch_size, self.mask[i], self.num_classes,
#                 self.memory_size, self.encode_model
#             ))
#         print(f"{self.dataset}: Complete")
 
#     def FedAvg(self, models):
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             param_list = [model.state_dict()[k].float() for model in models]
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model
 
#     def train_clients(self):
#         final_results = {} # 用于存储最后一次评估的结果——————————————————————
#         old_client_0 = []
 
#         # ========== 计算开销预估 ==========
#         model_bytes = self._get_model_bytes()
#         forward_flops_per_sample = self._estimate_forward_flops()
#         train_flops_per_sample = 2 * forward_flops_per_sample  # 前向 + 反向
#         print(f"\n{'='*50}")
#         print(f"[Overhead] Model size: {self._format_bytes(model_bytes)}")
#         print(f"[Overhead] Forward FLOPs per sample: {self._format_flops(forward_flops_per_sample)}")
#         print(f"{'='*50}\n")
        
#         for task_idx in range(self.task_num):
#             print(f"\n{self.dataset}: -------- Global Task {task_idx} -----------")
            
#             # 模拟GLFC中的global round循环 (这里简化为每个task训练几轮)
#             # 在原始GLFC中，global round是外部循环，task是内部状态。
#             # 这里适配你的代码结构：外层循环控制任务。
#             # 我们假设每个任务进行多次联邦聚合（例如5轮）
            
#             rounds_per_task = 5 # 可以根据需要调整
            
#             for round_idx in range(rounds_per_task):
#                 print(f"{self.dataset}:   Round {round_idx} in Task {task_idx}")
                
#                 # 1. 下发模型
#                 for client in self.clients:
#                     client.recive_global_models(self.global_model)
 
#                 # 下行通信量：全局模型
#                 self.total_comm_bytes += self.client_num * model_bytes
                
#                 # 2. 收集客户端梯度和模型
#                 client_models = []
#                 pool_grad = []
                
#                 # 随机选择客户端 (全选或部分)
#                 # clients_index = random.sample(range(self.client_num), int(self.client_num * 0.5))
#                 clients_index = range(self.client_num) 
 
#                 round_flops = 0
#                 for i in clients_index:
#                     client = self.clients[i]
                    
#                     # 客户端训练
#                     # 注意：GLFC中client需要知道自己是不是“旧”客户端(old_client_0)
#                     # 你的mask逻辑已经隐含了类别分配，这里简化传递
#                     # 0: old classes only (not strictly used here due to mask logic), 1: new classes
#                     # 你的setup是每个client在每个task有新数据
                    
#                     local_model_state, proto_grad = client.train_glfc(task_idx, self.global_model)
#                     client_models.append(client.model)
                    
#                     if proto_grad is not None:
#                         pool_grad.extend(proto_grad)
 
#                     # 累计本轮FLOPs
#                     client_flops = client.num_train_samples_this_round * self.epoch_local * train_flops_per_sample
#                     round_flops += client_flops
 
#                 self.total_train_flops += round_flops
 
#                 # 上行通信量：模型 + proto_grad
#                 self.total_comm_bytes += len(clients_index) * model_bytes
#                 if len(pool_grad) > 0:
#                     self.total_comm_bytes += self._get_proto_grad_bytes(pool_grad)
 
#                 # 3. 联邦聚合
#                 self.global_model = self.FedAvg(client_models)
#                 self.proxy_server.model = copy.deepcopy(self.global_model)
 
#                 # 4. Proxy Server处理 (重构数据)
#                 if len(pool_grad) > 0:
#                     print(f"{self.dataset}:     Proxy Server: Reconstructing with {len(pool_grad)} gradients...")
#                     self.proxy_server.dataloader(pool_grad)
#                     # GLFC通常用重构数据来monitor或select best model，这里我们让它更新monitor loader即可
            
#             # 任务结束，更新样本库
#             print(f"{self.dataset}:   Updating exemplar sets for all clients...")
#             for client in self.clients:
#                 client.update_new_set(task_idx)
 
#             final_round_of_task = (task_idx + 1) * rounds_per_task - 1
#             # 评估
#             # self.model_global_eval(task_idx)
#             final_results = self.model_global_eval(i)#————————————————————————————
 
#         final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
 
#         # ========== 打印开销统计 ==========
#         print(f"\n{'='*50}")
#         print(f"[Overhead Summary]")
#         print(f"  Total Training FLOPs:        {self._format_flops(self.total_train_flops)}")
#         print(f"  Total Communication Bytes:   {self._format_bytes(self.total_comm_bytes)}")
#         print(f"  Model Size (per transfer):   {self._format_bytes(model_bytes)}")
#         total_rounds = self.task_num * 5
#         print(f"  Total Rounds:                {total_rounds}")
#         print(f"{'='*50}\n")
 
#         return final_results # 返回最终结果给 main.py————————————————————
 
#     def model_global_eval(self, global_round):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         self.global_model.eval()
        
#         # 1. 计算所有任务的总准确率
#         acc_all = prediction(self.global_model, self.test_loader, self.device)
#         print(f'{self.dataset}: Round {global_round} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
 
#         # 2. 逐个计算每个 Task 的准确率
#         # 即使任务还没开始训练，也可以评估（准确率应该很低），这样可以看到从0增长的过程
#         for t in range(self.task_num):
#             # 收集该任务包含的所有类别
#             task_classes = set()
#             for i in range(self.client_num):
#                 # self.mask[i] 是该客户端的任务列表
#                 if t < len(self.mask[i]):
#                     task_classes.update(self.mask[i][t])
            
#             task_classes_list = list(task_classes)
            
#             if len(task_classes_list) > 0:
#                 acc_task = prediction_task(self.global_model, self.test_loader, self.device, task_classes_list)
#                 print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
#                 results[f'task_{t}'] = acc_task#——————————————————————————————
#             else:
#                 print(f'{self.dataset}: Global model on task {t} test accuracy: N/A')
 
#         self.global_model.train()
#         return results # 返回结果字典——————————————————————————————