import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from client import Client
# from models.VAE_PRETRAINED import VAE
from utils import prediction, prediction_task
import os

from ResNet import resnet18_cbam  # 使用ResNet作为分类模型


class Server(object):
    # def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, batch_size, test_loader, num_classes,
    #              client_mask):
    #     self.client_num = client_num
    #     self.task_num = task_num
    #     self.device = device
    #     self.test_loader = test_loader
    #     self.num_classes = num_classes  # 总类别数
    #     self.client_mask = client_mask

    #     # 全局分类模型
    #     self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
    #     self.clients = []
    def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
                 local_method,
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
        # 标准FedAvg实现
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            # 获取所有客户端模型对应参数
            param_list = [model.state_dict()[k] for model in models]
            # 确保参数的数据类型为浮点型
            param_list = [param.float() for param in param_list]
            # 堆叠参数并求平均值
            global_dict[k] = torch.stack(param_list, 0).mean(0)
            # global_dict[k] = torch.stack([model.state_dict()[k] for model in models], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model

    def train_clients(self):
        all_eval_results = [] # 用于存储最后一次评估的结果——————————————————————
        for i in range(self.task_num * 5):
            print(f"--------global round {i},task number {i // 5}-----------")

            # 分发全局模型
            for client in self.clients:
                client.recive_global_models(self.global_model)

            # 客户端本地训练
            client_models = []
            for client in self.clients:
                client.train(i)
                client_models.append(client.model)

            # 服务器聚合
            print("使用fedavg进行aggregation")
            self.global_model = self.FedAvg(client_models)

            # 评估全局模型
            if (i + 1) % 5 == 0:
                # self.model_global_eval(i)
                all_eval_results.append(self.model_global_eval(i))#————————————————————————————

        # 训练结束后保存最终模型
        final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
        torch.save(self.global_model.state_dict(), final_model_path)
        print(f"{self.dataset}: Saved final global model at {final_model_path}")

        print("All process complete")
        return all_eval_results # 返回最终结果给 main.py————————————————————

    def model_global_eval(self, taskid):
        results = {} # 创建字典存储结果——————————————————————————————————
        results['round'] = taskid
        # 整体测试集准确率 - 使用完整测试集
        acc_all = prediction(
            self.global_model,
            self.test_loader,
            self.device
        )
        print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        results['all_tasks'] = acc_all

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

        # self.compute_classifier_similarity(task_index)  # 已关闭：不再打印/保存分类器相似度矩阵

        # 保存当前任务结束后的全局模型
        model_path = os.path.join(self.save_path, f'{self.dataset}: global_model_task_{task_index}.pth')
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


    def compute_classifier_similarity(self, task_index):
            """
            训练完第 task_index 个任务后，对第 0 ~ task_index 个任务分别计算
            其包含类别的分类器权重余弦相似度矩阵，打印并保存为 CSV。
            """
            # 获取分类器（最后一层 fc）的权重，shape: (num_classes, feature_dim)
            fc_weight = self.global_model.fc.weight.data.clone().cpu()
    
            save_dir = os.path.join(self.save_path, 'classifier_similarity')
            os.makedirs(save_dir, exist_ok=True)
    
            for t in range(task_index + 1):
                # 汇总所有客户端在任务 t 上的类别
                all_task_classes = set()
                for client_mask in self.mask:
                    all_task_classes.update(client_mask[t])
                task_classes = sorted(list(all_task_classes))
    
                # 取出这些类别对应的分类器权重行
                class_weights = fc_weight[task_classes]  # (len(task_classes), feature_dim)
    
                # L2 归一化后做矩阵乘法得到余弦相似度矩阵
                normed_weights = torch.nn.functional.normalize(class_weights, p=2, dim=1)
                cos_sim_matrix = torch.mm(normed_weights, normed_weights.t()).numpy()
    
                # 用 pandas DataFrame 格式化，行列索引为类别编号
                df = pd.DataFrame(cos_sim_matrix,
                                index=[f'class_{c}' for c in task_classes],
                                columns=[f'class_{c}' for c in task_classes])
    
                print(f'\n[Classifier Similarity] After task {task_index}, '
                    f'cosine similarity of task {t} classes {task_classes}:')
                print(df.to_string())
    
                # 保存为 CSV
                save_file = os.path.join(
                    save_dir,
                    f'{self.dataset}_after_task{task_index}_sim_of_task{t}.csv'
                )
                df.to_csv(save_file)
                print(f'Saved to {save_file}')




# # 3turns

# import copy
# import random

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# from client import Client
# # from models.VAE_PRETRAINED import VAE
# from utils import prediction, prediction_task
# import os

# from ResNet import resnet18_cbam  # 使用ResNet作为分类模型


# class Server(object):
#     # def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, batch_size, test_loader, num_classes,
#     #              client_mask):
#     #     self.client_num = client_num
#     #     self.task_num = task_num
#     #     self.device = device
#     #     self.test_loader = test_loader
#     #     self.num_classes = num_classes  # 总类别数
#     #     self.client_mask = client_mask

#     #     # 全局分类模型
#     #     self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#     #     self.clients = []
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method,
#                  FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
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
#         # 标准FedAvg实现
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             # 获取所有客户端模型对应参数
#             param_list = [model.state_dict()[k] for model in models]
#             # 确保参数的数据类型为浮点型
#             param_list = [param.float() for param in param_list]
#             # 堆叠参数并求平均值
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#             # global_dict[k] = torch.stack([model.state_dict()[k] for model in models], 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model

#     def train_clients(self):
#         all_eval_results = [] # 用于存储每次评估的结果——————————————————————
#         for i in range(self.task_num * 5):
#             print(f"--------global round {i},task number {i // 5}-----------")

#             # 分发全局模型
#             for client in self.clients:
#                 client.recive_global_models(self.global_model)

#             # 客户端本地训练
#             client_models = []
#             for client in self.clients:
#                 client.train(i)
#                 client_models.append(client.model)

#             # 服务器聚合
#             print("使用fedavg进行aggregation")
#             self.global_model = self.FedAvg(client_models)

#             # 评估全局模型
#             if (i + 1) % 5 == 0:
#                 # self.model_global_eval(i)
#                 eval_result = self.model_global_eval(i)#————————————————————————————
#                 all_eval_results.append(eval_result)#————————————————————————————

#         # 训练结束后保存最终模型
#         final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"{self.dataset}: Saved final global model at {final_model_path}")

#         print("All process complete")
#         return all_eval_results # 返回所有评估结果给 main.py————————————————————

#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         results['round'] = taskid # 记录轮次——————————————————————————————
#         # 整体测试集准确率 - 使用完整测试集
#         acc_all = prediction(
#             self.global_model,
#             self.test_loader,
#             self.device
#         )
#         print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
#         results['all_tasks'] = acc_all # 存储总精度——————————————————————————

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

#         # self.compute_classifier_similarity(task_index)

#         # 保存当前任务结束后的全局模型
#         model_path = os.path.join(self.save_path, f'{self.dataset}: global_model_task_{task_index}.pth')
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
# # from models.VAE_PRETRAINED import VAE
# from utils import prediction, prediction_task
# import os
 
# from ResNet import resnet18_cbam  # 使用ResNet作为分类模型
 
 
# class Server(object):
#     # def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, batch_size, test_loader, num_classes,
#     #              client_mask):
#     #     self.client_num = client_num
#     #     self.task_num = task_num
#     #     self.device = device
#     #     self.test_loader = test_loader
#     #     self.num_classes = num_classes  # 总类别数
#     #     self.client_mask = client_mask
 
#     #     # 全局分类模型
#     #     self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#     #     self.clients = []
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method,
#                  FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
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
#                 # FLOPs = 2 * Cout * Hout * Wout * (Cin/groups) * Kh * Kw
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
 
#         # 根据数据集确定输入尺寸
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
#         # 标准FedAvg实现
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             # 获取所有客户端模型对应参数
#             param_list = [model.state_dict()[k] for model in models]
#             # 确保参数的数据类型为浮点型
#             param_list = [param.float() for param in param_list]
#             # 堆叠参数并求平均值
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#             # global_dict[k] = torch.stack([model.state_dict()[k] for model in models], 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model
 
#     def train_clients(self):
#         final_results = {} # 用于存储最后一次评估的结果——————————————————————
 
#         # ========== 计算开销预估 ==========
#         model_bytes = self._get_model_bytes()
#         forward_flops_per_sample = self._estimate_forward_flops()
#         # 训练FLOPs ≈ 2 × forward FLOPs（前向 + 反向）
#         train_flops_per_sample = 2 * forward_flops_per_sample
#         print(f"\n{'='*50}")
#         print(f"[Overhead] Model size: {self._format_bytes(model_bytes)}")
#         print(f"[Overhead] Forward FLOPs per sample: {self._format_flops(forward_flops_per_sample)}")
#         print(f"{'='*50}\n")
 
#         for i in range(self.task_num * 5):
#             print(f"--------global round {i},task number {i // 5}-----------")
 
#             # 分发全局模型（通信：下行）
#             for client in self.clients:
#                 client.recive_global_models(self.global_model)
#             # 下行通信量：server -> 每个client 发送一份模型
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # 客户端本地训练
#             client_models = []
#             round_flops = 0
#             for client in self.clients:
#                 client.train(i)
#                 client_models.append(client.model)
#                 # 累计本轮FLOPs：samples × epochs × flops_per_sample
#                 client_flops = client.num_train_samples_this_round * self.epoch_local * train_flops_per_sample
#                 round_flops += client_flops
 
#             self.total_train_flops += round_flops
 
#             # 上行通信量：每个client -> server 发送一份模型
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # 服务器聚合
#             print("使用fedavg进行aggregation")
#             self.global_model = self.FedAvg(client_models)
 
#             # 评估全局模型
#             if (i + 1) % 5 == 0:
#                 # self.model_global_eval(i)
#                 final_results = self.model_global_eval(i)#————————————————————————————
 
#         # 训练结束后保存最终模型
#         final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"{self.dataset}: Saved final global model at {final_model_path}")
 
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
 
#         print("All process complete")
#         return final_results # 返回最终结果给 main.py————————————————————
 
#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
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
 
#         self.compute_classifier_similarity(task_index)
 
#         # 保存当前任务结束后的全局模型
#         model_path = os.path.join(self.save_path, f'{self.dataset}: global_model_task_{task_index}.pth')
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
 
 
#     def compute_classifier_similarity(self, task_index):
#             """
#             训练完第 task_index 个任务后，对第 0 ~ task_index 个任务分别计算
#             其包含类别的分类器权重余弦相似度矩阵，打印并保存为 CSV。
#             """
#             # 获取分类器（最后一层 fc）的权重，shape: (num_classes, feature_dim)
#             fc_weight = self.global_model.fc.weight.data.clone().cpu()
    
#             save_dir = os.path.join(self.save_path, 'classifier_similarity')
#             os.makedirs(save_dir, exist_ok=True)
    
#             for t in range(task_index + 1):
#                 # 汇总所有客户端在任务 t 上的类别
#                 all_task_classes = set()
#                 for client_mask in self.mask:
#                     all_task_classes.update(client_mask[t])
#                 task_classes = sorted(list(all_task_classes))
    
#                 # 取出这些类别对应的分类器权重行
#                 class_weights = fc_weight[task_classes]  # (len(task_classes), feature_dim)
    
#                 # L2 归一化后做矩阵乘法得到余弦相似度矩阵
#                 normed_weights = torch.nn.functional.normalize(class_weights, p=2, dim=1)
#                 cos_sim_matrix = torch.mm(normed_weights, normed_weights.t()).numpy()
    
#                 # 用 pandas DataFrame 格式化，行列索引为类别编号
#                 df = pd.DataFrame(cos_sim_matrix,
#                                 index=[f'class_{c}' for c in task_classes],
#                                 columns=[f'class_{c}' for c in task_classes])
    
#                 print(f'\n[Classifier Similarity] After task {task_index}, '
#                     f'cosine similarity of task {t} classes {task_classes}:')
#                 print(df.to_string())
    
#                 # 保存为 CSV
#                 save_file = os.path.join(
#                     save_dir,
#                     f'{self.dataset}_after_task{task_index}_sim_of_task{t}.csv'
#                 )
#                 df.to_csv(save_file)
#                 print(f'Saved to {save_file}')