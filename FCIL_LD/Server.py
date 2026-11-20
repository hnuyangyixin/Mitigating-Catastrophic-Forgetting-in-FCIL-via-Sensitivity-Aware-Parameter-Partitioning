# """
# Server implementation for FCIL-LD
# Based on the paper: "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning 
# with a Low-Dimensional Projection Layer" (CVPR 2025 Workshop)
# """

# import copy
# import random
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# from client import Client
# from Generator import Generator, GeneratorTrainer
# from utils import prediction, prediction_task
# import os

# from ResNet import resnet18_cbam


# class Server(object):
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
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

#         # Global classification model with projection layer
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes, projection_dim=128)
        
#         # Previous task's global model (for knowledge distillation)
#         self.prev_global_model = None
        
#         # Data-free generator
#         self.generator = None
        
#         # Generator training parameters (from paper)
#         self.gen_latent_dim = 200
#         self.gen_lr = 0.005
#         self.gen_epochs = 1000  # Reduced for efficiency, paper uses 5000
#         self.alpha_div = 1.0
#         self.alpha_bn = 75.0
#         self.alpha_prior = 0.001
        
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
#         """Standard FedAvg implementation"""
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             param_list = [model.state_dict()[k] for model in models]
#             param_list = [param.float() for param in param_list]
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model

#     def train_generator(self, current_task):
#         """
#         Train the data-free generator on the server side (Section 3.1 in paper).
#         Called at the end of each task after model aggregation.
#         """
#         print(f"\n=== Training data-free generator for task {current_task} ===")
        
#         # Initialize or continue training generator
#         if self.generator is None:
#             # Determine image size based on dataset
#             if self.dataset in ['MNIST']:
#                 img_size = 32  # MNIST is resized to 32x32 in spliter
#             else:
#                 img_size = 32  # CIFAR-10/100 default
            
#             self.generator = Generator(
#                 latent_dim=self.gen_latent_dim,
#                 img_channels=3,
#                 img_size=img_size,
#                 num_classes=self.num_classes
#             )
        
#         # Create generator trainer
#         trainer = GeneratorTrainer(
#             generator=self.generator,
#             global_model=self.global_model,
#             device=self.device,
#             num_classes=self.num_classes,
#             alpha_div=self.alpha_div,
#             alpha_bn=self.alpha_bn,
#             alpha_prior=self.alpha_prior,
#             lr=self.gen_lr,
#             num_epochs=self.gen_epochs
#         )
        
#         # Train generator
#         self.generator = trainer.train(batch_size=64, verbose=True)
        
#         print(f"Generator training completed for task {current_task}")

#     def train_clients(self):
#         all_eval_results = [] # 用于存储最后一次评估的结果——————————————————————
#         for i in range(self.task_num * 5):
#             current_task = i // 5
#             print(f"\n========== Global round {i}, Task {current_task} ==========")

#             # Distribute global model, previous model, and generator to clients
#             for client in self.clients:
#                 client.recive_global_models(
#                     self.global_model, 
#                     self.prev_global_model,  # Previous task's model for KD
#                     self.generator  # Generator for replay
#                 )

#             # Client local training
#             client_models = []
#             for client in self.clients:
#                 client.train(i)
#                 client_models.append(client.model)

#             # Server aggregation using FedAvg
#             print("Performing FedAvg aggregation")
#             self.global_model = self.FedAvg(client_models)

#             # At the end of each task (every 5 rounds)
#             if (i + 1) % 5 == 0:
#                 # Evaluate global model
#                 all_eval_results.append(self.model_global_eval(i))#————————————————————————————
                
#                 # Save current global model as previous model for next task
#                 self.prev_global_model = copy.deepcopy(self.global_model)
#                 self.prev_global_model.eval()
#                 for param in self.prev_global_model.parameters():
#                     param.requires_grad = False
                
#                 # Train generator at the end of each task
#                 self.train_generator(current_task)

#         # Save final model
#         final_model_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_final_global_model.pth')
#         os.makedirs(self.save_path, exist_ok=True)
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"{self.dataset}: Saved final global model at {final_model_path}")
#         return all_eval_results # 返回最终结果给 main.py————————————————————
        
#         # Save generator
#         if self.generator is not None:
#             gen_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_generator.pth')
#             torch.save(self.generator.state_dict(), gen_path)
#             print(f"{self.dataset}: Saved generator at {gen_path}")

#         print("\nAll process complete")

#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         # Overall test accuracy
#         acc_all = prediction(
#             self.global_model,
#             self.test_loader,
#             self.device
#         )
#         print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')

#         # Per-task evaluation
#         task_index = taskid // 5
#         for t in range(task_index + 1):
#             all_task_classes = set()
#             for client_mask in self.mask:
#                 all_task_classes.update(client_mask[t])
#             task_classes = list(all_task_classes)
#             acc_task = self.evaluate_task(t, task_classes)
#             print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
#             results[f'task_{t}'] = acc_task#——————————————————————————————

#         # Save checkpoint
#         model_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_global_model_task_{task_index}.pth')
#         os.makedirs(self.save_path, exist_ok=True)
#         torch.save(self.global_model.state_dict(), model_path)
#         print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
#         return results # 返回结果字典——————————————————————————————

#     def evaluate_task(self, task_id, task_classes):
#         """Evaluate model on specific task"""
#         return prediction_task(
#             self.global_model,
#             self.test_loader,
#             self.device,
#             task_classes
#         )



# 3 turns

"""
Server implementation for FCIL-LD
Based on the paper: "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning 
with a Low-Dimensional Projection Layer" (CVPR 2025 Workshop)
"""

import copy
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from client import Client
from Generator import Generator, GeneratorTrainer
from utils import prediction, prediction_task
import os

from ResNet import resnet18_cbam


class Server(object):
    def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
                 local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
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

        # Global classification model with projection layer
        self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes, projection_dim=128)
        
        # Previous task's global model (for knowledge distillation)
        self.prev_global_model = None
        
        # Data-free generator
        self.generator = None
        
        # Generator training parameters (from paper)
        self.gen_latent_dim = 200
        self.gen_lr = 0.005
        self.gen_epochs = 1000  # Reduced for efficiency, paper uses 5000
        self.alpha_div = 1.0
        self.alpha_bn = 75.0
        self.alpha_prior = 0.001
        
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
        """Standard FedAvg implementation"""
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            param_list = [model.state_dict()[k] for model in models]
            param_list = [param.float() for param in param_list]
            global_dict[k] = torch.stack(param_list, 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model

    def train_generator(self, current_task):
        """
        Train the data-free generator on the server side (Section 3.1 in paper).
        Called at the end of each task after model aggregation.
        """
        print(f"\n=== Training data-free generator for task {current_task} ===")
        
        # Initialize or continue training generator
        if self.generator is None:
            # Determine image size based on dataset
            if self.dataset in ['MNIST']:
                img_size = 32  # MNIST is resized to 32x32 in spliter
            else:
                img_size = 32  # CIFAR-10/100 default
            
            self.generator = Generator(
                latent_dim=self.gen_latent_dim,
                img_channels=3,
                img_size=img_size,
                num_classes=self.num_classes
            )
        
        # Create generator trainer
        trainer = GeneratorTrainer(
            generator=self.generator,
            global_model=self.global_model,
            device=self.device,
            num_classes=self.num_classes,
            alpha_div=self.alpha_div,
            alpha_bn=self.alpha_bn,
            alpha_prior=self.alpha_prior,
            lr=self.gen_lr,
            num_epochs=self.gen_epochs
        )
        
        # Train generator
        self.generator = trainer.train(batch_size=64, verbose=True)
        
        print(f"Generator training completed for task {current_task}")

    def train_clients(self):
        all_eval_results = [] # 用于存储每次评估的结果——————————————————————
        for i in range(self.task_num * 5):
            current_task = i // 5
            print(f"\n========== Global round {i}, Task {current_task} ==========")

            # Distribute global model, previous model, and generator to clients
            for client in self.clients:
                client.recive_global_models(
                    self.global_model, 
                    self.prev_global_model,  # Previous task's model for KD
                    self.generator  # Generator for replay
                )

            # Client local training
            client_models = []
            for client in self.clients:
                client.train(i)
                client_models.append(client.model)

            # Server aggregation using FedAvg
            print("Performing FedAvg aggregation")
            self.global_model = self.FedAvg(client_models)

            # At the end of each task (every 5 rounds)
            if (i + 1) % 5 == 0:
                # Evaluate global model
                final_results = self.model_global_eval(i)#————————————————————————————
                all_eval_results.append(final_results)#————————————————————————————
                
                # Save current global model as previous model for next task
                self.prev_global_model = copy.deepcopy(self.global_model)
                self.prev_global_model.eval()
                for param in self.prev_global_model.parameters():
                    param.requires_grad = False
                
                # Train generator at the end of each task
                self.train_generator(current_task)

        # Save final model
        final_model_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_final_global_model.pth')
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.global_model.state_dict(), final_model_path)
        print(f"{self.dataset}: Saved final global model at {final_model_path}")
        return all_eval_results # 返回所有评估结果给 main.py————————————————————
        
        # Save generator
        if self.generator is not None:
            gen_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_generator.pth')
            torch.save(self.generator.state_dict(), gen_path)
            print(f"{self.dataset}: Saved generator at {gen_path}")

        print("\nAll process complete")

    def model_global_eval(self, taskid):
        results = {} # 创建字典存储结果——————————————————————————————————
        results['round'] = taskid # 记录轮次——————————————————————————————
        # Overall test accuracy
        acc_all = prediction(
            self.global_model,
            self.test_loader,
            self.device
        )
        print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        results['all_tasks'] = acc_all # 存储总精度——————————————————————————

        # Per-task evaluation
        task_index = taskid // 5
        for t in range(task_index + 1):
            all_task_classes = set()
            for client_mask in self.mask:
                all_task_classes.update(client_mask[t])
            task_classes = list(all_task_classes)
            acc_task = self.evaluate_task(t, task_classes)
            print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
            results[f'task_{t}'] = acc_task#——————————————————————————————

        # Save checkpoint
        model_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_global_model_task_{task_index}.pth')
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.global_model.state_dict(), model_path)
        print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
        return results # 返回结果字典——————————————————————————————

    def evaluate_task(self, task_id, task_classes):
        """Evaluate model on specific task"""
        return prediction_task(
            self.global_model,
            self.test_loader,
            self.device,
            task_classes
        )



# # 计算与通信开销


# """
# Server implementation for FCIL-LD
# Based on the paper: "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning 
# with a Low-Dimensional Projection Layer" (CVPR 2025 Workshop)
# """
 
# import copy
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
 
# from client import Client
# from Generator import Generator, GeneratorTrainer
# from utils import prediction, prediction_task
# import os
 
# from ResNet import resnet18_cbam
 
 
# class Server(object):
#     def __init__(self, client_num, task_num, dataset, lr, device, epoch_local, pseudo_samples, global_method,
#                  local_method, FE, subset, batch_size, alpha, test_dataset, class_num, mask, save_path):
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
 
#         # Global classification model with projection layer
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes, projection_dim=128)
        
#         # Previous task's global model (for knowledge distillation)
#         self.prev_global_model = None
        
#         # Data-free generator
#         self.generator = None
        
#         # Generator training parameters (from paper)
#         self.gen_latent_dim = 200
#         self.gen_lr = 0.005
#         self.gen_epochs = 1000  # Reduced for efficiency, paper uses 5000
#         self.alpha_div = 1.0
#         self.alpha_bn = 75.0
#         self.alpha_prior = 0.001
        
#         self.clients = []
 
#         # ========== 计算和通信开销统计 ==========
#         self.total_comm_bytes = 0       # 累计通信字节数
#         self.total_train_flops = 0      # 累计训练FLOPs
#         self._forward_flops = None      # 单次前向FLOPs（延迟计算）
 
#     @staticmethod
#     def _get_model_bytes_static(model):
#         """计算任意模型的参数+缓冲区总字节数"""
#         total_bytes = 0
#         for param in model.parameters():
#             total_bytes += param.nelement() * param.element_size()
#         for buf in model.buffers():
#             total_bytes += buf.nelement() * buf.element_size()
#         return total_bytes
 
#     def _get_model_bytes(self):
#         """计算全局模型的参数+缓冲区总字节数"""
#         return self._get_model_bytes_static(self.global_model)
 
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
#         """Standard FedAvg implementation"""
#         global_dict = self.global_model.state_dict()
#         for k in global_dict.keys():
#             param_list = [model.state_dict()[k] for model in models]
#             param_list = [param.float() for param in param_list]
#             global_dict[k] = torch.stack(param_list, 0).mean(0)
#         self.global_model.load_state_dict(global_dict)
#         return self.global_model
 
#     def train_generator(self, current_task):
#         """
#         Train the data-free generator on the server side (Section 3.1 in paper).
#         Called at the end of each task after model aggregation.
#         """
#         print(f"\n=== Training data-free generator for task {current_task} ===")
        
#         # Initialize or continue training generator
#         if self.generator is None:
#             # Determine image size based on dataset
#             if self.dataset in ['MNIST']:
#                 img_size = 32  # MNIST is resized to 32x32 in spliter
#             else:
#                 img_size = 32  # CIFAR-10/100 default
            
#             self.generator = Generator(
#                 latent_dim=self.gen_latent_dim,
#                 img_channels=3,
#                 img_size=img_size,
#                 num_classes=self.num_classes
#             )
        
#         # Create generator trainer
#         trainer = GeneratorTrainer(
#             generator=self.generator,
#             global_model=self.global_model,
#             device=self.device,
#             num_classes=self.num_classes,
#             alpha_div=self.alpha_div,
#             alpha_bn=self.alpha_bn,
#             alpha_prior=self.alpha_prior,
#             lr=self.gen_lr,
#             num_epochs=self.gen_epochs
#         )
        
#         # Train generator
#         self.generator = trainer.train(batch_size=64, verbose=True)
        
#         print(f"Generator training completed for task {current_task}")
 
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
#             current_task = i // 5
#             print(f"\n========== Global round {i}, Task {current_task} ==========")
 
#             # Distribute global model, previous model, and generator to clients
#             for client in self.clients:
#                 client.recive_global_models(
#                     self.global_model, 
#                     self.prev_global_model,  # Previous task's model for KD
#                     self.generator  # Generator for replay
#                 )
 
#             # 下行通信量：server -> 每个client
#             # 1) 全局模型（每轮都发）
#             self.total_comm_bytes += self.client_num * model_bytes
#             # 2) prev_global_model（与全局模型同架构，task>0时发送）
#             if self.prev_global_model is not None:
#                 self.total_comm_bytes += self.client_num * model_bytes
#             # 3) generator（task>0时发送）
#             if self.generator is not None:
#                 gen_bytes = self._get_model_bytes_static(self.generator)
#                 self.total_comm_bytes += self.client_num * gen_bytes
 
#             # Client local training
#             client_models = []
#             round_flops = 0
#             for client in self.clients:
#                 client.train(i)
#                 client_models.append(client.model)
#                 # 累计本轮FLOPs
#                 client_flops = client.num_train_samples_this_round * self.epoch_local * train_flops_per_sample
#                 round_flops += client_flops
 
#             self.total_train_flops += round_flops
 
#             # 上行通信量：每个client -> server 发送一份模型
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # Server aggregation using FedAvg
#             print("Performing FedAvg aggregation")
#             self.global_model = self.FedAvg(client_models)
 
#             # At the end of each task (every 5 rounds)
#             if (i + 1) % 5 == 0:
#                 # Evaluate global model
#                 final_results = self.model_global_eval(i)#————————————————————————————
                
#                 # Save current global model as previous model for next task
#                 self.prev_global_model = copy.deepcopy(self.global_model)
#                 self.prev_global_model.eval()
#                 for param in self.prev_global_model.parameters():
#                     param.requires_grad = False
                
#                 # Train generator at the end of each task
#                 self.train_generator(current_task)
 
#         # Save final model
#         final_model_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_final_global_model.pth')
#         os.makedirs(self.save_path, exist_ok=True)
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"{self.dataset}: Saved final global model at {final_model_path}")
 
#         # ========== 打印开销统计 ==========
#         print(f"\n{'='*50}")
#         print(f"[Overhead Summary]")
#         print(f"  Total Training FLOPs:        {self._format_flops(self.total_train_flops)}")
#         print(f"  Total Communication Bytes:   {self._format_bytes(self.total_comm_bytes)}")
#         print(f"  Model Size (per transfer):   {self._format_bytes(model_bytes)}")
#         if self.generator is not None:
#             print(f"  Generator Size:              {self._format_bytes(self._get_model_bytes_static(self.generator))}")
#         total_rounds = self.task_num * 5
#         print(f"  Total Rounds:                {total_rounds}")
#         print(f"{'='*50}\n")
 
#         return final_results # 返回最终结果给 main.py————————————————————
        
#         # Save generator
#         if self.generator is not None:
#             gen_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_generator.pth')
#             torch.save(self.generator.state_dict(), gen_path)
#             print(f"{self.dataset}: Saved generator at {gen_path}")
 
#         print("\nAll process complete")
 
#     def model_global_eval(self, taskid):
#         results = {} # 创建字典存储结果——————————————————————————————————
#         # Overall test accuracy
#         acc_all = prediction(
#             self.global_model,
#             self.test_loader,
#             self.device
#         )
#         print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
 
#         # Per-task evaluation
#         task_index = taskid // 5
#         for t in range(task_index + 1):
#             all_task_classes = set()
#             for client_mask in self.mask:
#                 all_task_classes.update(client_mask[t])
#             task_classes = list(all_task_classes)
#             acc_task = self.evaluate_task(t, task_classes)
#             print(f'{self.dataset}: Global model on task {t} test accuracy: {acc_task:.2f}%')
#             results[f'task_{t}'] = acc_task#——————————————————————————————
 
#         # Save checkpoint
#         model_path = os.path.join(self.save_path, f'{self.dataset}_FCIL_LD_global_model_task_{task_index}.pth')
#         os.makedirs(self.save_path, exist_ok=True)
#         torch.save(self.global_model.state_dict(), model_path)
#         print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
#         return results # 返回结果字典——————————————————————————————
 
#     def evaluate_task(self, task_id, task_classes):
#         """Evaluate model on specific task"""
#         return prediction_task(
#             self.global_model,
#             self.test_loader,
#             self.device,
#             task_classes
#         )