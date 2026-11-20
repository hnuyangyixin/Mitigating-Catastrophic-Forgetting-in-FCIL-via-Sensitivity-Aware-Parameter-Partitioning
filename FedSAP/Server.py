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
# import torch.nn.functional as F

# # 新增用于可视化
# try:
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#     HAS_SKLEARN = True
# except Exception:
#     HAS_SKLEARN = False


# class Server(object):
    
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
#         self.similarity_results = {}

#         # 全局分类模型
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.clients = []

#         self.current_task = 0  # 当前任务索引
#         self.current_task_classes = []  # 当前任务类别

#         # 初始化客户端
#         self.init_clients()

#         # 确保保存目录存在，并保存任务划分（mask）
#         os.makedirs(self.save_path, exist_ok=True)
#         try:
#             torch.save(self.mask[0], os.path.join(self.save_path, "task_mask.pt"))
#             print("保存任务划分到 task_mask.pt")
#         except Exception as e:
#             print(f"保存 task_mask 失败: {e}")


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
    

#     # 用来衔接分类器和特征提取器
#     def classifier_align(self, client_prototypes):
#         """
#         client_prototypes: list of dicts, 每个客户端的 {cls: P_c}
#         """
#         model = self.global_model
#         model.to(self.device)
#         model.train()

#         fc = model.fc2.weight  # [num_classes, dim]
#         optimizer = torch.optim.SGD([fc], lr=0.01)  # 只优化分类器

#         align_loss = 0.0

#         for proto_dict in client_prototypes:
#             for cls, P_c in proto_dict.items():
#                 w = fc[cls]
#                 P_c = P_c.to(self.device)
#                 align_loss += (1 - torch.nn.functional.cosine_similarity(w, P_c, dim=0))

#         align_loss = align_loss / max(1, len(client_prototypes))  # 平均化

#         optimizer.zero_grad()
#         align_loss.backward()
#         optimizer.step()

#         print(f"[Server] L_align = {align_loss.item():.4f}")


#     def train_clients(self):
#         for i in range(self.task_num * 5):
#             print(f"{self.dataset}: --------global round {i},task number {i // 5}-----------")

#             # 分发全局模型
#             for client in self.clients:
#                 client.recive_global_models(self.global_model)

#             # 客户端本地训练
#             client_models = []
#             for client in self.clients:
#                 client.phase2_train(i)
#                 client_models.append(client.model)

#             # 服务器聚合
#             print("{self.dataset}: 使用fedavg进行aggregation")
#             self.global_model = self.FedAvg(client_models)
           
#             # # ======== 新增：对齐分类器方向 ========
#             # client_protos = [client.get_prototype() for client in self.clients]
#             # self.classifier_align(client_protos)
#             # # =====================================

#             # 评估全局模型
#             if (i + 1) % 5 == 0:
#                 self.model_global_eval(i)

#         # 训练结束后保存最终模型
#         final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
#         torch.save(self.global_model.state_dict(), final_model_path)
#         print(f"{self.dataset}: Saved final global model at {final_model_path}")
        
#         print("All process complete")

    
#     def model_global_eval(self, taskid):
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

#         # self.compute_classifier_similarity(task_index)

#         # 保存当前任务结束后的全局模型
#         model_path = os.path.join(self.save_path, f'{self.dataset}: global_model_task_{task_index}.pth')
#         torch.save(self.global_model.state_dict(), model_path)
#         print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")

#         # #新加的
#         # try:
#         #     # 当前任务（本次训练结束的任务）
#         #     current_task_classes = list(self.mask[0][task_index])
#         #     self.compute_and_save_pairwise_cosine(current_task_classes, task_index, based_on_global_after_task=task_index)

#         #     # 历史任务（包含从 0 到 task_index-1 的每一个旧任务）
#         #     if task_index > 0:
#         #         for past_t in range(task_index):
#         #             past_classes = list(self.mask[0][past_t])
#         #             # 注意：这里仍然使用当前 self.global_model 来计算 past_classes 的相似度
#         #             self.compute_and_save_pairwise_cosine(past_classes, past_t, based_on_global_after_task=task_index)

#         #     # # 新增：基于最新全局模型，计算并保存 TSNE 图
#         #     # # 1) 最新任务对应的类的 TSNE
#         #     # self.compute_and_save_tsne_for_task(current_task_classes, task_index, based_on_global_after_task=task_index)

#         #     # # 2) 历史任务（每个旧任务）对应的类，但基于最新分类器和特征提取器
#         #     # if task_index > 0:
#         #     #     for past_t in range(task_index):
#         #     #         past_classes = list(self.mask[0][past_t])
#         #     #         self.compute_and_save_tsne_for_task(past_classes, past_t, based_on_global_after_task=task_index)

#         # except Exception as e:
#         #     print(f"计算相似度或TSNE时发生错误: {e}")


#     def evaluate_task(self, task_id, task_classes):
#         """评估模型在特定任务上的表现 - 使用完整标签空间"""
#         return prediction_task(
#             self.global_model,
#             self.test_loader,  # 使用完整测试集
#             self.device,
#             task_classes
#         )

#     # def compute_classifier_similarity(self, task_index):
#     #     """
#     #     训练完第 task_index 个任务后，对第 0 ~ task_index 个任务分别计算
#     #     其包含类别的分类器权重余弦相似度矩阵，打印并保存为 CSV。
#     #     """
#     #     # 获取分类器（最后一层 fc）的权重，shape: (num_classes, feature_dim)
#     #     fc_weight = self.global_model.fc.weight.data.clone().cpu()
 
#     #     save_dir = os.path.join(self.save_path, 'classifier_similarity')
#     #     os.makedirs(save_dir, exist_ok=True)
 
#     #     for t in range(task_index + 1):
#     #         # 汇总所有客户端在任务 t 上的类别
#     #         all_task_classes = set()
#     #         for client_mask in self.mask:
#     #             all_task_classes.update(client_mask[t])
#     #         task_classes = sorted(list(all_task_classes))
 
#     #         # 取出这些类别对应的分类器权重行
#     #         class_weights = fc_weight[task_classes]  # (len(task_classes), feature_dim)
 
#     #         # L2 归一化后做矩阵乘法得到余弦相似度矩阵
#     #         normed_weights = torch.nn.functional.normalize(class_weights, p=2, dim=1)
#     #         cos_sim_matrix = torch.mm(normed_weights, normed_weights.t()).numpy()
 
#     #         # 用 pandas DataFrame 格式化，行列索引为类别编号
#     #         df = pd.DataFrame(cos_sim_matrix,
#     #                           index=[f'class_{c}' for c in task_classes],
#     #                           columns=[f'class_{c}' for c in task_classes])
 
#     #         print(f'\n[Classifier Similarity] After task {task_index}, '
#     #               f'cosine similarity of task {t} classes {task_classes}:')
#     #         print(df.to_string())
 
#     #         # 保存为 CSV
#     #         save_file = os.path.join(
#     #             save_dir,
#     #             f'{self.dataset}_after_task{task_index}_sim_of_task{t}.csv'
#     #         )
#     #         df.to_csv(save_file)
#     #         print(f'Saved to {save_file}')









# 3turns

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
import torch.nn.functional as F

# 新增用于可视化
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class Server(object):
    
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
        self.similarity_results = {}

        # 全局分类模型
        self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.clients = []

        self.current_task = 0  # 当前任务索引
        self.current_task_classes = []  # 当前任务类别

        # 初始化客户端
        self.init_clients()

        # 确保保存目录存在，并保存任务划分（mask）
        os.makedirs(self.save_path, exist_ok=True)
        try:
            torch.save(self.mask[0], os.path.join(self.save_path, "task_mask.pt"))
            print("保存任务划分到 task_mask.pt")
        except Exception as e:
            print(f"保存 task_mask 失败: {e}")


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
    

    # 用来衔接分类器和特征提取器
    def classifier_align(self, client_prototypes):
        """
        client_prototypes: list of dicts, 每个客户端的 {cls: P_c}
        """
        model = self.global_model
        model.to(self.device)
        model.train()

        fc = model.fc2.weight  # [num_classes, dim]
        optimizer = torch.optim.SGD([fc], lr=0.01)  # 只优化分类器

        align_loss = 0.0

        for proto_dict in client_prototypes:
            for cls, P_c in proto_dict.items():
                w = fc[cls]
                P_c = P_c.to(self.device)
                align_loss += (1 - torch.nn.functional.cosine_similarity(w, P_c, dim=0))

        align_loss = align_loss / max(1, len(client_prototypes))  # 平均化

        optimizer.zero_grad()
        align_loss.backward()
        optimizer.step()

        print(f"[Server] L_align = {align_loss.item():.4f}")


    def train_clients(self):
        all_eval_results = [] # 用于存储每次评估的结果
        for i in range(self.task_num * 5):
            print(f"{self.dataset}: --------global round {i},task number {i // 5}-----------")

            # 分发全局模型
            for client in self.clients:
                client.recive_global_models(self.global_model)

            # 客户端本地训练
            client_models = []
            for client in self.clients:
                client.phase2_train(i)
                client_models.append(client.model)

            # 服务器聚合
            print("{self.dataset}: 使用fedavg进行aggregation")
            self.global_model = self.FedAvg(client_models)
           
            # # ======== 新增：对齐分类器方向 ========
            # client_protos = [client.get_prototype() for client in self.clients]
            # self.classifier_align(client_protos)
            # # =====================================

            # 评估全局模型
            if (i + 1) % 5 == 0:
                eval_result = self.model_global_eval(i)
                all_eval_results.append(eval_result)

        # 训练结束后保存最终模型
        final_model_path = os.path.join(self.save_path, f'{self.dataset}: final_global_model.pth')
        torch.save(self.global_model.state_dict(), final_model_path)
        print(f"{self.dataset}: Saved final global model at {final_model_path}")
        
        print("All process complete")
        return all_eval_results # 返回所有评估结果给 main.py

    
    def model_global_eval(self, taskid):
        results = {} # 创建字典存储结果
        results['round'] = taskid # 记录轮次
        # 整体测试集准确率 - 使用完整测试集
        acc_all = prediction(
            self.global_model,
            self.test_loader,
            self.device
        )
        print(f'{self.dataset}: Round {taskid} finishes, global model accuracy (all tasks): {acc_all:.2f}%')
        results['all_tasks'] = acc_all # 存储总精度

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
            results[f'task_{t}'] = acc_task

        # self.compute_classifier_similarity(task_index)

        # 保存当前任务结束后的全局模型
        model_path = os.path.join(self.save_path, f'{self.dataset}: global_model_task_{task_index}.pth')
        torch.save(self.global_model.state_dict(), model_path)
        print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
        return results # 返回结果字典

        # #新加的
        # try:
        #     # 当前任务（本次训练结束的任务）
        #     current_task_classes = list(self.mask[0][task_index])
        #     self.compute_and_save_pairwise_cosine(current_task_classes, task_index, based_on_global_after_task=task_index)

        #     # 历史任务（包含从 0 到 task_index-1 的每一个旧任务）
        #     if task_index > 0:
        #         for past_t in range(task_index):
        #             past_classes = list(self.mask[0][past_t])
        #             # 注意：这里仍然使用当前 self.global_model 来计算 past_classes 的相似度
        #             self.compute_and_save_pairwise_cosine(past_classes, past_t, based_on_global_after_task=task_index)

        #     # 新增：基于最新全局模型，计算并保存 TSNE 图
        #     # 1) 最新任务对应的类的 TSNE
        #     self.compute_and_save_tsne_for_task(current_task_classes, task_index, based_on_global_after_task=task_index)

        #     # 2) 历史任务（每个旧任务）对应的类，但基于最新分类器和特征提取器
        #     if task_index > 0:
        #         for past_t in range(task_index):
        #             past_classes = list(self.mask[0][past_t])
        #             self.compute_and_save_tsne_for_task(past_classes, past_t, based_on_global_after_task=task_index)

        # except Exception as e:
        #     print(f"计算相似度或TSNE时发生错误: {e}")


    def evaluate_task(self, task_id, task_classes):
        """评估模型在特定任务上的表现 - 使用完整标签空间"""
        return prediction_task(
            self.global_model,
            self.test_loader,  # 使用完整测试集
            self.device,
            task_classes
        )

    # def compute_and_save_pairwise_cosine(self, task_classes, task_index, based_on_global_after_task=None):
    #     """
    #     计算并打印给定任务类别集合在当前全局模型分类器（self.global_model）上的两两余弦相似度矩阵。
    #     - task_classes: 含该任务涉及的全局类标签（例如 [0,1,...,19]）
    #     - task_index: 该任务的索引（用于打印/保存文件名）
    #     - based_on_global_after_task: 可选，表示这是基于训练到哪个任务之后的 global_model 来计算的（用于文件命名）
    #     结果会保存到 self.similarity_results，并写入磁盘。
    #     """
    #     # 准备模型（使用当前最新 global_model）
    #     self.global_model.to(self.device)
    #     self.global_model.eval()

    #     with torch.no_grad():
    #         # 尝试直接取 fc 权重（常见）
    #         try:
    #             classifier_weight = self.global_model.fc2.weight.data.clone().detach()  # [num_classes, feat_dim]
    #         except Exception:
    #             # 备用方案：从 state_dict 中寻找第一个行数等于 num_classes 的 2D 张量
    #             sd = self.global_model.state_dict()
    #             found = None
    #             for k, v in sd.items():
    #                 if len(v.shape) == 2 and v.shape[0] == self.num_classes:
    #                     found = v.clone().detach()
    #                     break
    #             if found is None:
    #                 raise RuntimeError("无法在 global_model 中找到形状匹配分类器权重的参数 (num_classes, feat_dim).")
    #             classifier_weight = found

    #         # 提取所需类别的权重行
    #         class_indices = list(task_classes)
    #         W = classifier_weight[class_indices, :]  # shape: (k, feat_dim)
    #         # 归一化每一行向量
    #         W_norm = F.normalize(W.to(self.device), p=2, dim=1)  # 每一行单位化
    #         sim_matrix = torch.matmul(W_norm, W_norm.t())  # (k, k)
    #         sim_np = sim_matrix.cpu().numpy()

    #     # 打印结果（格式化）
    #     based_on_str = f"after_task_{based_on_global_after_task}" if based_on_global_after_task is not None else "latest"
    #     print(f"\n=== Task {task_index} classifier pairwise cosine similarities (based on global model {based_on_str}) ===")
    #     print(f"Classes involved: {class_indices}")
    #     k = len(class_indices)
    #     header = "    " + " ".join([f"{c:>6}" for c in class_indices])
    #     print(header)
    #     for i, c in enumerate(class_indices):
    #         row = " ".join([f"{sim_np[i, j]:6.4f}" for j in range(k)])
    #         print(f"{c:>4} {row}")

    #     # 保存到内存字典（记录元信息）
    #     entry_key = f"task{task_index}_based_on_{based_on_str}"
    #     self.similarity_results[entry_key] = {
    #         "classes": class_indices,
    #         "matrix": sim_np,
    #         "based_on": based_on_global_after_task
    #     }

    #     # 保存到文件：例如 similarity_task_0_based_on_global_after_task_1.npy
    #     based_name = based_on_str
    #     save_file = os.path.join(self.save_path, f"similarity_task_{task_index}_based_on_global_{based_name}.npy")
    #     try:
    #         np.save(save_file, sim_np)
    #         print(f"Saved pairwise cosine similarity matrix for task {task_index} (based on {based_name}) to {save_file}")
    #     except Exception as e:
    #         print(f"保存相似度矩阵文件失败: {e}")

    # def compute_and_save_tsne_for_task(self, task_classes, task_index, based_on_global_after_task=None, perplexity=30, n_iter=1000):
    #     """
    #     基于当前 self.global_model（最新的模型），提取测试集中特定类的特征（model 返回的第一个输出），并对这些类的特征做 t-SNE 可视化。
    #     - task_classes: 该任务的类别全局 id 列表
    #     - task_index: 任务索引（用于命名）
    #     - based_on_global_after_task: 用于标注该图片是基于训练到哪个任务后的全局模型
    #     """
    #     if not HAS_SKLEARN:
    #         print("sklearn 或 matplotlib 未安装，无法绘制 TSNE 图。请安装 scikit-learn 和 matplotlib。")
    #         return

    #     # 收集特征与标签
    #     self.global_model.to(self.device)
    #     self.global_model.eval()

    #     features = []
    #     labels = []
    #     with torch.no_grad():
    #         for inputs, labs in self.test_loader:
    #             inputs = inputs.to(self.device)
    #             # 这里假设 model 返回 (feature, logits)
    #             out = self.global_model(inputs)
    #             if isinstance(out, tuple) or isinstance(out, list):
    #                 feat = out[0]
    #             else:
    #                 # 如果模型只返回 logits，则尝试取中间层或抛错
    #                 raise RuntimeError("global_model forward must return (features, logits) for TSNE extraction.")

    #             feat = feat.cpu().numpy()
    #             labs = labs.numpy()
    #             features.append(feat)
    #             labels.append(labs)

    #     features = np.concatenate(features, axis=0)
    #     labels = np.concatenate(labels, axis=0)

    #     # 为每个请求的类集合筛选相应样本
    #     sel_mask = np.isin(labels, task_classes)
    #     sel_features = features[sel_mask]
    #     sel_labels = labels[sel_mask]

    #     if len(sel_labels) == 0:
    #         print(f"没有在测试集中找到任务 {task_index} 的任意样本，无法绘制 TSNE。")
    #         return

    #     # 为可视化，降到 50 维（可选）以加速 t-SNE
    #     if sel_features.shape[1] > 50:
    #         # 使用 PCA 先降维（用 sklearn 的 PCA，若不可用则直接用前 50 维）
    #         try:
    #             from sklearn.decomposition import PCA
    #             pca = PCA(n_components=50)
    #             sel_features_reduced = pca.fit_transform(sel_features)
    #         except Exception:
    #             sel_features_reduced = sel_features[:, :50]
    #     else:
    #         sel_features_reduced = sel_features

    #     tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(sel_labels)//3)), init='pca', max_iter=n_iter)
    #     tsne_emb = tsne.fit_transform(sel_features_reduced)

    #     # 绘图
    #     plt.figure(figsize=(8, 6))
    #     unique_classes = np.unique(sel_labels)
    #     for cls in unique_classes:
    #         idxs = sel_labels == cls
    #         plt.scatter(tsne_emb[idxs, 0], tsne_emb[idxs, 1], label=str(cls), s=8)
    #     plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

    #     based_on_str = f"after_task_{based_on_global_after_task}" if based_on_global_after_task is not None else "latest"
    #     title = f"TSNE Task {task_index} ({based_on_str})"
    #     plt.title(title)
    #     plt.tight_layout()

    #     save_name = os.path.join(self.save_path, f"tsne_task_{task_index}_classes_{min(task_classes)}_{max(task_classes)}_based_on_{based_on_str}.png")
    #     try:
    #         plt.savefig(save_name, dpi=200)
    #         plt.close()
    #         print(f"Saved TSNE plot for task {task_index} to {save_name}")
    #     except Exception as e:
    #         print(f"保存 TSNE 图失败: {e}")







# # 计算通信开销

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
# import torch.nn as nn
# import torch.nn.functional as F
 
# # 新增用于可视化
# try:
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#     HAS_SKLEARN = True
# except Exception:
#     HAS_SKLEARN = False
 
 
# class Server(object):
    
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
#         self.similarity_results = {}
 
#         # 全局分类模型
#         self.global_model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.clients = []
 
#         self.current_task = 0  # 当前任务索引
#         self.current_task_classes = []  # 当前任务类别
 
#         # 初始化客户端
#         self.init_clients()
 
#         # 确保保存目录存在，并保存任务划分（mask）
#         os.makedirs(self.save_path, exist_ok=True)
#         try:
#             torch.save(self.mask[0], os.path.join(self.save_path, "task_mask.pt"))
#             print("保存任务划分到 task_mask.pt")
#         except Exception as e:
#             print(f"保存 task_mask 失败: {e}")
 
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
    
 
#     # 用来衔接分类器和特征提取器
#     def classifier_align(self, client_prototypes):
#         """
#         client_prototypes: list of dicts, 每个客户端的 {cls: P_c}
#         """
#         model = self.global_model
#         model.to(self.device)
#         model.train()
 
#         fc = model.fc2.weight  # [num_classes, dim]
#         optimizer = torch.optim.SGD([fc], lr=0.01)  # 只优化分类器
 
#         align_loss = 0.0
 
#         for proto_dict in client_prototypes:
#             for cls, P_c in proto_dict.items():
#                 w = fc[cls]
#                 P_c = P_c.to(self.device)
#                 align_loss += (1 - torch.nn.functional.cosine_similarity(w, P_c, dim=0))
 
#         align_loss = align_loss / max(1, len(client_prototypes))  # 平均化
 
#         optimizer.zero_grad()
#         align_loss.backward()
#         optimizer.step()
 
#         print(f"[Server] L_align = {align_loss.item():.4f}")
 
 
#     def train_clients(self):
 
#         # ========== 计算开销预估 ==========
#         model_bytes = self._get_model_bytes()
#         forward_flops_per_sample = self._estimate_forward_flops()
#         train_flops_per_sample = 2 * forward_flops_per_sample  # 前向 + 反向
#         print(f"\n{'='*50}")
#         print(f"[Overhead] Model size: {self._format_bytes(model_bytes)}")
#         print(f"[Overhead] Forward FLOPs per sample: {self._format_flops(forward_flops_per_sample)}")
#         print(f"{'='*50}\n")
 
#         for i in range(self.task_num * 5):
#             print(f"{self.dataset}: --------global round {i},task number {i // 5}-----------")
 
#             # 分发全局模型
#             for client in self.clients:
#                 client.recive_global_models(self.global_model)
 
#             # 下行通信量
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # 客户端本地训练
#             client_models = []
#             round_flops = 0
#             for client in self.clients:
#                 client.phase2_train(i)
#                 client_models.append(client.model)
#                 # 累计本轮FLOPs
#                 client_flops = client.num_train_samples_this_round * self.epoch_local * train_flops_per_sample
#                 round_flops += client_flops
 
#             self.total_train_flops += round_flops
 
#             # 上行通信量
#             self.total_comm_bytes += self.client_num * model_bytes
 
#             # 服务器聚合
#             print("{self.dataset}: 使用fedavg进行aggregation")
#             self.global_model = self.FedAvg(client_models)
           
#             # # ======== 新增：对齐分类器方向 ========
#             # client_protos = [client.get_prototype() for client in self.clients]
#             # self.classifier_align(client_protos)
#             # # =====================================
 
#             # 评估全局模型
#             if (i + 1) % 5 == 0:
#                 self.model_global_eval(i)
 
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
 
    
#     def model_global_eval(self, taskid):
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
 
#         # self.compute_classifier_similarity(task_index)
 
#         # 保存当前任务结束后的全局模型
#         model_path = os.path.join(self.save_path, f'{self.dataset}: global_model_task_{task_index}.pth')
#         torch.save(self.global_model.state_dict(), model_path)
#         print(f"{self.dataset}: Saved global model for task {task_index} at {model_path}")
 
 
#     def evaluate_task(self, task_id, task_classes):
#         """评估模型在特定任务上的表现 - 使用完整标签空间"""
#         return prediction_task(
#             self.global_model,
#             self.test_loader,  # 使用完整测试集
#             self.device,
#             task_classes
#         )