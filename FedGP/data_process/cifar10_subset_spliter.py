# import random
# import numpy as np
# import torch
# from torchvision.datasets import CIFAR10
# from torchvision.transforms import transforms
# from tqdm import tqdm
# from PIL import Image
# from torch.utils.data import Subset, Dataset
# from typing import TypeVar, Sequence

# # 保持与原文件一致的类型定义
# T_co = TypeVar('T_co', covariant=True)

# class cifar10_Data_Spliter():

#     def __init__(self, client_num, task_num, private_class_num, input_size):
#         self.client_num = client_num
#         self.task_num = task_num
#         scale = (0.05, 1.0)
#         ratio = (3. / 4., 4. / 3.)
        
#         # 训练集数据增强
#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#         ])
        
#         # CIFAR-10 标准归一化参数
#         self.transform1 = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ])
        
#         self.private_class_num = private_class_num
#         self.input_size = input_size

#     # 分成client_num数目个subset,每个subset里包含了task个subsubset
#     def random_split(self):
#         # 根据数据集选择对应的类
#         # 如果是 MNIST，请改为 self.mnist_dataset = MNIST(...)
#         self.cifar10_dataset = CIFAR10(root='./local_datasets', train=True, download=True)
#         trainset = self.cifar10_dataset

#         # 1. 统计所有数据的索引
#         class_counts = torch.zeros(10)
#         class_label_indices = [[] for _ in range(10)] # 存储每个类别的所有索引
#         j = 0
#         for x, label in tqdm(trainset):
#             class_counts[label] += 1
#             class_label_indices[label].append(j)
#             j += 1
        
#         # 2. 确定私有类和公共类 (逻辑不变)
#         total_private_class_num = self.client_num * self.private_class_num
#         if total_private_class_num > 10:
#              raise ValueError("Private classes exceed total classes.")

#         public_class_num = 10 - total_private_class_num
#         class_public = [i for i in range(10)]
        
#         class_p = random.sample(class_public, total_private_class_num)
#         class_public = list(set(class_public) - set(class_p))

#         # 分配给每个客户端的类别池
#         client_class_pool = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0, self.client_num)]
#         for i in range(0, self.client_num):
#             client_class_pool[i].extend(class_public)
#             random.shuffle(client_class_pool[i])
#             # client_class_pool[i] 现在包含了该客户端能看到的所有类别 (例如 8 个类)

#         # 3. 计算 Dirichlet 分布比例 (逻辑不变)
#         dirichlet_perclass = {}
#         for i in class_public:
#             a = np.random.dirichlet(np.ones(self.client_num), 1)
#             while (a < 0.05).any(): 
#                 a = np.random.dirichlet(np.ones(self.client_num), 1)
#             dirichlet_perclass[i] = a[0]

#         # === 4. 关键修改：为客户端构建“类重叠、样本不重叠”的任务 ===
        
#         client_subset = [[] for _ in range(self.client_num)]
#         client_mask = [[] for _ in range(self.client_num)]

#         for i in range(self.client_num):
#             # A. 确定该客户端所有任务的类别安排
#             # 策略：保证每个任务至少有 2 个类。
#             # 我们将该客户端拥有的类列表重复拼接，然后按每任务2个类进行截取。
            
#             # 例如: pool = [0,1,2,3,4,5,6,7] (8类)
#             # 需要 5 个任务，每个任务 2 类 -> 需要 10 个槽位
#             # 扩展 pool -> [0,1,2,3,4,5,6,7, 0,1] (取前10个)
            
#             pool = client_class_pool[i]
#             needed_slots = self.task_num * 2 # 强制每个任务 2 个类
            
#             extended_pool = []
#             while len(extended_pool) < needed_slots:
#                 extended_pool.extend(pool)
#             extended_pool = extended_pool[:needed_slots] # 截断到刚好够用
            
#             # 统计每个类在这个扩展池中出现了几次，以便切分样本
#             # usage_map: {class_id: total_times_used}
#             usage_map = {}
#             for cls in extended_pool:
#                 usage_map[cls] = usage_map.get(cls, 0) + 1
            
#             # current_use_pointer: {class_id: current_usage_index}
#             current_use_pointer = {cls: 0 for cls in usage_map}

#             # B. 预处理该客户端的公共类样本池 (应用 Dirichlet)
#             # 为了方便切分，我们先把该客户端能分到的所有公共类索引提取出来
#             client_specific_indices = {} # {class_id: [idx1, idx2, ...]}
            
#             for cls in pool:
#                 if cls in class_public:
#                     # 公共类：按 Dirichlet 比例截取
#                     total_indices = class_label_indices[cls]
#                     count = int(int(len(total_indices)) * dirichlet_perclass[cls][i])
#                     # 简单的哈希采样或随机采样，这里为了逻辑简单使用 shuffle 后的切片
#                     # 注意：为了避免不同客户端抢同一个 index (虽然是 random_split 但这只是模拟)，
#                     # 严谨做法是基于 index 偏移。为了复刻原代码逻辑，我们随机采样：
#                     # *警告*: 原代码 logic 是 destructive 的 (remove)，这里我们假设 split 是独立的
#                     # 为了简化，我们假设每个客户端取自己的一份，互不影响 (模拟 Non-IID)
                    
#                     # 临时打乱当前类的所有索引
#                     indices_copy = list(total_indices)
#                     random.shuffle(indices_copy)
#                     # 简单的处理：这一步其实应该在客户端循环外做 strict partition，
#                     # 但为了兼容原代码结构，我们尽量复用。
#                     # 这里我们只取属于该客户端的份额
#                     client_specific_indices[cls] = indices_copy[:count]
#                 else:
#                     # 私有类：全部拿走
#                     client_specific_indices[cls] = list(class_label_indices[cls])

#             # C. 生成任务
#             for t in range(self.task_num):
#                 # 获取当前任务的 2 个类
#                 task_classes = extended_pool[t*2 : (t+1)*2]
#                 client_mask[i].append(task_classes)
                
#                 task_indices = []
                
#                 for cls in task_classes:
#                     # 获取该类属于该客户端的总样本
#                     all_indices = client_specific_indices[cls]
#                     total_samples = len(all_indices)
                    
#                     # 计算切分块的大小
#                     # 如果这个类要被用 2 次，就把样本除以 2，第一次用前 50%，第二次用后 50%
#                     num_chunks = usage_map[cls]
#                     chunk_size = total_samples // num_chunks
                    
#                     chunk_idx = current_use_pointer[cls]
                    
#                     # 计算起止位置
#                     start_idx = chunk_idx * chunk_size
#                     # 如果是最后一块，取到最后（防止除不尽丢失数据）
#                     if chunk_idx == num_chunks - 1:
#                         end_idx = total_samples
#                     else:
#                         end_idx = start_idx + chunk_size
                        
#                     selected_indices = all_indices[start_idx : end_idx]
#                     task_indices.extend(selected_indices)
                    
#                     # 更新指针
#                     current_use_pointer[cls] += 1
                
#                 # 打乱当前任务的数据
#                 random.shuffle(task_indices)
#                 client_subset[i].append(CustomedSubset(trainset, task_indices, self.transform))

#         return client_subset, client_mask

#     def process_testdata(self, surrogate_num):
#         # 测试集处理
#         self.cifar10_dataset = CIFAR10(root='./local_datasets', train=False, download=True)
#         testset = self.cifar10_dataset
        
#         class_counts = torch.zeros(10)
#         class_label = []
#         for i in range(10):
#             class_label.append([])
        
#         j = 0
#         for x, label in testset:
#             class_counts[label] += 1
#             class_label[label].append(j)
#             j += 1

#         surro_index = []
#         test_index = []
        
#         for i in tqdm(range(10)):
#             q = 0
#             unused_indice = set(class_label[i])
#             # 如果需要 surrogate 数据 (用于蒸馏等)，从测试集分出一部分
#             # 确保不越界
#             curr_surrogate = min(surrogate_num, len(unused_indice))
            
#             chosen_surro = random.sample(list(unused_indice), curr_surrogate)
#             surro_index.extend(chosen_surro)
            
#             for idx in chosen_surro:
#                 unused_indice.remove(idx)
            
#             test_index.extend(list(unused_indice))
            
#         # 使用 transform1 (测试集/验证集通常只做 Normalize)
#         surrodata = CustomedSubset(testset, surro_index, self.transform1)
#         testdata = CustomedSubset(testset, test_index, self.transform1)
        
#         return surrodata, testdata


# class CustomedSubset(Dataset[T_co]):
#     r"""
#     Subset of a dataset at specified indices.
#     """
#     dataset: Dataset[T_co]
#     indices: Sequence[int]

#     def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans) -> None:
#         self.indices = indices
#         self.data = []
#         self.targets = []
#         self.dataset = dataset
#         self.transform_pretrain = trans
#         # CIFAR10 默认 transform
#         self.transform_origin = transforms.Compose([
#                 transforms.RandomResizedCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
#         for i in self.indices:
#             self.data.append(dataset.data[i])
#             self.targets.append(dataset.targets[i])
            
#         self.data = np.array(self.data)
#         self.targets = np.array(self.targets)
#         self.target_transform = None

#     def __getitem__(self, idx):
#         img, target = self.data[idx], self.targets[idx]
#         img = Image.fromarray(img)

#         if self.transform_pretrain is not None:
#             img_pre = self.transform_pretrain(img)
#         else:
#             img_pre = self.transform_origin(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img_pre, target

#     def __len__(self):
#         return len(self.indices)

import random
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Subset, Dataset
from typing import TypeVar, Sequence

# 保持与原文件一致的类型定义
T_co = TypeVar('T_co', covariant=True)

class cifar10_Data_Spliter():

    def __init__(self, client_num, task_num, private_class_num, input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        # 训练集数据增强
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        
        # CIFAR-10 标准归一化参数
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.private_class_num = private_class_num
        self.input_size = input_size

    # 分成client_num数目个subset,每个subset里包含了task个subsubset
    def random_split(self):
        # 注意：这里我们不需要 build_transform，直接使用内部定义的transform逻辑，
        # 或者你需要确保 data_process.utils 里有 build_transform 且兼容。
        # 为保持逻辑一致，这里沿用 CustomedSubset 的逻辑，传入 None 或 self.transform
        
        self.cifar10_dataset = CIFAR10(root='./local_datasets', train=True, download=True)
        trainset = self.cifar10_dataset

        # 10个类别的数据分给客户端使用
        class_counts = torch.zeros(10) # 每个类的数量
        class_label = [] # 每个类的index
        for i in range(10):
            class_label.append([])
        
        j = 0
        # CIFAR10 dataset 遍历返回 (image, label)
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        
        # --- 核心划分逻辑 (与 CIFAR100 完全一致) ---
        total_private_class_num = self.client_num * self.private_class_num
        
        # 检查参数合理性
        if total_private_class_num > 10:
            raise ValueError(f"Total private classes ({total_private_class_num}) exceeds CIFAR10 classes (10). Reduce private_class_num.")

        public_class_num = 10 - total_private_class_num
        class_public = [i for i in range(10)]
        
        # 随机选出私有类
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        # 分配私有类给每个客户端
        class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0, self.client_num)]
        
        # 将公共类加入每个客户端的候选列表并打乱
        for i in range(0, self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        # 对每个客户端进行操作
        client_subset = [[] for i in range(0, self.client_num)]
        client_mask = [[] for i in range(0, self.client_num)]

        # 计算每个任务包含多少个类 (这取决于划分逻辑，需确保能整除)
        # 注意：如果 (public + private_per_client) / task_num 不是整数，这里会截断。
        # CIFAR10: 假设 private=1, public=7 -> 每个客户端有8个类。如果 task=4, 则每任务2类。
        # 如果使用默认 task=5, 且每个客户端可见所有10类(或部分)，需要确保逻辑通顺。
        # 这里的逻辑是基于 client 视角的总类别数。
        
        # 修正：class_private[i] 的长度是 private_class_num + public_class_num
        classes_per_client = self.private_class_num + public_class_num
        class_every_task = int(classes_per_client / self.task_num)
        
        if class_every_task == 0:
             # 防止除零或空任务，对于CIFAR10 task_num不能太大
            class_every_task = 1 

        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # 避免极端不平衡
            while (a < 0.05).any(): # 稍微放宽限制，因为CIFAR10数据量大但类少
                a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]

        # === 修复开始：使用余数分配逻辑 ===
        for i in range(0, self.client_num):
            # 获取当前客户端拥有的总类别数 (例如 8)
            total_classes_for_client = len(class_private[i])
            
            # 计算基础步长和余数
            # 例如 8 / 5 -> base=1, remainder=3
            # 前3个任务分2个类，后2个任务分1个类 -> 2+2+2+1+1 = 8
            base_size = total_classes_for_client // self.task_num
            remainder = total_classes_for_client % self.task_num

            start_idx = 0
            for j in range(0, self.task_num):
                index = []
                
                # 动态计算当前任务的类别数量
                # 如果 j < remainder，则当前任务多分配 1 个类
                current_task_size = base_size + (1 if j < remainder else 0)
                
                end_idx = start_idx + current_task_size
                class_this_task = class_private[i][start_idx : end_idx]
                
                # 更新下一次的起点
                start_idx = end_idx
                
                client_mask[i].append(class_this_task)

                for k in class_this_task:
                    if k in class_public:
                        count = int(int(class_counts[k]) * dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        if len(unused_indice) < count:
                            count = len(unused_indice)
                        chosen_indices = random.sample(list(unused_indice), count)
                        index.extend(chosen_indices)
                        for idx in chosen_indices:
                            class_label[k].remove(idx)
                    else: 
                        index.extend(class_label[k])
                
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, self.transform))
        # === 修复结束 ===

        return client_subset, client_mask

    def process_testdata(self, surrogate_num):
        # 测试集处理
        self.cifar10_dataset = CIFAR10(root='./local_datasets', train=False, download=True)
        testset = self.cifar10_dataset
        
        class_counts = torch.zeros(10)
        class_label = []
        for i in range(10):
            class_label.append([])
        
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1

        surro_index = []
        test_index = []
        
        for i in tqdm(range(10)):
            q = 0
            unused_indice = set(class_label[i])
            # 如果需要 surrogate 数据 (用于蒸馏等)，从测试集分出一部分
            # 确保不越界
            curr_surrogate = min(surrogate_num, len(unused_indice))
            
            chosen_surro = random.sample(list(unused_indice), curr_surrogate)
            surro_index.extend(chosen_surro)
            
            for idx in chosen_surro:
                unused_indice.remove(idx)
            
            test_index.extend(list(unused_indice))
            
        # 使用 transform1 (测试集/验证集通常只做 Normalize)
        surrodata = CustomedSubset(testset, surro_index, self.transform1)
        testdata = CustomedSubset(testset, test_index, self.transform1)
        
        return surrodata, testdata


class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans) -> None:
        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans
        # CIFAR10 默认 transform
        self.transform_origin = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
            
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.target_transform = None

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)
        else:
            img_pre = self.transform_origin(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_pre, target

    def __len__(self):
        return len(self.indices)