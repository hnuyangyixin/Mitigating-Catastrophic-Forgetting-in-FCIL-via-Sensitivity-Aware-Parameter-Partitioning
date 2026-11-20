"""TinyImageNet20C 数据划分器（新逻辑，独立于原 tinyimagenet_subset_spliter）。

需求（与原 TinyImageNet 划分逻辑分开，互不影响）：
- 固定 120 私有类 + 80 公共类（共 200 类）。
  → 20 客户端时 private_class_num=6（20×6=120），公共类=80。
- 每客户端可见类数 = 私有类 + 全部公共类 = 6 + 80 = 86。
- task_num=5 时每任务 int(86/5)=17 类，5×17=85 类被用上；
  每客户端排序后的“第 86 类”被丢弃 —— 该客户端不在任何任务里训练它。
- 被丢的类如果全局没有任何客户端训练到，则测试集也排除它（既不训练也不测试）。

关键设计：训练/测试在同一次划分里一致地决定（先 random_split 定下每客户端用到的类，
再据此构造测试集），避免原代码“训练和测试各自独立随机划分”导致的不一致。
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import TypeVar

# 复用原文件里的 TinyImageNet 数据集加载类（带自动下载），避免重复实现
from data_process.tinyimagenet_subset_spliter import TinyImageNet

T_co = TypeVar('T_co', covariant=True)

NUM_TOTAL_CLASSES = 200


class TinyImageNet20C_Data_Spliter():

    def __init__(self, client_num, task_num, private_class_num, input_size):
        self.client_num = client_num
        self.task_num = task_num
        self.private_class_num = private_class_num
        self.input_size = input_size

        # 训练增强（与原 TinyImageNet 一致的风格）
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 划分确定后记录：每客户端实际用到的类（前 85 个），以及全局用到的类并集。
        # 这两个在 random_split() 里被填充，供 process_testdata() 复用。
        self.client_used_classes = None     # list[set]，每客户端实际训练的类
        self.global_used_classes = None     # set，全局所有客户端训练到的类（测试集只保留这些）
        self.dropped_classes = None         # set，被全局丢弃的类（既不训练也不测试）

    def random_split(self):
        root_path = './local_datasets/tiny-imagenet-200'
        self.dataset = TinyImageNet(root=root_path, train=True, download=True)
        trainset = self.dataset
        N = NUM_TOTAL_CLASSES

        # 统计每类样本索引
        class_counts = torch.zeros(N)
        class_label = [[] for _ in range(N)]
        for j, label in enumerate(tqdm(trainset.targets, desc="Indexing Classes (20C)")):
            class_counts[label] += 1
            class_label[label].append(j)

        # --- 私有/公共类划分（120 私有 + 80 公共）---
        total_private_class_num = self.client_num * self.private_class_num
        assert total_private_class_num <= N, \
            f"client_num×private_class_num={total_private_class_num} 超过 {N} 类，无法划分"
        public_class_num = N - total_private_class_num  # 期望 = 80

        class_public_all = list(range(N))
        class_p = random.sample(class_public_all, total_private_class_num)
        class_public = list(set(class_public_all) - set(class_p))  # 80 个公共类

        # 分配私有类：每客户端 private_class_num 个
        class_private = [class_p[self.private_class_num * i: self.private_class_num * i + self.private_class_num]
                         for i in range(self.client_num)]

        # 每客户端 = 私有 + 全部公共，打乱
        for i in range(self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        client_subset = [[] for _ in range(self.client_num)]
        client_mask = [[] for _ in range(self.client_num)]

        # 每任务类数（向下取整）；末尾不足一任务的类被丢弃
        classes_per_client = len(class_private[0])          # 86
        class_every_task = int(classes_per_client / self.task_num)  # 17
        used_per_client = class_every_task * self.task_num          # 85（前85类用上，第86类丢）

        # Dirichlet 控制公共类在客户端间的非独立同分布。
        # 注意：客户端数较多时（如 20），不能用 while 强制每个客户端份额 >= 0.05，
        # 否则 dirichlet(ones(20)) 的均值正好是 1/20=0.05，几乎不可能全部 >= 0.05，
        # 会陷入死循环。改为按 0.5/client_num 自适应下限 + 有限次重采样。
        min_share = 0.5 / self.client_num
        dirichlet_perclass = {}
        for c in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            retry = 0
            while (a < min_share).any() and retry < 10:
                a = np.random.dirichlet(np.ones(self.client_num), 1)
                retry += 1
            dirichlet_perclass[c] = a[0]

        self.client_used_classes = [set() for _ in range(self.client_num)]

        for i in range(self.client_num):
            # 该客户端实际用到的类 = 前 used_per_client 个（第86类被丢）
            used_classes_i = class_private[i][:used_per_client]
            self.client_used_classes[i].update(int(c) for c in used_classes_i)

            for j in range(self.task_num):
                start = j * class_every_task
                end = start + class_every_task
                class_this_task = class_private[i][start:end]
                client_mask[i].append(class_this_task)

                index = []
                for k in class_this_task:
                    if k in class_public:
                        # 公共类：按 Dirichlet 比例从剩余样本抽
                        count = int(int(class_counts[k]) * dirichlet_perclass[k][i])
                        unused = set(class_label[k])
                        if len(unused) < count:
                            count = len(unused)
                        chosen = random.sample(list(unused), count)
                        index.extend(chosen)
                        for idx in chosen:
                            class_label[k].remove(idx)
                    else:
                        # 私有类：全部拿走
                        index.extend(class_label[k])

                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, self.transform))

        # 全局用到的类并集；被丢弃的类 = 200 类里没人训练到的
        self.global_used_classes = set()
        for s in self.client_used_classes:
            self.global_used_classes.update(s)
        self.dropped_classes = set(range(N)) - self.global_used_classes
        print(f"[TinyImageNet20C] 私有={total_private_class_num} 公共={public_class_num} "
              f"每客户端用 {used_per_client}/{classes_per_client} 类，"
              f"全局训练到 {len(self.global_used_classes)} 类，丢弃 {len(self.dropped_classes)} 类")

        return client_subset, client_mask

    def process_testdata(self, surrogate_num):
        """构造测试集：只保留全局训练到的类（被丢弃的类既不训练也不测试）。
        必须在 random_split() 之后调用（依赖其确定的 global_used_classes）。
        """
        assert self.global_used_classes is not None, \
            "必须先调用 random_split() 再 process_testdata()，以保证训练/测试类一致"

        root_path = './local_datasets/tiny-imagenet-200'
        testset = TinyImageNet(root=root_path, train=False, download=True)
        N = NUM_TOTAL_CLASSES

        class_label = [[] for _ in range(N)]
        for j, label in enumerate(testset.targets):
            class_label[label].append(j)

        surro_index = []
        test_index = []
        for c in tqdm(range(N), desc="Processing Test Data (20C)"):
            if c not in self.global_used_classes:
                continue  # 被丢弃的类：测试集排除
            unused = set(class_label[c])
            curr_surro = min(surrogate_num, len(unused))
            chosen_surro = random.sample(list(unused), curr_surro)
            surro_index.extend(chosen_surro)
            for idx in chosen_surro:
                unused.remove(idx)
            test_index.extend(list(unused))

        surrodata = CustomedSubset(testset, surro_index, self.transform_test)
        testdata = CustomedSubset(testset, test_index, self.transform_test)
        return surrodata, testdata


class CustomedSubset(Dataset[T_co]):
    def __init__(self, dataset, indices, trans) -> None:
        self.indices = indices
        self.dataset = dataset
        self.transform = trans
        self.data_paths = [dataset.data[i] for i in indices]
        self.targets = [dataset.targets[i] for i in indices]
        self.targets = np.array(self.targets)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        target = int(self.targets[idx])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (64, 64))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)
