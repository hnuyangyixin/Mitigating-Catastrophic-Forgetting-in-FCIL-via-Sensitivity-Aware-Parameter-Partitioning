import os
import random
from typing import TypeVar, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

T_co = TypeVar('T_co', covariant=True)


# === 1. 自定义 ImageNet-1K 数据集加载类（带自动下载 + 每类子采样） ===
# 思路与 TinyImageNet 完全一致：首次使用时自动下载并落地到本地目录，
# 之后从磁盘按路径读取。区别在于 ImageNet-1K 没有单个小 zip，这里用
# Hugging Face 的 datasets 流式下载，且每类只取 (train_per_class+test_per_class)
# 张，避免拉取完整 ~150GB 数据集。
class ImageNet1K(Dataset):
    def __init__(self, root, train=True, transform=None, download=False,
                 train_per_class=500, test_per_class=100,
                 hf_repo_id='benjamin-paine/imagenet-1k-256x256',
                 image_key='image', label_key='label'):
        self.root = root
        self.train = train
        self.transform = transform
        self.train_per_class = train_per_class
        self.test_per_class = test_per_class
        self.per_class = train_per_class + test_per_class
        self.hf_repo_id = hf_repo_id
        self.image_key = image_key
        self.label_key = label_key
        self.num_classes = 1000

        self.data = []      # 图像路径
        self.targets = []   # 标签

        # 自动下载（若本地尚未落地）
        if download:
            self.download_dataset()

        train_dir = os.path.join(root, 'train')
        if not os.path.isdir(train_dir):
            raise RuntimeError(
                f'ImageNet-1K not found at {train_dir}. '
                f'请设置 download=True 自动下载，或检查路径。')

        # 类别文件夹按名称排序，class_0000 -> 0 ... class_0999 -> 999
        classes = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

        desc = "Loading ImageNet-1K Train" if train else "Loading ImageNet-1K Test"
        for name in tqdm(classes, desc=desc):
            class_idx = self.class_to_idx[name]
            cdir = os.path.join(train_dir, name)
            img_names = sorted(os.listdir(cdir))
            # 每类预取：前 train_per_class 张做训练，紧随的 test_per_class 张做测试
            if self.train:
                selected = img_names[:self.train_per_class]
            else:
                selected = img_names[self.train_per_class:
                                     self.train_per_class + self.test_per_class]
            for fn in selected:
                self.data.append(os.path.join(cdir, fn))
                self.targets.append(class_idx)

        self.targets = np.array(self.targets)

    def download_dataset(self):
        """用 Hugging Face datasets 流式下载，每类落地 per_class 张到本地。"""
        train_root = os.path.join(self.root, 'train')
        marker = os.path.join(self.root, '.download_complete')
        if os.path.exists(marker):
            print("ImageNet-1K subset already exists.")
            return

        os.makedirs(train_root, exist_ok=True)
        print(f"ImageNet-1K not found at {self.root}, "
              f"streaming from Hugging Face: {self.hf_repo_id} ...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "需要 datasets 库才能自动下载 ImageNet-1K，请先 `pip install datasets`。")

        ds = load_dataset(self.hf_repo_id, split='train', streaming=True)

        # 扫描磁盘已有图片，支持断点续传：已下满的类直接跳过，只补缺失部分
        saved = [0] * self.num_classes
        for label in range(self.num_classes):
            cdir = os.path.join(train_root, f"class_{label:04d}")
            if os.path.isdir(cdir):
                saved[label] = min(self.per_class, len(
                    [fn for fn in os.listdir(cdir) if fn.lower().endswith('.jpeg')]))
        done_classes = sum(1 for s in saved if s >= self.per_class)
        total_target = self.num_classes * self.per_class
        saved_total = sum(saved)
        if saved_total > 0:
            print(f"检测到已存在 {saved_total} 张（{done_classes} 个类已下满），"
                  f"断点续传，仅下载缺失部分 ...")

        for example in ds:
            label = int(example[self.label_key])
            if label < 0 or label >= self.num_classes:
                continue
            if saved[label] >= self.per_class:
                continue

            cdir = os.path.join(train_root, f"class_{label:04d}")
            if saved[label] == 0:
                os.makedirs(cdir, exist_ok=True)

            img = example[self.image_key]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(cdir, f"{label:04d}_{saved[label]:04d}.JPEG"),
                     format="JPEG", quality=95)

            saved[label] += 1
            saved_total += 1
            if saved[label] == self.per_class:
                done_classes += 1
            if saved_total % 2000 == 0:
                print(f"  已保存 {saved_total}/{total_target} 张，"
                      f"完成 {done_classes}/{self.num_classes} 个类别")
            if done_classes >= self.num_classes:
                break

        incomplete = [i for i in range(self.num_classes) if saved[i] < self.per_class]
        if incomplete:
            print(f"警告：{len(incomplete)} 个类别样本不足 {self.per_class} 张："
                  f"{incomplete[:10]}{' ...' if len(incomplete) > 10 else ''}")
        # 写完成标记，避免重复下载
        with open(marker, 'w') as f:
            f.write(f"per_class={self.per_class}\n")
        print(f"下载并整理完成，共 {saved_total} 张，保存在 {train_root}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = int(self.targets[index])  # 转 python int，确保 collate 成 int64(Long)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# === 2. 划分器逻辑 (ImageNet1K_Data_Spliter) ===
# 划分逻辑与 TinyImageNet_Data_Spliter / cifar100_Data_Spliter 完全一致。
class ImageNet1K_Data_Spliter():

    def __init__(self, client_num, task_num, private_class_num, input_size,
                 train_per_class=500, test_per_class=100):
        self.client_num = client_num
        self.task_num = task_num
        self.private_class_num = private_class_num
        self.input_size = input_size
        self.train_per_class = train_per_class
        self.test_per_class = test_per_class
        self.total_class_num = 1000

        # 数据增强（与 TinyImageNet 保持一致的风格，强制缩放到模型输入尺寸）
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def random_split(self):
        root_path = './local_datasets/imagenet-1k'
        # === 自动下载 ===
        self.dataset = ImageNet1K(root=root_path, train=True, download=True,
                                  train_per_class=self.train_per_class,
                                  test_per_class=self.test_per_class)
        trainset = self.dataset
        N = self.total_class_num

        # 统计每个类别的索引
        class_counts = torch.zeros(N)
        class_label = [[] for _ in range(N)]
        for j, label in enumerate(tqdm(trainset.targets, desc="Indexing Classes")):
            class_counts[label] += 1
            class_label[label].append(j)

        # --- 核心划分逻辑（与 CIFAR100 / TinyImageNet 一致）---
        total_private_class_num = self.client_num * self.private_class_num
        public_class_num = N - total_private_class_num

        class_public = [i for i in range(N)]
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        # 分配私有类
        class_private = [class_p[self.private_class_num*i: self.private_class_num*i+self.private_class_num]
                         for i in range(0, self.client_num)]

        # 混入公共类
        for i in range(0, self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        client_subset = [[] for _ in range(0, self.client_num)]
        client_mask = [[] for _ in range(0, self.client_num)]

        # 计算每任务类别数
        classes_per_client = len(class_private[0])
        class_every_task = int(classes_per_client / self.task_num)

        # Dirichlet 分布（控制公共类在客户端间的非独立同分布程度）
        # 注意：客户端数较多时（如 30），不能再用 while 强制每个客户端份额 >= 0.05，
        # 否则 dirichlet(ones(30)) 几乎不可能满足该条件，会陷入近乎死循环。
        # 这里改为按 1/(2*client_num) 自适应下限，仅做有限次重采样，超过则接受。
        min_share = 0.5 / self.client_num
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            retry = 0
            while (a < min_share).any() and retry < 10:
                a = np.random.dirichlet(np.ones(self.client_num), 1)
                retry += 1
            dirichlet_perclass[i] = a[0]

        for i in range(0, self.client_num):
            for j in range(0, self.task_num):
                index = []
                start = j * class_every_task
                end = start + class_every_task

                if start >= len(class_private[i]):
                    break

                class_this_task = class_private[i][start:end]
                client_mask[i].append(class_this_task)

                for k in class_this_task:
                    if k in class_public:
                        # 公共类：按 Dirichlet 分配
                        count = int(int(class_counts[k]) * dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        if len(unused_indice) < count:
                            count = len(unused_indice)

                        chosen = random.sample(list(unused_indice), count)
                        index.extend(chosen)
                        for idx in chosen:
                            class_label[k].remove(idx)
                    else:
                        # 私有类：全部拿走
                        index.extend(class_label[k])

                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, self.transform))

        return client_subset, client_mask

    def process_testdata(self, surrogate_num):
        root_path = './local_datasets/imagenet-1k'
        self.dataset = ImageNet1K(root=root_path, train=False, download=True,
                                  train_per_class=self.train_per_class,
                                  test_per_class=self.test_per_class)
        testset = self.dataset
        N = self.total_class_num

        class_counts = torch.zeros(N)
        class_label = [[] for _ in range(N)]
        for j, label in enumerate(testset.targets):
            class_counts[label] += 1
            class_label[label].append(j)

        surro_index = []
        test_index = []

        for i in tqdm(range(N), desc="Processing Test Data"):
            unused_indice = set(class_label[i])
            curr_surro = min(surrogate_num, len(unused_indice))

            chosen_surro = random.sample(list(unused_indice), curr_surro)
            surro_index.extend(chosen_surro)
            for idx in chosen_surro:
                unused_indice.remove(idx)

            test_index.extend(list(unused_indice))

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
        target = int(self.targets[idx])  # 转 python int，确保 collate 成 int64(Long)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 简单的错误处理，防止坏图导致训练中断（后续 transform 会统一缩放）
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (256, 256))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)
