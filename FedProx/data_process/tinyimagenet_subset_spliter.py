import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from typing import TypeVar, Sequence
import requests
import zipfile
import io

T_co = TypeVar('T_co', covariant=True)

# === 1. 自定义 TinyImageNet 数据集加载类 (带自动下载) ===
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        self.url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

        # 检查是否需要下载
        if download:
            self.download_dataset()

        # 建立类别映射 (wnid -> 0-199)
        wnids_path = os.path.join(root, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise RuntimeError(f'Dataset not found at {root}. Please set download=True or check the path.')
        
        with open(wnids_path, 'r') as f:
            wnids = [x.strip() for x in f.readlines()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        if self.train:
            # 加载训练集: ./train/n0xxxx/images/xxxx.JPEG
            train_dir = os.path.join(root, 'train')
            for wnid in tqdm(wnids, desc="Loading TinyImageNet Train"):
                class_idx = self.class_to_idx[wnid]
                img_dir = os.path.join(train_dir, wnid, 'images')
                if not os.path.isdir(img_dir):
                    continue
                for filename in os.listdir(img_dir):
                    if filename.endswith('.JPEG'):
                        self.data.append(os.path.join(img_dir, filename))
                        self.targets.append(class_idx)
        else:
            # 加载验证集作为测试集: ./val/images/xxxx.JPEG + ./val/val_annotations.txt
            val_dir = os.path.join(root, 'val')
            img_dir = os.path.join(val_dir, 'images')
            anno_path = os.path.join(val_dir, 'val_annotations.txt')
            
            # 某些情况下val可能没有被正确解压或者结构不同，做健壮性检查
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split('\t')
                        filename = parts[0]
                        wnid = parts[1]
                        
                        if wnid in self.class_to_idx: # 确保类别存在
                            full_path = os.path.join(img_dir, filename)
                            if os.path.exists(full_path):
                                self.data.append(full_path)
                                self.targets.append(self.class_to_idx[wnid])

        self.targets = np.array(self.targets)

    def download_dataset(self):
        """自动下载并解压 TinyImageNet"""
        if os.path.exists(os.path.join(self.root, 'wnids.txt')):
            print("TinyImageNet dataset already exists.")
            return

        print(f"Dataset not found at {self.root}, downloading from {self.url}...")
        
        # 确保父目录存在
        parent_dir = os.path.dirname(self.root)
        os.makedirs(parent_dir, exist_ok=True)
        
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        # 使用 BytesIO 在内存中下载，避免存临时文件
        buffer = io.BytesIO()
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as t:
            for data in response.iter_content(block_size):
                t.update(len(data))
                buffer.write(data)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(buffer) as zf:
            # zip包里面已经包含 tiny-imagenet-200 文件夹
            # 所以解压到 local_datasets 即可
            zf.extractall(parent_dir)
        print("Download and extraction complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        
        # 必须转为 RGB，因为部分图片是灰度的
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# === 2. 划分器逻辑 (TinyImageNet_Data_Spliter) ===
class TinyImageNet_Data_Spliter():

    def __init__(self, client_num, task_num, private_class_num, input_size):
        self.client_num = client_num
        self.task_num = task_num
        self.private_class_num = private_class_num
        self.input_size = input_size 

        # 数据增强 (适配 ImageNet 均值)
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(input_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        self.transform = transforms.Compose([
            transforms.Resize(input_size),          # 先强制缩放到 32x32 (因为你的模型是针对CIFAR设计的)
            transforms.RandomCrop(input_size, padding=4), # 再进行轻微的随机裁剪增强
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def random_split(self):
        root_path = './local_datasets/tiny-imagenet-200'
        # === 关键修改：开启 download=True ===
        self.dataset = TinyImageNet(root=root_path, train=True, download=True)
        trainset = self.dataset

        # 统计 200 个类别的索引
        class_counts = torch.zeros(200)
        class_label = [[] for _ in range(200)]
        
        for j, label in enumerate(tqdm(trainset.targets, desc="Indexing Classes")):
            class_counts[label] += 1
            class_label[label].append(j)

        # --- 核心划分逻辑 (与 CIFAR100 一致) ---
        total_private_class_num = self.client_num * self.private_class_num
        public_class_num = 200 - total_private_class_num
        
        class_public = [i for i in range(200)]
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        # 分配私有类
        class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0, self.client_num)]
        
        # 混入公共类
        for i in range(0, self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        client_subset = [[] for i in range(0, self.client_num)]
        client_mask = [[] for i in range(0, self.client_num)]

        # 计算每任务类别数 
        classes_per_client = len(class_private[0])
        class_every_task = int(classes_per_client / self.task_num)

        # Dirichlet 分布
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            while (a < 0.05).any(): 
                a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]

        for i in range(0, self.client_num):
            for j in range(0, self.task_num):
                index = []
                start = j * class_every_task
                end = start + class_every_task
                
                if start >= len(class_private[i]): break
                
                class_this_task = class_private[i][start : end]
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
        root_path = './local_datasets/tiny-imagenet-200'
        # 测试集也开启 download=True (虽然random_split可能已经下载了，但保险起见)
        self.dataset = TinyImageNet(root=root_path, train=False, download=True)
        testset = self.dataset

        class_counts = torch.zeros(200)
        class_label = [[] for _ in range(200)]
        
        for j, label in enumerate(testset.targets):
            class_counts[label] += 1
            class_label[label].append(j)

        surro_index = []
        test_index = []
        
        for i in tqdm(range(200), desc="Processing Test Data"):
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
        target = self.targets[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 增加一个简单的错误处理，防止坏图导致训练中断
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (64, 64))

        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.indices)