import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import numpy as np
from PIL import Image
from ResNet import resnet18_cbam

def get_one_hot(target, num_class, device):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot

class Client(object):
    def __init__(self, dataset_name, lr, device, task_num, id, epoch_local, subset, batch_size, mask, num_classes, memory_size, encode_model):
        self.id = id
        self.dataset_name = dataset_name
        self.learning_rate = lr
        self.device = device
        self.task_num = task_num
        self.epoch_local = epoch_local
        # self.subset 可能是 Dataset 对象，也可能是 Dataset 对象的 list
        self.subset = subset 
        self.batch_size = batch_size
        self.mask = mask 
        self.num_classes = num_classes
        
        # GLFC components
        self.memory_size = memory_size
        self.exemplar_set = [] 
        self.exemplar_label_set = []
        self.encode_model = encode_model
        
        self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
        self.old_model = None
        
        self.train_loader = None
        self.current_classes = []
        self.learned_classes = []
        
        # 预处理：构建类索引映射
        # 结构: {label: [(dataset_index, sample_index), ...]}
        self.class_indices = {}
        print(f"Client {self.id}: Indexing data...")
        
        # 1. 统一将 subset 处理为 list of Datasets
        if isinstance(subset, list):
            # 检查列表内容是否为 Dataset (CIFAR100 情况)
            if len(subset) > 0 and (isinstance(subset[0], Dataset) or isinstance(subset[0], Subset)):
                self.datasets = subset
            else:
                # 可能是数据列表本身(不太可能，但在CIFAR10某些写法下可能)
                self.datasets = [subset]
        else:
            # 单个 Dataset (CIFAR10 情况)
            self.datasets = [subset]

        # 2. 遍历所有数据集构建索引
        for d_idx, dataset in enumerate(self.datasets):
            # 尝试直接获取 targets/labels 以加速
            targets = None
            if hasattr(dataset, 'targets'):
                targets = dataset.targets
            elif hasattr(dataset, 'labels'):
                targets = dataset.labels
            elif isinstance(dataset, Subset):
                # 如果是 Subset，尝试访问底层 dataset 的 targets
                if hasattr(dataset.dataset, 'targets'):
                     # Subset 的 targets 需要根据 indices 映射
                     full_targets = dataset.dataset.targets
                     targets = [full_targets[i] for i in dataset.indices]
                elif hasattr(dataset.dataset, 'labels'):
                     full_labels = dataset.dataset.labels
                     targets = [full_labels[i] for i in dataset.indices]

            if targets is not None:
                # 快速路径
                for s_idx, label in enumerate(targets):
                    label = int(label)
                    if label not in self.class_indices:
                        self.class_indices[label] = []
                    self.class_indices[label].append((d_idx, s_idx))
            else:
                # 慢速路径：逐个遍历 __getitem__
                # 仅在无法获取 targets 属性时使用
                for s_idx in range(len(dataset)):
                    item = dataset[s_idx]
                    # 兼容不同返回值: (img, label) 或 (idx, img, label)
                    if isinstance(item, (tuple, list)):
                        if len(item) >= 2:
                            label = item[-1] # 假设最后一个是 label
                        else:
                            label = item[0]
                    else:
                        label = item
                    
                    try:
                        label = int(label)
                        if label not in self.class_indices:
                            self.class_indices[label] = []
                        self.class_indices[label].append((d_idx, s_idx))
                    except:
                        continue

    def recive_global_models(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def get_image_class(self, label):
        # 返回指定类别的所有图片数据
        indices = self.class_indices.get(label, [])
        images = []
        
        for (d_idx, s_idx) in indices:
            dataset = self.datasets[d_idx]
            item = dataset[s_idx]
            
            # 提取图像
            img = None
            if isinstance(item, (tuple, list)):
                if len(item) == 2:
                    img = item[0] # (img, label)
                elif len(item) == 3:
                    img = item[1] # (idx, img, label)
                else:
                    img = item[0]
            else:
                img = item

            if img is None: continue

            # 转为 Numpy
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            elif isinstance(img, Image.Image):
                img = np.array(img)
            
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            images.append(img)
            
        return np.array(images)

    def train_glfc(self, task_id, model_old_global=None):
        self.current_classes = self.mask[task_id]
        
        # 1. 准备数据: 当前任务数据 + 样本库
        train_data = []
        train_labels = []
        
        # 添加当前任务数据
        for cls in self.current_classes:
            imgs = self.get_image_class(cls)
            if len(imgs) > 0:
                train_data.append(imgs)
                train_labels.append(np.full(len(imgs), cls))
        
        # 添加样本库数据
        if len(self.exemplar_set) > 0:
            for i, ex_imgs in enumerate(self.exemplar_set):
                if len(ex_imgs) > 0:
                    train_data.append(ex_imgs)
                    train_labels.append(np.full(len(ex_imgs), self.exemplar_label_set[i]))

        if len(train_data) > 0:
            train_data = np.concatenate(train_data)
            train_labels = np.concatenate(train_labels)
        else:
            print(f"Client {self.id} has no data for task {task_id}")
            return self.model.state_dict(), None

        # 简化的Dataset
        class GLFCDataset(Dataset):
            def __init__(self, x, y, device):
                self.x = x
                self.y = y
                self.device = device
            def __len__(self): return len(self.x)
            def __getitem__(self, idx):
                img = self.x[idx]
                # HWC -> CHW check
                # 只有当维度是3且通道不在第一位时才转置
                # CIFAR100 splitter返回的是 tensor (CHW)，转numpy后还是 CHW
                # 如果 img 是 (3, 32, 32)，img.shape[0] == 3，不转置 (正确)
                # 如果 img 是 (32, 32, 3)，img.shape[0] != 3，转置 (正确)
                if img.ndim == 3 and img.shape[0] != 3: 
                     img = np.transpose(img, (2, 0, 1))
                return torch.tensor(img).float(), torch.tensor(self.y[idx]).long()

        self.train_loader = DataLoader(GLFCDataset(train_data, train_labels, self.device), 
                                       batch_size=self.batch_size, shuffle=True)

        # 2. 设置旧模型
        if model_old_global is not None:
            self.old_model = copy.deepcopy(model_old_global)
            self.old_model.to(self.device)
            self.old_model.eval()

        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # 3. 训练循环
        for epoch in range(self.epoch_local):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                loss = self._compute_loss(images, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 4. 计算原型梯度
        proto_grad = self.proto_grad_sharing()
        
        return self.model.state_dict(), proto_grad

    def _compute_loss(self, imgs, label):
        _, output = self.model(imgs) 
        target = get_one_hot(label, self.num_classes, self.device)
        
        if self.old_model is None:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))
            return loss_cur
        else:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))
            
            with torch.no_grad():
                _, old_output = self.old_model(imgs)
                old_target = torch.sigmoid(old_output)
            
            loss_old = F.binary_cross_entropy_with_logits(output, old_target)
            return 0.5 * loss_cur + 0.5 * loss_old

    def efficient_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        N, C = pred.size(0), pred.size(1)
        class_mask = torch.zeros(N, C).to(self.device)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        
        target = get_one_hot(label, self.num_classes, self.device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        w = g.clone().fill_(1.)
        return w

    def update_new_set(self, task_id):
        self.learned_classes.extend(self.current_classes)
        self.learned_classes = list(set(self.learned_classes))
        
        if len(self.learned_classes) == 0: return

        m = int(self.memory_size / len(self.learned_classes))
        self._reduce_exemplar_sets(m)
        
        self.model.eval()
        for cls in self.current_classes:
            images = self.get_image_class(cls)
            if len(images) > 0:
                self._construct_exemplar_set(images, cls, m)
        self.model.train()

    def _construct_exemplar_set(self, images, label, m):
        if len(images) == 0: return
        
        tensor_images = []
        for img in images:
            # 确保输入到 model 的是 Tensor (N, C, H, W)
            if img.ndim == 3 and img.shape[0] != 3:
                img = np.transpose(img, (2, 0, 1))
            tensor_images.append(torch.tensor(img).float())
        
        if len(tensor_images) == 0: return
        
        tensor_input = torch.stack(tensor_images).to(self.device)
        
        with torch.no_grad():
            features, _ = self.model(tensor_input)
            features = F.normalize(features).cpu().numpy()
            
        class_mean = np.mean(features, axis=0)
        
        exemplar_indices = []
        now_class_mean = np.zeros((1, 512)) 
        
        select_num = min(m, len(images))
        
        for i in range(select_num):
            x = class_mean - (now_class_mean + features) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += features[index]
            exemplar_indices.append(index)
            features[index] = features[index] + 10000 

        self.exemplar_set.append(images[exemplar_indices])
        self.exemplar_label_set.append(label)

    def _reduce_exemplar_sets(self, m):
        for i in range(len(self.exemplar_set)):
            self.exemplar_set[i] = self.exemplar_set[i][:m]

    def proto_grad_sharing(self):
        proto_grad = []
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        for i, cls in enumerate(self.current_classes):
            images = self.get_image_class(cls)
            if len(images) == 0: continue
            
            if cls in self.exemplar_label_set:
                idx = self.exemplar_label_set.index(cls)
                if len(self.exemplar_set[idx]) > 0:
                    proto_img = self.exemplar_set[idx][0]
                else:
                    proto_img = images[0]
            else:
                proto_img = images[0]

            if proto_img.ndim == 3 and proto_img.shape[0] != 3:
                 proto_img = np.transpose(proto_img, (2, 0, 1))
            
            data = torch.tensor(proto_img).float().to(self.device)
            data = data.unsqueeze(0).requires_grad_(True)
            label = torch.tensor([cls]).long().to(self.device)
            
            self.encode_model.to(self.device)
            self.encode_model.eval() 
            
            outputs = self.encode_model(data)
            loss = criterion(outputs, label)
            
            dy_dx = torch.autograd.grad(loss, self.encode_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append(original_dy_dx)
            
        return proto_grad




# # 计算通信开销


# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset, Subset
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# from ResNet import resnet18_cbam

# def get_one_hot(target, num_class, device):
#     one_hot = torch.zeros(target.shape[0], num_class).to(device)
#     one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
#     return one_hot

# class Client(object):
#     def __init__(self, dataset_name, lr, device, task_num, id, epoch_local, subset, batch_size, mask, num_classes, memory_size, encode_model):
#         self.id = id
#         self.dataset_name = dataset_name
#         self.learning_rate = lr
#         self.device = device
#         self.task_num = task_num
#         self.epoch_local = epoch_local
#         # self.subset 可能是 Dataset 对象，也可能是 Dataset 对象的 list
#         self.subset = subset 
#         self.batch_size = batch_size
#         self.mask = mask 
#         self.num_classes = num_classes
        
#         # GLFC components
#         self.memory_size = memory_size
#         self.exemplar_set = [] 
#         self.exemplar_label_set = []
#         self.encode_model = encode_model
        
#         self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes)
#         self.old_model = None
        
#         self.train_loader = None
#         self.current_classes = []
#         self.learned_classes = []

#         # 用于统计每轮训练的样本数（计算FLOPs用）
#         self.num_train_samples_this_round = 0
        
#         # 预处理：构建类索引映射
#         # 结构: {label: [(dataset_index, sample_index), ...]}
#         self.class_indices = {}
#         print(f"Client {self.id}: Indexing data...")
        
#         # 1. 统一将 subset 处理为 list of Datasets
#         if isinstance(subset, list):
#             # 检查列表内容是否为 Dataset (CIFAR100 情况)
#             if len(subset) > 0 and (isinstance(subset[0], Dataset) or isinstance(subset[0], Subset)):
#                 self.datasets = subset
#             else:
#                 # 可能是数据列表本身(不太可能，但在CIFAR10某些写法下可能)
#                 self.datasets = [subset]
#         else:
#             # 单个 Dataset (CIFAR10 情况)
#             self.datasets = [subset]

#         # 2. 遍历所有数据集构建索引
#         for d_idx, dataset in enumerate(self.datasets):
#             # 尝试直接获取 targets/labels 以加速
#             targets = None
#             if hasattr(dataset, 'targets'):
#                 targets = dataset.targets
#             elif hasattr(dataset, 'labels'):
#                 targets = dataset.labels
#             elif isinstance(dataset, Subset):
#                 # 如果是 Subset，尝试访问底层 dataset 的 targets
#                 if hasattr(dataset.dataset, 'targets'):
#                      # Subset 的 targets 需要根据 indices 映射
#                      full_targets = dataset.dataset.targets
#                      targets = [full_targets[i] for i in dataset.indices]
#                 elif hasattr(dataset.dataset, 'labels'):
#                      full_labels = dataset.dataset.labels
#                      targets = [full_labels[i] for i in dataset.indices]

#             if targets is not None:
#                 # 快速路径
#                 for s_idx, label in enumerate(targets):
#                     label = int(label)
#                     if label not in self.class_indices:
#                         self.class_indices[label] = []
#                     self.class_indices[label].append((d_idx, s_idx))
#             else:
#                 # 慢速路径：逐个遍历 __getitem__
#                 # 仅在无法获取 targets 属性时使用
#                 for s_idx in range(len(dataset)):
#                     item = dataset[s_idx]
#                     # 兼容不同返回值: (img, label) 或 (idx, img, label)
#                     if isinstance(item, (tuple, list)):
#                         if len(item) >= 2:
#                             label = item[-1] # 假设最后一个是 label
#                         else:
#                             label = item[0]
#                     else:
#                         label = item
                    
#                     try:
#                         label = int(label)
#                         if label not in self.class_indices:
#                             self.class_indices[label] = []
#                         self.class_indices[label].append((d_idx, s_idx))
#                     except:
#                         continue

#     def recive_global_models(self, global_model):
#         self.model.load_state_dict(global_model.state_dict())

#     def get_image_class(self, label):
#         # 返回指定类别的所有图片数据
#         indices = self.class_indices.get(label, [])
#         images = []
        
#         for (d_idx, s_idx) in indices:
#             dataset = self.datasets[d_idx]
#             item = dataset[s_idx]
            
#             # 提取图像
#             img = None
#             if isinstance(item, (tuple, list)):
#                 if len(item) == 2:
#                     img = item[0] # (img, label)
#                 elif len(item) == 3:
#                     img = item[1] # (idx, img, label)
#                 else:
#                     img = item[0]
#             else:
#                 img = item

#             if img is None: continue

#             # 转为 Numpy
#             if isinstance(img, torch.Tensor):
#                 img = img.cpu().numpy()
#             elif isinstance(img, Image.Image):
#                 img = np.array(img)
            
#             if not isinstance(img, np.ndarray):
#                 img = np.array(img)
            
#             images.append(img)
            
#         return np.array(images)

#     def train_glfc(self, task_id, model_old_global=None):
#         self.current_classes = self.mask[task_id]
        
#         # 1. 准备数据: 当前任务数据 + 样本库
#         train_data = []
#         train_labels = []
        
#         # 添加当前任务数据
#         for cls in self.current_classes:
#             imgs = self.get_image_class(cls)
#             if len(imgs) > 0:
#                 train_data.append(imgs)
#                 train_labels.append(np.full(len(imgs), cls))
        
#         # 添加样本库数据
#         if len(self.exemplar_set) > 0:
#             for i, ex_imgs in enumerate(self.exemplar_set):
#                 if len(ex_imgs) > 0:
#                     train_data.append(ex_imgs)
#                     train_labels.append(np.full(len(ex_imgs), self.exemplar_label_set[i]))

#         if len(train_data) > 0:
#             train_data = np.concatenate(train_data)
#             train_labels = np.concatenate(train_labels)
#         else:
#             print(f"Client {self.id} has no data for task {task_id}")
#             return self.model.state_dict(), None

#         # 简化的Dataset
#         class GLFCDataset(Dataset):
#             def __init__(self, x, y, device):
#                 self.x = x
#                 self.y = y
#                 self.device = device
#             def __len__(self): return len(self.x)
#             def __getitem__(self, idx):
#                 img = self.x[idx]
#                 # HWC -> CHW check
#                 # 只有当维度是3且通道不在第一位时才转置
#                 # CIFAR100 splitter返回的是 tensor (CHW)，转numpy后还是 CHW
#                 # 如果 img 是 (3, 32, 32)，img.shape[0] == 3，不转置 (正确)
#                 # 如果 img 是 (32, 32, 3)，img.shape[0] != 3，转置 (正确)
#                 if img.ndim == 3 and img.shape[0] != 3: 
#                      img = np.transpose(img, (2, 0, 1))
#                 return torch.tensor(img).float(), torch.tensor(self.y[idx]).long()

#         self.train_loader = DataLoader(GLFCDataset(train_data, train_labels, self.device), 
#                                        batch_size=self.batch_size, shuffle=True)

#         # 记录本轮训练样本数
#         self.num_train_samples_this_round = len(train_data)

#         # 2. 设置旧模型
#         if model_old_global is not None:
#             self.old_model = copy.deepcopy(model_old_global)
#             self.old_model.to(self.device)
#             self.old_model.eval()

#         self.model.to(self.device)
#         self.model.train()
#         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

#         # 3. 训练循环
#         for epoch in range(self.epoch_local):
#             for images, labels in self.train_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 loss = self._compute_loss(images, labels)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
        
#         # 4. 计算原型梯度
#         proto_grad = self.proto_grad_sharing()
        
#         return self.model.state_dict(), proto_grad

#     def _compute_loss(self, imgs, label):
#         _, output = self.model(imgs) 
#         target = get_one_hot(label, self.num_classes, self.device)
        
#         if self.old_model is None:
#             w = self.efficient_old_class_weight(output, label)
#             loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))
#             return loss_cur
#         else:
#             w = self.efficient_old_class_weight(output, label)
#             loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))
            
#             with torch.no_grad():
#                 _, old_output = self.old_model(imgs)
#                 old_target = torch.sigmoid(old_output)
            
#             loss_old = F.binary_cross_entropy_with_logits(output, old_target)
#             return 0.5 * loss_cur + 0.5 * loss_old

#     def efficient_old_class_weight(self, output, label):
#         pred = torch.sigmoid(output)
#         N, C = pred.size(0), pred.size(1)
#         class_mask = torch.zeros(N, C).to(self.device)
#         ids = label.view(-1, 1)
#         class_mask.scatter_(1, ids, 1.)
        
#         target = get_one_hot(label, self.num_classes, self.device)
#         g = torch.abs(pred.detach() - target)
#         g = (g * class_mask).sum(1).view(-1, 1)

#         w = g.clone().fill_(1.)
#         return w

#     def update_new_set(self, task_id):
#         self.learned_classes.extend(self.current_classes)
#         self.learned_classes = list(set(self.learned_classes))
        
#         if len(self.learned_classes) == 0: return

#         m = int(self.memory_size / len(self.learned_classes))
#         self._reduce_exemplar_sets(m)
        
#         self.model.eval()
#         for cls in self.current_classes:
#             images = self.get_image_class(cls)
#             if len(images) > 0:
#                 self._construct_exemplar_set(images, cls, m)
#         self.model.train()

#     def _construct_exemplar_set(self, images, label, m):
#         if len(images) == 0: return
        
#         tensor_images = []
#         for img in images:
#             # 确保输入到 model 的是 Tensor (N, C, H, W)
#             if img.ndim == 3 and img.shape[0] != 3:
#                 img = np.transpose(img, (2, 0, 1))
#             tensor_images.append(torch.tensor(img).float())
        
#         if len(tensor_images) == 0: return
        
#         tensor_input = torch.stack(tensor_images).to(self.device)
        
#         with torch.no_grad():
#             features, _ = self.model(tensor_input)
#             features = F.normalize(features).cpu().numpy()
            
#         class_mean = np.mean(features, axis=0)
        
#         exemplar_indices = []
#         now_class_mean = np.zeros((1, 512)) 
        
#         select_num = min(m, len(images))
        
#         for i in range(select_num):
#             x = class_mean - (now_class_mean + features) / (i + 1)
#             x = np.linalg.norm(x, axis=1)
#             index = np.argmin(x)
#             now_class_mean += features[index]
#             exemplar_indices.append(index)
#             features[index] = features[index] + 10000 

#         self.exemplar_set.append(images[exemplar_indices])
#         self.exemplar_label_set.append(label)

#     def _reduce_exemplar_sets(self, m):
#         for i in range(len(self.exemplar_set)):
#             self.exemplar_set[i] = self.exemplar_set[i][:m]

#     def proto_grad_sharing(self):
#         proto_grad = []
#         criterion = nn.CrossEntropyLoss().to(self.device)
        
#         for i, cls in enumerate(self.current_classes):
#             images = self.get_image_class(cls)
#             if len(images) == 0: continue
            
#             if cls in self.exemplar_label_set:
#                 idx = self.exemplar_label_set.index(cls)
#                 if len(self.exemplar_set[idx]) > 0:
#                     proto_img = self.exemplar_set[idx][0]
#                 else:
#                     proto_img = images[0]
#             else:
#                 proto_img = images[0]

#             if proto_img.ndim == 3 and proto_img.shape[0] != 3:
#                  proto_img = np.transpose(proto_img, (2, 0, 1))
            
#             data = torch.tensor(proto_img).float().to(self.device)
#             data = data.unsqueeze(0).requires_grad_(True)
#             label = torch.tensor([cls]).long().to(self.device)
            
#             self.encode_model.to(self.device)
#             self.encode_model.eval() 
            
#             outputs = self.encode_model(data)
#             loss = criterion(outputs, label)
            
#             dy_dx = torch.autograd.grad(loss, self.encode_model.parameters())
#             original_dy_dx = list((_.detach().clone() for _ in dy_dx))
#             proto_grad.append(original_dy_dx)
            
#         return proto_grad