import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # 引入进度条库

class Proxy_Data(Dataset):
    def __init__(self):
        super(Proxy_Data, self).__init__()
        self.TestData = []
        self.TestLabels = []

    def getTestData(self, new_set, new_set_label):
        self.TestData = []
        self.TestLabels = []
        if len(new_set) > 0:
            for i in range(len(new_set)): # for each class
                imgs = new_set[i]
                labels = np.full(len(imgs), new_set_label[i])
                self.TestData.extend(imgs)
                self.TestLabels.extend(labels)

    def __getitem__(self, index):
        img = self.TestData[index]
        if img.ndim == 3 and img.shape[0] != 3:
             img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).float(), torch.tensor(self.TestLabels[index]).long()

    def __len__(self):
        return len(self.TestData)

class proxyServer:
    def __init__(self, device, learning_rate, numclass, feature_extractor, encode_model, test_transform=None):
        # === 修改 1: 减少迭代次数，加快速度 (测试时可设为 10-20，实际运行时建议 50-100) ===
        self.Iteration = 10 
        self.learning_rate = learning_rate
        self.encode_model = encode_model
        self.monitor_dataset = Proxy_Data()
        self.new_set = []
        self.new_set_label = []
        self.numclass = numclass
        self.device = device
        # === 修改 2: 限制每个类重构的图片数量 ===
        self.num_image = 5 
        self.pool_grad = None
        self.model = None 

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
            self.monitor_dataset.getTestData(self.new_set, self.new_set_label)

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            # 简单推断：根据梯度最大值对应的index推断label
            # w_single[-2] 是权重的梯度，通常 shape 为 [num_classes, hidden_dim]
            # 对 hidden_dim 求和，找最大的梯度反方向（梯度下降）或正方向
            # 这里沿用之前的逻辑
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().cpu().item()
            pool_label.append(pred)
        return pool_label

    def reconstruction(self):
        self.new_set, self.new_set_label = [], []
        pool_label = self.gradient2label()
        
        unique_labels = list(set(pool_label))
        print(f"    Proxy Server: Found labels {unique_labels} from gradients.")

        # === 修改 3: 添加外层进度条 ===
        for label_i in tqdm(unique_labels, desc="Reconstructing Classes"):
            grad_indices = [i for i, x in enumerate(pool_label) if x == label_i]
            
            augmentation = []
            # 限制每个类处理的梯度数量，防止过多
            process_indices = grad_indices[:self.num_image]

            for j_idx, j in enumerate(process_indices):
                grad_truth_temp = self.pool_grad[j]

                # 初始化 dummy data
                dummy_data = torch.randn((1, 3, 32, 32)).to(self.device).requires_grad_(True)
                label_pred = torch.Tensor([label_i]).long().to(self.device)

                optimizer = torch.optim.LBFGS([dummy_data, ], lr=0.1)
                criterion = nn.CrossEntropyLoss().to(self.device)

                recon_model = copy.deepcopy(self.encode_model)
                recon_model.to(self.device)
                recon_model.eval()
                
                # === 修改 4: 内部循环简化，去掉不必要的打印 ===
                for iters in range(self.Iteration):
                    def closure():
                        optimizer.zero_grad()
                        pred = recon_model(dummy_data)
                        dummy_loss = criterion(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                        grad_diff = 0
                        # 仅计算最后两层（FC层）的梯度差异以节省计算
                        # 假设 w_single 包含了所有层，通常只匹配最后几层效果更好且更快
                        # 这里匹配所有层
                        for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                            grad_diff += ((gx - gy) ** 2).sum()
                        
                        grad_diff.backward()
                        return grad_diff

                    optimizer.step(closure)
                
                # 收集生成的图片
                dummy_data_np = dummy_data.detach().cpu().numpy().squeeze(0) # [3, 32, 32]
                augmentation.append(dummy_data_np)
            
            if len(augmentation) > 0:
                self.new_set.append(np.array(augmentation))
                self.new_set_label.append(label_i)