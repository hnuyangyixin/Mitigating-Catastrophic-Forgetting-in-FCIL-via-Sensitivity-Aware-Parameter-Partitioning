"""
Client implementation for FCIL-LD
Based on the paper: "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning 
with a Low-Dimensional Projection Layer" (CVPR 2025 Workshop)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import resnet18_cbam
import random
from copy import deepcopy
import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision import transforms


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Eq. 6 in paper).
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: Projected features of shape (batch_size, projection_dim)
            labels: Ground truth labels of shape (batch_size,)
        Returns:
            SupCon loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        
        # Compute log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for subsequent tasks (Eq. 11 in paper).
    Aligns current model's projection with previous model's projection.
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_current, z_prev):
        """
        Args:
            z_current: Current model's projections (batch_size, projection_dim)
            z_prev: Previous model's projections (batch_size, projection_dim)
        Returns:
            Contrastive loss
        """
        batch_size = z_current.shape[0]
        device = z_current.device
        
        # Normalize
        z_current = F.normalize(z_current, dim=1)
        z_prev = F.normalize(z_prev, dim=1)
        
        # Compute similarity between each current sample and all previous samples
        similarity = torch.matmul(z_current, z_prev.T) / self.temperature
        
        # Diagonal elements are the positive pairs (same sample, different models)
        # Off-diagonal elements are negative pairs
        
        # Loss: -log(exp(sim_ii) / sum_j(exp(sim_ij)))
        exp_sim = torch.exp(similarity)
        loss = -torch.log(torch.diag(exp_sim) / exp_sim.sum(dim=1) + 1e-8)
        
        return loss.mean()


class Client(object):
    def __init__(self, dataset, lr, device, task_num, id, epoch_local, subset, batch_size, mask, num_classes):
        self.id = id
        self.task_num = task_num
        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.device = device
        self.dataset = dataset
        self.mask = mask
        self.num_classes = num_classes

        # Label mapping
        self.label_mapping = {}
        self.reverse_mapping = {}

        self.test_loader = []
        self.train_dataset = subset
        self.train_data = None

        # Model with projection layer
        self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes, projection_dim=128)
        self.global_model = None
        self.prev_model = None  # Previous task's global model (for KD)
        
        # Generator for replay
        self.generator = None
        
        # FCIL-LD hyperparameters
        self.tau = 0.1  # Temperature for contrastive loss
        self.alpha_KD = 1.0   # Weight for knowledge distillation loss
        self.alpha_con = 0.5  # Weight for contrastive loss
        self.alpha_FT = 0.5   # Weight for fine-tuning loss
        
        # Loss functions
        self.supcon_loss = SupConLoss(temperature=self.tau)
        self.contrastive_loss = ContrastiveLoss(temperature=self.tau)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Track current task
        self.current_task_id = -1
        
        # Data augmentation for SupCon
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        ])

    def recive_global_models(self, global_model, prev_model=None, generator=None):
        """Receive global model, previous model, and generator from server"""
        self.model.load_state_dict(global_model.state_dict())
        
        if prev_model is not None:
            self.prev_model = deepcopy(prev_model)
            self.prev_model.eval()
            for param in self.prev_model.parameters():
                param.requires_grad = False
        
        if generator is not None:
            self.generator = generator
            self.generator.eval()

    def get_data(self, task_id):
        if task_id % 5 == 0:
            self.task_id = task_id // 5
            self.current_class = self.mask[self.task_id]
            self.trainset = self.train_dataset[self.task_id]
            if self.task_id == 0:
                self.first_task = self.current_class

        print(f'{self.id} client, {task_id} task contains {self.current_class} classes, '
              f'there are {len(self.current_class)} classes')

        self.label_mapping = {cls: idx for idx, cls in enumerate(self.current_class)}
        self.reverse_mapping = {idx: cls for idx, cls in enumerate(self.current_class)}

        traindata, testdata = random_split(
            self.trainset,
            [int(len(self.trainset) * 0.8), len(self.trainset) - int(len(self.trainset) * 0.8)]
        )
        testdata = deepcopy(testdata)
        testloader = DataLoader(testdata, shuffle=True, batch_size=self.batch_size)

        self.test_loader.append(testloader)
        self.train_data = traindata
        self.current_task_id = self.task_id

    def _augment_batch(self, images):
        """Apply data augmentation to a batch of images for SupCon"""
        augmented = []
        for img in images:
            # Convert tensor to PIL, augment, convert back
            # For simplicity, we'll just add noise as augmentation
            aug_img = img + 0.1 * torch.randn_like(img)
            augmented.append(aug_img)
        return torch.stack(augmented)

    def train_initial_task(self):
        """
        Train on the initial task using SupCon loss (Section 3.2.1 in paper).
        """
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        
        self.model.to(self.device)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, 
                                    weight_decay=1e-5, momentum=0.9)
        
        print(f'Client {self.id} training initial task with SupCon loss')
        
        for epoch in range(self.epoch_local):
            epoch_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Create augmented version
                inputs_aug = self._augment_batch(inputs).to(self.device)
                
                # Concatenate original and augmented
                inputs_combined = torch.cat([inputs, inputs_aug], dim=0)
                labels_combined = torch.cat([labels, labels], dim=0)
                
                optimizer.zero_grad()
                
                # Forward pass
                features = self.model.forward_features(inputs_combined)
                projections = self.model.forward_projection(features)
                
                # SupCon loss on projection layer
                loss_supcon = self.supcon_loss(projections, labels_combined)
                
                # Classification head loss (with gradient blocked from feature extractor)
                # Only update classification head with CE loss
                features_detached = features.detach()
                logits = self.model.forward_classifier(features_detached)
                loss_head = self.ce_loss(logits, labels_combined)
                
                # Total loss (Eq. 8)
                loss = loss_head + self.tau * loss_supcon
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch [{epoch+1}/{self.epoch_local}], Loss: {epoch_loss/len(self.train_loader):.4f}')

    def train_subsequent_task(self):
        """
        Train on subsequent tasks using the novel loss function (Section 3.2.2 in paper).
        """
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        
        self.model.to(self.device)
        self.model.train()
        
        if self.prev_model is not None:
            self.prev_model.to(self.device)
            self.prev_model.eval()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=1e-5, momentum=0.9)
        
        print(f'Client {self.id} training subsequent task with FCIL-LD loss')
        
        # Generate synthetic samples if generator is available
        synthetic_data = None
        if self.generator is not None:
            self.generator.to(self.device)
            self.generator.eval()
            with torch.no_grad():
                num_synthetic = min(len(self.train_data), 500)  # Limit synthetic samples
                z = torch.randn(num_synthetic, self.generator.latent_dim).to(self.device)
                synthetic_images = self.generator(z)
                
                # Get pseudo-labels from previous model
                if self.prev_model is not None:
                    _, syn_logits = self.prev_model(synthetic_images)
                    synthetic_labels = syn_logits.argmax(dim=1)
                else:
                    synthetic_labels = z[:, :self.num_classes].argmax(dim=1)
                
                synthetic_data = (synthetic_images, synthetic_labels)
        
        for epoch in range(self.epoch_local):
            epoch_loss = 0.0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                total_loss = 0.0
                
                # === Component 1: Cross-entropy loss on new data (Eq. 9) ===
                features = self.model.forward_features(inputs)
                logits = self.model.forward_classifier(features)
                loss_ce = self.ce_loss(logits, labels)
                total_loss += loss_ce
                
                # === Component 2: Knowledge Distillation loss on synthetic samples (Eq. 10) ===
                if synthetic_data is not None and self.prev_model is not None:
                    # Sample a mini-batch from synthetic data
                    syn_idx = torch.randint(0, synthetic_data[0].size(0), (min(inputs.size(0), synthetic_data[0].size(0)),))
                    syn_images = synthetic_data[0][syn_idx]
                    
                    # Get features from current model
                    features_syn_current = self.model.forward_features(syn_images)
                    
                    # Get features from previous model (frozen)
                    with torch.no_grad():
                        features_syn_prev = self.prev_model.forward_features(syn_images)
                    
                    # Feature-level KD: compare outputs through previous model's classifier
                    with torch.no_grad():
                        # Get target: previous model's representation
                        target_logits = self.prev_model.forward_classifier(features_syn_prev)
                    
                    # Current model's features through previous classifier for comparison
                    current_through_prev = self.prev_model.forward_classifier(features_syn_current)
                    
                    loss_kd = self.mse_loss(current_through_prev, target_logits)
                    total_loss += self.alpha_KD * loss_kd
                
                # === Component 3: Contrastive loss in projection layer (Eq. 11) ===
                if self.prev_model is not None:
                    # Project features through both models
                    z_current = self.model.forward_projection(features)
                    
                    with torch.no_grad():
                        features_prev = self.prev_model.forward_features(inputs)
                        z_prev = self.prev_model.forward_projection(features_prev)
                    
                    loss_con = self.contrastive_loss(z_current, z_prev)
                    total_loss += self.alpha_con * loss_con
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            # === Component 4: Fine-tuning loss (Eq. 12) ===
            # Fine-tune classification head with combined real and synthetic data
            if synthetic_data is not None:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.fc2.parameters():
                    param.requires_grad = True
                
                optimizer_ft = torch.optim.SGD(self.model.fc2.parameters(), lr=self.learning_rate * 0.1)
                
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Combine with synthetic
                    syn_idx = torch.randint(0, synthetic_data[0].size(0), (min(inputs.size(0), synthetic_data[0].size(0)),))
                    syn_images = synthetic_data[0][syn_idx]
                    syn_labels = synthetic_data[1][syn_idx]
                    
                    combined_inputs = torch.cat([inputs, syn_images], dim=0)
                    combined_labels = torch.cat([labels, syn_labels], dim=0)
                    
                    optimizer_ft.zero_grad()
                    
                    with torch.no_grad():
                        features_combined = self.model.forward_features(combined_inputs)
                    logits_combined = self.model.forward_classifier(features_combined)
                    
                    loss_ft = self.ce_loss(logits_combined, combined_labels)
                    loss_ft.backward()
                    optimizer_ft.step()
                
                # Re-enable all gradients
                for param in self.model.parameters():
                    param.requires_grad = True
            
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch [{epoch+1}/{self.epoch_local}], Loss: {epoch_loss/len(self.train_loader):.4f}')

    def train(self, task_id):
        self.get_data(task_id)
        
        if self.task_id == 0:
            # Initial task: use SupCon loss
            self.train_initial_task()
        else:
            # Subsequent tasks: use FCIL-LD loss
            self.train_subsequent_task()
        
        print(f'Client {self.id} finished training')
        self.final_state = copy.deepcopy(self.model.state_dict())
        self.evaluate_local_models(task_id)

    def evaluate_local_models(self, index):
        self.model.load_state_dict(self.final_state)
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader[index]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, logits = self.model(inputs)
                
                _, pred_idx = torch.max(logits, 1)
                predicted_labels = pred_idx
                
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        acc = correct / total * 100
        print(f'Client {self.id}\'s local model on {index} task acc is {acc:.2f}%')
        return acc




# # 计算与通信开销


# """
# Client implementation for FCIL-LD
# Based on the paper: "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning 
# with a Low-Dimensional Projection Layer" (CVPR 2025 Workshop)
# """

# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ResNet import resnet18_cbam
# import random
# from copy import deepcopy
# import numpy as np
# from torch.utils.data import random_split, DataLoader, TensorDataset
# from torchvision import transforms


# class SupConLoss(nn.Module):
#     """
#     Supervised Contrastive Loss (Eq. 6 in paper).
#     """
#     def __init__(self, temperature=0.1):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
        
#     def forward(self, features, labels):
#         """
#         Args:
#             features: Projected features of shape (batch_size, projection_dim)
#             labels: Ground truth labels of shape (batch_size,)
#         Returns:
#             SupCon loss
#         """
#         device = features.device
#         batch_size = features.shape[0]
        
#         # Normalize features
#         features = F.normalize(features, dim=1)
        
#         # Compute similarity matrix
#         similarity = torch.matmul(features, features.T) / self.temperature
        
#         # Create mask for positive pairs (same label)
#         labels = labels.view(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)
        
#         # Remove diagonal (self-similarity)
#         logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
#         mask = mask * logits_mask
        
#         # Compute log-softmax
#         exp_logits = torch.exp(similarity) * logits_mask
#         log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
#         # Compute mean of log-likelihood over positive pairs
#         mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
#         # Loss is negative log-likelihood
#         loss = -mean_log_prob_pos.mean()
        
#         return loss


# class ContrastiveLoss(nn.Module):
#     """
#     Contrastive loss for subsequent tasks (Eq. 11 in paper).
#     Aligns current model's projection with previous model's projection.
#     """
#     def __init__(self, temperature=0.1):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature
        
#     def forward(self, z_current, z_prev):
#         """
#         Args:
#             z_current: Current model's projections (batch_size, projection_dim)
#             z_prev: Previous model's projections (batch_size, projection_dim)
#         Returns:
#             Contrastive loss
#         """
#         batch_size = z_current.shape[0]
#         device = z_current.device
        
#         # Normalize
#         z_current = F.normalize(z_current, dim=1)
#         z_prev = F.normalize(z_prev, dim=1)
        
#         # Compute similarity between each current sample and all previous samples
#         similarity = torch.matmul(z_current, z_prev.T) / self.temperature
        
#         # Diagonal elements are the positive pairs (same sample, different models)
#         # Off-diagonal elements are negative pairs
        
#         # Loss: -log(exp(sim_ii) / sum_j(exp(sim_ij)))
#         exp_sim = torch.exp(similarity)
#         loss = -torch.log(torch.diag(exp_sim) / exp_sim.sum(dim=1) + 1e-8)
        
#         return loss.mean()


# class Client(object):
#     def __init__(self, dataset, lr, device, task_num, id, epoch_local, subset, batch_size, mask, num_classes):
#         self.id = id
#         self.task_num = task_num
#         self.learning_rate = lr
#         self.epoch_local = epoch_local
#         self.batch_size = batch_size
#         self.device = device
#         self.dataset = dataset
#         self.mask = mask
#         self.num_classes = num_classes

#         # Label mapping
#         self.label_mapping = {}
#         self.reverse_mapping = {}

#         self.test_loader = []
#         self.train_dataset = subset
#         self.train_data = None

#         # Model with projection layer
#         self.model = resnet18_cbam(pretrained=False, num_classes=self.num_classes, projection_dim=128)
#         self.global_model = None
#         self.prev_model = None  # Previous task's global model (for KD)
        
#         # Generator for replay
#         self.generator = None

#         # 用于统计每轮训练的样本数（计算FLOPs用）
#         self.num_train_samples_this_round = 0
        
#         # FCIL-LD hyperparameters
#         self.tau = 0.1  # Temperature for contrastive loss
#         self.alpha_KD = 1.0   # Weight for knowledge distillation loss
#         self.alpha_con = 0.5  # Weight for contrastive loss
#         self.alpha_FT = 0.5   # Weight for fine-tuning loss
        
#         # Loss functions
#         self.supcon_loss = SupConLoss(temperature=self.tau)
#         self.contrastive_loss = ContrastiveLoss(temperature=self.tau)
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()
        
#         # Track current task
#         self.current_task_id = -1
        
#         # Data augmentation for SupCon
#         self.augment = transforms.Compose([
#             transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#             ], p=0.8),
#         ])

#     def recive_global_models(self, global_model, prev_model=None, generator=None):
#         """Receive global model, previous model, and generator from server"""
#         self.model.load_state_dict(global_model.state_dict())
        
#         if prev_model is not None:
#             self.prev_model = deepcopy(prev_model)
#             self.prev_model.eval()
#             for param in self.prev_model.parameters():
#                 param.requires_grad = False
        
#         if generator is not None:
#             self.generator = generator
#             self.generator.eval()

#     def get_data(self, task_id):
#         if task_id % 5 == 0:
#             self.task_id = task_id // 5
#             self.current_class = self.mask[self.task_id]
#             self.trainset = self.train_dataset[self.task_id]
#             if self.task_id == 0:
#                 self.first_task = self.current_class

#         print(f'{self.id} client, {task_id} task contains {self.current_class} classes, '
#               f'there are {len(self.current_class)} classes')

#         self.label_mapping = {cls: idx for idx, cls in enumerate(self.current_class)}
#         self.reverse_mapping = {idx: cls for idx, cls in enumerate(self.current_class)}

#         traindata, testdata = random_split(
#             self.trainset,
#             [int(len(self.trainset) * 0.8), len(self.trainset) - int(len(self.trainset) * 0.8)]
#         )
#         testdata = deepcopy(testdata)
#         testloader = DataLoader(testdata, shuffle=True, batch_size=self.batch_size)

#         self.test_loader.append(testloader)
#         self.train_data = traindata
#         self.current_task_id = self.task_id

#     def _augment_batch(self, images):
#         """Apply data augmentation to a batch of images for SupCon"""
#         augmented = []
#         for img in images:
#             # Convert tensor to PIL, augment, convert back
#             # For simplicity, we'll just add noise as augmentation
#             aug_img = img + 0.1 * torch.randn_like(img)
#             augmented.append(aug_img)
#         return torch.stack(augmented)

#     def train_initial_task(self):
#         """
#         Train on the initial task using SupCon loss (Section 3.2.1 in paper).
#         """
#         self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        
#         self.model.to(self.device)
#         self.model.train()
        
#         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, 
#                                     weight_decay=1e-5, momentum=0.9)
        
#         print(f'Client {self.id} training initial task with SupCon loss')
        
#         for epoch in range(self.epoch_local):
#             epoch_loss = 0.0
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 # Create augmented version
#                 inputs_aug = self._augment_batch(inputs).to(self.device)
                
#                 # Concatenate original and augmented
#                 inputs_combined = torch.cat([inputs, inputs_aug], dim=0)
#                 labels_combined = torch.cat([labels, labels], dim=0)
                
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 features = self.model.forward_features(inputs_combined)
#                 projections = self.model.forward_projection(features)
                
#                 # SupCon loss on projection layer
#                 loss_supcon = self.supcon_loss(projections, labels_combined)
                
#                 # Classification head loss (with gradient blocked from feature extractor)
#                 # Only update classification head with CE loss
#                 features_detached = features.detach()
#                 logits = self.model.forward_classifier(features_detached)
#                 loss_head = self.ce_loss(logits, labels_combined)
                
#                 # Total loss (Eq. 8)
#                 loss = loss_head + self.tau * loss_supcon
                
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
            
#             if (epoch + 1) % 5 == 0:
#                 print(f'  Epoch [{epoch+1}/{self.epoch_local}], Loss: {epoch_loss/len(self.train_loader):.4f}')

#     def train_subsequent_task(self):
#         """
#         Train on subsequent tasks using the novel loss function (Section 3.2.2 in paper).
#         """
#         self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        
#         self.model.to(self.device)
#         self.model.train()
        
#         if self.prev_model is not None:
#             self.prev_model.to(self.device)
#             self.prev_model.eval()
        
#         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
#                                     weight_decay=1e-5, momentum=0.9)
        
#         print(f'Client {self.id} training subsequent task with FCIL-LD loss')
        
#         # Generate synthetic samples if generator is available
#         synthetic_data = None
#         if self.generator is not None:
#             self.generator.to(self.device)
#             self.generator.eval()
#             with torch.no_grad():
#                 num_synthetic = min(len(self.train_data), 500)  # Limit synthetic samples
#                 z = torch.randn(num_synthetic, self.generator.latent_dim).to(self.device)
#                 synthetic_images = self.generator(z)
                
#                 # Get pseudo-labels from previous model
#                 if self.prev_model is not None:
#                     _, syn_logits = self.prev_model(synthetic_images)
#                     synthetic_labels = syn_logits.argmax(dim=1)
#                 else:
#                     synthetic_labels = z[:, :self.num_classes].argmax(dim=1)
                
#                 synthetic_data = (synthetic_images, synthetic_labels)
        
#         for epoch in range(self.epoch_local):
#             epoch_loss = 0.0
            
#             for batch_idx, (inputs, labels) in enumerate(self.train_loader):
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 optimizer.zero_grad()
#                 total_loss = 0.0
                
#                 # === Component 1: Cross-entropy loss on new data (Eq. 9) ===
#                 features = self.model.forward_features(inputs)
#                 logits = self.model.forward_classifier(features)
#                 loss_ce = self.ce_loss(logits, labels)
#                 total_loss += loss_ce
                
#                 # === Component 2: Knowledge Distillation loss on synthetic samples (Eq. 10) ===
#                 if synthetic_data is not None and self.prev_model is not None:
#                     # Sample a mini-batch from synthetic data
#                     syn_idx = torch.randint(0, synthetic_data[0].size(0), (min(inputs.size(0), synthetic_data[0].size(0)),))
#                     syn_images = synthetic_data[0][syn_idx]
                    
#                     # Get features from current model
#                     features_syn_current = self.model.forward_features(syn_images)
                    
#                     # Get features from previous model (frozen)
#                     with torch.no_grad():
#                         features_syn_prev = self.prev_model.forward_features(syn_images)
                    
#                     # Feature-level KD: compare outputs through previous model's classifier
#                     with torch.no_grad():
#                         # Get target: previous model's representation
#                         target_logits = self.prev_model.forward_classifier(features_syn_prev)
                    
#                     # Current model's features through previous classifier for comparison
#                     current_through_prev = self.prev_model.forward_classifier(features_syn_current)
                    
#                     loss_kd = self.mse_loss(current_through_prev, target_logits)
#                     total_loss += self.alpha_KD * loss_kd
                
#                 # === Component 3: Contrastive loss in projection layer (Eq. 11) ===
#                 if self.prev_model is not None:
#                     # Project features through both models
#                     z_current = self.model.forward_projection(features)
                    
#                     with torch.no_grad():
#                         features_prev = self.prev_model.forward_features(inputs)
#                         z_prev = self.prev_model.forward_projection(features_prev)
                    
#                     loss_con = self.contrastive_loss(z_current, z_prev)
#                     total_loss += self.alpha_con * loss_con
                
#                 total_loss.backward()
#                 optimizer.step()
#                 epoch_loss += total_loss.item()
            
#             # === Component 4: Fine-tuning loss (Eq. 12) ===
#             # Fine-tune classification head with combined real and synthetic data
#             if synthetic_data is not None:
#                 for param in self.model.parameters():
#                     param.requires_grad = False
#                 for param in self.model.fc2.parameters():
#                     param.requires_grad = True
                
#                 optimizer_ft = torch.optim.SGD(self.model.fc2.parameters(), lr=self.learning_rate * 0.1)
                
#                 for inputs, labels in self.train_loader:
#                     inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
#                     # Combine with synthetic
#                     syn_idx = torch.randint(0, synthetic_data[0].size(0), (min(inputs.size(0), synthetic_data[0].size(0)),))
#                     syn_images = synthetic_data[0][syn_idx]
#                     syn_labels = synthetic_data[1][syn_idx]
                    
#                     combined_inputs = torch.cat([inputs, syn_images], dim=0)
#                     combined_labels = torch.cat([labels, syn_labels], dim=0)
                    
#                     optimizer_ft.zero_grad()
                    
#                     with torch.no_grad():
#                         features_combined = self.model.forward_features(combined_inputs)
#                     logits_combined = self.model.forward_classifier(features_combined)
                    
#                     loss_ft = self.ce_loss(logits_combined, combined_labels)
#                     loss_ft.backward()
#                     optimizer_ft.step()
                
#                 # Re-enable all gradients
#                 for param in self.model.parameters():
#                     param.requires_grad = True
            
#             if (epoch + 1) % 5 == 0:
#                 print(f'  Epoch [{epoch+1}/{self.epoch_local}], Loss: {epoch_loss/len(self.train_loader):.4f}')

#     def train(self, task_id):
#         self.get_data(task_id)
        
#         # 记录本轮训练样本数
#         self.num_train_samples_this_round = len(self.train_data)
        
#         if self.task_id == 0:
#             # Initial task: use SupCon loss
#             self.train_initial_task()
#         else:
#             # Subsequent tasks: use FCIL-LD loss
#             self.train_subsequent_task()
        
#         print(f'Client {self.id} finished training')
#         self.final_state = copy.deepcopy(self.model.state_dict())
#         self.evaluate_local_models(task_id)

#     def evaluate_local_models(self, index):
#         self.model.load_state_dict(self.final_state)
#         self.model.eval()
#         correct, total = 0, 0
        
#         with torch.no_grad():
#             for inputs, labels in self.test_loader[index]:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 _, logits = self.model(inputs)
                
#                 _, pred_idx = torch.max(logits, 1)
#                 predicted_labels = pred_idx
                
#                 total += labels.size(0)
#                 correct += (predicted_labels == labels).sum().item()

#         acc = correct / total * 100
#         print(f'Client {self.id}\'s local model on {index} task acc is {acc:.2f}%')
#         return acc