"""
Data-free Generator for FCIL-LD
Based on the paper: "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning 
with a Low-Dimensional Projection Layer" (CVPR 2025 Workshop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    """
    Data-free generator for synthesizing samples from previous tasks.
    Used for replay to mitigate catastrophic forgetting.
    """
    def __init__(self, latent_dim=200, img_channels=3, img_size=32, num_classes=100):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Initial size for upsampling (for 32x32 images, start from 4x4)
        self.init_size = img_size // 8  # 4 for 32x32
        
        # First layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z):
        """
        Generate images from latent vectors.
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        Returns:
            Generated images of shape (batch_size, img_channels, img_size, img_size)
        """
        out = self.fc(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class GeneratorTrainer:
    """
    Trainer for the data-free generator on the server side.
    """
    def __init__(self, generator, global_model, device, num_classes, 
                 alpha_div=1.0, alpha_bn=75.0, alpha_prior=0.001,
                 lr=0.005, num_epochs=5000):
        self.generator = generator
        self.global_model = global_model
        self.device = device
        self.num_classes = num_classes
        self.alpha_div = alpha_div
        self.alpha_bn = alpha_bn
        self.alpha_prior = alpha_prior
        self.lr = lr
        self.num_epochs = num_epochs
        
        # Get BN statistics from the global model
        self.bn_stats = self._get_bn_statistics()
        
    def _get_bn_statistics(self):
        """Extract running mean and variance from BatchNorm layers"""
        bn_stats = []
        for module in self.global_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_stats.append({
                    'mean': module.running_mean.clone().detach(),
                    'var': module.running_var.clone().detach()
                })
        return bn_stats
    
    def _compute_bn_loss(self, model):
        """
        Compute batch normalization loss (Eq. 3 in paper).
        Minimizes KL divergence between BN statistics of real and synthetic samples.
        """
        bn_loss = 0.0
        idx = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if idx < len(self.bn_stats):
                    # Running stats from real data
                    real_mean = self.bn_stats[idx]['mean']
                    real_var = self.bn_stats[idx]['var']
                    
                    # Current stats from synthetic data (using running stats updated during forward)
                    syn_mean = module.running_mean
                    syn_var = module.running_var
                    
                    # KL divergence between two Gaussians
                    real_std = torch.sqrt(real_var + 1e-8)
                    syn_std = torch.sqrt(syn_var + 1e-8)
                    
                    kl = torch.log(syn_std / real_std) - 0.5 * (1 - (real_var + (real_mean - syn_mean)**2) / (syn_var + 1e-8))
                    bn_loss += kl.mean()
                    idx += 1
        
        if idx > 0:
            bn_loss = bn_loss / idx
        return bn_loss
    
    def _compute_diversity_loss(self, logits):
        """
        Compute diversity loss (Eq. 2 in paper).
        Maximizes information entropy of average class predictions.
        """
        # Average prediction across batch
        avg_pred = F.softmax(logits, dim=1).mean(dim=0)
        # Information entropy (negative because we want to maximize)
        entropy = -torch.sum(avg_pred * torch.log(avg_pred + 1e-8))
        # Return negative entropy (to minimize)
        return -entropy
    
    def _compute_smoothness_loss(self, images):
        """
        Compute smoothness prior loss (Eq. 4 in paper).
        Encourages smooth transitions between neighboring pixels.
        """
        # Simple total variation as smoothness measure
        diff_h = images[:, :, 1:, :] - images[:, :, :-1, :]
        diff_w = images[:, :, :, 1:] - images[:, :, :, :-1]
        return torch.mean(diff_h**2) + torch.mean(diff_w**2)
    
    def train(self, batch_size=64, verbose=True):
        """
        Train the generator using data-free knowledge distillation.
        """
        self.generator.to(self.device)
        self.global_model.to(self.device)
        self.global_model.eval()
        
        # Freeze global model
        for param in self.global_model.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        
        for epoch in range(self.num_epochs):
            self.generator.train()
            
            # Sample noise with pseudo-labels encoded
            z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
            
            # Encode target class in the first 'num_classes' dimensions
            # Use argmax of first num_classes dims as pseudo-label
            pseudo_labels = z[:, :self.num_classes].argmax(dim=1)
            
            # Generate images
            fake_images = self.generator(z)
            
            # Get predictions from frozen global model
            # Need to set model to train mode briefly to update BN stats
            self.global_model.train()
            _, logits = self.global_model(fake_images)
            self.global_model.eval()
            
            # Cross-entropy loss (Eq. 1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            
            # Diversity loss (Eq. 2)
            div_loss = self._compute_diversity_loss(logits)
            
            # BN loss (Eq. 3)
            bn_loss = self._compute_bn_loss(self.global_model)
            
            # Smoothness loss (Eq. 4)
            prior_loss = self._compute_smoothness_loss(fake_images)
            
            # Total loss (Eq. 5)
            total_loss = ce_loss + self.alpha_div * div_loss + \
                        self.alpha_bn * bn_loss + self.alpha_prior * prior_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 500 == 0:
                print(f'Generator Training Epoch [{epoch+1}/{self.num_epochs}], '
                      f'Loss: {total_loss.item():.4f}, CE: {ce_loss.item():.4f}, '
                      f'Div: {div_loss.item():.4f}, BN: {bn_loss.item():.4f}')
        
        # Re-enable gradients for global model
        for param in self.global_model.parameters():
            param.requires_grad = True
            
        return self.generator
    
    def generate_samples(self, num_samples, batch_size=64):
        """
        Generate synthetic samples from previous tasks.
        """
        self.generator.eval()
        samples = []
        labels = []
        
        with torch.no_grad():
            remaining = num_samples
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                z = torch.randn(current_batch, self.generator.latent_dim).to(self.device)
                
                fake_images = self.generator(z)
                _, logits = self.global_model(fake_images)
                pseudo_labels = logits.argmax(dim=1)
                
                samples.append(fake_images.cpu())
                labels.append(pseudo_labels.cpu())
                remaining -= current_batch
        
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)