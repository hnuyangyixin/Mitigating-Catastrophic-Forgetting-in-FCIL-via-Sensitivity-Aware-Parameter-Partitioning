# -*- coding: utf-8 -*-
"""
计算持续学习中：每个任务训练完成后的全局模型上，
各（已见）类别的 prototype 与 分类器 proxy 之间的夹角。

定义：
  - prototype(c)  : 用「当前任务的全局模型」的特征提取器，对类别 c 的所有测试样本
                    提取 512 维特征 (ResNet.forward 返回的第一个输出) 后取均值。
  - proxy(c)      : 分类器权重 fc2.weight[c]（512 维）。
                    说明：ResNet.forward 只用 fc2 产生 logits，训练/对齐也都作用在 fc2 上，
                    fc 自始至终未参与前向，保持随机初始化，因此 proxy 取 fc2。
  - 夹角           : angle(c) = arccos( cos(prototype(c), proxy(c)) )，单位为度。

流程：
  对 task t = 0..T-1：
    加载 CIFAR100_global_model_task_{t}.pth
    对任务 0..t 的每个类别 c，计算 prototype(c) 与 proxy(c) 的夹角。
  （即：新任务训练完后，重新计算所有「旧任务 + 当前任务」类别的夹角。）

用法：
    python compute_proto_proxy_angle.py
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from ResNet import resnet18_cbam
from data_process.utils import build_transform

# ----------------------------- 配置 -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_data")
CLASS_SPLIT_TXT = os.path.join(MODEL_DIR, "类别划分.txt")
DATA_ROOT = os.path.join(HERE, "local_datasets")

NUM_CLASSES = 100
NUM_TASKS = 5
INPUT_SIZE = 32
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME_TMPL = "CIFAR100_global_model_task_{}.pth"
OUT_TXT = os.path.join(MODEL_DIR, "proto_proxy_angle.txt")


# ----------------------- 解析每个任务的类别 -----------------------
def parse_task_classes(txt_path, num_tasks):
    """
    从「类别划分.txt」解析每个任务（task index = round // 5）包含的类别（3 个客户端并集）。
    文件中的形如：  "0 client，3 task contains [72, 92, ...] classes"
    其中 "3 task" 是 global round 编号，任务索引 = round // 5。
    返回: {task_index: sorted([class, ...])}
    """
    pattern = re.compile(r"(\d+)\s*client[，,]\s*(\d+)\s*task contains\s*\[([0-9,\s]+)\]")
    task_sets = {t: set() for t in range(num_tasks)}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            rnd = int(m.group(2))
            task_idx = rnd // 5
            if task_idx >= num_tasks:
                continue
            classes = [int(x) for x in m.group(3).split(",") if x.strip() != ""]
            task_sets[task_idx].update(classes)
    return {t: sorted(s) for t, s in task_sets.items()}


# ----------------------- 计算每个类别的 prototype -----------------------
def compute_prototypes(model, test_loader, num_classes, device):
    """
    用 model 的特征提取器（forward 的第一个返回值）对整个测试集提取特征，
    按类别累加并求均值，得到每个类别的 prototype。
    返回: protos [num_classes, feat_dim], counts [num_classes]
    """
    model.eval()
    feat_sum = None
    counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            feats, _ = model(inputs)            # feats: [B, feat_dim]
            feats = feats.cpu()
            if feat_sum is None:
                feat_sum = torch.zeros(num_classes, feats.shape[1])
            for c in labels.unique():
                c = int(c)
                mask = labels == c
                feat_sum[c] += feats[mask].sum(dim=0)
                counts[c] += int(mask.sum())
    protos = feat_sum.clone()
    nonzero = counts > 0
    protos[nonzero] = feat_sum[nonzero] / counts[nonzero].unsqueeze(1).float()
    return protos, counts


def angle_deg(vec_a, vec_b):
    """两个向量夹角（度）。"""
    cos = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos)).item()), float(cos.item())


def main():
    print(f"[设备] {DEVICE}")
    task_classes = parse_task_classes(CLASS_SPLIT_TXT, NUM_TASKS)
    print("[任务类别划分] (3 个客户端并集)")
    for t in range(NUM_TASKS):
        print(f"  task {t} ({len(task_classes[t])} 类): {task_classes[t]}")

    # 测试集（与训练一致：eval transform = ToTensor，无归一化）
    transform = build_transform(False, INPUT_SIZE)
    testset = CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    for t in range(NUM_TASKS):
        model_path = os.path.join(MODEL_DIR, MODEL_NAME_TMPL.format(t))
        if not os.path.exists(model_path):
            emit(f"\n模型不存在，跳过: {model_path}")
            continue

        emit("\n" + "=" * 70)
        emit(f"==== 任务 {t} 训练完成后的全局模型: {os.path.basename(model_path)} ====")
        emit("=" * 70)

        model = resnet18_cbam(pretrained=False, num_classes=NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)

        # 分类器 proxy（fc2）与 prototype
        proxy_weight = model.fc2.weight.data.detach().cpu()      # [num_classes, feat_dim]
        protos, counts = compute_prototypes(model, test_loader, NUM_CLASSES, DEVICE)

        # 对任务 0..t，计算该任务所有类别夹角的平均值
        for s in range(t + 1):
            tag = "当前任务" if s == t else "旧任务"
            angles = []
            for c in task_classes[s]:
                if counts[c] == 0:
                    continue
                ang, _ = angle_deg(protos[c], proxy_weight[c])
                angles.append(ang)
            if angles:
                emit(f"  {tag} {s}: 平均夹角 = {np.mean(angles):.2f}°  "
                     f"({len(angles)} 个类别)")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\n[已保存] 结果写入: {OUT_TXT}")


if __name__ == "__main__":
    main()
