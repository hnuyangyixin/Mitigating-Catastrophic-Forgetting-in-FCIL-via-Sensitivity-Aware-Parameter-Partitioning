# analyze_similarity.py
import os
import torch
import torch.nn.functional as F
from ResNet import resnet18_cbam


def load_classifier_weights(model_path, num_classes, device="cpu"):
    """加载模型并提取分类器权重"""
    model = resnet18_cbam(pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # 假设最后一层是 model.fc
    classifier_weight = model.fc.weight.data.clone().to(device)
    return classifier_weight


def cosine_similarity_matrix(weight):
    """计算类别间余弦相似度矩阵"""
    normed_weight = F.normalize(weight, dim=1)
    sim_matrix = torch.matmul(normed_weight, normed_weight.t())
    return sim_matrix


def analyze_similarity(num_classes, mask, device="cpu"):
    """
    分析任务完成后的类别余弦相似度
    :param num_classes: 总类别数
    :param mask: 每个任务的类别划分 (形如 [[task0_classes], [task1_classes], ...])
    :param device: 运行设备
    """
    for task_idx in range(len(mask)):
        model_path = os.path.join("model_data", f'global_model_task_{task_idx}.pth')
        if not os.path.exists(model_path):
            print(f"模型 {model_path} 不存在，跳过。")
            continue

        print(f"\n==== 分析任务 {task_idx} ====")

        # 加载分类器权重
        weight = load_classifier_weights(model_path, num_classes, device)
        sim_matrix = cosine_similarity_matrix(weight)

        # ① 最新任务包含类别之间的余弦相似度
        task_classes = mask[task_idx]
        print(f"任务 {task_idx} 的类别: {task_classes}")
        for i in range(len(task_classes)):
            for j in range(i + 1, len(task_classes)):
                c1, c2 = task_classes[i], task_classes[j]
                sim = sim_matrix[c1, c2].item()
                print(f" 类别 {c1} vs {c2}: 余弦相似度 = {sim:.4f}")

        # ② 过去所有已训练类别之间的相似度最低的十对
        past_classes = [c for t in range(task_idx + 1) for c in mask[t]]
        sub_matrix = sim_matrix[past_classes][:, past_classes]

        triu_indices = torch.triu_indices(len(past_classes), len(past_classes), offset=1)
        sims = sub_matrix[triu_indices[0], triu_indices[1]]
        sorted_vals, sorted_idx = torch.sort(sims)

        print(f"\n过去所有任务已见类别间相似度最低的十对：")
        for k in range(min(10, len(sorted_vals))):
            idx1, idx2 = triu_indices[0][sorted_idx[k]], triu_indices[1][sorted_idx[k]]
            c1, c2 = past_classes[idx1], past_classes[idx2]
            print(f" {k+1}. 类别 {c1} vs {c2}: 余弦相似度 = {sorted_vals[k].item():.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze classifier similarity across tasks")
    parser.add_argument("--num_classes", type=int,default=100, required=True, help="总类别数")
    parser.add_argument("--mask_file", type=str, default=task_mask.pt,required=True,
                        help="包含任务类别划分的 .pt 文件 (如 [[task0_classes],[task1_classes],...])")
    parser.add_argument("--device", type=str, default="gpu")

    args = parser.parse_args()

    mask = torch.load(args.mask_file)  # 假设保存的是 list[list[int]]
    analyze_similarity(args.num_classes, mask, args.device)
