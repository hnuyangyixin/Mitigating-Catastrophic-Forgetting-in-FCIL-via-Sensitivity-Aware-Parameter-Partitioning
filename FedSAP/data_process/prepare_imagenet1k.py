"""把 Hugging Face 上的 ImageNet-1K 流式整理成本项目所需的目录结构。

默认使用免登录、已预缩放到 256x256 的镜像 benjamin-paine/imagenet-1k-256x256
（字段名 image/label、标签顺序与官方完全一致，0=tench(n01440764)）。
模型训练时会再 resize 到 img_size，所以分辨率对结果无实质影响。

每个类别只取 `--per_class` 张（默认 600 = 500 训练 + 100 测试），
无需下载完整数据集，省带宽省磁盘。

输出结构（与 imagenet1k_subset_spliter.py 约定一致）：
    <out_dir>/train/class_0000/*.JPEG
    <out_dir>/train/class_0001/*.JPEG
    ...
    <out_dir>/train/class_0999/*.JPEG

说明：
- 整数标签 0..999 按 wnid 排序固定不变。这里直接用整数标签作为文件夹名
  （class_0000 ...），spliter 端按 sorted() 文件夹名取 index，两边一一对应，
  因此训练/评估无需关心具体 wnid。
- 默认输出目录是 FedSAP/local_datasets/imagenet-1k（相对脚本定位，与运行目录无关），
  正好是 spliter 默认读取的位置，整理完直接就能跑。

用法（从任意目录均可）：
    pip install datasets
    python FedSAP/data_process/prepare_imagenet1k.py            # 直接跑，默认免登录镜像
    python FedSAP/main.py --dataset ImageNet1K
"""

import argparse
import os

# 仓库根目录（本文件在 right_code/FedSAP/data_process/ 下，向上三级到 right_code）。
# 各方法的 main.py 都从 right_code/ 运行，spliter 用相对路径 ./local_datasets，
# 因此这里默认下到 right_code/local_datasets/imagenet-1k，与之共享同一份数据。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUT = os.path.join(_REPO_ROOT, 'local_datasets', 'imagenet-1k')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default=_DEFAULT_OUT,
                        help='输出根目录（会在其下创建 train/）。默认 right_code/local_datasets/imagenet-1k，'
                             '与各方法共享同一份数据')
    parser.add_argument('--per_class', type=int, default=600,
                        help='每个类别保留的图片数（默认 600 = 500 训练 + 100 测试）')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='类别总数（ImageNet-1K 为 1000）')
    parser.add_argument('--split', type=str, default='train',
                        help='从 HF 的哪个 split 取数据（建议 train）')
    parser.add_argument('--repo_id', type=str,
                        default='benjamin-paine/imagenet-1k-256x256',
                        help='HF 数据集仓库 id。默认免登录的 256x256 镜像；'
                             '若要原图可用官方 ILSVRC/imagenet-1k（需登录+同意许可）')
    parser.add_argument('--image_key', type=str, default='image',
                        help='样本里图像字段名（个别仓库可能为 img）')
    parser.add_argument('--label_key', type=str, default='label',
                        help='样本里标签字段名（个别仓库可能为 labels）')
    args = parser.parse_args()

    from datasets import load_dataset

    train_root = os.path.join(args.out_dir, 'train')
    os.makedirs(train_root, exist_ok=True)

    print(f"开始流式读取 {args.repo_id} 的 {args.split} split ...")
    ds = load_dataset(args.repo_id, split=args.split, streaming=True)

    # 扫描磁盘已有图片，支持断点续传：已下满的类直接跳过，只补缺失部分
    saved = [0] * args.num_classes
    for label in range(args.num_classes):
        cdir = os.path.join(train_root, f"class_{label:04d}")
        if os.path.isdir(cdir):
            saved[label] = min(args.per_class, len(
                [fn for fn in os.listdir(cdir) if fn.lower().endswith('.jpeg')]))
    done_classes = sum(1 for s in saved if s >= args.per_class)
    total_target = args.num_classes * args.per_class
    saved_total = sum(saved)
    if saved_total > 0:
        print(f"检测到已存在 {saved_total} 张（{done_classes} 个类已下满），"
              f"断点续传，仅下载缺失部分 ...")

    for example in ds:
        label = int(example[args.label_key])
        if label < 0 or label >= args.num_classes:
            continue
        if saved[label] >= args.per_class:
            continue

        class_dir = os.path.join(train_root, f"class_{label:04d}")
        if saved[label] == 0:
            os.makedirs(class_dir, exist_ok=True)

        img = example[args.image_key]
        # 部分图片为灰度 / CMYK，统一转 RGB；以 JPEG 保存
        if img.mode != "RGB":
            img = img.convert("RGB")
        out_path = os.path.join(class_dir, f"{label:04d}_{saved[label]:04d}.JPEG")
        img.save(out_path, format="JPEG", quality=95)

        saved[label] += 1
        saved_total += 1
        if saved[label] == args.per_class:
            done_classes += 1

        if saved_total % 1000 == 0:
            print(f"  已保存 {saved_total}/{total_target} 张，"
                  f"已完成 {done_classes}/{args.num_classes} 个类别")

        # 所有类别都取够了就停止
        if done_classes >= args.num_classes:
            print("所有类别均已取够，提前结束流式读取。")
            break

    # 汇总
    incomplete = [i for i in range(args.num_classes) if saved[i] < args.per_class]
    print(f"\n完成：共保存 {saved_total} 张到 {train_root}")
    print(f"已取满 {args.per_class} 张的类别数：{args.num_classes - len(incomplete)}/{args.num_classes}")
    if incomplete:
        print(f"警告：以下 {len(incomplete)} 个类别样本不足 {args.per_class} 张："
              f"{incomplete[:20]}{' ...' if len(incomplete) > 20 else ''}")
        print("（若 train split 某些类样本偏少，可减小 --per_class，或补充 validation split。）")

    # 写完成标记：spliter 见到此文件即跳过自身的自动下载，直接复用这里下好的数据
    marker = os.path.join(args.out_dir, '.download_complete')
    with open(marker, 'w') as f:
        f.write(f"per_class={args.per_class}\n")
    print(f"已写入完成标记：{marker}")


if __name__ == '__main__':
    main()
