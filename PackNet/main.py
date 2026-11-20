# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import time

# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
# from tqdm import tqdm

# from option import args_parser
# from ResNet import resnet18_cbam
# from Server import Server
# from data_process.cifar100_subset_spliter import cifar100_Data_Spliter
# from data_process.cifar10_subset_spliter import cifar10_Data_Spliter
# from data_process.tinyimagenet_subset_spliter import TinyImageNet_Data_Spliter
# from utils import setup_seed

# start = time.time()
# args = args_parser()
# setup_seed(args.seed)

# print("="*60)
# print("PackNet for Federated Class-Incremental Learning")
# print("="*60)
# print(f"Dataset: {args.dataset}")
# print(f"Number of clients: {args.num_clients}")
# print(f"Number of tasks: {args.tasks_global}")
# print(f"Prune ratio: {args.prune_ratio}")
# print(f"Local epochs: {args.epochs_local}")
# print(f"Retrain epochs: {args.retrain_epochs}")
# print("="*60)

# feature_extractor = None

# # 数据集转换
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# train_dataset = []
# test_dataset = None

# # 数据集处理
# if args.dataset == 'CIFAR100':
#     client_data, client_mask = cifar100_Data_Spliter(
#         client_num=args.num_clients,
#         task_num=args.tasks_global,
#         private_class_num=15,
#         input_size=args.img_size).random_split()

#     _, test_data = cifar100_Data_Spliter(
#         client_num=args.num_clients,
#         task_num=args.tasks_global,
#         private_class_num=15,
#         input_size=args.img_size).process_testdata(0)

#     test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
#     class_num = 100

# elif args.dataset == 'CIFAR10':
#     print("Processing CIFAR10...")
#     priv_num = 1 
    
#     client_data, client_mask = cifar10_Data_Spliter(
#         client_num=args.num_clients,
#         task_num=args.tasks_global,
#         private_class_num=priv_num, 
#         input_size=args.img_size).random_split()

#     _, test_data = cifar10_Data_Spliter(
#         client_num=args.num_clients,
#         task_num=args.tasks_global,
#         private_class_num=priv_num,
#         input_size=args.img_size).process_testdata(0)
    
#     test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
#     class_num = 10

# elif args.dataset == 'TinyImageNet':
#     print("Processing TinyImageNet-200...")
#     priv_num = 20
    
#     client_data, client_mask = TinyImageNet_Data_Spliter(
#         client_num=args.num_clients,
#         task_num=args.tasks_global,
#         private_class_num=priv_num,
#         input_size=args.img_size).random_split()

#     _, test_data = TinyImageNet_Data_Spliter(
#         client_num=args.num_clients,
#         task_num=args.tasks_global,
#         private_class_num=priv_num,
#         input_size=args.img_size).process_testdata(0)

#     test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
#     class_num = 200

# # 服务器初始化
# num_clients = args.num_clients
# global_method = args.global_method
# dataset_name = args.dataset
# local_method = args.local_method
# batch_size = args.batch_size

# epoch = args.epochs_local
# lr = args.learning_rate
# task_num = args.tasks_global
# pretrain_model = args.pretrained_model

# alpha = args.alpha
# device = args.device
# pseudo_samples = args.pseudo_samples
# save_path = './model_data'

# # 确保保存目录存在
# os.makedirs(save_path, exist_ok=True)

# if dataset_name == "CIFAR100":
#     class_num = 100
# elif dataset_name == "TinyImageNet":
#     class_num = 200
# else:
#     class_num = 10

# # 创建Server（使用PackNet方法）
# myServer = Server(
#     num_clients, task_num, dataset_name, lr, device, epoch, pseudo_samples,
#     global_method, local_method, feature_extractor, client_data, batch_size,
#     alpha, test_loader, class_num, client_mask, save_path,
#     prune_ratio=args.prune_ratio,
#     retrain_epochs=args.retrain_epochs,
#     task_id_epochs=args.task_id_epochs
# )

# myServer.init_clients()
# myServer.train_clients()

# end = time.time()
# print(f'\nTotal time: {end-start:.2f} seconds')


# print(f"\n[Final Overhead] Total Training FLOPs: {myServer.total_train_flops:.4e}")
# print(f"[Final Overhead] Total Communication Bytes: {myServer.total_comm_bytes:.4e}")



# 3 turns

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from option import args_parser
from ResNet import resnet18_cbam
from Server import Server
from data_process.cifar100_subset_spliter import cifar100_Data_Spliter
from data_process.cifar10_subset_spliter import cifar10_Data_Spliter
from data_process.tinyimagenet_subset_spliter import TinyImageNet_Data_Spliter
from data_process.tinyimagenet20c_subset_spliter import TinyImageNet20C_Data_Spliter
from data_process.imagenet1k_subset_spliter import ImageNet1K_Data_Spliter
from utils import setup_seed

start = time.time()
args = args_parser()


# ============ 终端输出同时写入日志文件 ============
import re as _re
# 这些“逐客户端”噪音行只在终端显示，不写入日志文件（终端要、文件不要）。
_NOISE_PATTERNS = [
    _re.compile(r'task contains'),
    _re.compile(r'is training on'),
    _re.compile(r'client finish training'),
    _re.compile(r'client finished training'),
    _re.compile(r'local model on .*acc is'),
    _re.compile(r'local model on task \d+:'),
    _re.compile(r'Task \d+: Phase [123]'),
    _re.compile(r'finished task \d+, learned classes'),
    _re.compile(r'Task \d+ Statistics:'),
    _re.compile(r'Client space:'),
]

class _Tee:
    """屏幕原样输出；文件按行过滤掉噪音行后再写入。"""
    def __init__(self, screen, logfile):
        self.screen = screen
        self.logfile = logfile
        self._buf = ''
    def write(self, data):
        # 屏幕始终原样输出
        self.screen.write(data)
        self.screen.flush()
        # 文件按整行过滤
        self._buf += data
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if not any(p.search(line) for p in _NOISE_PATTERNS):
                self.logfile.write(line + '\n')
                self.logfile.flush()
    def flush(self):
        self.screen.flush()
        self.logfile.flush()

import datetime as _dt
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir,
    f"PackNet_{args.dataset}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, 'w', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f"[日志] 本次运行的终端输出将保存到: {_log_path}")
# ================================================

def run_once(seed):
    """运行一次完整实验，返回所有评估结果列表"""
    setup_seed(seed)

    print("\n" + "="*60)
    print(f"Start Run with Seed: {seed}")
    print("PackNet for Federated Class-Incremental Learning")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of tasks: {args.tasks_global}")
    print(f"Prune ratio: {args.prune_ratio}")
    print(f"Local epochs: {args.epochs_local}")
    print(f"Retrain epochs: {args.retrain_epochs}")
    print("="*60)

    feature_extractor = None

    # 数据集转换
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = []
    test_dataset = None

    # 数据集处理
    if args.dataset == 'CIFAR100':
        client_data, client_mask = cifar100_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=15,
            input_size=args.img_size).random_split()

        _, test_data = cifar100_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=15,
            input_size=args.img_size).process_testdata(0)

        test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        class_num = 100

    elif args.dataset == 'CIFAR10':
        print("Processing CIFAR10...")
        priv_num = 1

        client_data, client_mask = cifar10_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size).random_split()

        _, test_data = cifar10_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size).process_testdata(0)

        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
        class_num = 10

    elif args.dataset == 'TinyImageNet':
        print("Processing TinyImageNet-200...")
        priv_num = 20

        client_data, client_mask = TinyImageNet_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size).random_split()

        _, test_data = TinyImageNet_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size).process_testdata(0)

        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
        class_num = 200

    elif args.dataset == 'TinyImageNet20C':
        print("Processing TinyImageNet20C (120私有+80公共, 20客户端新划分)...")
        # 客户端数固定 20；每客户端 6 私有类 -> 120 私有 + 80 公共
        args.num_clients = 20
        priv_num = 6

        # 训练与测试复用同一个 spliter 实例，保证“被丢弃的类”在两侧一致
        spliter = TinyImageNet20C_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size)
        client_data, client_mask = spliter.random_split()
        _, test_data = spliter.process_testdata(0)

        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
        # 标签空间仍是 200（被丢弃的类只是不出现在训练/测试数据里，分类头维度不变）
        class_num = 200

    elif args.dataset == 'ImageNet1K':
        print("Processing ImageNet-1K...")
        # 客户端数量固定为 30
        args.num_clients = 30
        # 每个客户端 30 个私有类，减少公共类以更接近类增量(降低公共类持续重训的影响)
        # 30 客户端 -> 900 私有类，剩余 100 公共类（共 1000 类），每客户端 130 类、26 类/任务
        priv_num = 30

        # 每类预取 500 张作训练、100 张作测试（原数据集过大，自动下载子集）
        client_data, client_mask = ImageNet1K_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size,
            train_per_class=500,
            test_per_class=100).random_split()

        _, test_data = ImageNet1K_Data_Spliter(
            client_num=args.num_clients,
            task_num=args.tasks_global,
            private_class_num=priv_num,
            input_size=args.img_size,
            train_per_class=500,
            test_per_class=100).process_testdata(0)

        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
        class_num = 1000  # 必须准确

    # 服务器初始化
    num_clients = args.num_clients
    global_method = args.global_method
    dataset_name = args.dataset
    local_method = args.local_method
    batch_size = args.batch_size

    epoch = args.epochs_local
    lr = args.learning_rate
    task_num = args.tasks_global
    pretrain_model = args.pretrained_model

    alpha = args.alpha
    device = args.device
    pseudo_samples = args.pseudo_samples
    save_path = './model_data'

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    if dataset_name == "CIFAR100":
        class_num = 100
    elif dataset_name == "TinyImageNet":
        class_num = 200
    elif dataset_name == "TinyImageNet20C":
        class_num = 200
    elif dataset_name == "ImageNet1K":
        class_num = 1000
    else:
        class_num = 10

    # 创建Server（使用PackNet方法）
    myServer = Server(
        num_clients, task_num, dataset_name, lr, device, epoch, pseudo_samples,
        global_method, local_method, feature_extractor, client_data, batch_size,
        alpha, test_loader, class_num, client_mask, save_path,
        prune_ratio=args.prune_ratio,
        retrain_epochs=args.retrain_epochs,
        task_id_epochs=args.task_id_epochs
    )

    myServer.init_clients()
    all_eval_results = myServer.train_clients()
    return all_eval_results

# 运行3次实验
NUM_RUNS = 1
SEEDS = [args.seed + i for i in range(NUM_RUNS)]
all_runs_results = []

for run_idx, seed in enumerate(SEEDS):
    results = run_once(seed)
    all_runs_results.append(results)
    torch.cuda.empty_cache()

end = time.time()
print(f'\nTotal time: {end-start:.2f} seconds')

# 计算并输出平均值和标准差
print("\n" + "#"*60)
print(f"FINAL STATISTICS ACROSS {NUM_RUNS} RUNS")
print("#"*60)

num_checkpoints = len(all_runs_results[0])
dataset_name = args.dataset

for ckpt_idx in range(num_checkpoints):
    round_id = all_runs_results[0][ckpt_idx]['round']

    # all_tasks accuracy
    all_tasks_vals = [all_runs_results[r][ckpt_idx]['all_tasks'] for r in range(NUM_RUNS)]
    mean_all = np.mean(all_tasks_vals)
    std_all = np.std(all_tasks_vals)
    print(f'{dataset_name}: Round {round_id}, Overall accuracy: {mean_all:.2f}% ± {std_all:.2f}%')

    # per-task accuracy
    task_index = round_id // 5
    for t in range(task_index + 1):
        key = f'task_{t}'
        task_vals = [all_runs_results[r][ckpt_idx][key] for r in range(NUM_RUNS)]
        mean_t = np.mean(task_vals)
        std_t = np.std(task_vals)
        print(f'{dataset_name}: Task {t} accuracy: {mean_t:.2f}% ± {std_t:.2f}%')

print("#"*60)