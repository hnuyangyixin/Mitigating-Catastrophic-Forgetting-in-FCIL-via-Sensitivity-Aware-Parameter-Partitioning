# Reproducibility notes

This release preserves the selected source files byte-for-byte. Packaging does not establish numerical reproduction of the paper.

## Paper settings

The supplied manuscript reports CIFAR-100 and TinyImageNet, 3 clients, 5 tasks, Adam, learning rate 0.001, batch size 32, 30 local epochs, three repeated experiments, and regularization coefficient 0.1. It also includes a 10-task CIFAR-100 comparison.

## Differences and constraints in the current source

- The paper states Dirichlet concentration 0.1. The current CIFAR-100 and TinyImageNet splitters use `np.ones(self.client_num)` (concentration 1). CIFAR-100 rejects splits with a proportion below 0.15. These settings have not been silently changed.
- Repetition counts in `main.py`: FedAvg = 2; FedProx = 3; PackNet = 1; GLFC = 1; FCIL_LD = 3; FedGP = 3; FedSAP = 3. Seeds are derived from the requested starting seed. Change `NUM_RUNS` to 3 for all methods when following the manuscript's repetition count.
- FedSAP uses five communication rounds per task, as hard-coded in `Server.py` and `client.py`.
- FedSAP's ICSR coefficient is hard-coded as 0.1 in `client.py`; the `--alpha` argument is not a general override of this coefficient or the data partition concentration.
- Changing the number of clients requires reviewing the fixed private-class counts and partition constraints, rather than only changing a command-line flag.
- Extra branches such as CIFAR10, MNIST, TinyImageNet20C, and ImageNet1K are retained from the original code; they are not the two main benchmarks described in the manuscript.
- The auxiliary `FedSAP/classifiar_prox.py` contains an unquoted `task_mask.pt` default and requires cleanup before standalone use. It is not imported by the main training entry point.
- No original environment lockfile or upstream baseline source/license records were available in the selected directories.

## Validation scope

All packaged Python files were parsed for syntax. Entry-point help checks and any additional validation are recorded in VALIDATION.md. No full GPU training or paper-result reproduction was performed.
