# LLM_PEFT：参数高效微调与评估

本项目基于 [AI6130_Assignment2](https://github.com/duyngtr16061999/AI6130_Assignment2)，在原项目基础上调整了模型准备和实验执行流程：先下载基础模型，再由 `task.json` 驱动批量微调与微调模型评估；也可以独立运行基础模型（baseline）评估。

> 主要流程：`download_models.py` → 编辑/检查 `task.json` → `run_tasks.py`（训练后自动评估微调模型）→ `evaluate_baseline.py`（评估未微调模型）。

## 项目结构

```text
LLM_PEFT/
├── download_models.py          # 下载并固定本地基础模型版本
├── task.json                   # 批量训练和微调后评估的任务清单
├── task_test.json              # 小规模/测试任务配置
├── run_tasks.py                # 按任务清单批量执行、记录状态并支持恢复
├── finetune.py                 # 单项 PEFT 训练实现
├── finetune_single_task.py     # 单项训练辅助入口
├── evaluate.py                 # 微调模型评估；run_tasks.py 会调用它
├── evaluate_baseline.py        # 批量评估未微调基础模型
├── evaluate_single_task.py     # 单任务评估辅助脚本
├── generate.py                 # 文本生成相关代码
├── export_hf_checkpoint.py     # 导出 Hugging Face 格式 checkpoint
├── export_state_dict_checkpoint.py # 导出/转换 state dict checkpoint
├── summarize_runs.py           # 汇总训练、评估和 baseline 结果
├── requirements.txt            # Python 依赖
├── models/                     # download_models.py 下载的本地基础模型
├── ft-training_set/            # 微调训练数据
├── dataset/                    # 评估数据；每个数据集目录含 test.json
├── peft/                       # 项目使用的 PEFT 源码
├── trained_models/              # 微调输出、adapter 和 checkpoint
└── runs/                       # 批次状态、日志、预测结果和 baseline 结果
```

## 环境准备

建议在独立的 Python 环境中安装依赖。项目依赖 PyTorch、Transformers、Datasets、PEFT 相关组件；GPU 训练时请安装与 CUDA 环境匹配的 PyTorch 版本。当前 `requirements.txt` 固定了部分版本（例如 PyTorch 2.3.1、Transformers 4.35.2）。

在项目目录执行：

```powershell
cd E:\Python_projects\LLM_PEFT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如 PowerShell 阻止激活脚本，可改用 Anaconda Prompt 激活环境，或在已激活的 Python 环境中直接执行安装命令。

## 运行流程

### 1. 下载基础模型

```powershell
python download_models.py
```

脚本会从 Hugging Face 下载以下模型到项目本地 `models/`，并记录下载时间、仓库和 revision 到 `models/models_manifest.json`：

- `openlm-research/open_llama_3b_v2` → `models/open_llama_3b_v2/`
- `TinyLlama/TinyLlama_v1.1` → `models/TinyLlama_v1.1/`

首次下载需要网络访问和足够的磁盘空间。训练及评估脚本启用了 Hugging Face 离线模式，会从本地模型目录读取文件；若模型缺失或下载不完整，需要先重新下载。

### 2. 配置批量任务

编辑项目根目录的 `task.json`。顶层常用字段：

- `run_name`：本批实验名称；默认输出目录为 `runs/<run_name>/`。
- `continue_on_error`：单个任务失败后是否继续其他任务。
- `tasks`：任务数组，每项需有唯一的 `task_id`。

每个任务可包含：

- `train`：传给 `finetune.py` 的训练参数，例如 `base_model`、`data_path`、`output_dir`、`adapter_name`、batch size、epoch 和早停设置。模型、训练数据和输出目录均使用相对于项目根目录的路径。
- `evaluations`：该训练任务完成后要执行的评估。每个评估需提供 `eval_id`、`model`、`adapter`、`dataset` 和 `base_model`。如果没有显式提供 `lora_weights`，调度器会使用同一任务的 `train.output_dir`。

当前 `task.json` 的示例任务使用 `math_7k.json` 训练 OpenLLaMA-3B 或 TinyLlama，并在 AQuA、AddSub 数据集上评估；配置同时包含 LoRA、AdapterP、单 epoch 和 early stopping 实验。评估数据需位于 `dataset/<数据集名>/test.json`。

`task_test.json` 可用于检查配置和流程。正式运行前应确认其中模型路径、数据路径和输出目录符合当前环境。

### 3. 批量微调并评估微调模型

```powershell
python run_tasks.py --config task.json
```

测试配置：

```powershell
python run_tasks.py --config task_test.json
```

调度器按任务顺序执行训练，然后调用 `evaluate.py` 对该任务配置的评估数据集生成预测。任务状态和日志会保存在 `runs/<run_name>/`。中断后重新运行相同命令可继续未完成任务和评估；训练 checkpoint 位于各任务的 `train.output_dir`。已记录过训练进度后，不要直接修改同一个 `task_id` 的训练参数；调整训练配置时使用新的 `task_id` 和 `output_dir`。

主要输出：

```text
runs/<run_name>/
├── state.json                 # 各任务、训练和评估状态
├── logs/                      # 训练和评估日志
└── results/<task_id>/         # 评估预测 JSON 和 JSONL 进度文件
```

完整模型训练产物和 adapter/checkpoint 写入 `task.json` 中对应任务的 `train.output_dir`，通常在 `trained_models/` 下。

### 4. 评估未微调基础模型（baseline）

```powershell
python evaluate_baseline.py
```

脚本读取根目录的 `task.json`，从其中收集基础模型、数据集和生成参数组合，加载基础模型但不加载 adapter，并评估对应测试集。默认结果目录为：

```text
runs/<run_name>/baseline_results/
```

每组配置会使用两种提示模板（`relaxed` 和 `peft_template`）各生成一份结果；JSONL 文件用于断点续跑，最终 JSON 文件保存完整预测和准确率相关字段。baseline 是独立流程，不会由 `run_tasks.py` 自动触发。

### 5. 查看和汇总结果

- 检查 `runs/<run_name>/state.json` 了解各训练任务和评估是否完成。
- 查看 `runs/<run_name>/logs/` 下相应日志排查运行问题。
- 查看 `runs/<run_name>/results/` 中的微调模型预测结果。
- 查看 `runs/<run_name>/baseline_results/` 中的基础模型结果。
- 运行汇总脚本查看已有实验信息：

```powershell
python summarize_runs.py
```

## 在 Linux/HPC 上运行

项目提供了 SLURM 示例脚本 `tc2run.sh`、`tc2run_2.sh` 和 `tc2run_baseline.sh`。脚本针对集群环境，使用前请检查分区、GPU、节点、conda 环境名和执行命令，并确认项目及模型数据在计算节点可访问。Windows 本地环境直接使用前述 `python ...` 命令。

## 许可与数据

项目根目录包含 `LICENSE` 和 `DATA_LICENSE`。模型、训练数据及评估数据可能分别适用各自的上游许可；使用或分发前请核对对应来源的许可条款。
