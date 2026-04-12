# base-llm-06b-seqmonkey-experiment

🧪 基于 `seq-monkey` 数据集从零开始（From Scratch）预训练 Qwen3-0.6B 模型的实验仓库。

---

## 🧪 项目简介
本项目旨在演示大语言模型（LLM）预训练的完整技术链路，包括数据预处理、分布式训练配置以及核心训练逻辑。

- **模型架构**：Qwen3-0.6B (From Scratch)
- **训练数据**：`seq-monkey` 数据集
- **训练优化**：基于 DeepSpeed ZeRO-2 实现多 GPU 显存优化，适配 A800/V100S 等多卡环境。
- **实验监控**：集成 SwanLab 进行实时训练指标监控。
- **项目定位**：**纯学习实验性质**。模型未经过大规模数据训练、验证与人类对齐，**不具备实际应用能力，禁止用于任何生产环境**。
- **模型权重**：[exp-base-model-0.6B-fp16-V1](https://modelscope.cn/models/gebaili/exp-base-model-0.6B-fp16-V1) (ModelScope)

---

## 📁 仓库文件说明
| 文件/目录 | 核心作用 |
| --- | --- |
| `offline_preprocess.py` | 数据集离线预处理脚本，将原始 JSONL 数据转换为模型训练格式 |
| `offline_preprocess.sh` | 预处理启动脚本，封装路径与参数配置 |
| `pretrain.py` | 核心预训练逻辑，包含模型初始化、数据加载与训练循环 |
| `pretrain.sh` | 预训练任务启动脚本，集成 DeepSpeed 分布式启动命令 |
| `ds_config_zero2.json` | DeepSpeed ZeRO-2 配置文件，优化显存与加速训练 |
| `requirements.txt` | 项目运行所需的依赖包列表 |

---

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.8+ 及 CUDA 环境，然后克隆仓库并安装依赖：

```bash
# 克隆仓库
git clone https://github.com/gezhao96/base-llm-06b-seqmonkey-experiment.git
cd base-llm-06b-seqmonkey-experiment

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理
在开始训练前，需要将原始数据处理为 `block_size` 长度的序列。

1. 修改 `offline_preprocess.sh` 中的 `--tokenizer_name`、`--train_files` 和 `--output_dir` 为你的本地路径。
2. 运行脚本：
```bash
bash offline_preprocess.sh
```

### 3. 模型预训练
使用 DeepSpeed 启动分布式训练。

1. 修改 `pretrain.sh` 中的路径参数（如 `--config_name`, `--tokenizer_name`, `--processed_data_dir`, `--output_dir`）。
2. 根据硬件情况调整 `per_device_train_batch_size` 和 `gradient_accumulation_steps`。
3. 启动训练：
```bash
bash pretrain.sh
```

---

## 🛠️ 核心配置说明
- **DeepSpeed ZeRO-2**：在 `ds_config_zero2.json` 中配置，有效降低单卡显存占用。
- **混合精度**：脚本默认开启 `fp16` 或 `bf16`（取决于硬件支持），提升训练效率。
- **SwanLab 监控**：训练指标会实时上传，需提前配置好 API Key。

---

## ⚖️ 许可证与免责声明
- 本项目采用 **Apache-2.0** 许可证。
- 本项目仅供学术交流与研究使用，作者不对因使用本项目代码或模型产生的任何后果负责。
