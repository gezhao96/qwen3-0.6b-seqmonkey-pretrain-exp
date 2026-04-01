# qwen3-0.6b-seqmonkey-pretrain-exp

🧪 从零基于 seq-monkey 数据集预训练 Qwen3-0.6B 模型的实验代码仓库 | 仅用于学习研究，勿应用于任何生产环境。

---

## 🧪 项目简介
本项目进行了 **Qwen3-0.6B 模型架构从零开始（From Scratch）** 的完整预训练全流程：
- 训练数据：`seq-monkey` 数据集
- 训练优化：基于 DeepSpeed ZeRO-2 实现多GPU显存优化，适配 A800/V100S 等多卡训练环境
- 项目定位：**纯学习实验性质**，仅用于熟悉大语言模型（LLM）预训练的完整技术链路，模型未经过大规模数据训练、验证与人类对齐，**不具备实际应用能力，禁止用于任何生产环境**

---

## 📁 仓库文件说明（对应你当前的所有文件）
| 文件/目录 | 核心作用 |
| --- | --- |
| `LICENSE` | 项目开源许可证（Apache-2.0） |
| `README.md` | 本项目说明文档 |
| `ds_config_zero2.json` | DeepSpeed ZeRO-2 训练配置文件，用于多卡训练的显存优化与训练加速 |
| `offline_preprocess.py` | 数据集离线预处理Python脚本，将原始`seq-monkey`数据转换为模型训练所需的标准格式 |
| `offline_preprocess.sh` | 数据集预处理的Shell启动脚本，封装预处理命令与可配置参数 |
| `pretrain.py` | 核心预训练Python脚本，实现模型初始化、数据加载、训练循环等核心逻辑 |
| `pretrain.sh` | 预训练任务的Shell启动脚本，封装DeepSpeed启动命令与训练超参数 |
| `requirements.txt` | 项目运行所需的Python依赖包列表 |

---
```bash
# 1. 克隆仓库
git clone https://github.com/gezhao96/qwen3-0.6b-seqmonkey-pretrain-exp.git
cd qwen3-0.6b-seqmonkey-pretrain-exp

# 2. 安装项目依赖
pip install -r requirements.txt
