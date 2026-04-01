

# 参数说明：
# --config_name: 模型配置目录，用于从配置初始化模型结构
# --tokenizer_name: 分词器目录
# --train_files: 训练数据文件路径（json/jsonl）
# --per_device_train_batch_size: 单卡 batch size
# --gradient_accumulation_steps: 梯度累积步数，有效 batch 会按该值放大
# --do_train: 启用训练
# --output_dir: 训练输出目录（checkpoint 与最终模型）
# --eval_strategy no: 关闭评估流程
# --learning_rate: 初始学习率
# --num_train_epochs: 训练轮数
# --warmup_steps: 学习率预热步数
# --logging_dir: 日志输出目录
# --logging_strategy steps: 按步数记录日志
# --logging_steps: 每隔多少步记录一次日志
# --save_strategy steps: 按步数保存 checkpoint
# --save_steps: 每隔多少步保存一次 checkpoint
# --preprocessing_num_workers: 数据预处理并行进程数
# --save_total_limit: 最多保留的 checkpoint 数量
# --seed: 随机种子
# --block_size: 文本切块长度（上下文窗口长度）
# --bf16: 使用 bfloat16 混合精度训练
# --gradient_checkpointing: 启用梯度检查点以节省显存
# --deepspeed: DeepSpeed 配置文件路径
# --report_to swanlab: 将训练指标上报到 swanlab

CUDA_VISIBLE_DEVICES=0,1
deepspeed pretrain.py \
    --config_name /home/ud202280977/LLM/models/Qwen3-0.6B-Base \
    --tokenizer_name /home/ud202280977/LLM/models/Qwen3-0.6B-Base \
    --processed_data_dir /home/ud202280977/LLM/dataset/processed_mobvoi_2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir /home/ud202280977/LLM/output/pretrain \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir /home/ud202280977/LLM/output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 12 \
    --fp16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \