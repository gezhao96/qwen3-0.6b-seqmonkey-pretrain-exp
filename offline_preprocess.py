'''
离线预处理脚本
作用：
1. 加载 json/jsonl 原始文本数据
2. 使用 tokenizer 做分词
3. 按 block_size 拼接并切块
4. 保存为 HuggingFace datasets 的磁盘格式，供训练阶段直接 load_from_disk 使用
'''

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List

import datasets
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, HfArgumentParser


logger = logging.getLogger(__name__)


@dataclass
class PreprocessArguments:
    """
    离线预处理参数
    """
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer 路径"}
    )
    train_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "原始训练数据路径，支持 json/jsonl，可传多个"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "离线预处理结果保存路径"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "文本块长度"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "预处理并行进程数"}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "若输出目录已存在，是否覆盖"}
    )


def main():
    parser = HfArgumentParser(PreprocessArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # 日志设置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.setLevel(logging.INFO)

    # 参数检查
    if args.tokenizer_name is None:
        raise ValueError("tokenizer_name 不能为空")
    if args.train_files is None or len(args.train_files) == 0:
        raise ValueError("train_files 不能为空")
    if args.output_dir is None:
        raise ValueError("output_dir 不能为空")

    if os.path.exists(args.output_dir):
        if args.overwrite_output_dir:
            logger.warning(f"输出目录已存在，准备覆盖: {args.output_dir}")
        else:
            raise ValueError(
                f"输出目录已存在: {args.output_dir}，如需覆盖请传 --overwrite_output_dir True"
            )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    logger.info("完成 tokenizer 加载")
    logger.info(f"tokenizer 配置地址：{args.tokenizer_name}")

    # 某些 tokenizer 没有 pad_token，离线切块虽不一定必须，但补上更稳妥
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer 无 pad_token，已自动设置为 eos_token")

    # 加载原始数据
    ds = load_dataset("json", data_files=args.train_files)
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{args.train_files}")
    logger.info(f'训练文件总数: {len(ds["train"])}')

    # 自动识别文本列
    column_names = list(ds["train"].features)
    logger.info("训练集特征：%s", column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]
    logger.info("默认文本列：%s", text_column_name)

    # tokenize 函数
    def tokenize_function(examples):
        texts = [str(item) for item in examples[text_column_name]]
        output = tokenizer(texts)
        return output

    # tokenize
    logger.info("开始进行 tokenize")
    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset"
    )
    logger.info("完成 tokenize")

    # block_size 处理
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning("tokenizer 支持大于 1K 的上下文长度，默认设置为 1K")
            block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"设定的块长 ({args.block_size}) 大于模型上下文长度 "
                f"{tokenizer.model_max_length}，将自动截断"
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    logger.info(f"最终 block_size = {block_size}")

    # 分块函数
    def group_texts(examples):
        # 拼接
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

        # 总长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # 裁成 block_size 的整数倍
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            total_length = 0

        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("开始进行文本分块")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"文本分块到 {block_size}",
        batch_size=40000,
    )
    logger.info("完成文本分块")

    # 保存
    if os.path.exists(args.output_dir) and args.overwrite_output_dir:
        import shutil
        shutil.rmtree(args.output_dir)

    logger.info(f"开始保存离线预处理结果到: {args.output_dir}")
    lm_datasets.save_to_disk(args.output_dir)
    logger.info("离线预处理完成")


if __name__ == "__main__":
    main()