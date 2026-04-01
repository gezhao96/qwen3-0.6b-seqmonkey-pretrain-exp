CUDA_VISIBLE_DEVICES=0 python offline_preprocess.py \
    --tokenizer_name /home/ud202280977/LLM/models/Qwen3-0.6B-Base \
    --train_files /home/ud202280977/LLM/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl \
    --output_dir /home/ud202280977/LLM/dataset/processed_mobvoi_2048 \
    --preprocessing_num_workers 8 \
    --block_size 2048 \
    --overwrite_output_dir True