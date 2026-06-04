# Monico Model — Training & Eval Commands

.PHONY: train eval tokenizer data clean

# Train 7B model (multi-GPU)
train-7b:
	torchrun --nproc_per_node=8 --nnodes=1 scripts/train.py \
		--model_size=7b \
		--seq_length=4096 \
		--batch_size=4 \
		--gradient_accumulation=8

# Train with DeepSpeed ZeRO-3 (large scale)
train-deepspeed:
	deepspeed --num_gpus=8 scripts/train.py \
		--deepspeed configs/deepspeed_zero3.json

# Train tokenizer on corpus
tokenizer:
	python scripts/train_tokenizer.py \
		--corpus_path=data/raw \
		--vocab_size=65536 \
		--output_path=tokenizer

# Run all evals
eval:
	python scripts/evaluate.py \
		--model_path=checkpoints/monico-7b-latest

# Process raw datasets
data:
	python scripts/process_datasets.py \
		--config=configs/dataset_config.yaml \
		--output_dir=data/processed

# Docker build
docker:
	docker build -t monico-model:latest .

clean:
	rm -rf data/processed checkpoints/tmp
