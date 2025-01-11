#!/bin/bash
#SBATCH --container-image ghcr.io\#misakagaming/graphcodebert-env
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -t 1-0:00
source /opt/python3/venv/base/bin/activate
echo "nice"
cd parser
bash build.sh
cd ..
source=coolgen
target=java
lr=1e-4
batch_size=8
beam_size=10
source_length=320
target_length=256
output_dir=saved_models/$source-$target
train_file=train_gen.txt,train_java.txt
dev_file=valid_gen.txt,valid_java.txt
train_directories=Training/train.coolgen,Training/train.java
dev_directories=Training/valid.coolgen,Training/valid.java
epochs=200
pretrained_model=Salesforce/codet5-base
tokenizer_name=tokenizer_coolgen

mkdir -p $output_dir
python3 run.py \
		--do_train \
		--do_eval \
		--model_type codet5 \
		--source_lang $source \
		--model_name_or_path $pretrained_model \
		--tokenizer_name $tokenizer_name \
		--config_name $pretrained_model \
		--train_directories $train_directories \
		--dev_directories $dev_directories \
		--output_dir $output_dir \
		--max_source_length $source_length \
		--max_target_length $target_length \
		--beam_size 10 \
		--train_batch_size $batch_size \
		--eval_batch_size $batch_size \
		--learning_rate 1e-4 \
		--num_train_epochs $epochs 2>&1 | tee $output_dir/train.log

batch_size=4
test_file=test_gen.txt,test_java.txt
test_directories=Training/test.coolgen,Training/test.java
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py \
        --do_test \
        --model_type codet5 \
        --source_lang $source \
        --model_name_or_path $pretrained_model \
        --tokenizer_name $tokenizer_name \
        --config_name $pretrained_model \
        --load_model_path $load_model_path \
        --dev_directories $dev_directories \
        --test_directories $test_directories \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size 2>&1| tee $output_dir/test.log
		
#python run.py --do_train --do_eval --model_type codet5 --source_lang coolgen --model_name_or_path Salesforce/codet5-base --tokenizer_name tokenizer_coolgen --config_name Salesforce/codet5-base --train_directories ../gen2java/Training/train.coolgen,../gen2java/Training/train.java --dev_directories ../gen2java/Training/valid.coolgen,../gen2java/Training/valid.java --output_dir saved_models/coolgen-java --max_source_length 320 --max_target_length 256 --beam_size 10 --train_batch_size 8 --eval_batch_size 8 --learning_rate 1e-4 --num_train_epochs 200 2>&1 | tee saved_models/coolgen-java/train.log