#!/bin/bash
#SBATCH --container-image ghcr.io\#misakagaming/586-project
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -t 5-0:00
source /opt/python3/venv/base/bin/activate
echo "nice"
cd graphcodebert
cd parser
bash build.sh
cd ..
source=java
target=cs
lr=1e-4
batch_size=16
beam_size=10
source_length=320
target_length=256
output_dir=saved_models/$source-$target
train_file=data/train.csv
dev_file=data/valid.csv
epochs=200
pretrained_model=microsoft/graphcodebert-base
tokenizer_name=tokenizer
pred_dir=/users/eray.erer/codet5/graphcodebert_archetype/predictions/$(date +%Y-%m-%d-%H-%M-%S)
mkdir -p $pred_dir
mkdir -p $output_dir

python3 run.py \
        --do_train \
        --do_eval \
        --model_type roberta \
        --source_lang $source \
        --target_lang $target \
        --model_name_or_path $pretrained_model \
        --tokenizer_name $pretrained_model \
        --config_name $pretrained_model \
        --train_filename $train_file \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size 10 \
        --train_batch_size $batch_size \
        --eval_batch_size $batch_size \
        --learning_rate 1e-4 \
        --num_train_epochs $epochs 2>&1 | tee $output_dir/train.log

batch_size=32
dev_file=data/valid.csv
test_file=data/test.csv
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py \
        --do_test \
        --model_type roberta \
        --source_lang $source \
        --target_lang $target \
        --model_name_or_path $pretrained_model \
        --tokenizer_name $pretrained_model \
        --config_name $pretrained_model \
        --load_model_path $load_model_path \
        --test_filename $test_file \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size 2>&1| tee $output_dir/test.log

cp -r $output_dir $pred_dir