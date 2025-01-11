#!/bin/bash
#SBATCH --container-image ghcr.io\#misakagaming/586-project
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -t 5-0:00
source /opt/python3/venv/base/bin/activate
echo "nice"
cd codebert_sum
lang=java #programming language
lr=5e-5
batch_size=16
beam_size=10
source_length=256
target_length=128
data_dir=./data
output_dir=model/$lang
train_file=$data_dir/train.csv
dev_file=$data_dir/valid.csv
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
pred_dir=/users/eray.erer/codebert_sum/predictions/$(date +%Y-%m-%d-%H-%M-%S)
#<<com
mkdir -p $pred_dir
mkdir -p $output_dir
mkdir -p $pred_dir/$output_dir


python run.py --do_train \
        --do_eval \
        --model_type roberta \
        --model_name_or_path $pretrained_model \
        --train_filename $train_file \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --eval_batch_size $batch_size \
        --learning_rate $lr \
        --train_steps $train_steps \
        --eval_steps $eval_steps>&1 | tee $output_dir/train.log


lang=java #programming language
beam_size=10
batch_size=32
source_length=256
target_length=128
output_dir=model/$lang
data_dir=./data
dev_file=$data_dir/valid.csv
test_file=$data_dir/test.csv
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $test_model \
        --dev_filename $dev_file \
        --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size>&1 | tee $output_dir/test.log
cp -r $output_dir $pred_dir
com
lang=cs #programming language
lr=5e-5
batch_size=16
beam_size=10
source_length=256
target_length=128
data_dir=./data
output_dir=model/$lang
train_file=$data_dir/train.csv
dev_file=$data_dir/valid.csv
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
mkdir -p $output_dir
#<<com
python run_gen.py --do_train \
        --do_eval \
        --model_type roberta \
        --model_name_or_path $pretrained_model \
        --train_filename $train_file \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --eval_batch_size $batch_size \
        --learning_rate $lr \
        --train_steps $train_steps \
        --eval_steps $eval_steps>&1 | tee $output_dir/train.log
com
lang=cs #programming language
beam_size=10
batch_size=32
source_length=256
target_length=128
output_dir=model/$lang
data_dir=./data
dev_file=$data_dir/valid.csv
test_file=$data_dir/test.csv
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
#<<com
python run_gen.py --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $test_model \
        --dev_filename $dev_file \
        --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size>&1 | tee $output_dir/test.log

#cp -r $output_dir/*.output $pred_dir
#cp -r $output_dir/*.gold $pred_dir
#cp -r $output_dir/*.log $pred_dir
cp -r $output_dir $pred_dir
com
test_model=$pred_dir/$lang/checkpoint-best-bleu/pytorch_model.bin
cp -r $pred_dir/*.output .

python run_gen.py --do_test_final \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $test_model \
        --dev_filename $dev_file \
        --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size>&1 | tee $output_dir/test_final.log

cp -r $output_dir/test_final* $pred_dir/$lang