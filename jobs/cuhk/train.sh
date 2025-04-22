#!/bin/bash

config_name="cuhk_dicl"
config_path="../../configs/dicl/${config_name}.py" 

CUDA_VISIBLE_DEVICES=6 python -u ../../tools/train.py ${config_path}
# CUDA_VISIBLE_DEVICES=1 python -u ../../tools/train.py ../../configs/dicl/cuhk_dicl.py >train_cuhk_dicl.txt 2>&1

config_name="cuhk_dicl"
config_path="../../configs/dicl/${config_name}.py"
num_epoch='26'
CUDA_VISIBLE_DEVICES=6 python -u ../../tools/test_personsearch.py $config_path  work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox >>result_${config_name}_${num_epoch}.txt 2>&1

num_epoch='25'
CUDA_VISIBLE_DEVICES=6 python -u ../../tools/test_personsearch.py $config_path  work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox >>result_${config_name}_${num_epoch}.txt 2>&1

num_epoch='24'
CUDA_VISIBLE_DEVICES=6 python -u ../../tools/test_personsearch.py $config_path  work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox >>result_${config_name}_${num_epoch}.txt 2>&1

num_epoch='23'
CUDA_VISIBLE_DEVICES=6 python -u ../../tools/test_personsearch.py $config_path  work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox >>result_${config_name}_${num_epoch}.txt 2>&1
