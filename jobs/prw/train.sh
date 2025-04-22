# !/bin/bash

config_name="prw_dicl"
config_path="../../configs/dicl/${config_name}.py" 

CUDA_VISIBLE_DEVICES=2 python -u ../../tools/train.py ${config_path} 
# CUDA_VISIBLE_DEVICES=1 python -u ../../tools/train.py ../../configs/dicl/prw_dicl.py >train_prw_dicl.txt 2>&1

config_name='prw_dicl'
num_epoch='26'
CUDA_VISIBLE_DEVICES=2 python ../../tools/test.py ../../configs/dicl/${config_name}.py work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox --out work_dirs/${config_name}/results_1000.pkl >work_dirs/${config_name}/log_tmp_${num_epoch}.txt 2>&1
CUDA_VISIBLE_DEVICES=2 python ../../tools/test_results_prw.py ${config_name} >work_dirs/${config_name}/result_${config_name}_${num_epoch}.txt 2>&1