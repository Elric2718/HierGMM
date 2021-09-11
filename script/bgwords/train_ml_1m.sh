DATA_PATH="../../../../data/ml-1m/ml-1m/train.csv"
ITERATION=20000
SNAPSHOT=2000
ARCH_CONF_FILE=`pwd`/conf_ml_1m.json
ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64`

MODEL='bgword'
GRADIENT_CLIP=5                     # !
LEARNING_RATE=0.001
BATCH_SIZE=1024

######################################################################
CHECKPOINT_PATH=bgword/bgword_model_newem/baseline_model_single_${BATCH_SIZE}
echo "Model save to ${CHECKPOINT_PATH}"
######################################################################


CUDA_VISIBLE_DEVICES=1 python ../../main/debug.py --data_file ${DATA_PATH}\
							   --max_steps ${ITERATION}\
                           	--snapshot ${SNAPSHOT}\
                           	--checkpoint_dir ${CHECKPOINT_PATH}\
                           	--learning_rate ${LEARNING_RATE}\
                           	--batch_size ${BATCH_SIZE}\
                           	--model ${MODEL}\
                           	--max_gradient_norm ${GRADIENT_CLIP}\
                           	--arch_config_path ${ARCH_CONF_FILE}\
                           	#--arch_config ${ARCH_CONFIG_CONTENT}


echo "Training done: ${CHECKPOINT_PATH}"
