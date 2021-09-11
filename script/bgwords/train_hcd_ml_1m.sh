TRAIN_PATH="../../../../data/ml-1m/ml-1m/train2_aug.csv"
TEST_PATH="../../../../data/ml-1m/ml-1m/test2_aug.csv"
ITERATION=2000
SNAPSHOT=2000
ARCH_CONF_FILE=`pwd`/conf_ml_1m.json
ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64`

MODEL='global_bgword'
GRADIENT_CLIP=5
LEARNING_RATE=0.001
BATCH_SIZE=1024

BUCKETS='bgword/bgword_model_newem/'
INIT_CHECKPOINT='hcd_model_Global'
INIT_STEP=40000
INIT_STEPS_VAL="40000,60000,80000"




ALGORITHM="clus2pred-eval" #
INFERENCE="pred"
OUTPUT="results/hcd-C5" #

N_GROUPS=5 #
N_SAMPLING=1
EM_ROUNDS=10 #
EM_UNSUP_WARNINGUP=10 #

UNSUP_TEMP=1.0


######################################################################
CHECKPOINT_PATH=bgword/bgword_model_newem/hcd-C5 #
echo "Model save to ${CHECKPOINT_PATH}"
######################################################################

#
CUDA_VISIBLE_DEVICES=5 python ../../main/run_hcd.py --train_file ${TRAIN_PATH}\
								--test_file ${TEST_PATH}\
                        --output_path ${OUTPUT}\
							   --max_steps ${ITERATION}\
                           	--snapshot ${SNAPSHOT}\
                           	--checkpoint_dir ${CHECKPOINT_PATH}\
                           	--learning_rate ${LEARNING_RATE}\
                           	--batch_size ${BATCH_SIZE}\
                           	--model ${MODEL}\
                              --algo ${ALGORITHM}\
                              --infer_strategy ${INFERENCE}\
                           	--max_gradient_norm ${GRADIENT_CLIP}\
                           	--arch_config_path ${ARCH_CONF_FILE}\
                           	--buckets ${BUCKETS}\
                           	--init_checkpoint ${INIT_CHECKPOINT}\
                              --init_step ${INIT_STEP}\
                              --init_steps_val ${INIT_STEPS_VAL}\
                           	--n_groups ${N_GROUPS}\
                           	--n_sampling ${N_SAMPLING}\
                              --EM_rounds ${EM_ROUNDS}\
                              --EM_unsup_warningup ${EM_UNSUP_WARNINGUP}\
                              --unsup_temp ${UNSUP_TEMP}\
                           	#--arch_config ${ARCH_CONFIG_CONTENT}                              

echo "Training done: ${CHECKPOINT_PATH}"
