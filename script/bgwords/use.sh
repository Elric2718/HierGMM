TRAIN_PATH="../../../../data/ml-1m/ml-1m/train2_aug.csv"
TEST_PATH="../../../../data/ml-1m/ml-1m/test2_aug.csv"
ITERATION=10000
MULTIPLIER=10000
SNAPSHOT=100000
ARCH_CONF_FILE=`pwd`/conf_ml_1m.json
ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64`

MODEL='global_bgword'
GRADIENT_CLIP=5                     
LEARNING_RATE=0.001
BATCH_SIZE=1024

BUCKETS='bgword/bgword_model_newem/'
INIT_CHECKPOINT='use_Global'
INIT_STEP=0


ALGORITHM="global"
INFERENCE="random"
OUTPUT="results/use"

N_GROUPS=1
N_SAMPLING=1
EM_ROUNDS=1
EM_UNSUP_WARNINGUP=1


######################################################################
CHECKPOINT_PATH=bgword/bgword_model_newem/use
echo "Model save to ${CHECKPOINT_PATH}"
######################################################################


######################################################################

timer=0
until [ 1 -gt 2 ]
do
counter=0
nvidia-smi | grep 16160 | cut -d'|' -f3 | cut -d'/' -f1 |sed 's/ //g' | sed 's/MiB//g' > gpu_stat.txt
while read -r line
do   
   if [ $line -lt 13000 ]
   then 
   echo "CUDA_VISIBLE_DEVICES=${counter} python ../../main/run_hcd.py --train_file ${TRAIN_PATH}\\" > job.sh
	echo "     						  --test_file ${TEST_PATH}\\" >> job.sh
   echo "                       --output_path ${OUTPUT}\\"  >> job.sh
	echo "			     			  --max_steps ${ITERATION}\\" >> job.sh
   echo "                       --multiplier ${MULTIPLIER}\\" >> job.sh
   echo "                         	--snapshot ${SNAPSHOT}\\" >> job.sh
   echo "                        	--checkpoint_dir ${CHECKPOINT_PATH}\\" >> job.sh
   echo "                        	--learning_rate ${LEARNING_RATE}\\" >> job.sh
   echo "                        	--batch_size ${BATCH_SIZE}\\" >> job.sh
   echo "                        	--model ${MODEL}\\" >> job.sh
   echo "                           --algo ${ALGORITHM}\\" >> job.sh
   echo "                           --infer_strategy ${INFERENCE}\\" >> job.sh
   echo "                        	--max_gradient_norm ${GRADIENT_CLIP}\\" >> job.sh
   echo "                        	--arch_config_path ${ARCH_CONF_FILE}\\" >> job.sh
   echo "                        	--buckets ${BUCKETS}\\" >> job.sh
   echo "                        	--init_checkpoint ${INIT_CHECKPOINT}\\" >> job.sh
   echo "                        	--init_step ${INIT_STEP}\\" >> job.sh
   echo "                        	--n_groups ${N_GROUPS}\\" >> job.sh
   echo "                        	--n_sampling ${N_SAMPLING}\\" >> job.sh
   echo "                           --EM_rounds ${EM_ROUNDS}\\" >> job.sh
   echo "                           --EM_unsup_warningup ${EM_UNSUP_WARNINGUP}\\" >> job.sh
                           	#--arch_config ${ARCH_CONFIG_CONTENT}    
   chmod 777 job.sh
   nohup ./job.sh > use.out 2>&1 &
   echo "Submit to GPU ${counter} at time ${timer}."
   sleep 2s
   fi  
   ((counter++))
   ((timer++))
done < gpu_stat.txt
sleep 5m
done  
#echo "Training done: ${CHECKPOINT_PATH}"
