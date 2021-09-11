cd ../..
rm model.tar.gz
tar -czf model.tar.gz ./config ./loader ./main ./model ./util ./module
#cd script/bgwords
#
#ITERATION=150000
#SNAPSHOT=1000
#ARCH_CONF_FILE=`pwd`/conf.json
#ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64`
#
#GRADIENT_CLIP=5                     # !
#LEARNING_RATE=0.001,0.5,60000
#BATCH_SIZE=1024
#
#######################################################################
#LEARNING_RATE=0.001
#CHECKPOINT_PATH=bgword/bgword_model_newem/baseline_model_em_${BATCH_SIZE}_v3
#
#INIT_STEP=50000
#INIT_CHECKPOINT=bgword/bgword_model_newem/baseline_model_em_${BATCH_SIZE}_v3
#######################################################################
#
#echo ${CHECKPOINT_PATH}
#
## use time dataset
#PROJECT=graph_embedding_dev
#ROLEARN=acs:ram::1406399764179220:role/intern-oss
#
#GPU=100
#TF_VERSION=tensorflow180
#
#echo "Model save to ${CHECKPOINT_PATH}"
#
##--if_new_model=False
#
#/Users/yang/soft/odps_clt_release_64/bin/odpscmd -e "use ${PROJECT};${V100_FLAG}pai \
#        -name ${TF_VERSION} -project algo_public \
#        -Dscript=\"file://`pwd`/../../model.tar.gz\" \
#        -Dtables=\"odps://graph_embedding/tables/alipay_bgwords_new_feature_dataset/ds=20210511_v2/part=train\" \
#        -DentryFile=\"main/train.py\" \
#        -DgpuRequired=\"${GPU}\"\
#        -Dbuckets=\"oss://graph-interns/?host=oss-cn-zhangjiakou.aliyuncs.com&role_arn=${ROLEARN}\" \
#        -DuserDefinedParameters=\"--model=bgword_em --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} --gpu=${GPU} --max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOINT_PATH} --init_checkpoint=${INIT_CHECKPOINT} --init_step=${INIT_STEP}  \" \
#        "
#
#echo "Training done: ${CHECKPOINT_PATH}"
