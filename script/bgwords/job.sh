CUDA_VISIBLE_DEVICES=3 python ../../main/run_hcd.py --train_file ../../../../data/ml-1m/ml-1m/train2_aug.csv\
     						  --test_file ../../../../data/ml-1m/ml-1m/test2_aug.csv\
                       --output_path results/use\
			     			  --max_steps 10000\
                       --multiplier 10000\
                         	--snapshot 100000\
                        	--checkpoint_dir bgword/bgword_model_newem/use\
                        	--learning_rate 0.001\
                        	--batch_size 1024\
                        	--model global_bgword\
                           --algo global\
                           --infer_strategy random\
                        	--max_gradient_norm 5\
                        	--arch_config_path /home/yyt193705/Projects/HierCloudDevice/code/rerank_model/script/bgwords/conf_ml_1m.json\
                        	--buckets bgword/bgword_model_newem/\
                        	--init_checkpoint use_Global\
                        	--init_step 0\
                        	--n_groups 1\
                        	--n_sampling 1\
                           --EM_rounds 1\
                           --EM_unsup_warningup 1\