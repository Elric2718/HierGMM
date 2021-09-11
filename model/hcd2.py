from __future__ import absolute_import
import argparse
import logging

import tensorflow as tf
import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import os
import sys
import copy
sys.path.append("../")



import model
import loader
from config import TrainConfig
from config import ArchitectureConfig
from config import LearningRate
from util import env
from util import path
from util import args_processing as ap
from util import ModeKeys


EPSILON=0.
GLOBAL="Global"
class HierCloudDevice():
	def __init__(self, 
		args,\
		algo = "global",\
		n_groups = 2,\
		config = None,\
		init_checkpoint = None,\
		global_checkpoints = None,\
		log_every=100,\
		continue_training=True,\
		logz=None):
		
		self.args = args
		self.algo = algo
		self.n_groups = n_groups
		self.logz = logz

		self.config = config
		self.init_checkpoint = init_checkpoint
		self.global_checkpoints = global_checkpoints
		self.log_every = log_every
		self.continue_training = continue_training

		self._configure_algo_modes()
		self.model_meta = model.get_model_meta(self.args.model)  # type: model.ModelMeta

	    # Load architecture configuration
		self.arch_conf = ap.parse_arch_config_from_args(self.model_meta, self.args)  # type: ArchitectureConfig

	    # load data
		self._configure_data()

		# build unsupervised models
		self._build_unsup_models()

		# build supervised models
		self.sup_model_dirs = []
		self.sup_last_checkpoints = global_checkpoints
		self.sup_this_checkpoints = global_checkpoints
		self._create_model_dir_checkpoint(1)

		#self._build_sup_models(config, init_checkpoint, global_checkpoints, log_every)

		# create output folder
		if not os.path.exists(self.args.output_path):
			os.makedirs(self.args.output_path)

	def _configure_algo_modes(self):
		if self.algo == "global":		
			assert self.n_groups == 1, "Global model should only have one group."
			self.group_sets = ["Global"]
			self.training_algo_modes = ("M-sup" for _ in range(self.args.EM_rounds))			
		elif self.algo == "clus2pred":
			assert self.n_groups > 1, "Clus2Prediction model should have more than one group."
			self.group_sets = [f"clus2pred_G{group_id}" for group_id in range(self.n_groups)] # add global model in case
			self.training_algo_modes = (["E-unsup", "M-unsup"][iter%2] if iter < 2 * self.args.EM_unsup_warningup else "M-sup" 
				for iter in range(2 * self.args.EM_unsup_warningup + self.args.EM_rounds))
		elif self.algo == "hier":
			assert self.n_groups > 1, "Hierarchical model should have more than one group."
			self.group_sets = [f"hier_G{group_id}" for group_id in range(self.n_groups)]  # add global model in case
			self.training_algo_modes = (["E-unsup", "M-unsup"][iter%2] if iter < 2 * self.args.EM_unsup_warningup else ["E-both", "M-both"][(iter - 2 * self.args.EM_unsup_warningup)%2]
				for iter in range(2 * self.args.EM_unsup_warningup + 2 * self.args.EM_rounds))
		elif self.algo == "moe":
			assert self.n_groups > 1, "Hierarchical model should have more than one group."
			self.group_sets = [f"moe_G{group_id}" for group_id in range(self.n_groups)]  # add global model in case
			self.training_algo_modes = (["E-unsup", "M-unsup"][iter%2] if iter < 2 * self.args.EM_unsup_warningup else ["E-both", "M-both"][(iter - 2 * self.args.EM_unsup_warningup)%2]
				for iter in range(2 * self.args.EM_unsup_warningup + 2 * self.args.EM_rounds))
		elif self.algo == "global-eval":		
			assert self.n_groups == 1, "Global model should only have one group."
			self.group_sets = ["Global"]
			self.training_algo_modes = (["M-sup", "Eval"][iter%2] for iter in range(2 * self.args.EM_rounds))			
		elif self.algo == "clus2pred-eval":
			assert self.n_groups > 1, "Clus2Prediction model should have more than one group."
			self.group_sets = [f"clus2pred_G{group_id}" for group_id in range(self.n_groups)]  # add global model in case
			self.training_algo_modes = (["E-unsup", "M-unsup"][iter%2] if iter < 2 * self.args.EM_unsup_warningup else ["M-sup", "Eval"][(iter-2 * self.args.EM_unsup_warningup)%2]
				for iter in range(2 * self.args.EM_unsup_warningup + 2 * self.args.EM_rounds))
		elif self.algo == "hier-eval":
			assert self.n_groups > 1, "Hierarchical model should have more than one group."
			self.group_sets = [f"hier_G{group_id}" for group_id in range(self.n_groups)]  # add global model in case
			self.training_algo_modes = (["E-unsup", "M-unsup"][iter%2] if iter < 2 * self.args.EM_unsup_warningup else ["E-both", "M-both", "Eval"][(iter - 2 * self.args.EM_unsup_warningup)%3]
				for iter in range(2 * self.args.EM_unsup_warningup + 3 * self.args.EM_rounds))
		elif self.algo == "moe-eval":
			assert self.n_groups > 1, "Hierarchical model should have more than one group."
			self.group_sets = [f"moe_G{group_id}" for group_id in range(self.n_groups)]  # add global model in case
			self.training_algo_modes = (["E-unsup", "M-unsup"][iter%2] if iter < 2 * self.args.EM_unsup_warningup else ["E-both", "M-both", "Eval"][(iter - 2 * self.args.EM_unsup_warningup)%3]
				for iter in range(2 * self.args.EM_unsup_warningup + 3 * self.args.EM_rounds))
		else:
			raise NotImplementedError
		# add global models for evaluation
		global_count = 0	
		for global_step in self.args.init_steps_val.split(","):
			self.group_sets.append(GLOBAL + f"-{int(global_step)}")
			global_count+=1 
		self.num_global_models = global_count

		
	def _configure_data(self): 
		#self.required_fields = set(self.model_meta.data_loader_builder.required_fields()) # index of used features
		fields_category_idxs = [[self.arch_conf.model_config.all_fields.index(str(x)) for x in cat] 
								for cat in self.arch_conf.model_config.fields_category]
		self.user_idxs = fields_category_idxs[0]
		self.item_idxs = fields_category_idxs[1]
		self.hist_idxs = fields_category_idxs[2]
		self.label_idx = self.arch_conf.model_config.all_fields.index(str('label'))
		self.emb_idx = self.arch_conf.model_config.all_fields.index(str('emb'))
	
	def _build_unsup_models(self): 
		# build GMM
		self.mus = {}
		self.sigma2s = {}
		self.pop_weights = np.ones(shape = [self.n_groups])/self.n_groups
		

	def _initialize_unsup_models(self):	# could be cleaned up by passing emb instead of using self.emb
		d_e = self.emb.shape[1]
		emb_col_max = self.emb.max(axis = 0) # 
		emb_col_min = self.emb.min(axis = 0) # 
		emb_col_std = self.emb.std(axis = 0) # 
		for group_id in range(self.n_groups):
			#self.mus[self.group_sets[group_id]] = np.random.uniform(size = [d_e]) * (emb_col_max - emb_col_min) + emb_col_min
			self.mus[self.group_sets[group_id]] = np.squeeze(np.quantile(self.emb, q = (group_id + 1)/(self.n_groups + 1), axis = 0))
			self.sigma2s[self.group_sets[group_id]] = np.ones(shape = [d_e]) * emb_col_std

	def _build_sup_models(self, config, init_checkpoint, model_dir, log_every):
		#assert set(self.group_sets[self.n_groups:]) == set([key for key in global_checkpoints.keys()]), "Global model names not matched."
		if not config:
			config = tf.estimator.RunConfig(
				session_config=tf.ConfigProto(
					gpu_options=tf.GPUOptions(allow_growth=False),
					allow_soft_placement=True,
					),
				save_checkpoints_steps=self.args.snapshot,
				keep_checkpoint_max=40,
				train_distribute=None
				)

		# build models (supervised)
		"""
		self.rank_models = {}
		self.estimators = {}
		for group_id, group_name in enumerate(self.group_sets):
			self.rank_models[self.group_sets[group_id]] = self.model_meta.model_builder(
				arch_config=self.arch_conf,
				train_config=TrainConfig(
					learning_rate=LearningRate(self.args.learning_rate),
					batch_size=self.args.batch_size,
					max_gradient_norm=self.args.max_gradient_norm,
					log_every=log_every,
					init_checkpoint=init_checkpoint if group_id < self.n_groups else global_checkpoints[group_name]
					)
				)

			self.estimators[self.group_sets[group_id]] = tf.estimator.Estimator(
				model_fn=self.rank_models[self.group_sets[group_id]].model_fn,
				model_dir=self.args.checkpoint_dir + '_' + self.group_sets[group_id],
				config=config
	        )
	      """

		rank_model = self.model_meta.model_builder(
			arch_config=self.arch_conf,
			train_config=TrainConfig(
				learning_rate=LearningRate(self.args.learning_rate),
				batch_size=self.args.batch_size,
				max_gradient_norm=self.args.max_gradient_norm,
				log_every=log_every,
				init_checkpoint=init_checkpoint
				)
			)

		estimator = tf.estimator.Estimator(
			model_fn=rank_model.model_fn,
			model_dir=model_dir,
			config=config
		)

		return estimator

	def _create_model_dir_checkpoint(self, counter):
		self.sup_model_dirs.append({})
		for group_id, group_name in enumerate(self.group_sets):
			self.sup_model_dirs[-1][group_name] = os.path.join(self.args.checkpoint_dir, group_name + f"_{counter}")

		for group_id in range(self.n_groups):
			if len(self.sup_model_dirs) == 1:
				self.sup_this_checkpoints[self.group_sets[group_id]] = self.init_checkpoint # for evaluation and prediction			
				self.sup_last_checkpoints[self.group_sets[group_id]] = self.init_checkpoint # for training
			else:
				self.sup_this_checkpoints[self.group_sets[group_id]] = ap.get_init_checkpoint_file(self.sup_model_dirs[-2][group_name], '', None) 
				self.sup_last_checkpoints[self.group_sets[group_id]] = self.sup_this_checkpoints[self.group_sets[group_id]] if self.continue_training else self.init_checkpoint


		
	def _train(self, file_path):
		# load data
		working_data = pd.read_csv(file_path, sep = ";", header = None)
		self.emb = np.array([x.split(',') for x in working_data[self.emb_idx].values]).astype(float)
		self.labels = working_data[self.label_idx].values.astype(int)
		self._initialize_unsup_models()

		mode_counter = defaultdict(int)
		for self.step_mode in self.training_algo_modes:	
			mode_counter[self.step_mode] += 1
			### logging	
			self.logz.log_tabular("Step", self.step_mode + "-" + str(mode_counter[self.step_mode]) + "-" + str(EM_iter))
			### logging
			if self.step_mode.startswith("E-"):
				print("=======" * 10 + "\n" + f"Take E step: {mode_counter[self.step_mode]}-{EM_iter}.\n" + "=======" * 10)
				# E steppe
				self._take_E_step(file_path) 

			if self.step_mode.startswith("M-"):
				print("=======" * 10 + "\n" + f"Take M step: {mode_counter[self.step_mode]}-{EM_iter}.\n" + "=======" * 10)									
				# M step
				self._take_M_step(file_path)
				# create model dir and init_checkpoint for next round		
				if not self.step_mode.endswith("unsup"): 
					self._create_model_dir_checkpoint(mode_counter[self.step_mode])
			### logging
			self.logz.dump_tabular()
			### logging

	def _train_and_eval(self, file_path, test_file, infer_strategy):
		assert self.algo.endswith("eval"), "Need 'eval'-type algorithm to use train_and_eval."
		# load data
		working_data = pd.read_csv(file_path, sep = ";", header = None)
		self.emb = np.array([x.split(',') for x in working_data[self.emb_idx].values]).astype(float)
		self.labels = working_data[self.label_idx].values.astype(int)
		self._initialize_unsup_models()

		self.output = {"Round":[], "size": [], "prop": [], self.algo: [], GLOBAL+"-all":[],\
		 GLOBAL+"-RocAUC-mac": [], self.algo + "-RocAUC-mac": [],\
		 GLOBAL+"-RocAUC-mic": [], self.algo + "-RocAUC-mic": []}
		for global_name in self.group_sets[self.n_groups:]:
			self.output[global_name] = []
		n_test = sum(1 for _ in open(test_file))
		mode_counter = defaultdict(int)
		for EM_iter, self.step_mode in enumerate(self.training_algo_modes):	
			### logging	
			self.logz.log_tabular("Step", self.step_mode + "-" + str(mode_counter[self.step_mode]) + "-" + str(EM_iter))
			### logging
			if self.step_mode.startswith("E-"):
				print("=======" * 10 + "\n" + f"Take E step: {mode_counter[self.step_mode]}-{EM_iter}.\n" + "=======" * 10)
				# E steppe
				self._take_E_step(file_path) 

				### store parameters
				pd.DataFrame(self.sample_weights).to_csv(os.path.join(self.args.output_path, f"sample_weights_{EM_iter}.csv"), sep = ",")
				pd.DataFrame(self.sample_weights_tilde).to_csv(os.path.join(self.args.output_path, f"sample_weights_tilde_{EM_iter}.csv"), sep = ",")
			if self.step_mode.startswith("M-"):
				print("=======" * 10 + "\n" + f"Take M step: {mode_counter[self.step_mode]}-{EM_iter}.\n" + "=======" * 10)
				# M step
				self._take_M_step(file_path)
				# create model dir and init_checkpoint for next round
				if not self.step_mode.endswith("unsup"): 
					self._create_model_dir_checkpoint(mode_counter[self.step_mode])
			if self.step_mode.startswith("Eval"):
				print("=======" * 10 + "\n" + f"Evaluation: {mode_counter[self.step_mode]}-{EM_iter}.\n" + "=======" * 10)
				test_labels = np.squeeze(pd.read_csv(test_file, sep = ";", header = None)[self.label_idx].values.astype(int))
				# counter
				mode_counter[self.step_mode] += 1
				self.output["Round"].append(mode_counter[self.step_mode] - 1)				
				
				# size
				assignments, assign_weights = self._infer_cluster(test_file, infer_strategy)   
				sizes = ','.join([str(g_id) + ":" + str(np.sum(assignments == g_id)) if assignments is not None else f"0:{n_test}" for g_id in range(self.n_groups)]) 
				self.logz.log_tabular("Assignment", sizes)
				self.output["size"].append(sizes)

				pd.DataFrame(assignments).to_csv(os.path.join(self.args.output_path, f"assignments_{infer_strategy}_{EM_iter}.csv"), sep = ",")
				pd.DataFrame(assign_weights).to_csv(os.path.join(self.args.output_path, f"weights_{infer_strategy}_{EM_iter}.csv"), sep = ",")

				# prop
				props = ','.join([str(g_id) + ":" + str(round(np.mean(test_labels[np.where(assignments == g_id)[0]]), 4)) if assignments is not None else f"0:{round(np.mean(test_labels), 4)}" for g_id in range(self.n_groups)]) 
				self.output['prop'].append(props)

				# global on each partition
				results_global = {self.group_sets[self.n_groups + global_id]: {} for global_id in range(self.num_global_models)}
				for group_id in range(self.n_groups):	
					for global_id in range(self.num_global_models):
						results_global[self.group_sets[self.n_groups + global_id]][group_id] = self._eval(test_file, assignments, group_id, self.n_groups+global_id)
				for global_name in self.group_sets[self.n_groups:]:
					self.output[global_name].append(','.join([str(key) + ":" + str(round(val['loss'], 4)) for key, val in results_global[global_name].items()]))
				
				# algo on each partition
				results_algo = {}
				for group_id in range(self.n_groups):	
					results_algo[group_id] = self._eval(test_file, assignments, group_id, group_id)
				self.output[self.algo].append(','.join([str(key) + ":" + str(round(val['loss'], 4)) for key, val in results_algo.items()]))

				# global on all data
				results_global_all = []
				for global_id in range(self.num_global_models):
					results_global_all.append(self._eval(test_file, None, 0, self.n_groups + global_id))
				self.output[GLOBAL + "-all"].append(','.join([str(round(val['loss'], 4)) for val in results_global_all]))
				

				# ROC-AUC of global (macro)
				results_global_rocauc = []
				for global_id, global_name in enumerate(self.group_sets[self.n_groups:]):
					scores = np.array([prediction['score'] for _, prediction in enumerate(self._predict(test_file, None, 0, global_id + self.n_groups))])
					results_global_rocauc.append(roc_auc_score(test_labels, scores, average = "macro"))
				self.output[GLOBAL + "-RocAUC-mac"].append(','.join([str(round(val, 4)) for val in results_global_rocauc]))

				# ROC-AUC of algo (macro)
				results_algo_rocauc = []
				for group_id in range(self.n_groups):
					if np.sum(np.array(assignments) == group_id) == 0:
						continue
					par_scores = np.array([prediction['score'] for prediction in self._predict(test_file, assignments, group_id, group_id)])
					par_scores = np.transpose([np.where(np.array(assignments) == group_id)[0], par_scores])										
					results_algo_rocauc.append(par_scores)
				results_algo_rocauc = np.concatenate(results_algo_rocauc, axis = 0)
				results_algo_rocauc = roc_auc_score(test_labels, np.squeeze(results_algo_rocauc[np.argsort(results_algo_rocauc[:,0]), 1]), average = "macro")
				self.output[self.algo + "-RocAUC-mac"].append(round(results_algo_rocauc, 4))

				# ROC-AUC of global (micro)
				results_global_rocauc = []
				for global_id, global_name in enumerate(self.group_sets[self.n_groups:]):
					scores = np.array([prediction['score'] for _, prediction in enumerate(self._predict(test_file, None, 0, global_id + self.n_groups))])
					results_global_rocauc.append(roc_auc_score(test_labels, scores, average = "micro"))
				self.output[GLOBAL + "-RocAUC-mic"].append(','.join([str(round(val, 4)) for val in results_global_rocauc]))

				# ROC-AUC of algo (micro)
				results_algo_rocauc = []
				for group_id in range(self.n_groups):
					if np.sum(np.array(assignments) == group_id) == 0:
						continue
					par_scores = np.array([prediction['score'] for prediction in self._predict(test_file, assignments, group_id, group_id)])
					par_scores = np.transpose([np.where(np.array(assignments) == group_id)[0], par_scores])
					results_algo_rocauc.append(par_scores)
				results_algo_rocauc = np.concatenate(results_algo_rocauc, axis = 0)
				results_algo_rocauc = roc_auc_score(test_labels, np.squeeze(results_algo_rocauc[np.argsort(results_algo_rocauc[:,0]), 1]), average = "micro")
				self.output[self.algo + "-RocAUC-mic"].append(round(results_algo_rocauc, 4))



				pd.DataFrame(self.output).to_csv(os.path.join(self.args.output_path, self.algo + ".csv"), sep = ";")

			### logging
			self.logz.dump_tabular()
			### logging


	def _take_E_step(self, file_path):
		# compute sample weights (unsupervised version)
		self.sample_weights_tilde = np.reshape(self.pop_weights, newshape = [1, self.n_groups]) * \
								self._prob_matrix(file_path, np.expand_dims(np.squeeze(self.labels), axis = 1), "E-unsup") + EPSILON # np.array [n x K]		
		self.sample_weights_tilde = self.sample_weights_tilde/np.sum(self.sample_weights_tilde, axis = 1, keepdims = True)# np.array; [n x K]	


		# compute sample weights
		if self.algo.startswith("moe") and not self.step_mode.endswith("unsup"):
			sample_multiplier = self.sample_weights_tilde			
		else:
			sample_multiplier = np.reshape(self.pop_weights, newshape = [1, self.n_groups])  
		self.sample_weights = sample_multiplier * self._prob_matrix(file_path, np.expand_dims(np.squeeze(self.labels), axis = 1), self.step_mode) + EPSILON # np.array [n x K]		
		self.sample_weights = self.sample_weights/np.sum(self.sample_weights, axis = 1, keepdims = True)# np.array; [n x K]						
		self.pop_weights = np.squeeze(np.mean(self.sample_weights, axis = 0)) # np.array; [K]
		
		### logging
		self.logz.log_tabular("Population Weights", ','.join([str(round(val, 4)) for val in self.pop_weights]))
		### logging

	def _prob_matrix(self, file_path, labels, prob_mode, debug = False):			
		working_data = pd.read_csv(file_path, sep = ";", header = None)
		emb_dat = np.array([x.split(',') for x in working_data[self.emb_idx].values]).astype(float)
		n_samples = emb_dat.shape[0]
		d_e = emb_dat.shape[1]

		if type(labels) is int:
			labels = np.ones([n_samples, 1]) * labels 
		else:
			labels = np.expand_dims(np.squeeze(labels), axis = 1)

		if debug:
			import pdb; pdb.set_trace()			
		sup_logits = {}
		unsup_logits = {}
		for group_id in range(self.n_groups):
			# supervised learning
			if prob_mode in ("E-sup", "E-both", "t-sup", "t-both"):
				prob = np.array([[prediction['score']] for _, prediction in enumerate(self._predict(file_path, None, 0, group_id))])
				sup_logits[self.group_sets[group_id]] = np.log(np.abs(prob + (-1) * (1 - labels))) # np.array [n x 1]
			else:
				sup_logits[self.group_sets[group_id]] = np.zeros([n_samples, 1])

    		# unsupervised learning
			if prob_mode in ("E-unsup", "E-both", "t-unsup", "t-both"):
				unsup_logits[self.group_sets[group_id]] = self._GMM_loglik(group_id, emb_dat) # np.array [n x 1] 
			else:
				unsup_logits[self.group_sets[group_id]] = np.zeros([n_samples, 1])

		### logging
		unsup_logits_matrix = np.concatenate([unsup_logits[self.group_sets[group_id]] for group_id in range(self.n_groups)], axis = 1) # np.array [n x K]

		random_indexes = np.random.choice(len(unsup_logits_matrix), size = 10)	
		self.logz.log_tabular("Random Unsup Logits", '|'.join([','.join([str(round(val, 4)) for val in unsup_logits_matrix[idx]]) for idx in random_indexes]))

		sup_logits_matrix = np.concatenate([sup_logits[self.group_sets[group_id]] for group_id in range(self.n_groups)], axis = 1) # np.array [n x K]
		self.logz.log_tabular("Random Sup Logits", '|'.join([','.join([str(round(val, 4)) for val in sup_logits_matrix[idx]]) for idx in random_indexes]))
		### logging

		logits_matrix = np.concatenate([unsup_logits[self.group_sets[group_id]] + sup_logits[self.group_sets[group_id]] for group_id in range(self.n_groups)], axis = 1) # np.array [n x K]
		output = np.exp(logits_matrix - np.ndarray.max(logits_matrix, axis = 1, keepdims = True))
		return output

	def _GMM_loglik(self, group_id, emb_dat):
		d_e = emb_dat.shape[1]
		Xi = emb_dat # np.array [n x d_e]
		mean_squared = np.square(Xi - np.reshape(self.mus[self.group_sets[group_id]], newshape = [1, d_e])) # np.array [n x d_e]
		loglik = -0.5 * np.sum(mean_squared/np.reshape(self.sigma2s[self.group_sets[group_id]], newshape = [1, d_e]), axis = 1, keepdims = True) \
				 -0.5 * np.sum(np.log(self.sigma2s[self.group_sets[group_id]] + EPSILON)) \
				 -0.5 * d_e * np.log(2 * np.pi)

		return loglik/self.args.unsup_temp # np.array [n x 1]

	def _take_M_step(self, file_path):
		# unsupervised learning
		if self.step_mode in ("M-unsup", "M-both"):			
			for group_id in range(self.n_groups):		
				if not self.algo.startswith("moe") or self.step_mode.endswith("unsup"):	
					wik = self.sample_weights[:,group_id] 
				else:
					wik = self.sample_weights[:,group_id] - self.sample_weights_tilde[:,group_id]  # np.array [n x 1]

				Xi = self.emb # np.array [n x d_e]  # could be cleaned up by passing emb instead of using self.emb
				self.mus[self.group_sets[group_id]] =  np.matmul(np.transpose(wik), Xi)/np.sum(wik + EPSILON) # np.array [1 x d_e] 

				mean_squared = np.square(Xi - self.mus[self.group_sets[group_id]]) # np.array [n x d_e]
				self.sigma2s[self.group_sets[group_id]] = np.matmul(np.transpose(wik), mean_squared)/np.sum(wik + EPSILON)# np.array [1 x d_e]

				self.mus[self.group_sets[group_id]] = np.squeeze(self.mus[self.group_sets[group_id]]) # np.array [d_e]
				self.sigma2s[self.group_sets[group_id]] = np.abs(np.squeeze(self.sigma2s[self.group_sets[group_id]])) # np.array [d_e]; take abs in case the sigma2s are negative for moe

				if np.isnan(np.sum(self.mus[self.group_sets[group_id]])) or np.isnan(np.sum(self.sigma2s[self.group_sets[group_id]])):
					import pdb; pdb.set_trace()
		# supervised models
		if self.step_mode in ("M-sup", "M-both"):	
			for group_id in (gid for _ in range(self.args.n_sampling) for gid in range(self.n_groups)):	# improvable: should update self.sup_init_checkpoints if n_sampling > 1
				if self.algo not in ["global", "global-eval"]:
					if group_id == 0:
						# sampling
						assignments = self._group_sampling() # np.array; [n]	
						### logging
						self.logz.log_tabular("Assignment", ','.join([str(g_id) + ":" + str(np.sum(assignments == g_id)) for g_id in range(self.n_groups)]))
						### logging

					row_ids = np.where(assignments == group_id)[0] # np.array; [n_k]
					# partition csv file
					working_file = file_path.replace(".csv", "_" + self.group_sets[group_id] + ".csv")
					self.partition_csv(input_path=file_path, 
						output_path=working_file, 
						row_ids = row_ids)
					
				else:
					working_file = file_path

				estimator = self._build_sup_models(self.config,\
				 									self.sup_last_checkpoints[self.group_sets[group_id]],\
				 									self.sup_model_dirs[-1][self.group_sets[group_id]],\
				  									self.log_every)
				
				# supervised learning
				estimator.train(
					self.model_meta.data_loader_builder(
						arch_config=self.arch_conf,
						mode=ModeKeys.TRAIN,
						source=loader.CsvDataSource(file_path=working_file, delimiter=";"),
						shuffle=10,
						batch_size=self.args.batch_size,
						prefetch=100,
						parallel_calls=4,
						repeat=None
						).input_fn,
					steps=self.args.max_steps * self.args.multiplier
					)

				if working_file != file_path:
					os.remove(working_file)

	def _group_sampling(self):	
		try:
			return np.array([np.random.choice(len(ws), size = 1, p = ws) for ws in self.sample_weights])
		except:
			import pdb;
			pdb.set_trace()
		#return np.argmax(self.sample_weights, axis = 1)

	@staticmethod
	def partition_csv(input_path, output_path, row_ids):
		working_data = pd.read_csv(input_path, sep = ";", header = None)
		working_data.loc[row_ids].to_csv(output_path, sep = ";", header = False, index = False)

	def _predict(self, file_path, assignments, group_id, model_id):
		if assignments is None:
			working_file = file_path
			assert group_id == 0, "When assignments is None, group_id can only be 0."
		else:
			row_ids = np.where(assignments == group_id)[0] # np.array; [n_k]
			if len(row_ids) == 0:
				return {{"score": 0.}}

			# partition csv file
			working_file = file_path.replace(".csv", "_" + self.group_sets[group_id] + ".csv")
			self.partition_csv(input_path=file_path, 
				output_path=working_file,
				row_ids = row_ids)

		estimator = self._build_sup_models(self.config,\
				 							self.sup_this_checkpoints[self.group_sets[model_id]],\
				 							self.sup_model_dirs[-1][self.group_sets[model_id]],\
				  							self.log_every)
		result = estimator.predict(
				self.model_meta.data_loader_builder(
	            	arch_config=self.arch_conf,
		            mode=ModeKeys.PREDICT,
	            	source=loader.CsvDataSource(file_path=working_file, delimiter=";"),
		            shuffle=0,
	    	        batch_size=self.args.batch_size,
    	    	    prefetch=100,
        	    	parallel_calls=4,
	            	repeat=1,
        			).input_fn,
				checkpoint_path=self.sup_this_checkpoints[self.group_sets[model_id]]
    		)
		result = [val for val in result]

		if working_file != file_path:
			os.remove(working_file)
		return result

	def _infer_cluster(self, file_path, strategy = "random"): #improvable		
		if self.algo in ["global", "global-eval"]:
			return None
		elif self.algo in ["clus2pred", "clus2pred-eval"]:
			prob_mode = "t-unsup"
		elif self.algo in ["hier", "hier-eval"]:
			prob_mode = "t-both"
		elif self.algo in ["moe", "moe-eval"]:
			prob_mode = "t-unsup"
		### logging
		self.logz.log_tabular("Step mode", prob_mode + "-inference")
		### logging

		if strategy == "random":
			n_test = sum(1 for _ in open(file_path))
			return np.random.choice(self.n_groups, size = n_test), np.array([])
		elif strategy == "max-max":
			assignments = {}
			probs = {}
			#import pdb; pdb.set_trace()
			for label in [0, 1]:				
				prob_matrix = self._prob_matrix(file_path, label, prob_mode) # np.array [n_test x K]		
				sample_weights = np.reshape(self.pop_weights, newshape = [1, self.n_groups]) * prob_matrix  + EPSILON# np.array [n_test x K]		
				sample_weights = sample_weights/np.sum(sample_weights, axis = 1, keepdims = True) # np.array [n_test x K]	
				probs[label] = np.max(sample_weights, axis = 1, keepdims = True) # np.array; [n_test, 1]
				assignments[label] = np.argmax(sample_weights, axis = 1)
			labels = np.argmax(np.concatenate([probs[0], probs[1]], axis = 1), axis = 1) # np.array; [n]
			return np.array([assignments[label][idx] for idx, label in enumerate(labels)]), np.array([]) # np.array; [n]
		elif strategy == "clustering":
			# inference
			prob_matrix = self._prob_matrix(file_path, 0, prob_mode) # np.array [n_test x K]		
			sample_weights_tilde = np.reshape(self.pop_weights, newshape = [1, self.n_groups]) * prob_matrix  + EPSILON# np.array [n_test x K]		
			sample_weights_tilde = sample_weights_tilde/np.sum(sample_weights_tilde, axis = 1, keepdims = True) # np.array [n_test x K]	
			return np.argmax(sample_weights_tilde, axis = 1), sample_weights_tilde
		elif strategy == "pred":
			# prediction by global model
			prob = np.array([[prediction['score']] for _, prediction in enumerate(self._predict(file_path, None, 0, self.n_groups))]) # np.array; [n_test, 1]
			labels = np.argmax(np.concatenate([1 - prob, prob], axis = 1), axis = 1)

			# inference
			prob_matrix = self._prob_matrix(file_path, labels, prob_mode) # np.array [n_test x K]		
			sample_weights = np.reshape(self.pop_weights, newshape = [1, self.n_groups]) * prob_matrix  + EPSILON# np.array [n_test x K]		
			sample_weights = sample_weights/np.sum(sample_weights, axis = 1, keepdims = True) # np.array [n_test x K]	

			### logging
			random_indexes = np.random.choice(len(sample_weights), size = 10)			
			self.logz.log_tabular("Random Sample Weights", '|'.join([','.join([str(round(val, 4)) for val in sample_weights[idx]]) for idx in random_indexes]))
			### logging

			return np.argmax(sample_weights, axis = 1), sample_weights
		elif strategy == "truth":
			working_data = pd.read_csv(file_path, sep = ";", header = None)
			labels = working_data[self.label_idx].values.astype(int)			

			# inference
			prob_matrix = self._prob_matrix(file_path, labels, prob_mode) # np.array [n_test x K]		
			sample_weights = np.reshape(self.pop_weights, newshape = [1, self.n_groups]) * prob_matrix  + EPSILON# np.array [n_test x K]		
			sample_weights = sample_weights/np.sum(sample_weights, axis = 1, keepdims = True) # np.array [n_test x K]

			### logging
			random_indexes = np.random.choice(len(sample_weights), size = 10)			
			self.logz.log_tabular("Random Sample Weights", '|'.join([','.join([str(round(val, 4)) for val in sample_weights[idx]]) for idx in random_indexes]))
			### logging

			return np.argmax(eval_sample_weights, axis = 1), sample_weights



	def _eval(self, file_path, assignments, group_id, model_id):
		if assignments is None:
			working_file = file_path
			assert group_id == 0, "When assignments is None, group_id can only be 0."
		else:
			row_ids = np.where(assignments == group_id)[0] # np.array; [n_k]
			if len(row_ids) == 0:
				return {"loss": 0.}

			# partition csv file
			working_file = file_path.replace(".csv", "_" + self.group_sets[group_id] + ".csv")
			self.partition_csv(input_path=file_path, 
				output_path=working_file,
				row_ids = row_ids)

		estimator = self._build_sup_models(self.config,\
				 							self.sup_this_checkpoints[self.group_sets[model_id]],\
				 							self.sup_model_dirs[-1][self.group_sets[model_id]],\
				  							self.log_every)

		result = estimator.evaluate(
        		input_fn=self.model_meta.data_loader_builder(
	            	arch_config=self.arch_conf,
	    	        mode=ModeKeys.EVAL,
    	    	    source=loader.CsvDataSource(file_path=working_file, delimiter=";"),
            		shuffle=0,
		            batch_size=self.args.batch_size,
    		        prefetch=100,
        		    parallel_calls=4,
            		repeat=1
		        ).input_fn,
    		    steps=None,
    		    checkpoint_path=self.sup_this_checkpoints[self.group_sets[model_id]]
    		    )
		if working_file != file_path:
			os.remove(working_file)
				
		return result
				
			

		

