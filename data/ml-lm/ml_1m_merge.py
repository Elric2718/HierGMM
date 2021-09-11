import sys
sys.path.append("../../")
import os
import pandas as pd
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", type=str, help="embedding file")
    parser.add_argument("--org_files", type=str, help="original file")

    return parser.parse_known_args()[0]


def main():
    args = parse_args()

    emb_df = pd.read_csv(args.emb_file, sep = ";", names = ["uidx", "iidx_hist", "emb"])
    for file_name in args.org_files.split(','):    	
    	import pdb;pdb.set_trace()
    	dat_df = pd.read_csv(file_name, sep = ";", names = ["uidx", "gender", "age", "occupation", "iidx", "hist_seq_length", "iidx_hist"])
    	aug_dat_df = pd.merge(emb_df, dat_df, on = ['uidx', 'iidx_hist'], how = 'inner')
    	aug_dat_df = aug_dat_df.join(aug_dat_df['iidx'].str.split('::', expand = True))
    	del aug_dat_df['iidx']
    	aug_dat_df	= aug_dat_df.set_index(['uidx', 'gender', 'age', 'occupation', 'hist_seq_length', 'iidx_hist', 'emb'])\
    					.stack()\
    					.reset_index()
    	aug_dat_df.columns = ['uidx', 'gender', 'age', 'occupation', 'hist_seq_length', 'iidx_hist', 'emb', 'label', 'iidx']
    	aug_dat_df['label'] = aug_dat_df['label'].apply(lambda x: 1 - int(x > 0))
    	aug_dat_df = aug_dat_df[['uidx', 'gender', 'age', 'occupation', 'iidx', 'hist_seq_length', 'iidx_hist', 'emb', 'label']]

    	aug_dat_df.to_csv(file_name.replace('.csv', '_aug.csv'), sep = ";", header = False, index = False)

if __name__ =='__main__':
	main()



