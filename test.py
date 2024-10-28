import zipfile
import os

zip_file_path   = '/content/drive/MyDrive/data/open.zip'
extract_to_path = '/content/sample_data/data'

os.makedirs(extract_to_path, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print("압축 해제가 완료되었습니다.")

import argparse

def get_argments():

    parser = argparse.ArgumentParser(description="Speech Recognition")

    # =============== parser with data ===================== #
    parser.add_argument('--SR', type=int, default=32000, help="sampling_ratio")
    parser.add_argument('data_path', type==str, default='/content/sample_data/data/open', help='data_path')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation data ratio')
    # =================================================================== #
    
    #================= Other Arguments ================================#
    parser.add_argument('--is_inference', type=bool, default=False, help='use only when inference')
    parser.add_argument('--is_embedding', type=bool, default=False, help='use only when embedding')
    parser.add_argument('--fold_iter', type=int, default=-1, help='use only when kfold')
    #==================================================================#

    args = parser.parse_args()

    return args 

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold


# create: validation_index
def get_validation_index(args, train_df:pd.DataFrame) -> int:
    """
    Maintain consistency in dataset splitting by saving and reusing validation indices
    """

    # original: train/valid
    if args.fold_iter == -1:
        num_of_validation     = int(len(train_df) * args.valid_ratio)
        validation_index_path = f'{args.data_path}/valid_indices_{num_of_validation}'

        if os.path.exists(validation_index_path):
            validation_index = np.loadtxt(validation_index_path).astype(int)
        else:
            # without replacement: indices
            validation_index  = np.random.choice(train_df.index, size=num_of_validation, replace=False).astype(int)
            np.savetxt(validation_index_path, validation_index, delimiter=',')

    # k-fold: train/valid
    else:
        validation_index_path = f'{args.data_path}/5kfold_valid_indices_{args.fold_iter}.csv'


        if os.path.exists(validation_index_path):
            validation_index = np.loadtxt(validation_index_path).astype(int)
        else: 
            kf    = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(train_df))

            for i in range(5):
                traind_indices, validation_indices = folds[i]
                validation_index_path = f'{args.data_path}/5kfold_valid_indices_{i}.csv'
                np.savetxt(validation_index_path, validation_indices, delimiter=',')

            validation_index = folds[args.fold_iter][1]


    return validation_index

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from typing import Tuple, Any
import torch
import pickle


# create: train, valid, test datasets & with or without inference
def get_dataset(args) -> Tuple[Dataset, Dataset, Dataset, Any]:
    train_path = os.path.join(args.data_path, "./train.csv")
    train_df   = pd.read_csv(train_path)

    validation_index = get_validation_index(args, train_df)

    valid_df = train_df.loc[validation_index]
    train_df = train_df.drop(validation_index)

    test_path = os.path.join(args.data_path, './test.csv')
    test_df   = pd.read_csv(test_path)

    print(f"# of train: {len(train_df)}, # of valid: {len(valid_df)}")

    # use when inference
    if args.is_inference:
        train_dataset = None
        valid_dataset = None
        test_dataset  = AudioDataset(args, test_df, test=True)
    else:
        train_dataset = AudioDataset(args, train_df, train=True)
        valid_dataset = AudioDataset(args, test_df, test=True)
        test_dataset  = None

    data_collactor = CustomDataCollator(args.model_name)


    return (train_dataset, valid_dataset, test_dataset, data_collactor)
