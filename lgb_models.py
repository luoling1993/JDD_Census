#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import rmsle

warnings.filterwarnings('ignore')

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")


def get_data(name):
    if name not in ['dwell', 'flow_in', 'flow_out']:
        raise ValueError()

    file_name = os.path.join(ETL_DATA_PATH, '{}_features.csv'.format(name))
    df = pd.read_csv(file_name)
    return df


def lgb_model(name, train_data, test_data, params, nflod):
    columns = train_data.columns
    remove_columns = [name, 'date_dt', 'district_code', 'city_code']
    features_columns = [column for column in columns if column not in remove_columns]

    train_features = train_data[features_columns]
    train_labels = train_data[name]

    test_features = test_data[features_columns]

    kfolder = KFold(n_splits=nflod, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    for train_index, test_index in kfold:
        k_x_train = train_features.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_features.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        gbm = lgb.LGBMRegressor(**params)
        gbm = gbm.fit(k_x_train, k_y_train,
                      eval_metric="mse",
                      eval_set=[(k_x_train, k_y_train),
                                (k_x_test, k_y_test)],
                      eval_names=["train", "valid"],
                      early_stopping_rounds=100,
                      verbose=True)

        preds = gbm.predict(test_features, num_iteration=gbm.best_iteration_)

        preds_list.append(preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))

    return preds_list


def model_main():
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "max_bin": 425,
        "subsample_for_bin": 20000,
        "objective": 'regression',
        # "metric": 'l1',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False
    }

    test_length = 98 * 15

    dwell_df = get_data(name='dwell')
    train_dwell = dwell_df[:-test_length]
    test_dwell = dwell_df[-test_length:]

    preds_df = test_dwell[['date_dt', 'city_code', 'district_code']]

    dwell_preds = lgb_model('dwell', train_dwell, test_dwell, lgb_parms, nflod=5)
    preds_df['dwell'] = dwell_preds

    flow_in_df = get_data(name='flow_in')
    train_flow_in = flow_in_df[:-test_length]
    test_flow_in = flow_in_df[-test_length:]
    flow_in_preds = lgb_model('flow_in', train_flow_in, test_flow_in, lgb_parms, nflod=5)
    preds_df['flow_in'] = flow_in_preds

    flow_out_df = get_data(name='flow_out')
    train_flow_out = flow_out_df[:-test_length]
    test_flow_out = flow_out_df[-test_length:]
    flow_out_preds = lgb_model('flow_out', train_flow_out, test_flow_out, lgb_parms, nflod=5)
    preds_df['flow_out'] = flow_out_preds

    # validate_preds = preds_df[['dwell', 'flow_in', 'flow_out']][test_length:]
    #
    # validate_dwell_data = test_dwell[['dwell']][test_length:]
    # dwell_rmsle = rmsle(validate_dwell_data['dwell'], validate_preds['dwell'])
    # print('dwell rmsle: {}'.format(dwell_rmsle))
    #
    # validate_flow_in_data = test_flow_in[['flow_in']][test_length:]
    # flow_in_rmsle = rmsle(validate_flow_in_data['flow_in'], validate_preds['flow_in'])
    # print('flow_in rmsle: {}'.format(flow_in_rmsle))
    #
    # validate_flow_out_data = test_flow_out[['flow_out']][test_length:]
    # flow_out_rmsle = rmsle(validate_flow_out_data['flow_out'], validate_preds['flow_out'])
    # print('flow_out rmsle: {}'.format(flow_out_rmsle))
    #
    # rmsle_score = np.sqrt(np.sum([dwell_rmsle, flow_in_rmsle, flow_out_rmsle]) / (15 * 98 * 3))
    # print('rmsle score: {}'.format(rmsle_score))
    #
    # preds_df = preds_df[:test_length]

    preds_df.to_csv('prediction2.csv', index=False, header=False)


if __name__ == '__main__':
    model_main()
