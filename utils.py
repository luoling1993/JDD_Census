#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def auto_balance(data=None):
    if data is None:
        predicttion_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        df = pd.read_csv('prediction.csv', names=predicttion_columns, header=-1)
    else:
        df = data

    balance_df = pd.DataFrame()
    date_dt_values = df['date_dt'].unique()

    for date_dt in date_dt_values:
        sub_df = df[df['date_dt'] == date_dt]
        flow_in_sum = sub_df['flow_in'].sum()
        flow_out_sum = sub_df['flow_out'].sum()

        flow_in_rate = np.mean([flow_in_sum, flow_out_sum]) / flow_in_sum
        flow_out_rate = np.mean([flow_in_sum, flow_out_sum]) / flow_out_sum

        sub_df['flow_in'] = sub_df['flow_in'].apply(lambda item: item * flow_in_rate)
        sub_df['flow_out'] = sub_df['flow_out'].apply(lambda item: item * flow_out_rate)

        balance_df = pd.concat([balance_df, sub_df], axis=0, ignore_index=True)

    balance_df.to_csv('blance.csv', index=False, header=False)


def rmsle(y_true, y_pred):
    return np.sum(np.power(np.log1p(y_true) - np.log1p(y_pred), 2))


def fuse():
    df1_columns = ['date_dt', 'city_code', 'district_code', 'dwell1', 'flow_in1', 'flow_out1']
    df1 = pd.read_csv('prediction1.csv', header=-1, names=df1_columns)

    df2_columns = ['date_dt', 'city_code', 'district_code', 'dwell2', 'flow_in2', 'flow_out2']
    df2 = pd.read_csv('prediction2.csv', header=-1, names=df2_columns)

    df = pd.merge(df1, df2, how='left', on=['date_dt', 'city_code', 'district_code'])
    df['dwell'] = (df['dwell1'] + df['dwell2']) / 2
    df['flow_in'] = (df['flow_in1'] + df['flow_in2']) / 2
    df['flow_out'] = (df['flow_out1'] + df['flow_out2']) / 2

    df = df.drop(columns=['dwell1', 'flow_in1', 'flow_out1', 'dwell2', 'flow_in2', 'flow_out2'])
    print(df.columns)

    df.to_csv('prediction.csv', index=False, header=False)


if __name__ == '__main__':
    # fuse()
    auto_balance()
