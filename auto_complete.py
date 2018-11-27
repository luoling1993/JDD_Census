#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通过ARIMA模型补充接下来15天的数据
"""
import datetime
import os
import warnings

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

warnings.filterwarnings('ignore')

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")

date_dt = list()
init_date = datetime.date(2018, 3, 2)
for delta in range(15):  # test date length
    _date = init_date + datetime.timedelta(days=delta)
    date_dt.append(_date.strftime('%Y%m%d'))

train_date_dt = list()
train_init_date = datetime.date(2017, 6, 1)
for delta in range(274):  # train date length
    _date = train_init_date + datetime.timedelta(days=delta)
    train_date_dt.append(_date.strftime('%Y%m%d'))


def base_arima(ts):
    arima_model = auto_arima(ts, start_p=1, max_p=9, start_q=1, max_q=9, max_d=5,
                             start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=5,
                             m=7,
                             trace=True,
                             seasonal=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
    preds = arima_model.predict(n_periods=15)
    preds = pd.Series(preds)
    return preds


def complete_flow():
    flow_df_name = os.path.join(RAW_DATA_PATH, 'flow_train.csv')
    flow_df = pd.read_csv(flow_df_name)
    flow_df = flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

    district_code_values = flow_df['district_code'].unique()
    preds_df = pd.DataFrame()
    tmp_df_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']

    for district_code in district_code_values:
        sub_df = flow_df[flow_df['district_code'] == district_code]
        city_code = sub_df['city_code'].iloc[0]

        predict_columns = ['dwell', 'flow_in', 'flow_out']
        tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
        tmp_df['city_code'] = city_code
        tmp_df['district_code'] = district_code

        for column in predict_columns:
            ts_log = np.log(sub_df[column])

            preds = base_arima(ts_log)
            preds = np.exp(preds)
            tmp_df = pd.concat([tmp_df, preds], axis=1)

        tmp_df.columns = tmp_df_columns
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

    preds_df = preds_df.sort_values(by=['date_dt'])

    preds_df_name = os.path.join(ETL_DATA_PATH, 'arima_flow_train.csv')
    preds_df.to_csv(preds_df_name, index=False)


def get_cnt_list(sub_df):
    cnt_list = list()
    # sub_df = sub_df[['date_dt', 'cnt']]
    sub_df = sub_df.copy()

    tmp_index = 0
    for index, (_, item) in enumerate(sub_df.iterrows()):
        _date_dt = str(int(item['date_dt']))
        cnt = item['cnt']

        if _date_dt == train_date_dt[index + tmp_index]:
            cnt_list.append(np.log(cnt * 1000))
            continue

        while True:
            tmp_index += 1

            if _date_dt == train_date_dt[index + tmp_index]:
                cnt_list.append(0.0)
                break
            else:
                cnt_list.append(0.0)

        cnt_list.append(np.log(cnt))

    # The last date is not 20180301
    if len(cnt_list) != 274:
        extra_date_num = 274 - len(cnt_list)
        cnt_list.extend([0.0] * extra_date_num)

    return cnt_list


def complete_transition():
    """
    This function will cost much time
    about 98 * 97 * 5 / 3600 = 13 hours
    """
    transition_df_name = os.path.join(RAW_DATA_PATH, 'transition_train.csv')
    transition_df = pd.read_csv(transition_df_name)
    transition_df = transition_df.sort_values(
        by=['o_city_code', 'o_district_code', 'd_city_code', 'd_district_code', 'date_dt'])

    o_district_code_values = transition_df['o_district_code'].unique()
    preds_df = pd.DataFrame()

    for o_district_code in o_district_code_values:
        for d_district_code in o_district_code_values:
            if o_district_code == d_district_code:
                continue

            sub_df = transition_df[(transition_df['o_district_code'] == o_district_code) & (
                    transition_df['d_district_code'] == d_district_code)]
            o_city_code = sub_df['o_city_code'].iloc[0]
            d_city_code = sub_df['d_city_code'].iloc[0]

            tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
            tmp_df['o_city_code'] = o_city_code
            tmp_df['o_district_code'] = o_district_code
            tmp_df['d_city_code'] = d_city_code
            tmp_df['d_district_code'] = d_district_code

            cnt_list = get_cnt_list(sub_df)
            preds = base_arima(cnt_list)
            preds = np.exp(preds) / 1000

            new_preds = list()
            for pred in preds:
                if pred < 0:
                    new_preds.append(0.0)
                else:
                    new_preds.append(pred)

            tmp_df['cnt'] = new_preds
            preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

    preds_df = preds_df.sort_values(by=['date_dt'])

    preds_df_name = os.path.join(ETL_DATA_PATH, 'arima_transition_train.csv')
    preds_df.to_csv(preds_df_name, index=False)


if __name__ == '__main__':
    # complete_flow()
    complete_transition()
