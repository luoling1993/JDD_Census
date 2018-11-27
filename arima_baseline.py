#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import warnings

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

warnings.filterwarnings('ignore')

flow_df = pd.read_csv('data/RawData/flow_train.csv')
flow_df = flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

date_dt = list()
init_date = datetime.date(2018, 3, 2)
for delta in range(15):
    _date = init_date + datetime.timedelta(days=delta)
    date_dt.append(_date.strftime('%Y%m%d'))

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
        ts_log = np.log(1 + sub_df[column])
        arima_model = auto_arima(ts_log, start_p=1, max_p=9, start_q=1, max_q=9, max_d=5,
                                 start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=5,
                                 m=7, random_state=2018,
                                 trace=True,
                                 seasonal=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

        preds = arima_model.predict(n_periods=15)
        preds = pd.Series(preds)
        preds = np.exp(preds) - 1
        tmp_df = pd.concat([tmp_df, preds], axis=1)

    tmp_df.columns = tmp_df_columns
    preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

preds_df = preds_df.sort_values(by=['date_dt'])
preds_df.to_csv('prediction1.csv', index=False, header=False)
