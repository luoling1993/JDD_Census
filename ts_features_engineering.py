#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import gc
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")


def get_train_data(name):
    if name not in ['flow', 'transition']:
        raise ValueError('name should be `flow` or `transition`, but get `{}`'.format(name))

    raw_data_name = os.path.join(RAW_DATA_PATH, '{}_train.csv'.format(name))
    etl_data_name = os.path.join(ETL_DATA_PATH, 'arima_{}_train.csv'.format(name))

    raw_data = pd.read_csv(raw_data_name)
    etl_data = pd.read_csv(etl_data_name)

    train_data = pd.concat([raw_data, etl_data], axis=0, ignore_index=True)
    return train_data


class FlowProcessing(object):
    @staticmethod
    def _get_days_week(item):
        item = str(item)
        item_date = datetime.datetime.strptime(item, '%Y%m%d')
        days_week = item_date.weekday()
        return days_week

    @staticmethod
    def _get_stats_item(stats_dict):
        stats_items_list = list()
        date_dt = stats_dict['date_dt']

        for key, item in stats_dict['stats'].items():
            stats_item_list = list()

            stats_item_list.append(date_dt)  # date_dt
            stats_item_list.append(key)  # district_code
            stats_item_list.append(item['days_1'][0])  # last value

            # days_7 max
            days_7_max = max(item['days_7'])
            stats_item_list.append(days_7_max)

            # days_7 min
            days_7_min = min(item['days_7'])
            stats_item_list.append(days_7_min)

            # days_7 mean
            days_7_mean = np.mean(item['days_7'])
            stats_item_list.append(days_7_mean)

            # days_15 max
            days_15_max = max(item['days_15'])
            stats_item_list.append(days_15_max)

            # days_15 min
            days_15_min = min(item['days_15'])
            stats_item_list.append(days_15_min)

            # days_15 mean
            days_15_mean = np.mean(item['days_15'])
            stats_item_list.append(days_15_mean)

            # days_30 max
            days_30_max = max(item['days_30'])
            stats_item_list.append(days_30_max)

            # days_30 min
            days_30_min = min(item['days_30'])
            stats_item_list.append(days_30_min)

            # days_30 mean
            days_30_mean = np.mean(item['days_30'])
            stats_item_list.append(days_30_mean)

            stats_items_list.append(stats_item_list)

        return stats_items_list

    @staticmethod
    def _update_stats_dict(stats_dict, district_code, column_item):
        if district_code not in stats_dict['stats'].keys():
            stats_dict['stats'][district_code] = dict(days_1=list(), days_7=list(),
                                                      days_15=list(), days_30=list())

        if len(stats_dict['stats'][district_code]['days_1']) == 1:
            stats_dict['stats'][district_code]['days_1'].pop(0)
            stats_dict['stats'][district_code]['days_1'].append(column_item)
        else:
            stats_dict['stats'][district_code]['days_1'].append(column_item)

        if len(stats_dict['stats'][district_code]['days_7']) == 7:
            stats_dict['stats'][district_code]['days_7'].pop(0)
            stats_dict['stats'][district_code]['days_7'].append(column_item)
        else:
            stats_dict['stats'][district_code]['days_7'].append(column_item)

        if len(stats_dict['stats'][district_code]['days_15']) == 15:
            stats_dict['stats'][district_code]['days_15'].pop(0)
            stats_dict['stats'][district_code]['days_15'].append(column_item)
        else:
            stats_dict['stats'][district_code]['days_15'].append(column_item)

        if len(stats_dict['stats'][district_code]['days_30']) == 30:
            stats_dict['stats'][district_code]['days_30'].pop(0)
            stats_dict['stats'][district_code]['days_30'].append(column_item)
        else:
            stats_dict['stats'][district_code]['days_30'].append(column_item)

        return stats_dict

    def _get_stats_df(self, df, column):
        if column not in ['dwell', 'flow_in', 'flow_out']:
            raise ValueError()

        init_date = 20170701  # int is enough
        stats_dict = dict()
        stats_list = list()
        stats_df_columns = ['date_dt', 'district_code', 'days_1', 'days_7_max', 'days_7_min', 'days_7_mean',
                            'days_15_max', 'days_15_min', 'days_15_mean', 'days_30_max', 'days_30_min', 'days_30_mean']
        df = df.copy()

        for _, item in df.iterrows():
            date_dt = item['date_dt']
            district_code = item['district_code']
            column_item = item[column]

            if 'date_dt' not in stats_dict.keys():
                stats_dict['date_dt'] = date_dt
                stats_dict['stats'] = dict()

            if date_dt != stats_dict['date_dt']:
                stats_dict = self._update_stats_dict(stats_dict, district_code, column_item)
                stats_dict['date_dt'] = date_dt

                if date_dt < init_date:
                    continue

                stats_items_list = self._get_stats_item(stats_dict)
                stats_list.extend(stats_items_list)

            else:
                stats_dict = self._update_stats_dict(stats_dict, district_code, column_item)

        stats_df = pd.DataFrame(data=stats_list, columns=stats_df_columns)
        return stats_df

    def processing(self):
        flow_df = get_train_data(name='flow')

        flow_df['days_week'] = flow_df['date_dt'].apply(self._get_days_week)
        flow_df['city_code_copy'] = flow_df['city_code']
        flow_df['district_code_copy'] = flow_df['district_code']

        flow_df = pd.get_dummies(flow_df, columns=['days_week', 'city_code_copy', 'district_code_copy'])

        dwell_df = flow_df.copy()
        dwell_df = dwell_df.drop(columns=['flow_in', 'flow_out'])

        flow_in_df = flow_df.copy()
        flow_in_df = flow_in_df.drop(columns=['dwell', 'flow_out'])

        flow_out_df = flow_df.copy()
        flow_out_df = flow_out_df.drop(columns=['dwell', 'flow_in'])

        del flow_df
        gc.collect()

        dwell_stats_df = self._get_stats_df(dwell_df, column='dwell')
        dwell_df = pd.merge(dwell_df, dwell_stats_df, how='left', on=['date_dt', 'district_code'])
        dwell_df = dwell_df[dwell_df['date_dt'] >= 20170701]
        dwell_df_name = os.path.join(ETL_DATA_PATH, 'dwell_features.csv')
        dwell_df.to_csv(dwell_df_name, index=False)

        flow_in_stats_df = self._get_stats_df(flow_in_df, column='flow_in')
        flow_in_df = pd.merge(flow_in_df, flow_in_stats_df, how='left', on=['date_dt', 'district_code'])
        flow_in_df = flow_in_df[flow_in_df['date_dt'] >= 20170701]
        flow_in_df_name = os.path.join(ETL_DATA_PATH, 'flow_in_features.csv')
        flow_in_df.to_csv(flow_in_df_name, index=False)

        flow_out_stats_df = self._get_stats_df(flow_out_df, column='flow_out')
        flow_out_df = pd.merge(flow_out_df, flow_out_stats_df, how='left', on=['date_dt', 'district_code'])
        flow_out_df = flow_out_df[flow_out_df['date_dt'] >= 20170701]
        flow_out_df_name = os.path.join(ETL_DATA_PATH, 'flow_out_features.csv')
        flow_out_df.to_csv(flow_out_df_name, index=False)


if __name__ == '__main__':
    flow_processing = FlowProcessing()
    flow_processing.processing()
