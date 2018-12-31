# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:26:18 2018

@author: yelin0618
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

#load data.
order_book_data = pd.read_csv('/Users/meihuaren/personal/OR_2018fall/Courses/E4720 Deep Learning/project_coding/Team E_code/INTC_2012-06-21_34200000_57600000_orderbook_5.csv', header=None)
message_data = pd.read_csv('/Users/meihuaren/personal/OR_2018fall/Courses/E4720 Deep Learning/project_coding/Team E_code/INTC_2012-06-21_34200000_57600000_message_5.csv', header=None)

#input_date = datetime(year=2012, month=6, day=21).timestamp()

#order_book_data['Time'] = message_data.loc[:,0].apply(lambda x: datetime.fromtimestamp(x + input_date))

#order_book_data.set_index('Time', drop=True, inplace=True)

order_book_data['mid'] = (order_book_data.loc[:,0] + order_book_data.loc[:,2]) / 2

#construct 'fake' datetime 
end_date = datetime(year=2012, month=6, day=22).timestamp()

order_book_data['RevTime'] = message_data.loc[:,0].apply(lambda x: datetime.fromtimestamp(end_date - x)).values

order_book_data.set_index('RevTime', drop=True, inplace=True)

rev_order_book_data = order_book_data.loc[:,'mid']

#reverse timestamp
rev_order_book_data = rev_order_book_data[::-1]

#set horizon
horizon = '1s'

#calculate the midprice change for given horizon
result = rev_order_book_data.rolling(horizon).apply(lambda x: np.sign(x[0] - x[-1]))

#reverse back
result = result[::-1]

order_book_data.loc[:, 'label'] = result.values.astype(int)

#calculate correct datatime
input_date = datetime(year=2012, month=6, day=21).timestamp()

order_book_data['Time'] = message_data.loc[:,0].apply(lambda x: datetime.fromtimestamp(x + input_date)).values

order_book_data.set_index('Time', drop=True, inplace=True)

order_book_data.drop(columns=['mid'],inplace=True)

order_book_data.to_csv('INTC_orderbook_1s_horizon.csv', index=True)