
# coding: utf-8

# In[2]:
#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math
import sys

import OrderUtil as ou

'''class to build the fill probability matrix'''
class FillProbabilitySimulator:
    def __init__(self, orderbook, order_tuple, numupdates=None, timeunit=None, t_start=34200, t_end=57600):
        '''
        Ordertuple is represented as follows:
        (bid_level, ask_level) are the levels we would like to place orders
        '''
        self._orderbook = orderbook
        self._midprice_df = orderbook.get_midprice_data(numupdates=numupdates, timeunit=timeunit,
                                                        t_start=t_start, t_end=t_end)
        self._order_tuple = order_tuple
        self._t_starts = self._midprice_df.index
        self._N = len(self._midprice_df)
        self._fill_prob_matrx = pd.DataFrame(np.zeros((4, 3)),
                                             index=['none_executed', 'bid_executed', 'ask_executed', 'bid_and_ask_executed'],
                                             columns=[-1, 0, 1])
        self._fill_prob_matrx.index.name = 'orders_executed'
        self._fill_prob_matrx.columns.name = 'y'
        if(numupdates is None):
            self._timeupdates = True
            self._unit = timeunit
        else:
            self._timeupdates = False
            self._unit = numupdates
            
        self._quantity = {-1: 0, 0:0, 1:0}
    
    def _generate_order_pair(self):
        rand_int = np.random.randint(1, self._N - 2)
        timestamp = self._midprice_df.index[rand_int]
        if self._timeupdates:
            self._bid_order = ou.TimeOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[0],
                                       is_buy=True, delta_t=self._unit)
            self._ask_order = ou.TimeOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[1],
                                       is_buy=False, delta_t=self._unit)
        else:
            self._bid_order = ou.BookUpdatesOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[0],
                                               is_buy=True, numupdates=self._unit)
            self._ask_order = ou.BookUpdatesOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[1],
                                               is_buy=False, numupdates=self._unit)
        return rand_int
    
    def _evaluate_order_pair(self):
        rand_int = self._generate_order_pair()
        self._bid_order.update()
        self._ask_order.update()
        midprice_movement = self._midprice_df.iloc[rand_int, 2]
        self._quantity[midprice_movement] += 1
        if self._bid_order.order_type("expired") or self._ask_order.order_type("expired"):
            print("one order expired, not a fair data point")
            return False
        if self._bid_order.order_type("executed") and self._ask_order.order_type("executed"):
            self._fill_prob_matrx.loc['bid_and_ask', midprice_movement] += 1.0
            return True
        if self._bid_order.order_type("executed") and not self._ask_order.order_type("executed"):
            self._fill_prob_matrx.loc['bid', midprice_movement] += 1.0
            return True
        if not self._bid_order.order_type("executed") and self._ask_order.order_type("executed"):
            self._fill_prob_matrx.loc['ask', midprice_movement] += 1.0
            return True
        if not self._bid_order.order_type("executed") and not self._ask_order.order_type("executed"):
            self._fill_prob_matrx.loc['none', midprice_movement] += 1.0
            return True
        
        raise Exception("unhandled order case")
        
    def generate_matrix(self, num_samples):
        self._fill_prob_matrx = pd.DataFrame(np.zeros((4, 3)), index=['none', 'bid', 'ask', 'bid_and_ask'],
                                             columns=[-1, 0, 1])
        self._quantity = {-1: 0, 0:0, 1:0}
        count = 0.0;
        while(count < num_samples):
            if self._evaluate_order_pair():
                count += 1.0
            if(count % 1000 == 0):
                print("samples processed so far: " + str(count))
        self._fill_prob_matrx /= self._fill_prob_matrx.sum()
    def get_prob_matrix(self):
        return self._fill_prob_matrx
    
    def get_quantities(self):
        return self._quantity
    

