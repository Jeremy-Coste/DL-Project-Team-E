'''module contains a class for fill probabilities and a class for loss function estimation'''
# coding: utf-8

# In[2]:
#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math
import sys

import OrderUtil as ou

# coding: utf-8

'''class to build the fill probability matrix and pnls for given trades'''
class FillProbabilitySimulator:
    def __init__(self, orderbook, order_tuple, numupdates=None, timeunit=None,
                 t_start=3420.01, t_end=57600, uniform_sampling=True, midprice_df=None, ticks=None, rel_queue=None):
        '''
        Ordertuple is represented as follows:
        (list_of_bid_levels, list_of_ask_levels) are the levels we would like to place orders
        '''
        self._uniform_sampling = uniform_sampling
        self._orderbook = orderbook
        self._ticks = ticks
        self._relative_queue = rel_queue
        if midprice_df is None:
            self._midprice_df = self._orderbook.get_midprice_data(numupdates=numupdates, timeunit=timeunit,
                                                                  t_start=t_start, t_end=t_end, tick_size=self._ticks)
        else:
            self._midprice_df = midprice_df
        self._order_tuple = order_tuple
        self._shape_orders = (len(order_tuple[0]), len(order_tuple[1]))
        self._N = len(self._midprice_df)
        self._num_rows = (1 + self._shape_orders[0])*(1 + self._shape_orders[1])
        
        self._index = ['bid_level_0_ask_level_0']
        self._index += ['bid_level_0_ask_level_' + str(self._order_tuple[1][j]) 
                        for j in range(self._shape_orders[1])]
        self._index += ['bid_level_' + str(self._order_tuple[0][i]) + '_ask_level_0'
                        for i in range(self._shape_orders[0])]
        self._index += ['bid_level_' + str(order_tuple[0][i]) + '_ask_level_' + str(order_tuple[1][j])
                        for i in range(self._shape_orders[0]) for j in range(self._shape_orders[1])]
        
        self._fill_prob_matrix = pd.DataFrame()
        self._cum_prob_matrix = pd.DataFrame()
        self._split_midprice_to_sample = {}
        self._midprice_df.reset_index(inplace=True)
        for i in range(-1, 2, 1):
            self._split_midprice_to_sample[i] = self._midprice_df.loc[self._midprice_df['y_1'] == i]['timestamp']
        self._midprice_df.set_index('timestamp', inplace=True)
        self._pnls = pd.DataFrame()
        
        if(numupdates is None):
            self._timeupdates = True
            self._unit = timeunit
        else:
            self._timeupdates = False
            self._unit = numupdates
            
        self._quantity = {-1: 0, 0:0, 1:0}
        self._y1_ind = self._midprice_df.columns.get_loc('y_1')
        self._bp1_ind = self._midprice_df.columns.get_loc('bp1')
        self._ap1_ind = self._midprice_df.columns.get_loc('ap1')
    
    def _generate_order_pair(self, mid=0):
        if self._uniform_sampling:
            rand_int = np.random.randint(1, len(self._split_midprice_to_sample[mid]) - 2)
            rand_int = self._split_midprice_to_sample[mid].index[rand_int]
            midprice_movement = self._midprice_df.values[rand_int, self._y1_ind]
            if(midprice_movement != mid):
                raise Exception("problem with importance sampling")
        else:
            rand_int = np.random.randint(1, self._N - 2)
            midprice_movement = self._midprice_df.values[rand_int, self._y1_ind]
        timestamp = self._midprice_df.index[rand_int]
        self._bid_orders = []
        self._ask_orders = []
        if self._timeupdates:
            #create our orders
            for i in range(self._shape_orders[0]):
                self._bid_orders += [ou.TimeOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[0][i],
                                                  is_buy=True, delta_t=self._unit, relative_queue=self._relative_queue)]
                
            for i in range(self._shape_orders[1]):
                self._ask_orders += [ou.TimeOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[1][i],
                                                  is_buy=False, delta_t=self._unit, relative_queue=self._relative_queue)]
        else:
            for i in range(self._shape_orders[0]):
                self._bid_orders += [ou.BookUpdatesOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[0][i],
                                                         is_buy=True, numupdates=self._unit,
                                                         relative_queue=self._relative_queue)]
            for i in range(self._shape_orders[1]):
                self._ask_orders += [ou.BookUpdatesOrder(self._orderbook, timestamp=timestamp, level=self._order_tuple[1][i],
                                                         is_buy=False, numupdates=self._unit,
                                                         relative_queue=self._relative_queue)]
        return midprice_movement, rand_int
    
    def _evaluate_order_pair(self, midprice_type):
        midprice_movement, rand_int = self._generate_order_pair(midprice_type)
        for i in range(self._shape_orders[0]):
            self._bid_orders[i].update()
        for i in range(self._shape_orders[1]):
            self._ask_orders[i].update()
                                     
        self._quantity[midprice_movement] += 1
        bids_exec = 0
        asks_exec = 0
        for i in range(self._shape_orders[0]):
            if self._bid_orders[i].order_type("expired"):
                print("one order expired, not a fair data point")
                return False
        for i in range(self._shape_orders[1]):
            if self._ask_orders[i].order_type("expired"):
                print("one order expired, not a fair data point")
                return False
        #fair data point
        for i in range(self._shape_orders[0] - 1, -1, -1):
            if self._bid_orders[i].order_type("executed"):
                bids_exec = self._order_tuple[0][i]
                break
        for i in range(self._shape_orders[1] - 1, -1, -1):
            if self._ask_orders[i].order_type("executed"):
                asks_exec = self._order_tuple[1][i]
                break
                
        self._fill_prob_matrix.loc['bid_level_' + str(bids_exec) + '_ask_level_' + str(asks_exec), midprice_movement] += 1.0
        pnl = self._get_pnl(rand_int + 1)
        self._pnls.loc[:, midprice_movement] += pnl
        return True
    
    def _get_pnl(self, next_ind):
        mkt_bid_ask = (self._midprice_df.values[next_ind, self._bp1_ind],
                       self._midprice_df.values[next_ind, self._ap1_ind])
        long_position = 0.0
        short_position = 0.0
        cash_flow = 0.0
        #now we see which orders are filled and adjust for profit
        for bid in self._bid_orders:
            if bid.order_type("executed"):
                price = bid.get_order_price()
                cash_flow -= price
                long_position += 1
        for ask in self._ask_orders:
            if ask.order_type("executed"):
                price = ask.get_order_price()
                cash_flow += price
                short_position += 1
                
        #net long
        if long_position > short_position:
            cash_flow += mkt_bid_ask[0]*(long_position - short_position)
            
        #net short 
        elif short_position > long_position:
            cash_flow -= mkt_bid_ask[1]*(short_position - long_position)
            
        return cash_flow
    
    def _get_cum_prob(self, bid_level, ask_level, midprice):
        if bid_level == 0 and ask_level == 0:
            return 1.0
        
        prob = 0.0
        if bid_level == 0:
            for j in self._order_tuple[1]:
                if j >= ask_level:
                    prob += self._fill_prob_matrix.loc['bid_level_0_ask_level_' + str(j), midprice]
                    
        if ask_level == 0:
            for i in self._order_tuple[0]:
                if i >= bid_level:
                    prob += self._fill_prob_matrix.loc['bid_level_' + str(i) + '_ask_level_0', midprice]
                    
        for i in self._order_tuple[0]:
            for j in self._order_tuple[1]:
                if(i >= bid_level and j >= ask_level):
                    prob += self._fill_prob_matrix.loc['bid_level_' + str(i) + '_ask_level_' + str(j), midprice]
        return prob
    
    def _set_cum_prob_matrix(self):
        self._cum_prob_matrix = pd.DataFrame(np.zeros((self._num_rows , 3)),
                                             index=self._index,
                                             columns=[-1, 0, 1])
        for col in self._cum_prob_matrix.columns:
            self._cum_prob_matrix.loc['bid_level_0_ask_level_0'] = self._get_cum_prob(0, 0, col)
            for i in self._order_tuple[0]:
                self._cum_prob_matrix.loc['bid_level_' + str(i) +
                                          '_ask_level_0', col] = self._get_cum_prob(i, 0, col)
            for j in self._order_tuple[1]:
                self._cum_prob_matrix.loc['bid_level_0_ask_level_' +
                                          str(j), col] = self._get_cum_prob(0, j, col)
            for i in self._order_tuple[0]:
                for j in self._order_tuple[1]:
                    self._cum_prob_matrix.loc['bid_level_' + str(i) +
                                              '_ask_level_' + str(j), col] = self._get_cum_prob(i, j, col)
        self._cum_prob_matrix.index.name = 'orders_executed'
        self._cum_prob_matrix.columns.name = 'y'
        return self._cum_prob_matrix
        
    def generate_matrices(self, num_samples):
        self._fill_prob_matrix = pd.DataFrame(np.zeros((self._num_rows, 3)), index=self._index,
                                             columns=[-1, 0, 1])
        pnl_ind = ['bids: ' + str(self._order_tuple[0]) + ',  asks: ' + str(self._order_tuple[1])]
        self._pnls = pd.DataFrame(np.zeros((1, 3)), index=pnl_ind, columns=[-1, 0, 1])
        self._pnls.columns.name = 'y'
        self._pnls.index.name = 'orders placed'
        self._quantity = {-1: 0, 0:0, 1:0}
        count = 0.0;
        while(count < num_samples):
            if self._evaluate_order_pair(count % 3 - 1):
                count += 1.0
            if(count % 1000 == 0):
                print("samples processed so far: " + str(count))
                
        self._fill_prob_matrix /= self._fill_prob_matrix.sum()
        self._pnls /= count
        self._pnls.fillna(0.0, inplace=True)
        self._fill_prob_matrix.fillna(0.0, inplace=True)
        self._fill_prob_matrix.index.name = 'orders_executed'
        self._fill_prob_matrix.columns.name = 'y'
        self._set_cum_prob_matrix()
        
    def get_cum_prob_matrix(self):
        return self._cum_prob_matrix
        
    def get_prob_matrix(self):
        return self._fill_prob_matrix
    
    def get_quantities(self):
        return self._quantity
    
    def get_pnls(self):
        return self._pnls

class LossFunction:
    '''class used to build our loss function for a given trading strategy'''
    def __init__(self, strategy, book, numupdates=None, timeunit=None,
                 t_start=34200.01, t_end=57600, uniform_sampling=True, ticks=None, rel_queue=None):
        '''
        get the loss function for a given trading strategy
        Strategy is a dict of the form: {y_hat: ([bids_list], [asks_list])} for each yhat value. 
        '''
        self._relative_queue = rel_queue
        self._strategy = strategy
        self._uniform_sampling = uniform_sampling
        self._numupdates = numupdates
        self._timeunit = timeunit
        self._t_start = t_start
        self._t_end = t_end
        self._orderbook = book
        self._ticks = ticks
        self._midprice_df = self._orderbook.get_midprice_data(numupdates=numupdates, timeunit=timeunit,
                                                              t_start=t_start, t_end=t_end, tick_size=self._ticks)
        self._prob_matrix_dict = {}
        self._cum_prob_matrix_dict = {}
        self._loss_matrix = pd.DataFrame()
        
    def generate_loss_function_and_fill_probabilities(self, num_samples):
        self._loss_matrix = pd.DataFrame(index=[-1, 0, 1], columns=[-1, 0, 1])
        self._prob_matrix_dict = {}
        self._cum_prob_matrix_dict = {}
        
        self._loss_matrix.index.name = 'y_predicted'
        self._loss_matrix.columns.name = 'y_true:'
        for key in self._strategy:
            print("simulating strategy for case yhat = " + str(key) + "...\n")
            self._probsimulator = FillProbabilitySimulator(numupdates=self._numupdates,
                                                           order_tuple=self._strategy[key], 
                                                           t_end=self._t_end, timeunit=self._timeunit,
                                                           t_start=self._t_start, ticks=self._ticks,
                                                           orderbook=self._orderbook,
                                                           midprice_df=self._midprice_df,
                                                           uniform_sampling=self._uniform_sampling,
                                                           rel_queue=self._relative_queue)
            
            self._probsimulator.generate_matrices(num_samples)
            self._loss_matrix.loc[key] = -1.0*self._probsimulator.get_pnls().values
            self._prob_matrix_dict[key] = self._probsimulator.get_prob_matrix()
            self._cum_prob_matrix_dict[key] = self._probsimulator.get_cum_prob_matrix()
            
        
    def get_loss_matrix(self):
        return self._loss_matrix
        
    def get_fill_probabilities(self):
        return self._prob_matrix_dict
    
    def get_cumulative_fill_probabilities(self):
        return self._cum_prob_matrix_dict
        
        
    