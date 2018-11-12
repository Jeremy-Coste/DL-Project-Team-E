
# coding: utf-8

# In[1]:

#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math

class OrderBook():
    '''class to hold our orderbook and orderbook messages'''
    def __init__(self, message_filename, orderbook_filename, path=os.getcwd()):
        '''
        messsages contains all updates to limit order book
        limit_order_book contains the state of the book at all timestamps
        n is number of levels
        
        '''
        self._messages = pd.read_csv(os.path.join(path, 'data', message_filename), header=None)
        self._limit_order_book = pd.read_csv(os.path.join(path, 'data', orderbook_filename), header=None)
        
        if self._messages.shape[0] != self._limit_order_book.shape[0]:
            raise Exception("The two data files do not contain the same number of rows")
        
        self._n = int(self._limit_order_book.shape[1]/4)
        self._number_data_points = self._limit_order_book.shape[0]
        
        self._messages.columns = ['timestamp', 'type', 'order_id', 'size', 'price', 'direction']
        self.direction = {1: 'buy', -1: 'sell'}
        self.types = {1: 'limit', 2: 'partial_cancellation', 3: 'total_cancellation',
                       4: 'exec_visible_limit', 5: 'exec_hidden_limit', 7: 'trading_halt'}
        self._label_dataframes()
        
        self.tstart = self._messages.index.min()
        self.tend = self._messages.index.max()
        
        self.size = self._messages.shape[0]
        self._timestamp_series = self._messages.index
        
    def get_midprice_data(self, numupdates=None, timeunit=None, t_start=34200.1, t_end=57599.9):
        '''
        Here we compute midprices, and store y(previous), y(current), and y(next)
        If we wish to slice by time, set by_message_updates = False and
                enter timeunit=1 for seconds, 1e-3 for microseconds and so on
        '''
        if t_start < self.tstart or t_end > self.tend:
            raise Exception("Invalid time")
        if numupdates is None and timeunit is None:
            raise Exception("Must set either numupdates or timeunit")
        if not numupdates is None and not timeunit is None:
            raise Exception("Must set either numupdates or timeunit but not both")
            
        midprices = pd.DataFrame((self._limit_order_book['ap1'] + self._limit_order_book['bp1'])/2.0,
                                     index= self._limit_order_book.index, columns=['midprice'])
        if timeunit is None:
            midprices = midprices.iloc[0::numupdates]
            midprices = midprices.loc[t_start:t_end]
            midprices.reset_index(inplace=True)
            midprices.drop_duplicates(subset='timestamp', keep='last', inplace=True)
            midprices.set_index('timestamp', inplace=True)
        else:
            timestamps = []
            count = 1
            for ind in midprices.index.tolist():
                if (ind - t_start)/(1.0*timeunit) >= count:
                    count += 1.0
                    timestamps += [True]
                else:
                    timestamps += [False]
                    
            midprices = midprices.loc[timestamps]
            midprices.reset_index(inplace=True)
            midprices.drop_duplicates(subset='timestamp', keep='last', inplace=True)
            midprices.set_index('timestamp', inplace=True)
            count = 0
            t = t_start
            timestamps = []
            prices = []
            while(count < len(midprices)):
                t += timeunit
                if(t > midprices.index[count]):
                    new_mid = midprices.values[count, 0]
                    count += 1
                    timestamps += [t]
                    prices += [new_mid]
                    continue
                if not timestamps:
                    continue
                timestamps += [t]
                prices += [new_mid]
                
            midprices = pd.DataFrame(prices, index=timestamps, columns=['midprice'])
            midprices.index.name = 'timestamp'
            midprices = midprices.loc[t_start:t_end]
                
                
        midprices['y_0'] = np.sign(midprices['midprice'] - midprices['midprice'].shift(1)) 
        midprices['y_1'] = midprices['y_0'].shift(-1)
        midprices['y_prev'] = midprices['y_0'].shift(1)

        return midprices  
        
    def _label_dataframes(self):
        '''label dataframe columns and set index to timestamp for speed'''
        columns = []
        for i in range(1, self._n + 1):
            columns += ['ap' + str(i)]
            columns += ['aq' + str(i)]
            columns += ['bp' + str(i)]
            columns += ['bq' + str(i)] 
            
        self._limit_order_book.columns = columns
        self._limit_order_book['timestamp'] = self._messages['timestamp']
        self._messages.set_index('timestamp', inplace=True)
        self._limit_order_book.set_index('timestamp', inplace=True)
    
    def get_message(self, position):
        return self._messages.values[position]
    
    def get_book_state(self, position):
        return self._limit_order_book.values[position]
    
    def get_current_time(self, position):
        return self._timestamp_series.values[position]
        
    def limit_order_book(self):
        return self._limit_order_book;
    
    def messages(self):
        return self._messages
    
    def midprices(self):
        return self._midprices
    
    def num_levels(self):
        return self._n
        
