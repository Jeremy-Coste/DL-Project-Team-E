
# coding: utf-8

# In[1]:

#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math

class OrderBook():
    '''class to hold our orderbook and orderbook messages'''
    def __init__(self, message_filename, orderbook_filename, path=os.getcwd(), memory_size=10):
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
        
        self._memory_size = memory_size
        self._stored_keys = {'book': set(), 'messages': set(), 'timestamps': set()}
        self._stored_timestamps = {}
        self._stored_bookstates = {}
        self._stored_messages = {}
        
    def get_midprice_data(self, numupdates=1, t_start=34200.1,
                          t_end=57599.9, tick_size=None, next_move=False):
        '''
        Here we compute midprices, and store y(previous), y(current), and y(next)
        '''
        if t_start < self.tstart or t_end > self.tend:
            raise Exception("Invalid time")
           
        midprices = pd.DataFrame((self._limit_order_book['ap1'] + self._limit_order_book['bp1'])/2.0,
                                  index= self._limit_order_book.index, columns=['midprice'])
        midprices['ap1'] = self._limit_order_book['ap1']
        midprices['aq1'] = self._limit_order_book['aq1']
        midprices['bp1'] = self._limit_order_book['bp1']
        midprices['bq1'] = self._limit_order_book['bq1']
        midprices['index_position'] = range(len(midprices))
        
        midprices = midprices.iloc[0::numupdates]
        midprices = midprices.loc[t_start:t_end]
        midprices.reset_index(inplace=True)
        midprices.drop_duplicates(subset='timestamp', keep='last', inplace=True)
        midprices.set_index('timestamp', inplace=True)
   
        midprices['y_0'] = np.sign(midprices['midprice'] - midprices['midprice'].shift(1))
        midprices['iloc'] = [i for i in range(len(midprices))]
        if not tick_size is None:
            #we only count midprice movements that widen the spread (or maintain spread size)
            bools = (abs(midprices['ap1'].shift(1) - midprices['bp1'].shift(1)) <=
                     1.01*abs(midprices['ap1'] - midprices['bp1']))
            
            midprices['y_0'] *= bools.astype(int)
            
        midprices['movement'] = midprices['y_0'].copy()
        if next_move:
            #look for next midprice move
            change_df = midprices.loc[midprices['y_0'] != 0][['y_0', 'iloc']]
            change_df.set_index('iloc', inplace=True)
            change_df= change_df.reindex_axis(range(len(midprices)), method='backfill')
            midprices['y_0'] = change_df.values
            
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
    
    def get_bid_ask(self, timestamp):
        '''get best bid and ask at given timestamp'''
        ref_time = self._limit_order_book.loc[:timestamp].index[-1]
        
        #get index position so we can grab data from orderbook dataframes more quickly
        index_pos = self._limit_order_book.index.get_loc(ref_time)
        
        #always assume if looking up an existing time in dataframe we look at "last" value at that time
        if isinstance(index_pos, slice):
            index_pos = index_pos.stop - 1
        
        book_state = self._limit_order_book.values[index_pos]
        best_bid = book_state[2]
        best_ask = book_state[0]
        
        return (best_bid, best_ask)
    
    def get_message(self, position):
        if not position in self._stored_messages:
            if len(self._stored_keys['messages']) > self._memory_size:
                ind_to_delete = min(self._stored_keys['messages'])
                del self._stored_messages[ind_to_delete]
                self._stored_keys['messages'].remove(ind_to_delete)
            self._stored_messages[position] = self._messages.values[position]
            self._stored_keys['messages'].add(position)
            
            
        return self._stored_messages[position]
    
    def get_book_state(self, position):
        if not position in self._stored_bookstates:
            if len(self._stored_keys['book']) > self._memory_size:
                ind_to_delete = min(self._stored_keys['book'])
                del self._stored_bookstates[ind_to_delete]
                self._stored_keys['book'].remove(ind_to_delete)
            self._stored_bookstates[position] = self._limit_order_book.values[position]
            
            
        return self._stored_bookstates[position]
    
    def get_current_time(self, position):
        if not position in self._stored_timestamps:
            if len(self._stored_keys['timestamps']) > self._memory_size:
                ind_to_delete = min(self._stored_keys['timestamps'])
                del self._stored_timestamps[ind_to_delete]
                self._stored_keys['timestamps'].remove(ind_to_delete)
            self._stored_timestamps[position] = self._timestamp_series.values[position]
            
        return self._stored_timestamps[position]
            
        
    def limit_order_book(self):
        return self._limit_order_book;
    
    def messages(self):
        return self._messages
    
    def midprices(self):
        return self._midprices
    
    def num_levels(self):
        return self._n
    
    def clear_memory(self):
        self._stored_keys = {'book': set(), 'messages': set(), 'timestamps': set()}
        self._stored_timestamps = {}
        self._stored_bookstates = {}
        self._stored_messages = {}
        