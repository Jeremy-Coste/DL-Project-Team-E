'''Module contains class to hold our order book data'''
# coding: utf-8

# In[1]:

#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math

class OrderBook():
    '''
    class to hold our orderbook and orderbook messages
    '''
    def __init__(self, message_filename, orderbook_filename, path=os.getcwd(), memory_size=10):
        '''
        initialize class
        Note that np.array.values is significantly faster (a couple orders of magnitude)
              if all data types in our array are of type int as opposed to floats.
        So it makes sense to convert all columns in the csv files to an integer type 
        (we set timestamp floats as our Pandas DataFrame index 
        so we can still take advantage of numpy speed when accessing values in our DataFrames)

        inputs
        --------
        message_filename: String. Name of csv file containing all order updates 
                          See "https://lobsterdata.com/info/DataSamples.php"
        orderook_filename: String. Name of csv file containing state of orderbook for all timestamps
                           found in messages file
        path: String. Directory to "data" folder containing our data files
        memory_size: Positive int. Number of messages and states stored in dictionaries at a time for faster lookup.
        '''
        self._messages = pd.read_csv(os.path.join(path, 'data', message_filename), header=None)
        self._limit_order_book = pd.read_csv(os.path.join(path, 'data', orderbook_filename), header=None)
        
        #check if the length of our dataframes are the same size
        if self._messages.shape[0] != self._limit_order_book.shape[0]:
            raise Exception("The two data files do not contain the same number of rows")
        
        #n stores the number of levels in our limit order book
        self._n = int(self._limit_order_book.shape[1]/4)

        #number of data points in book
        self._number_data_points = self._limit_order_book.shape[0]

        #these are the names and ordertype codes of each column given in lobster data files 
        self._messages.columns = ['timestamp', 'type', 'order_id', 'size', 'price', 'direction']
        self.direction = {1: 'buy', -1: 'sell'}
        self.types = {1: 'limit', 2: 'partial_cancellation', 3: 'total_cancellation',
                       4: 'exec_visible_limit', 5: 'exec_hidden_limit', 7: 'trading_halt'}

        #simply name our rows and columns
        self._label_dataframes()
        
        #get earliest and latest timestamp (represented as seconds since midnight as a float)
        self.tstart = self._messages.index.min()
        self.tend = self._messages.index.max()
        
        self.size = self._messages.shape[0]
        self._timestamp_series = self._messages.index
        
        self._memory_size = memory_size

        #dictionaries to store recent rows in orderbook data for fast repetitive lookup
        self._stored_keys = {'book': set(), 'messages': set(), 'timestamps': set()}
        self._stored_timestamps = {}
        self._stored_bookstates = {}
        self._stored_messages = {}
        
    def get_midprice_data(self, numupdates=1, t_start=34200.1,
                          t_end=57599.9, count_only_wider_spreads=False, next_move=False):
        '''
        Here we compute midprices, and store y(previous), y(current), and y(next)

        inputs
        --------
        numupdates: int. We pull every row in dataframe that is a multiple of numupdates.
        t_start:  float. Time to start pulling data
        t_end: float. End time.
        count_only_wider_spreads: bool. Set to true to count only midprice movements 
                                  where current bid-ask spread >= previous bid-ask spread

        next_move: bool. Set to true if we would like y to be defined as the next midprice move instead of the current move. 

        output
        --------
        midprices: DataFrame. Contains best bid ask information, timestamps,
                   current y_0, previous y_prev and next y_1 midprice movement as desired
        '''
        if t_start < self.tstart or t_end > self.tend:
            raise Exception("Invalid time")

           
        #add a midprice row
        midprices = pd.DataFrame((self._limit_order_book['ap1'] + self._limit_order_book['bp1'])/2.0,
                                  index= self._limit_order_book.index, columns=['midprice'])

        #add best bid ask information 
        midprices['ap1'] = self._limit_order_book['ap1']
        midprices['aq1'] = self._limit_order_book['aq1']
        midprices['bp1'] = self._limit_order_book['bp1']
        midprices['bq1'] = self._limit_order_book['bq1']
        midprices['index_position'] = range(len(midprices))
        
        #keep rows that are multiples of numupdates
        midprices = midprices.iloc[0::numupdates]

        #slice for time range
        midprices = midprices.loc[t_start:t_end]
        midprices.reset_index(inplace=True)

        '''
        Important! keep last time entry when multiple rows with the same timestamp are present 
        as we cannot make any trades or be in any state as the
        orderbook is updated all at once for a given timestamp
        '''
        midprices.drop_duplicates(subset='timestamp', keep='last', inplace=True)
        midprices.set_index('timestamp', inplace=True)

        #get midprice movement
        midprices['y_0'] = np.sign(midprices['midprice'] - midprices['midprice'].shift(1))
        midprices['iloc'] = [i for i in range(len(midprices))]
        if count_only_wider_spreads:
            #we only count midprice movements that widen the spread (or maintain spread size)
            bools = (abs(midprices['ap1'].shift(1) - midprices['bp1'].shift(1)) <=
                     1.01*abs(midprices['ap1'] - midprices['bp1']))
            
            midprices['y_0'] *= bools.astype(int)
            
        midprices['movement'] = midprices['y_0'].copy()
        if next_move:
            #look for next midprice move
            change_df = midprices.loc[midprices['y_0'] != 0][['y_0', 'iloc']]
            change_df.set_index('iloc', inplace=True)

            #backfill all entries with the next midprice move
            change_df= change_df.reindex_axis(range(len(midprices)), method='backfill')
            midprices['y_0'] = change_df.values
            
        #shift y_0 to get y_1 and y_prev
        midprices['y_1'] = midprices['y_0'].shift(-1)
        midprices['y_prev'] = midprices['y_0'].shift(1)

        return midprices  
        
    def _label_dataframes(self):
        '''label dataframe columns and set index to timestamp for speed'''
        columns = []

        #our dataframe labeling works as follows
        #"ap" reprents ask price "aq" represents ask quantity and i is the level (starting from 1)
        for i in range(1, self._n + 1):
            columns += ['ap' + str(i)]
            columns += ['aq' + str(i)]
            columns += ['bp' + str(i)]
            columns += ['bq' + str(i)] 
            
        self._limit_order_book.columns = columns
        self._limit_order_book['timestamp'] = self._messages['timestamp']

        #we set the index for both our dataframes as the timestamp
        self._messages.set_index('timestamp', inplace=True)
        self._limit_order_book.set_index('timestamp', inplace=True)
    
    def get_bid_ask(self, timestamp):
        '''
        get best bid and ask prices at given timestamp

        inputs
        --------
        timestamp: Float. time to fetch best bid ask

        output
        --------
        tuple with best_bid_price, best_ask_price
        '''
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
        '''
        get message for given position

        inputs
        -------
        position: Integer. index location in original limit order book to fetch message

        output
        --------
        stored_messages[position]: Numpy array. Conatins each column of messages DataFrame for given position
        '''

        #we store the last memory number of keys in stored_messages, so if they are requested again,
        #lookup time is faster
        if not position in self._stored_messages:
            if len(self._stored_keys['messages']) > self._memory_size:
                ind_to_delete = min(self._stored_keys['messages'])
                del self._stored_messages[ind_to_delete]
                self._stored_keys['messages'].remove(ind_to_delete)
            self._stored_messages[position] = self._messages.values[position]
            self._stored_keys['messages'].add(position)
            
            
        return self._stored_messages[position]
    
    def get_book_state(self, position):
        '''
        get state of limit order book at position
        Same as above but inputs limit_order_book state for given position
        '''
        if not position in self._stored_bookstates:
            if len(self._stored_keys['book']) > self._memory_size:
                ind_to_delete = min(self._stored_keys['book'])
                del self._stored_bookstates[ind_to_delete]
                self._stored_keys['book'].remove(ind_to_delete)
            self._stored_bookstates[position] = self._limit_order_book.values[position]
            
            
        return self._stored_bookstates[position]
    
    def get_current_time(self, position):
        '''
        get current time for given position
        same as above but returns a float representing the timestamp value at given position in orderbook data
        '''
        if not position in self._stored_timestamps:
            if len(self._stored_keys['timestamps']) > self._memory_size:
                ind_to_delete = min(self._stored_keys['timestamps'])
                del self._stored_timestamps[ind_to_delete]
                self._stored_keys['timestamps'].remove(ind_to_delete)
            self._stored_timestamps[position] = self._timestamp_series.values[position]
            
        return self._stored_timestamps[position]
            
        
    def limit_order_book(self):
        '''function to call entire limit order book'''
        return self._limit_order_book;
    
    def messages(self):
        '''grab all messages'''
        return self._messages
    
    def num_levels(self):
        '''get number of levels in book'''
        return self._n
    
    def clear_memory(self):
        '''clear stored memory if desired'''
        self._stored_keys = {'book': set(), 'messages': set(), 'timestamps': set()}
        self._stored_timestamps = {}
        self._stored_bookstates = {}
        self._stored_messages = {}
        