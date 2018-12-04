
# coding: utf-8

# In[1]:

#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math


class Order():
    '''base class to track and execute our orders'''
    def __init__(self, orderbook, timestamp, level, is_buy, index_ref=None):
        '''
        when order is placed we create the order object
        in cases where the entered timestamp is exactly equal to an order in the book, 
        we place our order immediately beforehand
        '''
        if timestamp > orderbook.tend or timestamp < orderbook.tstart:
            raise Exception("invalid time")
        self._orderstate = 'open'
        self._level = level
        self._orderbook = orderbook
        self._is_buy = is_buy
        self.types = {1: 'limit', 2: 'partial_cancellation', 3: 'total_cancellation',
                       4: 'exec_visible_limit', 5: 'exec_hidden_limit', 7: 'trading_halt'}
        
        self._set_col_labels()
        ref_time = self._orderbook.limit_order_book().loc[:timestamp].index[-1]
        
        #get index position so we can grab data from orderbook dataframes more quickly
        if index_ref is None:
            self._index_pos = self._orderbook.limit_order_book().index.get_loc(ref_time)

            #always assume if entering at an existing time in dataframe we enter "last"
            if isinstance(self._index_pos, slice):
                self._index_pos = self._index_pos.stop - 1
        else:
            self._index_pos = index_ref  
        self._price = self._orderbook.get_book_state(self._index_pos)[self._label_price]
        
        #set start time and start index
        self._t_start = timestamp
        self._t = timestamp
        self._start_index = self._index_pos
        
        
        self._queue_position = self._orderbook.get_book_state(self._index_pos)[self._label_quantity]
        self._queue_start = self._queue_position
        self._num_updates = 0
        self._set_reference_id()

        
    def _set_col_labels(self):
        if self._is_buy:
            self._label_price = self._orderbook.limit_order_book().columns.get_loc('bp' + str(self._level))
            self._label_quantity = self._orderbook.limit_order_book().columns.get_loc('bq' + str(self._level))
        else:
            self._label_price = self._orderbook.limit_order_book().columns.get_loc('ap' + str(self._level))
            self._label_quantity = self._orderbook.limit_order_book().columns.get_loc('aq' + str(self._level))
            
    def _set_closing_stats(self, time, index):
        self._t_end = time
        self._end_index = self._index_pos
        
    #order ids are in conviniently in order of orderflow
    #so no need to randomize cancellations/exeution queue position!
    def _set_reference_id(self):
        ind = self._index_pos + 1
        set_id = False
        while(True):
            try:
                message = self._orderbook.get_message(ind)
            except:
                raise Exception("reference id not set")
            if message[0] == 1:
                self._order_reference_id = message[1] - 0.5
                set_id = True
                break
            ind += 1
        if set_id:
            return
        raise Exception("reference id not set")
           
    def _check_level(self, book_state):
        '''
        check to see if level changed during most recent update
        note that if level changed, it is not possible that we moved in the queue unless our order was filled
        '''
        #check if level changed in most recent update
        if book_state[self._label_price] == self._price:
            return False
        #we moved
        if self._is_buy:
            k = 2
        else:
            k = 0
        #check to see if our limit price exists on any level
        for i in range(k, 4*self._orderbook.num_levels(), 4):
            if self._price == book_state[i]:
                self._level = int((i - k)/4) + 1
                self._set_col_labels()
                return True
            else:
                continue
        #handle case where either our order was filled (entire bid level wiped out)
        #or we fell off deepest level
        if self._is_buy and self._price < book_state[k + 4*self._orderbook.num_levels() - 4]:
            #so much adverse selection that our order fell off the book
            self._orderstate = "cancelled"
            self._set_closing_stats(self._t, self._index_pos)
            return True
        if self._is_buy and self._price > book_state[k]:
            #level wiped out
            self._orderstate = "executed" 
#             print("order executed")
            self._set_closing_stats(self._t, self._index_pos)
            return True
        if not self._is_buy and self._price > book_state[k + 4*self._orderbook.num_levels() - 4]:
            self._orderstate = "cancelled"
            self._set_closing_stats(self._t, self._index_pos)
            return True
        if not self._is_buy and self._price < book_state[k]:
            #level wiped out
            self._orderstate = "executed"
#             print("order executed")
            self._set_closing_stats(self._t, self._index_pos)
            return True
        
        #rare case where our entire level was wiped out and we are floating between two levels...
        #we go to less competitive level and place our order at front of queue
        if self._is_buy:
            for i in range(k, 4*self._orderbook.num_levels(), 4):
                if self._price > book_state[i]:
                    self._level = int((i - k)/4) + 1
                    self._price = book_state[i]
                    self._set_col_labels()
                    self._set_reference_id()
                    self._queue_position = 0
                    return True
                
        for i in range(k, 4*self._orderbook.num_levels(), 4):
            if self._price < book_state[i]:
                self._level = int((i - k)/4) + 1
                self._price = book_state[i]
                self._set_col_labels()
                self._set_reference_id()
                self._queue_position = 0
                return True
            
        raise Exception("Unhandled case exists")
        
    def _check_queue(self, book_state):
        #if we hit the front of the queue and we are at level 1 consider as executed     
        if self._queue_position <= 0 and self._level == 1:
            self._orderstate = "executed"
            self._set_closing_stats(self._t, self._index_pos)
            return True
        return False

    
    def _process_message(self):
        #grab new updated data
        '''
        something still wrong with queue movement, either some logic is off
        or order_ids are not all tracked correctly in our data file
        '''
        if not self._orderstate == "open":
            return
        
        self._num_updates += 1
        self._index_pos += 1
        
        if self._index_pos >= self._orderbook.size:
            print("reached end of book")
            self._set_closing_stats(self._t, self._index_pos)
            self._orderstate == "expired"
            return
        
        message = self._orderbook.get_message(self._index_pos)
        book_state = self._orderbook.get_book_state(self._index_pos)
        self._t = self._orderbook.get_current_time(self._index_pos)
        order_type = message[0]
        
        #if we moved levels we do not have to process the actual updated message
        if self._check_level(book_state):
            if self._check_queue(book_state):
                return
            else:
                return
        
        #if we hit the front of the queue and we are at level 1 consider as executed
        if self._check_queue(book_state):
            return
        
        #do not care if current order != our price
        if message[3] != self._price:
            return
        
        if order_type == 5:
            #execution of hidden limit order, no queue movement
            return
        
        if self._is_buy and message[4] == -1 or not self._is_buy and message[4] == 1:
            print("can this happen!?!?")
            print(message)
            print(self._orderbook.get_book_state(self._index_pos - 1))
            print(self._orderbook.get_book_state(self._index_pos))
            print(self._orderbook.get_book_state(self._index_pos + 1))
            return
        
        #new limit order on our level, do nothing
        if order_type == 1:
            return
        
        if order_type == 7:
            #trading halt indicator
            print("Trading halt!")
            return
        
        #order behind our order
        if message[1] > self._order_reference_id:
            #if it got executed, we execute ours (means we didnt follow queue correctly)
            if message[1] == 4:
                self._orderstate = "executed"
                self._set_closing_stats(self._t, self._index_pos)
            return
        
        #order ahead of us on queue
        if message[1] <= self._order_reference_id:
            self._queue_position -= message[2]
            if self._check_queue(book_state):
                return
            #fail safe for cases we somehow fall behind back of queue
            if self._queue_position > book_state[self._label_quantity]:
                self._queue_position = book_state[self._label_quantity]
            return
        
        if order_type not in self.types:
            raise Exception("unhandled order type given")
        
        raise Exception("unhandled message exists")
      
    #the below functions are to be used outside of class
    
    def order_type(self, str_ordertype):
        '''checks ordertype'''
        if self._orderstate == str_ordertype:
            return True
        else:
            return False
        
    def get_opening_stats(self):
        return {'price' : self._price, 'start_time' : self._t_start, 'is_buy': self._is_buy,
                'queue_position' : self._queue_start,
                'orderstate' : "open", 'level' : self._level, 'start_index': self._start_index}
    
    def get_current_stats(self):
        return {'price' : self._price, 'time' : self._t, 'is_buy': self._is_buy,
                'queue_position' : self._queue_position,
                'orderstate' : self._orderstate, 'level' : self._level,
                'current_index': self._index_pos}
    
    def get_closing_stats(self):
        return {'price' : self._price, 'time' : self._t_end, 'is_buy': self._is_buy,
                'queue_position' : 0,
                'orderstate' : self._orderstate, 'level' : self._level,
                'end_index': self._end_index}
    
    def get_order_price(self):
        return self._price
    
    def get_current_level(self):
        return self._level
    
    def get_current_time(self):
        return self._t
    
    def get_queue_position(self):
        return self._queue_position
    
    def set_queue_position(self, pos):
        self._queue_position = pos
        
    def get_relative_queue(self):
        return self._queue_position/(1.0*self._orderbook.get_book_state(self._index_pos)[self._label_quantity])
    
    def cancel_order(self):
        if self._orderstate == 'open':
            self._orderstate = 'cancelled'
    
class TimeOrder(Order):
    '''Use this class for strategies that reevaluate orders by given timestep delta_t (in seconds)'''
    def __init__(self, orderbook, timestamp, level, is_buy, delta_t,
                 index_ref=None):
        Order.__init__(self, orderbook, timestamp, level, is_buy,
                       index_ref=index_ref)
        self._dt = delta_t
        self._time_to_step = self._t_start
        
    def update(self):
        self._time_to_step += self._dt
        self._t_next = self._orderbook.get_current_time(self._index_pos + 1)
        while(self._t_next <= self._time_to_step and self._orderstate == "open"):
            self._process_message()
            self._t_next = self._orderbook.get_current_time(self._index_pos + 1)
    
class BookUpdatesOrder(Order):
    '''use this class for strategies that reevaluate orders by number updates to limit order book'''
    def __init__(self, orderbook, timestamp, level, is_buy, numupdates, index_ref=None):
        Order.__init__(self, orderbook, timestamp, level, is_buy,
                       index_ref=index_ref)
        self._numupdates = numupdates
        self._index_to_step = self._start_index
        
    def update(self):
        self._index_to_step += self._numupdates
        while(self._index_pos < self._index_to_step and self._orderstate == "open"):
            self._process_message()

class IndexTrackedOrder(Order):
    '''use this class for strategies that build orders around a given index_reference series for backtesting'''
    def __init__(self, orderbook, level, is_buy, index_ser, ind_start):
        self._index_ser = index_ser.astype(int)
        self._current_index_ref = ind_start
        self._book_position = self._index_ser.values[self._current_index_ref]
        Order.__init__(self, orderbook=orderbook, level=level, is_buy=is_buy, index_ref=self._book_position,
                       timestamp=self._index_ser.index[ind_start])
        
        
    def update(self):
        self._current_index_ref += 1
        self._book_position = self._index_ser.values[self._current_index_ref]
        while(self._index_pos < self._book_position and self._orderstate == "open"):
            self._process_message()
            
    def get_current_index(self):
        return self._current_index_ref
            
                        
        