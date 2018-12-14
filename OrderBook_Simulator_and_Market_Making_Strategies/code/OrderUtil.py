'''Module contains functionality to handle order placement and execution'''
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
        initialize class, note that functionality for quantity is not in place
        we assume here we trade one lot (100 shares) at a time.

        inputs
        --------
        orderbook: OrderBook object. Contains the limit order book data associated with given equity symbol.
        timestamp: Float. Contains the time we enter the order (will round up to nearest time in orderbook data)
        level: positive int. What level we place our order (1 = inside market, 2 = 1 level away from best bid/ask, ...)
        is_buy: bool. Set to true for our order to be a bid, false for an ask.
        index_ref: Int. Set to an index reference if desired to place our order at that specific location in orderbook data.
        '''

        #timestamp out of orderbook range
        if timestamp > orderbook.tend or timestamp < orderbook.tstart:
            raise Exception("invalid time")

        #tracks state of our order, our order is now open
        self._orderstate = 'open'
        self._level = level
        self._orderbook = orderbook
        self._is_buy = is_buy

        #types of orders given in lobster data files
        self.types = {1: 'limit', 2: 'partial_cancellation', 3: 'total_cancellation',
                       4: 'exec_visible_limit', 5: 'exec_hidden_limit', 7: 'trading_halt'}
        
        #label our columns
        self._set_col_labels()

        #get the entry in our orderbook closest to the inputted timestamp
        ref_time = self._orderbook.limit_order_book().loc[:timestamp].index[-1]
        
        #get index position so we can grab data from orderbook dataframes more quickly
        if index_ref is None:
            #we now track out order by index position for faster lookup
            self._index_pos = self._orderbook.limit_order_book().index.get_loc(ref_time)

            #always assume if entering at an existing time in dataframe we enter "last"
            if isinstance(self._index_pos, slice):
                self._index_pos = self._index_pos.stop - 1
        else:
            self._index_pos = index_ref

        #here we get the price that reflects the current level and ordertype (buy or sell) that we placed
        self._price = self._orderbook.get_book_state(self._index_pos)[self._label_price]
        
        #set start time and start index
        self._t_start = timestamp
        self._t = timestamp
        self._start_index = self._index_pos
        
        #place our order at back of queue on given level. We track this queue value as we move through time
        self._queue_position = self._orderbook.get_book_state(self._index_pos)[self._label_quantity]
        self._queue_start = self._queue_position
        self._num_updates = 0

        #set order reference id
        self._set_reference_id()

        
    def _set_col_labels(self):
        '''function to set out label_price and label_quantity as integers for faster orderbook lookup'''
        if self._is_buy:
            self._label_price = self._orderbook.limit_order_book().columns.get_loc('bp' + str(self._level))
            self._label_quantity = self._orderbook.limit_order_book().columns.get_loc('bq' + str(self._level))
        else:
            self._label_price = self._orderbook.limit_order_book().columns.get_loc('ap' + str(self._level))
            self._label_quantity = self._orderbook.limit_order_book().columns.get_loc('aq' + str(self._level))
            
    def _set_closing_stats(self, time, index):
        '''set our order statistics at the time our order was closed'''
        self._t_end = time
        self._end_index = self._index_pos
        
    #order ids in lobster data files are conviniently in ascending order vs time order placed
    #so no need to randomize cancellations/exeution queue position!
    #we simply use a "reference_id" to determine if our order has higher or lower queue priority over other orders
    def _set_reference_id(self):
        '''set order reference id'''
        ind = self._index_pos + 1
        set_id = False
        #we get the next new limit order ID and consider all earlier IDs to be ahead of use in queue
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
        check to see if level changed during most recent single update to limit order book
        note that if level changed, it is not possible that we moved in the queue unless our order was filled (level wiped out)
        '''
        #check if level changed in most recent update, if the level has changed, the price in the orderbook no longer matches our price
        if book_state[self._label_price] == self._price:
            return False
        #we moved

        #simple index shift
        if self._is_buy:
            k = 2
        else:
            k = 0

        #check to see if our limit price exists on any level, if so we move to that level
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
            #so much adverse price movement that our order fell off the book
            #we cancel our order as we can no longer track this order correctly
            self._orderstate = "cancelled"
            self._set_closing_stats(self._t, self._index_pos)
            return True
        if self._is_buy and self._price > book_state[k]:
            #level wiped out, here our level no longer exists but our bid price is higher than best_bid
            #implies our order was just executed
            self._orderstate = "executed" 
#             print("order executed")
            self._set_closing_stats(self._t, self._index_pos)
            return True

        #same as above two condiitional statements but for an ask order
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
        
        #rare case where our entire level was wiped out and we are floating between two levels in book
        #we go to less competitive level and place our order at front of queue
        #note that this will occur far more often with illiquid stocks...
        if self._is_buy:
            for i in range(k, 4*self._orderbook.num_levels(), 4):
                if self._price > book_state[i]:
                    self._level = int((i - k)/4) + 1
                    #our price changes to less competitive price but we are placed at front of queue
                    self._price = book_state[i]
                    self._set_col_labels()
                    self._set_reference_id()
                    self._queue_position = 0
                    return True
                
        for i in range(k, 4*self._orderbook.num_levels(), 4):
            if self._price < book_state[i]:
                self._level = int((i - k)/4) + 1
                #our price changes to less competitive price but we are placed at front of queue
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

    
    def process_message(self):
        '''
        The main function of this class. Each time this function is called, we process one update to the limit order book
        We check if our order was executed, if we moved levels, if our queue position changed, or we fell off the book.
        Each time we increment our self._index_pos to point to next row in orderbook data.
        '''
        #if order already is closed we don't do anything
        if not self._orderstate == "open":
            self._num_updates += 1
            self._index_pos += 1
            return
        
        #increment to next row in orderbook data
        self._num_updates += 1
        self._index_pos += 1
        
        #reached end of book, order set as expired
        if self._index_pos >= self._orderbook.size:
            print("reached end of book")
            self._set_closing_stats(self._t, self._index_pos)
            self._orderstate == "expired"
            return
        
        #grab message and state of limit order book at given index_position reference
        message = self._orderbook.get_message(self._index_pos)
        book_state = self._orderbook.get_book_state(self._index_pos)

        #get the current time
        self._t = self._orderbook.get_current_time(self._index_pos)

        #get order_type of given message
        order_type = message[0]
        
        #if we moved levels we do not have to process the actual updated message
        if self._check_level(book_state):
            #check if we hit front of queue if so execute order
            if self._check_queue(book_state):
                return
            else:
                return
        
        #if we hit the front of the queue and we are at level 1 consider as executed
        if self._check_queue(book_state):
            return
        
        #do not care if current order is not equal to our price and we dont move a level
        #has no direct impact on our order
        if message[3] != self._price:
            return
        #execution of hidden limit order, no queue movement
        if order_type == 5:
            return
        
        #this has never happened but if it does we raise an excption
        #for less liquid stocks this might be a possibility
        if self._is_buy and message[4] == -1 or not self._is_buy and message[4] == 1:
            print("can this happen!?!?")
            raise Exception("why did this happen opposite order type observed on our level!")
            return
        
        #new limit order on our level, do nothing
        if order_type == 1:
            return
        
        #unhandled trading halt indicator
        if order_type == 7:
            raise Exception("Trading Halt!")
            return
        
        #limit order in new message is behind our order (since its ID is larger)
        if message[1] > self._order_reference_id:
            #if the limit order got executed, we execute ours (means we didnt follow queue correctly)
            if message[1] == 4:
                self._orderstate = "executed"
                self._set_closing_stats(self._t, self._index_pos)
            return
        
        #order ahead of us on queue
        if message[1] <= self._order_reference_id:
            #we move up in queue by amount equal to size of limit order
            #we do not care if this order was exeucted or cancelled
            self._queue_position -= message[2]

            #now check queue
            #it is true that we could be 0 on queue but our order was not yet executed, we just assume here it is
            if self._check_queue(book_state):
                return
            #fail safe for cases we somehow fall behind back of queue, we set our queue position to back of queue
            if self._queue_position > book_state[self._label_quantity]:
                self._queue_position = book_state[self._label_quantity]
            return
        
        #if we reach this code we have an unhandled case...
        if order_type not in self.types:
            raise Exception("unhandled order type given")
        
        raise Exception("unhandled message exists")
    
    def order_type(self, str_ordertype):
        '''
        checks ordertype

        input
        --------
        str_ordertype: String. Ordertype we would like to check (ex: "open", "executed",...)

        output
        --------
        True if types match false otherwise.
        '''
        if self._orderstate == str_ordertype:
            return True
        else:
            return False
        
    def get_opening_stats(self):
        '''retrieve statistics for our order at the time we placed the order'''
        return {'price' : self._price, 'start_time' : self._t_start, 'is_buy': self._is_buy,
                'queue_position' : self._queue_start,
                'orderstate' : "open", 'level' : self._level, 'start_index': self._start_index}
    
    def get_current_stats(self):
        '''get current order statistics'''
        return {'price' : self._price, 'time' : self._t, 'is_buy': self._is_buy,
                'queue_position' : self._queue_position,
                'orderstate' : self._orderstate, 'level' : self._level,
                'current_index': self._index_pos}
    
    def get_closing_stats(self):
        '''get statistics at the close'''
        return {'price' : self._price, 'time' : self._t_end, 'is_buy': self._is_buy,
                'queue_position' : 0,
                'orderstate' : self._orderstate, 'level' : self._level,
                'end_index': self._end_index}
    
    def get_order_price(self):
        '''get order price'''
        return self._price
    
    def get_current_level(self):
        '''get current level of our order'''
        return self._level
    
    def get_current_time(self):
        '''get current time'''
        return self._t
    
    def get_queue_position(self):
        '''get current queue position'''
        return self._queue_position
    
    def set_queue_position(self, pos):
        '''set the queue position if desired'''
        self._queue_position = pos
        
    def get_relative_queue(self):
        '''get relative queue position (ex: 0.05 means we are in front 5% of queue)'''
        return self._queue_position/(1.0*self._orderbook.get_book_state(self._index_pos)[self._label_quantity])
    
    def cancel_order(self):
        '''cancel our order if it is still open'''
        if self._orderstate == 'open':
            self._orderstate = 'cancelled'
            
    def get_current_index(self):
        '''get the current index position in the orderbook'''
        return self._index_pos
    
class TimeOrder(Order):
    '''Use this class for orders that are updated by a constant time step'''
    def __init__(self, orderbook, timestamp, level, is_buy, delta_t,
                 index_ref=None):
        '''
        Initialization is the same as the Order class except we have:
        delta_t: float. Timestep for evaluating orders, in seconds.
        '''
        Order.__init__(self, orderbook, timestamp, level, is_buy,
                       index_ref=index_ref)

        #set dt
        self._dt = delta_t
        self._time_to_step = self._t_start
        
    def update(self):
        '''Here we update our Order for given step self._dt in time'''
        self._time_to_step += self._dt
        self._t_next = self._orderbook.get_current_time(self._index_pos + 1)
        #keep processing a row in orderbook until we waited self._dt seconds
        while(self._t_next <= self._time_to_step and self._orderstate == "open"):
            self.process_message()
            self._t_next = self._orderbook.get_current_time(self._index_pos + 1)
    
class BookUpdatesOrder(Order):
    '''use this class for orders updated after a given fixed number of rows of orderbook data is updated'''
    def __init__(self, orderbook, timestamp, level, is_buy, numupdates, index_ref=None):
        '''
        Initialization is the same as the Order class except we have:
        numupdates: Int. Fixed number of updates to orderbook to wait until we stop updating our order.
        '''

        Order.__init__(self, orderbook, timestamp, level, is_buy,
                       index_ref=index_ref)
        self._numupdates = numupdates
        self._index_to_step = self._start_index
        
    def update(self):
        '''Keep processing messages "self._numupdates" number of times'''
        self._index_to_step += self._numupdates
        while(self._index_pos < self._index_to_step and self._orderstate == "open"):
            self.process_message()
