'''Module used to construct and backtest our Market Making Trading Strategy'''

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
class TradingStrategyBacktester: 
    '''
    Class to backtest our trading strategy
    Our Market Making Strategy is implemented here hoever one can implement their own strategy
    by changing some functionality regarding when we place trades and the signals themselves.

    Please refer to documentation concerning our Market Making Strategy before going though the code
    '''
    def __init__(self, book, strategy, midprice_df, latency=(0, 0),
                 set_edge_queue=None, tick_size=None,
                 max_exposure_dict={0:0, 1:0, -1:0}):
        '''
        Initialize class to test our strategy on limit order book data.

        inputs
        --------

        book: OrderBook object. limit order book data for given equity.
        strategy: Dictionary. The format is of the form {y_hat: ([bids_list], [asks_list])}
                  ex: {1: ([1, 2, 3], [2, 3])} implies that when we predict an upward midprice movement,
                  we place (or keep) buy order on levels 1, 2, and 3 and ask orders on levels 2 and 3.
                  For the given strategy this is {1:([1,2,3,4,5], [1,2,3,4,5]), -1:([1,2,3,4,5], [1,2,3,4,5])}
        midprice_df: DataFrame. Output of OrderBook.get_midprice_data(args) with one additional column
                     We must add a column named "y_predict" which represent our predictions at the given timestamp.
        latency: Tuple. latency[0] reprents the time delay we receive limit order book data (meaining market quotes)
                        latency[1] represents the time delay in sending/cancelling orders to exchange as well as the time delay
                                   to receive updates regarding our orders from the exchange.
        tick_size: Int. Refers to standard or typical bid-ask spread size in the orderbook data. 
                   For the given strategy, we only activate when the current bid-ask spread is less than or equal to tick_size.
        max_exposure_dict: Dictionary of the form {y_hat: exposure_integer} representing the max number of shares
                           we can be long or short for the given y_hat prediction. 
                           For the given strategy this is {1: 1, -1: 1}

        '''

        #set tracked tick size
        self._tick_size = tick_size
        self._orderbook = book

        #set latency 
        self._latency = latency

        self._midprice_df = midprice_df.copy()

        '''
        latency[1] is the delay in sending/receiving orders...
        below we input one row in our midprice DataFrame spaced in time by a minimum of 2*latency[1]
        We take double the order_to_exchange latency since we do not reevaluate our positions and strategy until receiving order updates
        from the exchange regarding our previous evaluation time. It takes latency[1] time for orders to get to exchange and then latency[1] for
        exchange to notify us of our order status.
        '''
        if latency[1] > 0:
            new_ind = []
            self._midprice_df['timestamp'] = [math.ceil(ind/(2*latency[1]))*2*latency[1] 
                                              for ind in self._midprice_df.index]
            self._midprice_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
            self._midprice_df.set_index('timestamp', inplace=True, drop=True)

        #convert all columns in midprice_df to integers so numpy array lookup is fast
        for col in self._midprice_df.columns:
            self._midprice_df[col] = self._midprice_df[col].astype(int)

        self._strategy = strategy
        self._tstart = self._midprice_df.index.min()
        self._tend = self._midprice_df.index.max()

        #these two dictionaries will hold our orders
        self._bid_orders = {}
        self._ask_orders = {}
        
        #indices stored for faster lookup (not sure if this is any more efficient)
        self._mov_ind = self._midprice_df.columns.get_loc('movement')
        self._bp1_ind = self._midprice_df.columns.get_loc('bp1')
        self._ap1_ind = self._midprice_df.columns.get_loc('ap1')
        self._bq1_ind = self._midprice_df.columns.get_loc('bq1')
        self._aq1_ind = self._midprice_df.columns.get_loc('aq1')
        self._predict_ind = self._midprice_df.columns.get_loc('y_predict')
        self._mid_ind = self._midprice_df.columns.get_loc('midprice')
        self._y0_ind = self._midprice_df.columns.get_loc('y_0')
        self._index_series = self._midprice_df['index_position'].copy()

        #set exposure levels
        self._max_exposure = max_exposure_dict

        #initialize some signals, reference indices, and positions to 0
        self._long_position = 0.0
        self._short_position = 0.0

        #mkt move set to true when we first expirience a market move
        self._mkt_move = False

        #current bid-ask spread a tick wide
        self._mkt_spread = False

        #our main signal based off our RNN preictions
        self._midprice_prediction_signal = 0

        #reference indices to orderbook to factor in latency 
        self._entry_exit_ind = 0
        self._positions_as_of = 0

        #reference time
        self._reference_time = 0

        #activate and cancel bid_ask below track decisions made to level 1 orders
        self._activate_bidask = [False, False]
        self._cancel_bidask = [False, False]

        #is our strategy active? 1 for yes 0 for no
        self._active = 0

        #when set to true we close all our open positions in current step
        self._close = False
        
        
    def _set_entry_exit_ind(self, ind):
        '''
        time an order placed or cancelled now is actually placed or cancelled
        note that the actual current time is midprice_df.index[ind] + latency[0]   
        '''
        self._reference_time = self._midprice_df.index[ind] + self._latency[1] + self._latency[0]
        self._entry_exit_ind = self._index_series.values[ind]
        self._entry_exit_ind += 1
        while self._orderbook.get_current_time(self._entry_exit_ind) <= self._reference_time:
            self._entry_exit_ind += 1
        self._entry_exit_ind -= 1
        
    def _set_positions_known_time(self, ind):
        '''
        time positions are known when we reach next step. this is the next midprice time,
        plus our quote delay minus our exchange_to_trader delay 

        positions_as_of represents the index in the orderbook data that we can access at the next step regarding our trade status
        '''
        reference_time = self._midprice_df.index[ind + 1] - self._latency[1] + self._latency[0]
        self._positions_as_of = self._index_series.values[ind]
        self._positions_as_of += 1
        while self._orderbook.get_current_time(self._positions_as_of) <= reference_time:
            self._positions_as_of += 1
        self._positions_as_of -= 1
        
        
    def _arrange_keys(self):
        '''
        this simply update the levels of our orders and drops all closed (cancelled or executed) orders
        as we no longer need to track those
        '''
        new_bids = {}
        new_asks = {}
        for bid in range(1, self._orderbook.num_levels() + 1):
            keys = list(self._bid_orders.keys())
            for key in sorted(keys, reverse=True):
                if self._bid_orders[key].get_current_level() == bid and self._bid_orders[key].order_type("open"):
                    new_bids[bid] = self._bid_orders[key]
                                 
        for ask in range(1, self._orderbook.num_levels() + 1):
            keys = list(self._ask_orders.keys())
            for key in sorted(keys, reverse=True):
                if self._ask_orders[key].get_current_level() == ask and self._ask_orders[key].order_type("open"):
                    new_asks[ask] = self._ask_orders[key]
                    
        self._bid_orders = new_bids
        self._ask_orders = new_asks

    def _step(self, ind, y_pred, y_actual):
        '''
        We call this function at each timestep (minimum timestep is 2*latency[1])
        note that the actual current time at beginning of every step is midprice_df.index[ind] + latency[0]
        Each function called below controls a different part of the backtest
        '''
        #cash flow from given step
        cash_flow = 0.0
        
        #set order entry index (index reference to when we can place new orders)
        self._set_entry_exit_ind(ind)
        
        #update signals (all updtes are as of current time - latency[0])
        self._update_all_signals(ind, y_pred, y_actual)  

        #go through strategy logic and determine what to do
        self._activate_and_deactivate_strategy(ind, y_pred, y_actual)
        
        #now we step to exit_entry_ind which is latency[1] further in time, udating all open orders
        self._update_orders(self._entry_exit_ind)
        
        #exchange closes positions we decided to close above with latency[1] delay
        cash_flow += self._close_positions(ind, y_pred, y_actual)

        #for executed orders we get some cash flow
        cash_flow += self._execute_orders(y_pred, y_actual)

        #for orders and level 1 orders we decided to place/cancel above now reach exchange
        self._place_orders(ind, y_pred, y_actual)
        self._place_level_1_orders(ind, y_pred, y_actual)
        
        #set positions known time, this is midprice_df.index[ind + 1] + latency[0] - latency[1]
        #this represents what information regarding our cash flows and positions we will know by beginning of next step
        self._set_positions_known_time(ind)
        
        #update all open orders to positions known time 
        #(self._positions_as_of is an index to orderbook to represent desired time)
        self._update_orders(self._positions_as_of)

        #now execute all orders up until positions known time to reflect exchange notifications we have at beginning of next step
        cash_flow += self._execute_orders(y_pred, y_actual)
        
        #return total cash_flow for given step
        return cash_flow
    
    def _update_all_signals(self, ind, y_pred, y_actual):
        '''first function for each step in time, update signals based on market data up until latency[0] time before current time'''
        mkt_prev_bid_ask = (self._midprice_df.values[ind, self._bp1_ind],
                             self._midprice_df.values[ind, self._ap1_ind])
        spread = mkt_prev_bid_ask[1] - mkt_prev_bid_ask[0]
        
        if spread <= self._tick_size*1.01:
            self._mkt_spread = True
        else:
            self._mkt_spread = False
            self._close = True

        #when midprice movement occurs, we close out our position
        if self._midprice_df.values[ind, self._mov_ind] != 0:
            self._mkt_move = True
            self._active = 0
            self._close = True
            self._midprice_prediction_signal = 0
            
        #as our strategy explains we begin keeping track of a running sum of our RNN predictions
        #when bid-ask spread reverts to normal tick size   
        if self._mkt_spread:      
            self._midprice_prediction_signal += y_pred[0]
                    
            
    def _activate_and_deactivate_strategy(self, ind, y_pred, y_actual):
        '''most strategy logic regarding the decision to be made at the current time is done here'''

        #cancel level 1 open bid if our strategy not active and expect a drop in midprice
        if 1 in self._bid_orders and self._active == 0 and self._midprice_prediction_signal <= 0 and self._mkt_spread:
            self._cancel_bidask[0] = True
        
        #cancel level 1 open ask if strategy not active and expect a rise in midprice
        if 1 in self._ask_orders and self._active == 0 and self._midprice_prediction_signal >= 0 and self._mkt_spread:
            self._cancel_bidask[1] = True      
        
        #if we currently have a position open in market and spread is a tick wide and we are not about to close positions based on previous signals
        if self._long_position + self._short_position != 0 and self._mkt_spread and not self._close:

            #activate strategy if expect down movment and we are short
            if self._active == 0 and self._short_position > self._long_position and self._midprice_prediction_signal < 0:
                self._active = 1

                #place new level 1 bid if we do not have one open
                if 1 in self._bid_orders:
                    if not self._bid_orders[1].order_type("open"):
                        #we must wait until latency[1] time has passed to place this order but we keep track of it
                        self._activate_bidask[0] = True
                else:
                    self._activate_bidask[0] = True   

            #close our positions if we now predict price will move the wrong way                    
            elif self._short_position > self._long_position and self._midprice_prediction_signal >= 0:
                self._close = True
                self._active = 0
                
            #same as 2 previous conditional statements but regarding ask level 1
            elif self._active == 0 and self._short_position < self._long_position and self._midprice_prediction_signal > 0:
                self._active = 1
                if 1 in self._ask_orders:
                    if not self._ask_orders[1].order_type("open"):
                        self._activate_bidask[1] = True
                else:
                    self._activate_bidask[1] = True
                        
                        
            elif self._short_position < self._long_position and self._midprice_prediction_signal <= 0:
                self._close = True
                self._active = 0
                
        #here we immediately close all outstanding positions if the current market spread is greater than a tick      
        if self._long_position + self._short_position != 0 and not self._mkt_spread:
            self._close = True
                 
        
    def _update_orders(self, ind):
        #now we update our orders up to index reference in DataFrame
        for bid in self._bid_orders:
            #process new orderbook messages untul we reach the passed index value
            while(self._bid_orders[bid].get_current_index() < ind and self._bid_orders[bid].order_type("open")):
                self._bid_orders[bid].process_message()
        for ask in self._ask_orders:
            while(self._ask_orders[ask].get_current_index() < ind and self._ask_orders[ask].order_type("open")):
                self._ask_orders[ask].process_message()
            
        #now we attempt to cancel orders if requested and still possible
        #note if orders were executed already the "cancel_order() call will fail to cancel the order"
        if self._cancel_bidask[0]:
            self._bid_orders[1].cancel_order()
            self._cancel_bidask[0] = False
        if self._cancel_bidask[1]:
            self._ask_orders[1].cancel_order()
            self._cancel_bidask[1] = False
                 
    def _close_positions(self, ind, y_pred, y_actual):
        '''
        We close positions here, at market price (must buy at ask and sell at bid) if we go over our max exposure level
        or our strategy/signals inform us to close positions at beginning of time step

        We also reset some signals when we close
        '''
        cash_flow = 0.0
        mkt_bid_ask = (self._orderbook.get_book_state(self._entry_exit_ind)[2],
                       self._orderbook.get_book_state(self._entry_exit_ind)[0])
        
        #net long
        if self._long_position > self._short_position:
            num_to_close = 0
            if self._long_position > self._max_exposure[y_pred[1]] or self._close:
                num_to_close = (self._long_position - self._short_position)
                self._active = 0
                self._close = False
                self._midprice_prediction_signal = 0
            if num_to_close > 0:
                cash_flow += mkt_bid_ask[0]*num_to_close
                self._long_position -= num_to_close
                
        #net short 
        elif self._short_position > self._long_position:
            #close out if we hit our max market exposure level
            num_to_close = 0
            if self._short_position > self._max_exposure[y_pred[1]] or self._close:
                num_to_close = (self._short_position - self._long_position)
                self._active = 0
                self._close = False
                self._midprice_prediction_signal = 0
            if num_to_close > 0:
                cash_flow -= mkt_bid_ask[1]*num_to_close
                self._short_position -= num_to_close
                
        #close as many positions as possible
        while self._long_position >= 1 and self._short_position >=1:

            self._long_position -= 1
            self._short_position -= 1
            self._active = 0
            self._close = False
            self._midprice_prediction_signal = 0
            
            
        self._close = False
        return cash_flow

    def _execute_orders(self, y_pred, y_actual):
        cash_flow = 0.0
        #now we see which orders are filled and adjust our cash flows
        for bid in self._bid_orders.keys():
            if self._bid_orders[bid].order_type("executed"):
                price = self._bid_orders[bid].get_order_price()
                cash_flow -= price
                self._long_position += 1


        for ask in self._ask_orders.keys():
            if self._ask_orders[ask].order_type("executed"):

                price = self._ask_orders[ask].get_order_price()
                cash_flow += price
                self._short_position += 1

        #finally drop executed orders from our orders dictionaries and update key levels 
        self._arrange_keys()

        return cash_flow
    
    def _place_orders(self, ind, y_pred, y_actual):
        '''Here we place orders (excluding level 1) as requested'''
        
        active_strategy = self._strategy[y_pred[0]]
        
        bids_to_add = active_strategy[0]
        asks_to_add = active_strategy[1]
        
        #check if there exist a previously placed order (from last update) on levels we want to place orders!
        #as these orders will be ahead of new orders in queue
        if not y_pred[-1] is None:
            #reset order keys, dropping closed/executed orders
            self._arrange_keys()
            keep_bids = {}
            keep_asks = {}

            #find all levels we would like to have open orders (and we currently do not have an open order there)
            for bid in bids_to_add:
                keys = list(self._bid_orders.keys())
                for key in sorted(keys, reverse=True):
                    if self._bid_orders[key].order_type("open") and self._bid_orders[key].get_current_level() == bid:
                        keep_bids[bid] = self._bid_orders[key]
                        
            for ask in asks_to_add:
                keys = list(self._ask_orders.keys())
                for key in sorted(keys, reverse=True):
                    if self._ask_orders[key].order_type("open") and self._ask_orders[key].get_current_level() == ask:
                        keep_asks[ask] = self._ask_orders[key]
                      
            self._bid_orders = keep_bids
            self._ask_orders = keep_asks
                        
        #add new non-level 1 bid orders  
        for bid in bids_to_add:
            if bid in self._bid_orders or bid == 1:
                continue
            #new order placed at reference time which is the time a placed order at beginning of time step reaches the exchange
            #delay is latency[1]
            self._bid_orders[bid] = ou.Order(orderbook=self._orderbook, level=bid, timestamp=self._reference_time,
                                             index_ref=self._entry_exit_ind, is_buy=True)
        #same as above but for asks
        for ask in asks_to_add:
            if ask in self._ask_orders or ask == 1:
                continue
            
            self._ask_orders[ask] = ou.Order(orderbook=self._orderbook, level=ask, timestamp=self._reference_time,
                                             index_ref=self._entry_exit_ind, is_buy=False)
                 
    def _place_level_1_orders(self, ind, y_pred, y_actual):
        '''
        Here we place/put in a cancel request for our level one orders as dictated by our strategy
        '''
        cash_flow = 0

        #we do not do anything when spread > a tick
        if not self._mkt_spread:
            return

        #when we have a market move or we just activated our strategy we may place a level 1 order
        if self._mkt_move or self._activate_bidask[0] or self._activate_bidask[1]:
            bid_price = self._midprice_df.values[ind, self._bp1_ind]  
            ask_price = self._midprice_df.values[ind, self._ap1_ind]
            new_bid = True
            new_ask = True
            
            #reset order keys, dropping closed/executed orders
            self._arrange_keys()

            #place new orders at level 1 if the order price does not exist and we recently had a midprice move
            #or as dicatedby activate_bid_ask (from strategy decision)
            if 1 not in self._bid_orders and 1 in self._strategy[y_pred[0]][0]:
                if self._midprice_prediction_signal > 0 or self._activate_bidask[0]:
                    #new order placed at time reached exchange
                    self._bid_orders[1] = ou.Order(orderbook=self._orderbook, level=1, timestamp=self._reference_time,
                                                   index_ref=self._entry_exit_ind, is_buy=True)
                    
                    new_bid = True
                    self._activate_bidask[0] = False
                    
            #same as above butfor ask
            if 1 not in self._ask_orders and 1 in self._strategy[y_pred[0]][1]:
                if self._midprice_prediction_signal < 0 or self._activate_bidask[1]:
                    self._ask_orders[1] = ou.Order(orderbook=self._orderbook, level=1, timestamp=self._reference_time,
                                                   index_ref=self._entry_exit_ind, is_buy=False)
                    new_ask = True
                    self._activate_bidask[1] = False
                    
            if new_bid or new_ask:
                self._mkt_move = False

        #reset order keys, dropping closed/executed orders      
        self._arrange_keys()
    
    def run_strategy(self, tstart=None, tend=None):
        '''
        function to call outside of class to run the strategy from tstart to tend

        inputs
        --------
        tstart: Int. Time in seconds since midnight to start running our strategy.
        tend: Int. Time to end running strategy.
        '''
        if tstart is None:
            tstart = self._tstart
        if tend is None:
            tend = self._tend
            
        #get reference index in orderbook to start
        #pick last one if multiple indices returned for gven time
        start_ind = self._midprice_df.index.get_loc(tstart)
        if isinstance(start_ind, slice):
            start_ind = start_ind.stop - 1
            
        end_ind = self._midprice_df.index.get_loc(tend)
        if isinstance(end_ind, slice):
            end_ind = end_ind.stop - 1
        
        #y_actual represents the previous, current (just observed), and next midprice move
        y_actual = {}
        y_actual[-1] = None
        y_actual[0] = self._midprice_df.values[start_ind, self._y0_ind]

        #same as above but our predictions (so y_pred[0] is a prediction for next midprice move)
        y_pred = {}
        y_pred[-1] = None
        y_pred[0] = self._midprice_df.values[start_ind, self._predict_ind]
        self._close = False
        cum_pnl = 0
        pnl_ser =  [0.0]

        #Now we step through time
        for i in range(start_ind, end_ind - 1):
            if i == end_ind - 2:
                #when we get to end of loop we must close all open positions
                self._close = True
            #update y
            y_pred[1] = self._midprice_df.values[i + 1, self._predict_ind]
            y_actual[1] = self._midprice_df.values[i + 1, self._y0_ind]
            
            #Now we step 2*latency[1] in time and return the cash_flow for the given timeframe
            pnl = self._step(i, y_pred, y_actual)
            
            #add it to our cumulative cash_flow
            cum_pnl += pnl
            
            #current time after the step above
            time = self._midprice_df.index[i + 1] + self._latency[0]
            
            #here we adjust our cumulative pnl series to account for outstanding long/short positions active in market
            midprice = self._midprice_df.values[i + 1, self._mid_ind]
            adjust = self._long_position*midprice - self._short_position*midprice
            pnl_ser += [cum_pnl + adjust]
            
            #update y and y_pred
            y_pred[-1] = y_pred[0]
            y_actual[-1] = y_actual[0]
            y_pred[0] = y_pred[1]
            y_actual[0] = y_actual[1]
            
            #print where we are every 10000 steps
            if (i - start_ind) % 10000 == 0:
                print('Current time:            ' + str(time))
                print('Current cumulative pnl:  ' + str(cum_pnl + adjust))
           
        #now we load our list into a pandas pnl series     
        self._pnl_ser = pd.Series()
        self._pnl_ser.index.name = 'timestamp'
        self._pnl_ser = pd.Series(pnl_ser, index=self._midprice_df.index[start_ind:end_ind] + self._latency[0])   

        #change to pnl (from comulative pnl)          
        self._pnl_ser = self._pnl_ser - self._pnl_ser.shift(1)
        self._pnl_ser.fillna(0.0, inplace=True)
        self._orderbook.clear_memory()
            
    def get_cumulative_pnl_series(self):
        '''returns cumulative (sum) of pnl series'''
        return self._pnl_ser.cumsum()
    
    def get_pnl_series(self):
        '''returns pnl series'''
        return self._pnl_ser

class BacktesterSimulator():
    '''
    Given a strategy we build a simulator to simulate pnl perfomance for given midprice prediction accuracy
    '''
    def __init__(self, book, strategy, midprice_df,
                 set_edge_queue=False, tick_size=None, latency=(0, 0), 
                 max_exposure_dict={0:0, 1:0, -1:0}, accuracy_rate=0.5):
        '''
        inputs are the same as TradingStrategySimulator class with one additional argument
        accuracy_rate: Float between 0 and 1, the accuracy rate of predicting correct midprice movement

        **Note: These arbitrary prediction accuracies are spread out uniformly across data samples.
                In practice, the RNN predictions (for current midprice movement) are heavily correlated with the previous few predictions
                as data points close in time are very similar. In result, performance given below will be better than actual performance
                for real predictions with the same accuracy rate.
                This is why we decided to ammend our strategy to predict "Next" midprice move for a given data point and not the current move.
                (meaning we remove the 0 label data points before feeding data into the RNN,
                and then fill in missing data points with our latest prediction)
        '''

        #randomize our predictions
        simulator_help = {}

        #list of "alternative predictions" for a given key when we predict wrong
        for key in strategy:
            other_keys = [k for k in strategy if k != key]
            simulator_help[key] = other_keys
        
        #correct predictions
        self._midprice_movement = midprice_df['y_1'].copy()

        #We set up Uniform(0, N) random variables
        #Probability of each Uniform(0, N) <= 1000000 is approximately our accuracy rate
        N = round(1000000/accuracy_rate)

        #We set Uniform(0, M) random variables
        #When we predict wrong, we randomly choose 1 of M "incorrect" labels to assign to our prediction series
        M = len(strategy.keys()) - 1

        #generate Uniorm(0, N) for all data samples
        random_generator = [np.random.randint(0, N) for i in range(len(midprice_df))]
        bernoulli_generator = [np.random.randint(0, M) for i in range(len(midprice_df))]

        #if the Uniform(0, N) <= 1000000 we set the correct prediction, otherwise we randomly pick a different wrong label for our prediction
        self._rand_predictions = pd.Series([self._midprice_movement.values[i] if random_generator[i] <= 1000000 
                                            else simulator_help[self._midprice_movement.values[i]][bernoulli_generator[i]]
                                            for i in range(len(midprice_df))], index=midprice_df.index)
        
        #add our prediction as 'y_predict' 
        midprice_df['y_predict'] = self._rand_predictions
        
                                           
        #build backtester based on random predictions
        self._strategy_simulator = TradingStrategyBacktester(book=book, strategy=strategy,
                                                             midprice_df=midprice_df,
                                                             set_edge_queue=set_edge_queue,
                                                             max_exposure_dict=max_exposure_dict,
                                                             tick_size=tick_size, latency=latency)
        
    def run_strategy_simulation(self):
        '''run the strategy'''
        self._strategy_simulator.run_strategy()
        
    def get_pnl_series(self):
        '''get the pnl series'''
        return self._strategy_simulator.get_pnl_series()