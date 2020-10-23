import datetime as dt
import itertools as it
import logging
import sqlite3
import time
from threading import Lock, Thread
from typing import List
import uuid

import alpaca_trade_api as alpaca
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from wshandler import WSHandler

logging.basicConfig(
    filename=f"logs/{dt.datetime.now().isoformat(timespec='seconds')}.log",
    format='%(asctime)s %(filename)-16s:%(lineno)-3s %(levelname)-7s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class Cointegration:
    """
    This cointegration trading strategy attemps to find stationary
    relationships in the spreads between comparable equities. A 
    spread is stationary if the historical mean oscillates about 0. 
    
    When we identify stationarity, we then test to see if the current 
    spread value is 'unlikely' (determined by a z-score). Should the 
    algorithm deem that the current prices are an anomaly, it will
    open opposing positions to try to capialize on future price convergence.
    """

    def __init__(self):
        # connection parameters
        self.key_id = 'PKM19ZH5QLFUUQZOXFCY'
        self.secret_key = 'hcRBPwpUlQl40BSIJuvMJ7v9y0IEt8btVHUEIJrE'
        self.base_url = 'https://paper-api.alpaca.markets'
        self.data_url = 'https://data.alpaca.markets'

        # db parameters
        self.db = 'fills.db'
        self.table = 'cointegration'

        # user-defined strategy parameters
        self.timeframe = '15Min'
        self.periods = 64
        self.sigmas = 2
        self.usd_trade_clip = 1000
        self.usd_stop_loss = 100
        self.usd_take_profit = 100

        self._check_strategy_params()

        # thread safety
        self.lock = Lock()

        # init REST connection
        self.rest = alpaca.REST(
            key_id=self.key_id,
            secret_key=self.secret_key,
            base_url=self.base_url
        )

        # maps pairs of submitted order_ids to each other
        self._submitted = {}
        # mark order_ids that are part of closing a position
        self._closing = set()

    def _check_strategy_params(self) -> None:
        """
        Run some safety checks on the user-defined parameters
        to make sure they are valid inputs
        """
        params = [
            self.sigmas, self.usd_trade_clip, self.usd_stop_loss, 
            self.usd_take_profit
        ]
        for p in params:
            assert p > 0
            assert isinstance(p, int) or isinstance(p, float)

    def _place_pair_trade(self, _long: str, _short: str, _latest: pd.Series) -> None:
        """
        Wrapper function for placing a pair of opposing orders. Contains
        logic for handling sides, quantities, and order sequence.
        """
        long_price = _latest.at[_long]
        short_price = _latest.at[_short]

        long_qty = round(self.usd_trade_clip / long_price)
        short_qty = round(self.usd_trade_clip / short_price)

        # quit early if we want to trade a 0 lot
        if long_qty == 0 or short_qty == 0:
            msg = (
                f'Cannot submit zero quantity: '
                f'BUY {long_qty} {_long} @ ~${long_price} '
                f'SELL {short_qty} {_short} @ ~${short_price}'
            )
            logger.warning(msg)
            return

        # generate unique order ids
        sell_id = str(uuid.uuid4())
        buy_id = str(uuid.uuid4())

        # always try to place short order first to catch HTB problems
        try:
            logger.info(f'Submitting market sell {_short} on {short_qty} lots')
            self.rest.submit_order(
                symbol=_short,
                qty=short_qty,
                side='sell',
                type='market',
                time_in_force='day',
                client_order_id=sell_id
            )

            logger.info(f'Submitting market buy {_long} on {long_qty} lots')
            self.rest.submit_order(
                symbol=_long,
                qty=long_qty,
                side='buy',
                type='market',
                time_in_force='day',
                client_order_id=buy_id
            )
        except alpaca.rest.APIError as e:
            logger.warning(e)
            return False

        # record submitted orders as pairs
        self.lock.acquire()
        self._submitted[sell_id] = {
            'entry_price': short_price,
            'pair_id': buy_id
        }
        self._submitted[buy_id] = {
            'entry_price': long_price,
            'pair_id': sell_id
        }
        self.lock.release()

    def _verify_database(self) -> None:
        """
        Verify that our local database postions match with the Alpaca
        dashboard positions
        """
        logger.info('Verifying database')

        # map symbol to Alpaca position
        symbol_to_qty_alpaca = {}
        open_pos = {p.symbol: p for p in self.rest.list_positions()}
        for symbol, pos in open_pos.items():
            symbol_to_qty_alpaca[symbol] = int(pos.qty)

        symbol_to_qty_db = {}
        cursor = self.conn.cursor()
        query = f'SELECT symbol, SUM(qty) FROM {self.table} GROUP BY symbol'
        for symbol, qty in cursor.execute(query):
            symbol_to_qty_db[symbol] = qty

        dicts = {
            'alpaca': symbol_to_qty_alpaca,
            'db': symbol_to_qty_db
        }
        for canonical, comparison in [('alpaca', 'db'), ('db', 'alpaca')]:
            for s, q in dicts[canonical].items():
                if s not in dicts[comparison] or q != dicts[comparison][s]:
                    msg = (
                        f'Position mismatch! {canonical} contains {q} {s} '
                        f'but {comparison} does not'
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

    def start(self) -> None:
        """
        Harness of the entire trading program. Handles KeyBoardInterrupt
        as well as spins off threads to concurrently receive data and make 
        trading decisions.
        """
        # load S&P stock universe and make industry pairs
        universe = pd.read_csv('universe.csv')
        f = lambda x: pd.DataFrame(
            list(it.combinations(x.values, 2)),
            columns=['Primary','Secondary']
        )
        self.pairs = (universe.groupby('GICS Sub Industry')['Symbol']
            .apply(f)
            .reset_index(level=1, drop=True)
            .reset_index()
        )
        logger.info(f'Generated {len(self.pairs)} pairs')

        # initialize db connection here so we can wrap with finally
        try:
            self.conn = sqlite3.connect(self.db)

            # start websocket handler in seperate thread
            handler = WSHandler(
                self.key_id, self.secret_key, self.base_url,
                self.data_url, self.db, self.table, self.lock,
                self._submitted, self._closing
            )
            handler.run()

            while True:
                # make sure our database matches Alpaca position before trading
                self._verify_database()
                logger.info('Database check passed!')

                self._check_for_exit_signals()

                # sleep so any pending exit trades have time to be excecuted and recorded
                time.sleep(2)
                logger.info(f'Current position checked, looking for new trades...')

                self._trade()
        except RuntimeError:
            print('RuntimeError encountered. Check logs for details.')
        except KeyboardInterrupt:
            print('\nExiting on KeyboardInterrupt...')
        finally:
            self.rest.cancel_all_orders()
            self.conn.close()

    def _check_for_exit_signals(self) -> None:
        """
        Look through all of the open trades and determine if we've hit our
        stop-limit or take-profit for any of them
        """
        # prepare data by querying database
        fills = {}
        quotes = {}
        cursor = self.conn.cursor()
        for row in cursor.execute(f'SELECT * FROM {self.table}'):
            order_id, symb, qty, price, pair_id, trade_date, _ = row
            fills[order_id] = {
                'order_id': order_id,
                'symbol': symb,
                'qty': qty,
                'price': price,
                'pair_id': pair_id,
                'trade_date': dt.datetime.fromisoformat(trade_date[:26])
            }

            quotes[symb] = self.rest.get_last_quote(symb)

        # used a couple of times later on
        today = dt.date.today()

        # define before we enter loop
        def _calc_pnl(quantity: int, price: float, bid: float, ask: float) -> float:
            """
            Calculate the pnl for a trade given quantity (pos or neg),
            market bid/ask, and entry_price
            """
            exit_price = bid if quantity > 0 else ask
            return (exit_price - price) * quantity

        # dict of [order_id, order] to exit
        liquidate = {}
        for order_id, fill in fills.items():
            # if a open position doesn't have a recorded pair, we will liquidate
            if fill['pair_id'] not in fills:
                msg = (
                    f"Found unpaired position. {fill['symbol']} with "
                    f"id={order_id} and trade_date={fill['trade_date']} "
                    "missing pair"
                )
                logger.info(msg)

                if fill['trade_date'].date() < today:
                    liquidate[order_id] = fill

            # now check pairs for pnl stops (initiate with longs only)
            elif fill['qty'] > 0:
                quote = quotes[fill['symbol']]
                bid, ask = quote.bidprice, quote.askprice
                pnl = _calc_pnl(fill['qty'], fill['price'], bid, ask)

                other_id = fill['pair_id']
                other = fills[other_id]
                other_quote = quotes[other['symbol']]
                other_bid, other_ask = other_quote.bidprice, other_quote.askprice
                other_pnl = _calc_pnl(other['qty'], other['price'], other_bid, other_ask)

                total_pnl = pnl + other_pnl
                msg = (
                    f"PnL for {fill['symbol']}/{other['symbol']} "
                    f"pair is ${total_pnl:.4f}. Markets are ${ask-bid:.2f} and "
                    f"${other_ask-other_bid:.2f} wide, respectively"
                )
                logger.info(msg)

                td = fill['trade_date'].date()
                otd = other['trade_date'].date()
                previously_opened = td < today and otd < today

                sl = total_pnl < -self.usd_stop_loss
                tp = total_pnl > self.usd_take_profit
                hit_pnl_stop = sl or tp

                if previously_opened and hit_pnl_stop:
                    liquidate[order_id] = fill
                    liquidate[other_id] = other

        # now that we've marked which positions to liquidate, we do it
        for order_id, fill in liquidate.items(): 
            logger.info(f"Liquidating {fill['qty']} {fill['symbol']}")
            self.lock.acquire()
            closing_order = self.rest.submit_order(
                symbol=fill['symbol'],
                qty=abs(fill['qty']),
                side='sell' if fill['qty'] > 0 else 'buy',
                type='market',
                time_in_force='day'
            )
            self._closing.add(closing_order.client_order_id)
            self.lock.release()

    def _trade(self) -> None:
        """
        The core trading logic depends on finding a statistically stationary 
        spread between two equities. Once a pair has been identified, we open
        a market neutral (long/short) position and hold for at least one day.
        """

        def is_stationary(X: pd.Series, cutoff: float=0.05) -> bool:
            """
            Use Augmented Dickie-Fuller test to determine if the time
            series is likely stationary.
            """
            pvalue = adfuller(X)[1]
            return pvalue < cutoff

        def _is_new_trade(_long: str, _short: str) -> bool:
            """
            Check if the long/short position we are trying to open
            is one we already have on
            """
            for _id, old in _pos.items():
                if old['symbol'] == _long and old['qty'] > 0:
                    if old['pair_id'] in _pos:
                        other = _pos[old['pair_id']]
                        if other['symbol'] == _short and other['qty'] < 0:
                            msg = (
                                f'Already own +{_long} -{_short} position'
                            )
                            logger.info(msg)
                            return False
            return True

        cursor = self.conn.cursor()
        _pos = {
            order_id: {
                'order_id': order_id,
                'symbol': symb,
                'qty': qty,
                'price': price,
                'pair_id': pair_id,
                'trade_date': dt.datetime.fromisoformat(trade_date[:26])
            } for order_id, symb, qty, price, pair_id, trade_date, _ in 
            cursor.execute(f'SELECT * FROM {self.table}')
        }

        # then we look to open new positions
        for idx, industry, a, b in self.pairs.itertuples():
            data = self.rest.get_barset(
                symbols=[a, b],
                timeframe=self.timeframe,
                limit=self.periods,
                until=dt.datetime.today().isoformat()
            ).df.dropna()

            # exit program if we don't have enough cash, to avoid margin fees
            account = self.rest.get_account()
            bp = float(account.regt_buying_power)
            if bp < 1.5 * 2 * self.usd_trade_clip:
                msg = (
                    f'Insufficient buying power to open new trade: '
                    f'bp={bp} and required={1.5 * 2 * self.usd_trade_clip}'
                )
                logger.warning(msg)
                return

            # Select close data to compare
            closes = data.xs('close', axis='columns', level=1)
            hedge_ratio = (closes[a] / closes[b]).mean()
            tseries = closes[a] - hedge_ratio * closes[b]
            
            if is_stationary(tseries):
                z = ((tseries - tseries.mean()) / tseries.std()).iat[-1]

                msg = (
                    f'Pair #{idx} >> stationarity in {a} - '
                    f'{hedge_ratio:.4f} * {b} in {industry}. '
                    f'z={z:.4f} vs. cutoff of {self.sigmas}'
                )
                logger.info(msg)

                # make sure we're not doubling down
                if z > self.sigmas and _is_new_trade(b, a):
                    #spread is over-priced so short front leg
                    self._place_pair_trade(
                        _long=b, 
                        _short=a, 
                        _latest=closes.iloc[-1]
                    )
                
                # make sure we're not doubling down
                elif z < -self.sigmas and _is_new_trade(a, b):
                    # spread is under-priced so long front leg
                    self._place_pair_trade(
                        _long=a, 
                        _short=b, 
                        _latest=closes.iloc[-1]
                    )

            # throttle requests so we don't bump up against API limit
            time.sleep(0.5)


if __name__ == "__main__":
    trader = Cointegration()
    trader.start()
