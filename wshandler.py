import asyncio
import datetime as dt
import logging
import sqlite3
from threading import Lock, Thread
from typing import Dict, Set
import time

import alpaca_trade_api as alpaca

logger = logging.getLogger(__name__)


# maybe make this sublass StreamConn one day? Not sure
# how the decorator would work though
class WSHandler:

    def __init__(self, key_id: str, secret_key: str, base_url: str,
        data_url: str, db: str, table: str, lock: Lock, 
        submitted: Dict[str, str], closing: Set[str]
    ):

        self.conn = alpaca.StreamConn(
            key_id=key_id,
            secret_key=secret_key,
            base_url=base_url,
            data_url=data_url,
            data_stream='alpacadatav1'
        )
        self.db = db
        self.table = table
        self.lock = lock
        self._submitted = submitted
        self._closing = closing
        self.thread = None
        logger.info('WSHandler initialized')

    
    def run(self) -> None:
        """
        Starts the websocket subscription to run in a seperate thread
        """

        @self.conn.on('^trade_updates$')
        async def on_trade_update(conn, channel, update):
            sql = sqlite3.connect(self.db)

            _id = update.order['client_order_id']
            symbol = update.order['symbol']
            msg = (
                f"Received event '{update.event}' for "
                f"order_id={_id}, symbol={symbol}"
            )
            logger.info(msg)

            self.lock.acquire()
            new_order_details = self._submitted.get(False)
            is_closing_order = _id in self._closing
            self.lock.release()

            if new_order_details:
                if update.event == 'fill':
                    query = (
                        f'INSERT INTO {self.table} (order_id, symbol, qty,'
                        ' price, pair_id, trade_date, entry_price) VALUES '
                        '(?, ?, ?, ?, ?, ?, ?)'
                    )
                    qty = int(update.order['qty'])
                    if update.order['side'] == 'sell':
                        qty *= -1
                    vals = (
                        _id, symbol, qty, float(update.price), 
                        new_order_details[_id]['pair_id'], update.timestamp,
                        new_order_details[_id]['entry_price']
                    )
                    cursor = sql.cursor()
                    cursor.execute(query, vals)
                    sql.commit()
                    logger.info(f'Inserted order_id={_id} into {self.db}.{self.table}')

            elif is_closing_order:
                if update.event == 'fill':
                    query = f'DELETE FROM {self.table} WHERE order_id="?"'
                    cursor = sql.cursor()
                    cursor.execute(query, (_id,))
                    sql.commit()
                    logger.info(f'Executed {query}')
            sql.close()

        thread = Thread(target=self.conn.run, args=(['trade_updates'],))
        thread.daemon = True
        thread.start()
