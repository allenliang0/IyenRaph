import queue
import pandas as pd
import sentiment
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import time, datetime
from ibapi.client import Contract, Order

from lightweight_charts import Chart

from threading import Thread


INITIAL_SYMBOL = "TSLA"
INITIAL_TIMEFRAME = '5 mins'

default_host = '127.0.0.1'
default_client_id = 1
paper_trading_port = 7497
live_trading_port = 7496
live_trading = False
trading_port = paper_trading_port
if live_trading:
    trading_port = live_trading_port


data_queue = queue.Queue()

class IBClient(EWrapper, EClient):

    def __init__(self, host, port, client_id):
        EClient.__init__(self, self)

        self.connect(host, port, client_id)
        thread = Thread(target=self.run)
        thread.start()

    def error(self, req_id, code, msg):
        if code in [2104, 2106, 2158]:
            print(msg)
        else:
            print('Error{}:{}'.format(code, msg))

    def nextValidId(self, orderId:int):
        super().nextValidId(orderId)
        self.order_id = orderId
        print((f'next avalible id is {self.order_id}'))

    def orderStatus(self, order_id , status:str, filled:float,
                    remaining:float, avgFillPrice:float, permId:int,
                    parentId:int, lastFillPrice:float, clientId:int,
                    whyHeld:str, mktCapPrice: float):
        print(f'order status: {order_id} {status} {filled} {remaining} {avgFillPrice} {permId} {parentId} {lastFillPrice} '
              f'{clientId} ')

    def historicalData(self, req_id, bar):
        print(bar)


        t = datetime.datetime.fromtimestamp(int(bar.date))

        # creation bar dictionary for each bar received
        data = {
            'date': t,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': int(bar.volume)
        }

        print(data)

        # Put the data into the queue
        data_queue.put(data)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print(f"end of data {start} {end}")

        update_chart()

def update_chart():
    try:
        bars = []
        while True:  # Keep checking the queue for new data
            data = data_queue.get_nowait()
            bars.append(data)
    except queue.Empty:
        print("empty queue")
    finally:
        # once we have received all the data, convert to pandas dataframe
        df = pd.DataFrame(bars)
        print(df)

        # set the data on the chart
        if not df.empty:
            chart.set(df)

            # once we get the data back, we don't need a spinner anymore
            #chart.spinner(False)

def on_timeframe_selection(chart):
    print('selcted timeframe')
    print(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)

def on_search(chart, search_string):
    get_bar_data(search_string, chart.topbar['timeframe'].value)
    chart.topbar['symbol'].set(search_string)

def take_screenshot(key):
    img = chart.screenshot()
    t = time.time()
    with open(f"scrrenshot-{t}.png", 'wb') as f:
        f.write(img)

def place_order(key):

    symbol = chart.topbar['symbol'].value.strip().upper()

    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.currency = 'USD'
    contract.exchange = 'SMART'

    order = Order()
    order.orderType = "MKT"
    order.eTradeOnly = False
    order.firmQuoteOnly = False  # Avoids similar issues
    order.outsideRth = True
    order.totalQuantity = 1

    client.reqIds(-1)
    time.sleep(1)

    if key == 'o':
        print('buy order')
        order.action = 'BUY'

    if key == 'p':
        print('sell order')
        order.action = 'SELL'

    if client.order_id:
        print("got order id, placing order")
        client.placeOrder(client.order_id, contract, order)


def get_bar_data(symbol, timeframe):

    print(f"getting bar data for {symbol} {timeframe}")

    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    what_to_show = 'TRADES'

    client.reqHistoricalData(
        2, contract, '', '10 D', timeframe, what_to_show, True, 2, False, []
    )

    time.sleep(1)

    chart.watermark(symbol)

if __name__ == '__main__':
    client = IBClient(default_host, trading_port, default_client_id)
    time.sleep(1)

    chart = Chart(toolbox=True, width=1000, inner_width=1, inner_height=1)
    chart.legend(True)


    chart.hotkey('shift','o', place_order)
    chart.hotkey('shift', 'p', place_order)


    chart.topbar.textbox('symbol', INITIAL_SYMBOL)
    chart.topbar.switcher('timeframe',('5 mins', '15 mins', '30 mins', '1 hour'),default=INITIAL_TIMEFRAME, func=on_timeframe_selection)

    chart.events.search += on_search

    get_bar_data(INITIAL_SYMBOL, INITIAL_TIMEFRAME)

    chart.show(block=True)
    time.sleep(1)

print('hi')