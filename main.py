import queue
import pandas as pd

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import time, datetime
from ibapi.client import Contract

from lightweight_charts import Chart

from threading import Thread

init_symbol = "TSLA"
INITAL_SYMBOL = "TSLA"

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
        EClient.__init__(self,self)

        self.connect(host, port, client_id)
        thread = Thread(target=self.run)
        thread.start()

    def error(self, req_id, code, msg):
        if code in [2104, 2106, 2158]:
            print(msg)
        else:
            print('Error{}:{}'.format(code, msg))

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

    get_bar_data(INITAL_SYMBOL, '5 mins')

    chart.show(block=True)
    time.sleep(1)