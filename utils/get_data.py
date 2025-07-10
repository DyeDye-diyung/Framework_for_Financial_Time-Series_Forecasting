import akshare as ak
import pandas as pd

apple_stock_df = ak.stock_us_hist(symbol='105.AAPL', adjust='')
apple_stock_df.rename(columns={
    '日期': 'Date',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume',
    '成交额': 'Turnover',
    '振幅': 'Volatility',
    '涨跌幅': 'PercentageChange',
    '涨跌额': 'Change',
    '换手率': 'TurnoverRatio',
}, inplace=True)
apple_stock_df.drop(columns=['Turnover'], inplace=True)
apple_stock_df = apple_stock_df.set_index(['Date'])
apple_stock_df.index = pd.to_datetime(apple_stock_df.index)
apple_stock_df.to_csv('Apple.csv', index=True)
print(apple_stock_df.shape)

microsoft_stock_df = ak.stock_us_hist(symbol='105.MSFT', adjust='')
microsoft_stock_df.rename(columns={
    '日期': 'Date',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume',
    '成交额': 'Turnover',
    '振幅': 'Volatility',
    '涨跌幅': 'PercentageChange',
    '涨跌额': 'Change',
    '换手率': 'TurnoverRatio',
}, inplace=True)
microsoft_stock_df.drop(columns=['Turnover'], inplace=True)
microsoft_stock_df = microsoft_stock_df.set_index(['Date'])
microsoft_stock_df.index = pd.to_datetime(microsoft_stock_df.index)
microsoft_stock_df.to_csv('Microsoft.csv', index=True)
print(microsoft_stock_df.shape)

maotai_stock_df = ak.stock_zh_a_hist(symbol='600519', adjust='')
maotai_stock_df.rename(columns={
    '日期': 'Date',
    '股票代码': 'StockCode',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume',
    '成交额': 'Turnover',
    '振幅': 'Volatility',
    '涨跌幅': 'PercentageChange',
    '涨跌额': 'Change',
    '换手率': 'TurnoverRatio',
}, inplace=True)
maotai_stock_df.drop(columns=['Turnover'], inplace=True)
maotai_stock_df.drop(columns=['StockCode'], inplace=True)
maotai_stock_df = maotai_stock_df.set_index(['Date'])
maotai_stock_df.index = pd.to_datetime(maotai_stock_df.index)
maotai_stock_df.to_csv('MaoTai.csv', index=True)
print(maotai_stock_df.shape)

hsbc_stock_df = ak.stock_hk_hist(symbol='00005', adjust='')
hsbc_stock_df.rename(columns={
    '日期': 'Date',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume',
    '成交额': 'Turnover',
    '振幅': 'Volatility',
    '涨跌幅': 'PercentageChange',
    '涨跌额': 'Change',
    '换手率': 'TurnoverRatio',
}, inplace=True)
hsbc_stock_df.drop(columns=['Turnover'], inplace=True)
hsbc_stock_df = hsbc_stock_df.set_index(['Date'])
hsbc_stock_df.index = pd.to_datetime(hsbc_stock_df.index)
hsbc_stock_df.to_csv('HSBC.csv', index=True)
print(hsbc_stock_df.shape)
