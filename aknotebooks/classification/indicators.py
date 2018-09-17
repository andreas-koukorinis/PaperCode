import numpy as np
import pandas as pd


# Exponential moving average
def exp_mvavg(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()

    a = np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a


# Moving average convergence/divergence
def macd(data, fast=12, slow=26):
    macd = exp_mvavg(data, fast) - exp_mvavg(data, slow)
    signal = exp_mvavg(macd, 9)
    divergence = macd - signal

    return macd, signal, divergence


# Relative strength index
def rsi(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down;
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


#Chaikin Oscillator- As a money flow oscillator, CMF can be used in conjunction with pure price oscillators, such as MACD or RSI. As with all indicators, Chaikin Money Flow should not be used as a stand-alone indicator.
def ChaikinMoneyFlow(df):
    MF_Multiplier = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])
    MF_Volume= pd.Series(MF_Multiplier* df['Volume']/pd.rolling_sum(df['Volume'],period=10))
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')
    df = df.join(Chaikin)
    return df

#Money Flow Index and Ratio
def MoneyFlowIndexRatio(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MoneyFlowRatio = pd.Series(PosMF / TotMF)
    MoneyFlowIndexRatio = pd.Series(pd.rolling_mean(MoneyFlowRatio, n), name = 'MFI_' + str(n))
    df = df.join(MoneyFlowIndexRatio)
    return df

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df



def concatenate(*args):
    ndata = len(args)
    output = args[0]
    for i in range(1, ndata):
        output = np.vstack((output, args[i]))
    return output


def set_var(data):
    # output = np.asmatrix(data)
    return data / np.max(abs(data))

#
# # Return the feature set as a list of objects
# def feature_set(date, closep, highp, lowp, openp, volume):
#     ema7 = set_var(EMA(closep, 7))
#     ema50 = set_var(EMA(closep, 50))
#     ema200 = set_var(EMA(closep, 200))
#     rsi = set_var(RSI(closep, n=14))
#     adx = set_var(ADX(date, closep, highp, lowp, openp, volume))
#     macd, signal, der = MACD(closep, fast=12, slow=26)
#     macd = set_var(macd)
#     signal = set_var(signal)
#     der = set_var(der)
#     high = set_var(highp)
#     low = set_var(lowp)
#     close = set_var(closep)
#     vol = set_var(volume)
#
#     # Return as single items for easy analysis
#     inputs = []
#     inputs.append([ema7, "EMA7"])
#     inputs.append([ema50, "EMA50"])
#     inputs.append([ema200, "EMA200"])
#     inputs.append([rsi, "RSI"])
#     inputs.append([adx, "ADX"])
#     inputs.append([macd, "MACD"])
#     inputs.append([signal, "Signal"])
#     inputs.append([der, "Derivative"])
#     inputs.append([high, "High"])
#     inputs.append([low, "Low"])
#     inputs.append([close, "Close"])
#     inputs.append([vol, "Volume"])
#
#     outputs = close
#
#     return inputs, outputs