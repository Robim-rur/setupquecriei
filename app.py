import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Scanner – Swing Trade Tendência de Alta (Manual Definitivo)")

# ============================================================
# LISTA FIXA DE ATIVOS (AJUSTADA)
# - HASH11 removido
# - FIIs removidos
# ============================================================

ativos_scan = sorted(set([
"RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
"BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
"COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
"ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
"HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
"MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
"RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
"UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA","AURE3.SA","BHIA3.SA","CASH3.SA",
"CVCB3.SA","DIRR3.SA","ENAT3.SA","GMAT3.SA","IFCM3.SA","INTB3.SA","JHSF3.SA","KEPL3.SA","MOVI3.SA","ORVR3.SA",
"PETZ3.SA","PLAS3.SA","POMO4.SA","POSI3.SA","RANI3.SA","RAPT4.SA","STBP3.SA","TEND3.SA","TUPY3.SA",
"BRSR6.SA","CXSE3.SA",

# BDRs
"AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA","NVDC34.SA","MELI34.SA",
"BABA34.SA","DISB34.SA","PYPL34.SA","JNJB34.SA","PGCO34.SA","KOCH34.SA","VISA34.SA","WMTB34.SA","NIKE34.SA",
"ADBE34.SA","AVGO34.SA","CSCO34.SA","COST34.SA","CVSH34.SA","GECO34.SA","GSGI34.SA","HDCO34.SA","INTC34.SA",
"JPMC34.SA","MAEL34.SA","MCDP34.SA","MDLZ34.SA","MRCK34.SA","ORCL34.SA","PEP334.SA","PFIZ34.SA","PMIC34.SA",
"QCOM34.SA","SBUX34.SA","TGTB34.SA","TMOS34.SA","TXN34.SA","UNHH34.SA","UPSB34.SA","VZUA34.SA","ABTT34.SA",
"AMGN34.SA","AXPB34.SA","BAOO34.SA","CATP34.SA","HONB34.SA",

# ETFs
"BOVA11.SA","IVVB11.SA","SMAL11.SA","GOLD11.SA","DIVO11.SA","NDIV11.SA","SPUB11.SA"
]))

# ============================================================
# INDICADORES
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def stochastic(df, k=14, d=3, smooth=3):
    low_min = df['Low'].rolling(k).min()
    high_max = df['High'].rolling(k).max()
    k_raw = 100 * (df['Close'] - low_min) / (high_max - low_min)
    k_s = k_raw.rolling(smooth).mean()
    d_s = k_s.rolling(d).mean()
    return k_s, d_s

def dmi_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (low.diff() < 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return plus_di, minus_di, adx

# ============================================================
# REGRAS DO MANUAL
# ============================================================

def candle_alinhado(df):
    c = df.iloc[-1]
    rng = c.High - c.Low
    if rng == 0:
        return False

    corpo = abs(c.Close - c.Open)
    sombra_inf = min(c.Open, c.Close) - c.Low
    fechamento_pos = (c.Close - c.Low) / rng

    if c.Close <= df.iloc[-2].Close:
        return False
    if fechamento_pos < 0.8:
        return False
    if corpo <= (rng - corpo):
        return False
    if sombra_inf > corpo * 0.5:
        return False
    if 0.4 <= fechamento_pos <= 0.6:
        return False

    return True

def pullback_curto(df):
    ultimos = df.iloc[-4:-1]
    return (ultimos['Close'] < ultimos['Open']).sum() <= 3

def nao_perdeu_ema9(df):
    ema9 = ema(df['Close'], 9)
    return (df.iloc[-4:-1]['Low'] > ema9.iloc[-4:-1]).all()

def sem_esticamento(df):
    ema69 = ema(df['Close'], 69)
    dist = abs(df.iloc[-1].Close - ema69.iloc[-1]) / ema69.iloc[-1]
    return dist <= 0.08

def fora_resistencia(df):
    max20 = df['High'].iloc[-21:-1].max()
    return df.iloc[-1].Close < max20 * 0.99

# ============================================================
# PROCESSAMENTO
# ============================================================

resultados = []

progress = st.progress(0.0)

for i, ticker in enumerate(ativos_scan):

    try:
        dfd = yf.download(ticker, period="300d", interval="1d", progress=False)
        dfw = yf.download(ticker, period="3y", interval="1wk", progress=False)

        if len(dfd) < 80 or len(dfw) < 30:
            continue

        # =======================
        # SEMANAL (SEMANA FECHADA)
        # =======================
        dfw = dfw.dropna()
        dfw = dfw.iloc[:-1]

        dfw['EMA69'] = ema(dfw['Close'], 69)
        wk, wd = stochastic(dfw)
        w_plus, w_minus, w_adx = dmi_adx(dfw)

        if not (dfw['Close'].iloc[-1] > dfw['EMA69'].iloc[-1]):
            continue

        if not (dfw['EMA69'].iloc[-1] > dfw['EMA69'].iloc[-2]):
            continue

        if wk.iloc[-1] < wk.iloc[-2]:
            continue

        if w_plus.iloc[-1] <= w_minus.iloc[-1]:
            continue

        # =======================
        # DIÁRIO
        # =======================
        dfd = dfd.dropna()

        dfd['EMA69'] = ema(dfd['Close'], 69)

        dk, dd = stochastic(dfd)
        d_plus, d_minus, d_adx = dmi_adx(dfd)

        if dk.iloc[-1] <= dd.iloc[-1]:
            continue

        if d_plus.iloc[-1] <= d_minus.iloc[-1]:
            continue

        if not candle_alinhado(dfd):
            continue

        if not pullback_curto(dfd):
            continue

        if not nao_perdeu_ema9(dfd):
            continue

        if not sem_esticamento(dfd):
            continue

        if not fora_resistencia(dfd):
            continue

        resultados.append({
            "Ativo": ticker.replace(".SA",""),
            "Fechamento": round(dfd.iloc[-1].Close,2),
            "ADX diário": round(d_adx.iloc[-1],2),
            "D+ diário": round(d_plus.iloc[-1],2),
            "D- diário": round(d_minus.iloc[-1],2),
            "Estoc %K": round(dk.iloc[-1],2),
            "Estoc %D": round(dd.iloc[-1],2)
        })

    except:
        pass

    progress.progress((i+1)/len(ativos_scan))

st.subheader("Ativos alinhados ao manual – sinal no diário com confirmação no semanal")

if len(resultados) == 0:
    st.warning("Nenhum ativo passou em todas as regras do manual hoje.")
else:
    df_res = pd.DataFrame(resultados).sort_values("Ativo")
    st.dataframe(df_res, use_container_width=True)
