from __future__ import annotations
__removed_setor_focus = None
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go           
import MetaTrader5 as mt5
from typing import Dict, List, Optional
from datetime import datetime, time, timedelta, date
import statsmodels.api as sm                 
import yfinance as yf
import pytz
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.stattools import coint, adfuller
import itertools
from plotly.subplots import make_subplots
from typing import Dict
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence



st.set_page_config(page_title="Rios Setorial", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .block-container {padding-left: 1rem !important; padding-right: 1rem !important; max-width: 1800px;}
    </style>
    """,
    unsafe_allow_html=True,
)



try:
    from zoneinfo import ZoneInfo
    BRA_TZ = ZoneInfo("America/Sao_Paulo")
except Exception:
    BRA_TZ = None  # fallback se não houver zoneinfo

# --- Inicializa MT5 (se ainda não estiver)
try:
    if not mt5.initialize():
        st.warning("MT5 não inicializado. Abra o MetaTrader 5 e recarregue o app.")
except Exception:
    pass


# === Login do TradingView ===
TV_USERNAME_CONFIG = "riosasset"
TV_PASSWORD_CONFIG = "atgdm12newom9365%*"

try:
    from tvdatafeed import Interval as TvInterval  # type: ignore[import]
    from tvdatafeed import TvDatafeed              # type: ignore[import]
except Exception:
    try:
        from tvDatafeed import Interval as TvInterval  # type: ignore[import]
        from tvDatafeed import TvDatafeed              # type: ignore[import]
    except Exception:  # depende do ambiente
        TvInterval = None
        TvDatafeed = None  # type: ignore[assignment]


class Interval(Enum):
    """Fallback caso o tvdatafeed não esteja disponível."""
    in_daily = "1d"
    in_weekly = "1w"


if TvInterval is not None:  # se existir, usa o oficial
    Interval = TvInterval  # type: ignore[misc]


_tv: TvDatafeed | None = None  # singleton
_TV_CREDENTIALS: dict[str, str | None] = {"user": None, "password": None}

def _empty_series(name: str | None = None) -> pd.Series:
    s = pd.Series(dtype=float)
    if name:
        s.name = name
    return s


def configure_tv_credentials(user: str | None, password: str | None) -> None:
    """Configura (e invalida se preciso) o singleton do tvdatafeed."""
    global _tv, _TV_CREDENTIALS
    user = (user or "").strip()
    password = (password or "").strip()
    if not user or not password:
        raise ValueError("Informe usuario e senha do TradingView para continuar.")
    if (
        _TV_CREDENTIALS.get("user") != user
        or _TV_CREDENTIALS.get("password") != password
    ):
        _tv = None
    _TV_CREDENTIALS = {"user": user, "password": password}


# tenta carregar as credenciais hardcoded logo no import; se não forem ajustadas, ignora
try:
    configure_tv_credentials(TV_USERNAME_CONFIG, TV_PASSWORD_CONFIG)
except ValueError:
    pass

def _resolve_tv_credentials(
    user: str | None = None, password: str | None = None
) -> tuple[str, str]:
    env_user = os.getenv("TV_USERNAME") or os.getenv("TRADINGVIEW_USERNAME")
    env_pass = os.getenv("TV_PASSWORD") or os.getenv("TRADINGVIEW_PASSWORD")
    hardcoded_user = TV_USERNAME_CONFIG.strip()
    hardcoded_pass = TV_PASSWORD_CONFIG.strip()
    if "COLOQUE" in hardcoded_user.upper():
        hardcoded_user = ""
    if "COLOQUE" in hardcoded_pass.upper():
        hardcoded_pass = ""

    resolved_user = (
        user
        or _TV_CREDENTIALS.get("user")
        or env_user
        or hardcoded_user
        or ""
    ).strip()
    resolved_pass = (
        password
        or _TV_CREDENTIALS.get("password")
        or env_pass
        or hardcoded_pass
        or ""
    ).strip()
    if not resolved_user or not resolved_pass:
        raise RuntimeError(
            "Credenciais do TradingView são obrigatórias. "
            "Edite TV_USERNAME_CONFIG / TV_PASSWORD_CONFIG ou use as variáveis de ambiente."
        )
    return resolved_user, resolved_pass


def _tv_client(user: str | None = None, password: str | None = None) -> TvDatafeed:
    if TvDatafeed is None:
        raise RuntimeError("tvdatafeed não está instalado; instale-o para usar TradingView.")
    resolved_user, resolved_pass = _resolve_tv_credentials(user, password)
    global _tv
    if _tv is None:
        _tv = TvDatafeed(username=resolved_user, password=resolved_pass)
    return _tv


def _flatten_universe(setores_dict) -> list:
    seen, out = set(), []
    for lst in setores_dict.values():
        for tk in lst:
            if tk not in seen:
                seen.add(tk); out.append(tk)
    return out


def _tv_interval(freq_key: str) -> Interval:
    fk = (freq_key or "D").upper()
    if fk == "W":
        return Interval.in_weekly
    return Interval.in_daily

def _get_price_series_tv(symbol: str, exchange: str, bars: int, freq_key: str) -> pd.Series:
    """
    Puxa OHLC do TradingView e retorna Close como Series.
    symbol: 'USOIL', 'GOLD', 'US10Y', 'DXY', 'ZN1!' etc.
    exchange: 'TVC', 'CBOT', 'COMEX', 'NYMEX' etc.
    """
    try:
        tv = _tv_client()
    except RuntimeError as exc:
        logging.warning("TVDatafeed indisponível: %s", exc)
        return _empty_series(f"{exchange}:{symbol}")

    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=_tv_interval(freq_key),
            n_bars=int(bars),
        )
        if df is None or df.empty:
            return _empty_series(f"{exchange}:{symbol}")
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = f"{exchange}:{symbol}"
        return s.sort_index()
    except Exception as exc:
        logging.error("Falha ao buscar %s:%s no TradingView: %s", exchange, symbol, exc)
        return _empty_series(f"{exchange}:{symbol}")

def _get_price_series_unified(meta: dict, bars: int, freq_key: str) -> pd.Series:
    """
    meta deve conter:
      - 'source' == 'tv' e 'tv_symbol' + 'tv_exchange'
    """
    if meta.get("source") == "tv":
        sym = meta.get("tv_symbol", "")
        ex = meta.get("tv_exchange", "")
        s_tv = _get_price_series_tv(sym, ex, bars=bars, freq_key=freq_key)
        if s_tv is not None and not s_tv.dropna().empty:
            return s_tv

    logging.warning("Meta sem configuração de TradingView ou sem dados: %s", meta)
    return _empty_series()




# ================================
# Helpers de símbolo (MT5/yfinance)
# ================================
def _ensure_tz_naive_index(x):
    """
    Remove timezone do índice (Series ou DataFrame) se existir.
    Retorna o mesmo tipo de x.
    """
    if isinstance(x, pd.Series):
        idx = x.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            # converte para BRT antes, depois remove tz (evita "pular" horário)
            tz = BRA_TZ if BRA_TZ is not None else "America/Sao_Paulo"
            x.index = idx.tz_convert(tz).tz_localize(None)
        return x
    elif isinstance(x, pd.DataFrame):
        idx = x.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            tz = BRA_TZ if BRA_TZ is not None else "America/Sao_Paulo"
            x.index = idx.tz_convert(tz).tz_localize(None)
        return x
    return x  # outros tipos: retorna como está

def _select_symbol(sym: str) -> str | None:
    """
    Garante que o símbolo existe/está visível no MT5.
    Retorna o mesmo 'sym' se ok, caso contrário None.
    """
    try:
        info = mt5.symbol_info(sym)
        if info is None:
            if not mt5.symbol_select(sym, True):
                return None
            info = mt5.symbol_info(sym)
        elif not info.visible:
            if not mt5.symbol_select(sym, True):
                return None
        return sym
    except Exception:
        return None


def _yf_symbol(sym: str) -> str:
    """
    Mapeia tickers da B3 para o sufixo '.SA' no yfinance.
    Deixa índices (começam com ^), pares FX (contêm '/'), e =X intocados.
    """
    if sym.endswith(".SA") or sym.endswith("=X") or sym.startswith("^") or ("/" in sym):
        return sym
    return sym + ".SA"


# ================================
# Último tick do MT5 (live)
# ================================
def _mt5_last_price(symbol: str) -> tuple[float, datetime | None]:
    """
    Lê o último preço 'vivo' do MT5 com fallback:
    - last; senão (bid+ask)/2; senão bid; senão ask.
    Retorna (preco_float ou np.nan, timestamp_em_timezone_BR).
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return np.nan, None

        # preço
        last = getattr(tick, "last", np.nan)
        if np.isnan(last) or last <= 0:
            bid = getattr(tick, "bid", np.nan)
            ask = getattr(tick, "ask", np.nan)
            if not np.isnan(bid) and not np.isnan(ask) and bid > 0 and ask > 0:
                px = (bid + ask) / 2.0
            elif not np.isnan(bid) and bid > 0:
                px = bid
            elif not np.isnan(ask) and ask > 0:
                px = ask
            else:
                px = np.nan
        else:
            px = float(last)

        # horário do tick (UTC -> BRA_TZ)
        t_utc = datetime.utcfromtimestamp(int(getattr(tick, "time", 0))).replace(tzinfo=pytz.UTC)
        t_br = t_utc.astimezone(BRA_TZ)
        return float(px), t_br
    except Exception:
        return np.nan, None

def _ensure_tz_naive(s: pd.Series) -> pd.Series:
    """Se o índice for DatetimeIndex com timezone, remove o tz (converte p/ naive)."""
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.tz_convert("America/Sao_Paulo").tz_localize(None)
    return s


def _is_b3_session_now() -> bool:
    """Verifica se estamos dentro do horário de negociação da B3 (10h às 18h30 BRT)."""
    now_br = datetime.now(BRA_TZ)
    return time(10, 0) <= now_br.time() <= time(18, 30)


def _overlay_last_tick_daily(s: pd.Series, symbol_for_mt5: str) -> pd.Series:
    """
    Substitui o último valor da série diária pelo último tick do MT5,
    apenas durante o pregão (heurística) e se houver tick válido.
    Retorna sempre uma Series com índice tz-naive.
    """
    if s is None or s.dropna().empty:
        return s

    # só tenta overlay durante pregão
    if not _is_b3_session_now():
        return s

    px, t_br = _mt5_last_price(symbol_for_mt5)
    if np.isnan(px) or t_br is None:
        return s

    s = s.copy().sort_index()

    # garante índice tz-naive
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    try:
        last_idx = s.index[-1]
        last_date = last_idx.date()
    except Exception:
        return s

    today = t_br.date()
    if today == last_date:
        # substitui o ponto mais recente
        s.iloc[-1] = float(px)
    elif today > last_date:
        # adiciona ponto de hoje (meia-noite, tz-naive)
        ts = pd.Timestamp(today)  # 00:00 do dia
        # se já existir esse índice por algum motivo, apenas substitui
        if ts in s.index:
            s.loc[ts] = float(px)
        else:
            # concatena uma nova observação
            add = pd.Series([float(px)], index=pd.DatetimeIndex([ts]), name=s.name)
            s = pd.concat([s, add]).sort_index()

    return s



# ================================
# yfinance: extrai coluna de fechamento robustamente
# (se você já tem essa função no arquivo, pode manter a sua)
# ================================
def _close_from_yf_df(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Extrai a coluna de fechamento do df do yfinance, lidando com MultiIndex."""
    if df is None or df.empty:
        return pd.Series(dtype=float, name=symbol)

    if isinstance(df.columns, pd.MultiIndex):
        # ('Close', <ticker>)
        try:
            if "Close" in df.columns.get_level_values(0):
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s[symbol] if symbol in s.columns else s.iloc[:, 0]
                return s.rename(symbol).astype(float)
        except Exception:
            pass
        # (<ticker>, 'Close')
        try:
            if "Close" in df.columns.get_level_values(-1):
                s = df.xs("Close", axis=1, level=-1)
                s = s[symbol] if symbol in s.columns else s.iloc[:, 0]
                return s.rename(symbol).astype(float)
        except Exception:
            pass
        # fallback: “achata” os nomes
        flat = df.copy()
        flat.columns = ["_".join(map(str, c)).strip() for c in flat.columns.to_list()]
        for key in (f"{symbol}_Close", "Close", "Adj Close", f"{symbol}_Adj Close"):
            if key in flat.columns:
                return flat[key].rename(symbol).astype(float)
        return pd.Series(dtype=float, name=symbol)

    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col:
        return df[col].rename(symbol).astype(float)
    return pd.Series(dtype=float, name=symbol)


# ================================
# yfinance: baixa e reamostra
# ================================
def _get_price_series_yf(symbol: str, bars: int, freq_key: str) -> pd.Series:
    """
    Baixa preços via yfinance e reamostra para:
      - Diário: 'B' (business day) com ffill
      - Semanal: 'W-FRI' (última de sexta)
    Retorna Series (index datetime, name=symbol) ou vazia.
    """
    def _resample(s: pd.Series, fk: str) -> pd.Series:
        fk = (fk or "D").upper()
        if fk == "W":
            return s.resample("W-FRI").last().dropna()
        return s.asfreq("B").ffill().dropna()

    try:
        days_back = max(120, int(bars * 1.8))
        start_dt = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_back)

        # A) download com start
        df = yf.download(
            tickers=symbol,
            start=start_dt.date(),
            auto_adjust=True,
            progress=False,
            threads=True,
            interval="1d",
            group_by="column",
        )
        s = _close_from_yf_df(df, symbol)
        if not s.dropna().empty:
            return _resample(s.rename(symbol).astype(float).sort_index(), freq_key)

        # B) Ticker().history
        tk = yf.Ticker(symbol)
        df = tk.history(period="10y", auto_adjust=True)
        s = _close_from_yf_df(df, symbol)
        if not s.dropna().empty:
            return _resample(s.rename(symbol).astype(float).sort_index(), freq_key)

        # C) download(period="10y")
        df = yf.download(
            tickers=symbol,
            period="10y",
            auto_adjust=True,
            progress=False,
            threads=True,
            interval="1d",
            group_by="column",
        )
        s = _close_from_yf_df(df, symbol)
        if not s.dropna().empty:
            return _resample(s.rename(symbol).astype(float).sort_index(), freq_key)

        # D) sem auto_adjust
        df = yf.download(
            tickers=symbol,
            period="10y",
            auto_adjust=False,
            progress=False,
            threads=True,
            interval="1d",
            group_by="column",
        )
        s = _close_from_yf_df(df, symbol)
        if not s.dropna().empty:
            return _resample(s.rename(symbol).astype(float).sort_index(), freq_key)

        return pd.Series(dtype=float, name=symbol)

    except Exception:
        return pd.Series(dtype=float, name=symbol)


def _get_price_series_mt5(symbol: str, bars: int, freq_key: str) -> pd.Series:
    """
    Busca preços no MT5 diretamente (D1) e devolve série já reamostrada:
      - 'D' -> calendário de negócios ('B') com ffill
      - 'W' -> 'W-FRI' (sexta)
    Sempre retorna índice tz-naive (sem timezone).
    """
    try:
        import MetaTrader5 as mt5
    except Exception:
        return pd.Series(dtype=float, name=symbol)

    # Resolve/seleciona o símbolo no servidor
    sym_resolved = _select_symbol(symbol)  # use a sua; se não tiver, troque por mt5.symbol_select(...)
    if not sym_resolved:
        return pd.Series(dtype=float, name=symbol)

    # Tenta copiar "bars + folga"
    n = int(max(5, min(4000, (bars or 300) + 20)))
    try:
        rates = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_D1, 0, n)
    except Exception:
        rates = None

    if rates is None or len(rates) == 0:
        return pd.Series(dtype=float, name=symbol)

    # Monta Series (close) com índice datetime e sem timezone
    df = pd.DataFrame(rates)
    if df.empty or "time" not in df or "close" not in df:
        return pd.Series(dtype=float, name=symbol)

    idx = pd.to_datetime(df["time"], unit="s")
    # garante tz-naive
    try:
        if getattr(idx, "tz", None) is not None:
            # se vier tz-aware, converte p/ BRT e remove tz
            idx = idx.tz_convert(BRA_TZ or "America/Sao_Paulo").tz_localize(None)
    except Exception:
        # se não suportar tz_convert (normal no MT5), apenas garante naive
        idx = idx.tz_localize(None) if hasattr(idx, "tz_localize") else idx

    s = pd.Series(df["close"].astype(float).values, index=idx, name=symbol).sort_index()
    s = s.dropna()

    # Reamostra conforme freq_key
    fk = (freq_key or "D").upper()
    if fk == "W":
        s = s.resample("W-FRI").last()
    else:
        s = s.asfreq("B").ffill()

    # tira sobras e garante tz-naive de novo por segurança
    s = s.dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    return s

def _returns_window(px_df: pd.DataFrame,
                    bench: pd.Series,
                    winN: int) -> pd.DataFrame:
    """
    Monta uma matriz de retornos (todas as ações + benchmark)
    e recorta os últimos winN pontos.
    """
    rets_all = px_df.pct_change().dropna(how="all")
    bench_ret = bench.pct_change().rename(BENCH_USED)

    rets_all, bench_ret = rets_all.align(bench_ret, join="inner", axis=0)
    rt_win = pd.concat([rets_all, bench_ret], axis=1).dropna()
    return rt_win.tail(winN)



def _get_price_series(symbol: str, bars: int, freq_key: str) -> pd.Series:
    """
    Única porta de entrada de séries de preços:
      1) Tenta MT5 (várias variações do símbolo)
      2) Se falhar/curto, cai para yfinance (com mapeamento .SA)
      3) Se a frequência for diária, faz overlay do último tick do MT5 no ponto mais recente.
    Sempre retorna índice tz-naive.
    """
    fk = (freq_key or "D").upper()

    def _resample(s: pd.Series) -> pd.Series:
        if s is None or s.dropna().empty:
            return pd.Series(dtype=float, name=symbol)
        s = s.sort_index()
        # padroniza tz antes de resample
        s = _ensure_tz_naive_index(s)
        if fk == "W":
            return s.resample("W-FRI").last().dropna()
        return s.asfreq("B").ffill().dropna()

    # Candidatos MT5 (sym, sem .SA, com .SA)
    mt5_candidates = [symbol]
    if symbol.endswith(".SA"):
        mt5_candidates.append(symbol.replace(".SA", ""))
    else:
        mt5_candidates.append(symbol + ".SA")

    # 1) MT5
    resolved_for_tick = None
    for cand in mt5_candidates:
        s_resolved = _select_symbol(cand)
        if not s_resolved:
            continue
        s_mt5 = _get_price_series_mt5(s_resolved, bars, fk)  # sua função MT5
        if s_mt5 is not None and not s_mt5.dropna().empty and s_mt5.shape[0] >= 5:
            # padroniza tz-naive
            s_mt5 = _ensure_tz_naive_index(s_mt5)
            # overlay apenas em base diária
            if fk == "D":
                resolved_for_tick = s_resolved
                s_mt5 = _overlay_last_tick_daily(s_mt5, resolved_for_tick)
            # resample e nome
            return _resample(s_mt5).rename(symbol)

    # 2) yfinance (com mapeamento .SA)
    yf_sym = _yf_symbol(symbol)
    s_yf = _get_price_series_yf(yf_sym, bars, fk)  # sua função yf
    if s_yf is not None and not s_yf.dropna().empty:
        # padroniza tz-naive (yfinance costuma vir tz-aware)
        s_yf = _ensure_tz_naive_index(s_yf)
        if fk == "D":
            # tenta overlay (se o símbolo existir no MT5)
            for cand in mt5_candidates:
                if _select_symbol(cand):
                    resolved_for_tick = cand
                    break
            if resolved_for_tick:
                s_yf = _overlay_last_tick_daily(s_yf, resolved_for_tick)
        return _resample(s_yf).rename(symbol)

    # Falhou geral
    return pd.Series(dtype=float, name=symbol)


# ================================
# Mapeia dict {label: symbol} -> DataFrame
# ================================
def _fetch_series_from_map(label2sym: dict, bars: int, freq_key: str):
    """
    Constrói DataFrame de preços a partir de {label: symbol} usando _get_price_series.
    Retorna (df, debug_dict).
    """
    tentativas, sucessos, falhas = [], [], []
    series = []

    for label, sym in label2sym.items():
        tentativas.append(sym)
        s = _get_price_series(sym, bars=bars, freq_key=freq_key)
        if s is not None and not s.dropna().empty:
            series.append(s.rename(label))
            sucessos.append(sym)
        else:
            falhas.append(sym)

    df = pd.concat(series, axis=1) if series else pd.DataFrame()
    dbg = {"tentativas": tentativas, "sucessos": sucessos, "falhas": falhas}
    return df, dbg


# ================================
# UI: Seleção e montagem das classes
# (ajuste CLASSES_MAP conforme seu universo)
# ================================

# Exemplo de mapa mais amplo (inclui USD e ETFs em USD)
# =============================
# Universo (10 setores definidos)
# =============================
SETORES_ATIVOS: Dict[str, List[str]] = {
    "Bens Industriais": ["WEGE3","EMBR3","MOTV3","RAIL3","GGPS3","POMO4","MRSA3B","FRAS3","VAMO3","MILS3"],
    "Consumo Cíclico": ["RENT3","LREN3","SMFT3","CYRE3","CURY3","DIRR3","VIVA3","ALPA4","WHRL4","MGLU3"],
    "Consumo Não Cíclico": ["ABEV3","GMAT3","ASAI3","MDIA3","RAIZ4","BEEF3","SLCE3","TTEN3","AGRO3","CAML3"],
    "Financeiro": ["ITUB4","BPAC11","BBDC4","BBAS3","BBSE3","PSSA3","MULT3"],
    "Materiais Básicos": ["VALE3","SUZB3","GGBR4","CMIN3","KLBN11","GOAU4","CSNA3","UNIP6","USIM5","BRKM5"],
    "Petróleo, Gás e Biocombustíveis": ["PETR4","PRIO3","VBBR3","UGPA3","CSAN3","RECV3","BRAV3"],
    "Saúde": ["RDOR3","HAPV3","HYPE3","FLRY3","ODPV3","BLAU3","PGMN3","PNVL3","ONCO3","DASA3"],
    "Tecnologia da Informação": ["TOTS3","LWSA3","BMOB3","MOSI3"],
    "Telecomunicações": ["VIVT3","TIMS3","INTB3","DESK3","FIQE3","BRST3"],
    "Utilities": ["AXIA3","ELET3","SBSP3","CPFE3","EQTL3","CPLE6","CMIG4","NEOE3","EGIE3","ENEV3"],
}



# Benchmark fixo
BENCHMARK = "BOVA11"

# Parâmetros padrão
N_RATIO = 90
N_MOM = 30
TRAIL = 8
LOOKBACK = 252  # base diária; reamostramos p/ semanal se escolhido
TF = mt5.TIMEFRAME_D1

# =============================
# Funções auxiliares
# =============================
@st.cache_resource(show_spinner=False)
def mt5_init() -> bool:
    return bool(mt5.initialize())

@st.cache_data(show_spinner=False)
def list_symbols() -> List[str]:
    arr = mt5.symbols_get()
    return [s.name for s in arr] if arr else []





@st.cache_data(show_spinner=False, ttl=300)
def _cached_rates_range(sym_resolved: str, start_dt, end_dt):
    rates = mt5.copy_rates_range(sym_resolved, mt5.TIMEFRAME_D1, start_dt, end_dt)
    return None if rates is None else pd.DataFrame(rates)

@st.cache_data(show_spinner=False, ttl=300)
def _cached_rates_from_pos(sym_resolved: str, bars: int):
    rates = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_D1, 0, int(bars))
    return None if rates is None else pd.DataFrame(rates)

# ==== UTILITÁRIOS LONG & SHORT (cointegração + Kalman) ====


def _align_2(y: pd.Series, x: pd.Series):
    yy, xx = y.align(x, join="inner")
    m = yy.notna() & xx.notna()
    return yy[m], xx[m]

def eg_pvalue_beta(y: pd.Series, x: pd.Series):
    """
    Engle-Granger numa janela y ~ alpha + beta*x + u -> (pvalue, beta_ols)
    Séries já alinhadas e sem NaN.
    """
    if len(y) < 25 or len(x) < 25:
        return np.nan, np.nan
    X = np.vstack([np.ones(len(x)), x.values]).T
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]  # [alpha, beta]
    beta_ols = float(b[1])
    try:
        _, pval, _ = coint(y.values, x.values)
    except Exception:
        pval = np.nan
    return float(pval), beta_ols

def kalman_tv_beta(y: pd.Series, x: pd.Series,
                   q_alpha: float = 1e-6, q_beta: float = 1e-5,
                   r: float = 1e-4,
                   init_alpha: float | None = None,
                   init_beta: float | None = None):
    """
    Filtro de Kalman com estado [alpha_t, beta_t] para y_t = alpha_t + beta_t*x_t + eps.
    Retorna dict com alpha, beta, spread (inovação) e var_spread.
    """
    y, x = _align_2(y, x)
    n = len(y)
    if n < 5:
        return {"alpha": pd.Series(dtype=float), "beta": pd.Series(dtype=float),
                "spread": pd.Series(dtype=float), "var_spread": pd.Series(dtype=float)}

    F = np.eye(2)
    Q = np.diag([q_alpha, q_beta])
    H = np.zeros((1, 2))
    R = np.array([[r]])

    a = np.zeros(2)
    if init_alpha is not None: a[0] = init_alpha
    if init_beta  is not None: a[1] = init_beta
    P = np.eye(2) * 1.0

    alpha_hist, beta_hist, spread_hist, var_hist = [], [], [], []
    x_vals, y_vals = x.values, y.values
    idx = y.index

    for t in range(n):
        # previsão
        a = F @ a
        P = F @ P @ F.T + Q

        # observação
        H[0, 0] = 1.0
        H[0, 1] = x_vals[t]

        y_hat = H @ a
        innov = y_vals[t] - y_hat[0]         # spread (inovação)
        S = H @ P @ H.T + R                  # var da inovação (1x1)
        K = (P @ H.T) / S                    # ganho (2x1)

        # atualização
        a = a + (K.flatten() * innov)
        P = (np.eye(2) - K @ H) @ P

        alpha_hist.append(a[0])
        beta_hist.append(a[1])
        spread_hist.append(innov)
        var_hist.append(S[0, 0])

    return {
        "alpha": pd.Series(alpha_hist, index=idx, name="alpha"),
        "beta": pd.Series(beta_hist,   index=idx, name="beta"),
        "spread": pd.Series(spread_hist, index=idx, name="spread"),
        "var_spread": pd.Series(var_hist, index=idx, name="var_spread"),
    }

def zscore(series: pd.Series, win: int = 60, method: str = "rolling"):
    s = series.dropna()
    if s.empty:
        return s.rename("z")
    if method == "ewma":
        lam = 0.94
        mu = s.ewm(alpha=1-lam, adjust=False).mean()
        var = (s - mu).ewm(alpha=1-lam, adjust=False).var(bias=False)
        sd = np.sqrt(var)
        z = (s - mu) / (sd.replace(0, np.nan))
        return z.rename("z")
    else:
        mu = s.rolling(win, min_periods=win//2).mean()
        sd = s.rolling(win, min_periods=win//2).std(ddof=1)
        z = (s - mu) / sd.replace(0, np.nan)
        return z.rename("z")

def scan_pairs_cointegration(prices: pd.DataFrame,
                             windows: list[int],
                             alpha: float = 0.05,
                             min_passes: int = 5) -> pd.DataFrame:
    """
    Vasculha todos os pares das colunas de 'prices' para as janelas em 'windows'.
    Retorna pares que passam em >= min_passes, com z-score atual (via Kalman).
    """
    cols = list(prices.columns)
    results = []

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            y_name, x_name = cols[i], cols[j]
            passes = 0
            passed_ws, pvals, betas = [], [], []

            for W in windows:
                px = prices[[y_name, x_name]].dropna()
                if len(px) < W:
                    pvals.append(np.nan); betas.append(np.nan)
                    continue
                yW = px[y_name].iloc[-W:]
                xW = px[x_name].iloc[-W:]
                yW, xW = _align_2(yW, xW)
                pval, beta_ols = eg_pvalue_beta(yW, xW)
                pvals.append(pval); betas.append(beta_ols)
                if pd.notna(pval) and pval < alpha:
                    passes += 1
                    passed_ws.append(W)

            if passes >= min_passes:
                # calcula z-score atual via Kalman na janela mais longa aprovada
                W0 = max(passed_ws) if passed_ws else max(windows)
                px = prices[[y_name, x_name]].dropna()
                yW = px[y_name].iloc[-W0:]; xW = px[x_name].iloc[-W0:]
                yW, xW = _align_2(yW, xW)
                kal = kalman_tv_beta(yW, xW)
                sprd = kal["spread"]
                z_now = zscore(sprd, win=min(60, max(20, W0//3))).iloc[-1] if not sprd.empty else np.nan

                results.append({
                    "pair": f"{y_name} ~ {x_name}",
                    "y": y_name,
                    "x": x_name,
                    "passes": passes,
                    "windows_passed": passed_ws,
                    "min_pvalue": np.nanmin(pvals) if len(pvals) else np.nan,
                    "beta_ols_mediana": np.nanmedian(betas) if len(betas) else np.nan,
                    "zscore_atual": float(z_now) if pd.notna(z_now) else np.nan
                })

    if not results:
        return pd.DataFrame(columns=["pair","passes","windows_passed","min_pvalue","beta_ols_mediana","zscore_atual"])

    df = pd.DataFrame(results)
    df = df.sort_values(["passes", "min_pvalue"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return df
# ==== FIM UTILITÁRIOS L&S ====


    
def _resolve_benchmark() -> str:
    """Resolve o nome exato do benchmark no servidor e salva em session_state."""
    key = "bench_symbol_resolved"
    if key in st.session_state and st.session_state[key]:
        return st.session_state[key]
    sym = _select_symbol(BENCHMARK)  # sua função que tenta match por exato/prefix/contains
    if sym:
        mt5.symbol_select(sym, True)  # garantir 'mostrado'
        st.session_state[key] = sym
        return sym
    return ""  # força erro tratável

def fetch_benchmark_series(d0: date, d1: date) -> pd.Series:
    """
    Busca D1 do benchmark com robustez:
    - garante symbol_select ativo
    - tenta range; se vazio, cai para barras; faz 2 retries
    - recorta por data
    """
    sym = _resolve_benchmark()
    if not sym:
        return pd.Series(dtype=float)

    mt5.symbol_select(sym, True)  # manter no Market Watch sempre que for usar

    start_dt = datetime.combine(d0, datetime.min.time())
    end_dt   = datetime.combine(d1, datetime.max.time())

    # 2 tentativas com fallback
    for attempt in range(2):
        df = _cached_rates_range(sym, start_dt, end_dt)
        if df is not None and not df.empty:
            break
        # fallback por barras (margem +40 dias, limitado)
        days = max(5, (d1 - d0).days + 40)
        df = _cached_rates_from_pos(sym, min(4000, days))
        if df is not None and not df.empty:
            break
        time.sleep(0.2)  # pequeno respiro para re-tentar
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()[["close"]]
    df = df.loc[(df.index.date >= d0) & (df.index.date <= d1)]
    if df.empty:
        return pd.Series(dtype=float)
    df.columns = [sym]
    return df.iloc[:, 0].rename(BENCHMARK)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_sym_range(sym_resolved: str, start_dt, end_dt):
    rates = mt5.copy_rates_range(sym_resolved, mt5.TIMEFRAME_D1, start_dt, end_dt)
    return None if rates is None else pd.DataFrame(rates)

@st.cache_data(show_spinner=False, ttl=300)
def _cached_sym_from_pos(sym_resolved: str, bars: int):
    rates = mt5.copy_rates_from_pos(sym_resolved, mt5.TIMEFRAME_D1, 0, int(bars))
    return None if rates is None else pd.DataFrame(rates)



def align_outer(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Alinha por 'outer join' para não zerar o setor quando 1 ticker falha.
    Preferimos preservar o máximo de colunas e datas e tratar NaNs no retorno.
    """
    out = None
    for d in dfs:
        if d is None or d.empty:
            continue
        out = d if out is None else out.join(d, how="outer")
    return out if out is not None else pd.DataFrame()

def to_freq_series(s: pd.Series, freq: str) -> pd.Series:
    if s is None or s.empty:
        return s
    if freq == "W":
        return s.resample("W-FRI").last().dropna()
    return s

def bars_for_freq(freq: str, lookback_days: int) -> int:
    """Quantidade de barras D1 a buscar no MT5 para suportar a frequência escolhida."""
    if freq == "W":
        # 4 anos úteis garantem >200 semanas para janelas longas
        return max(lookback_days * 4, 252 * 4 + 50)
    # Diário: lookback + margem
    return lookback_days + 80


def to_freq_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if freq == "W":
        return df.resample("W-FRI").last().dropna(how="all")
    return df

def sector_equal_weight_index(tickers: List[str], freq: str = "W", bars: int | None = None) -> pd.Series:
    """
    Índice equal-weight por setor usando a função unificada _get_price_series(...).
    - freq: 'W' (semanal) ou 'D' (diário)
    - bars: se None, estimamos a partir do LOOKBACK global
    """
    fk = (freq or "W").upper()
    need_bars = bars if bars is not None else bars_for_freq("D", LOOKBACK)  # sempre baixamos base diária suficiente

    # monta DataFrame de preços com _get_price_series
    series = []
    for tk in tickers:
        s = _get_price_series(tk, bars=need_bars, freq_key="D")  # baixa em D para garantir overlay intraday
        if s is not None and not s.dropna().empty:
            series.append(s.rename(tk))

    px = pd.concat(series, axis=1) if series else pd.DataFrame()
    if px.empty or px.shape[1] == 0:
        return pd.Series(dtype=float)

    # agrega para a frequência desejada
    if fk == "W":
        px = px.resample("W-FRI").last()
    else:
        # diário: business day com ffill
        px = px.asfreq("B").ffill()

    px = px.dropna(how="all").ffill()

    # cobertura mínima
    base_min = 20 if fk == "W" else 60
    min_rows = max(base_min, int(0.25 * len(px)))
    good_cols = [c for c in px.columns if px[c].notna().sum() >= min_rows]
    px = px[good_cols]
    if px.shape[1] < 2:
        return pd.Series(dtype=float)

    # retornos e média EW (com mínimo de participantes por data)
    rets = px.pct_change()
    min_k = max(2, int(np.ceil(0.30 * rets.shape[1])))
    counts = rets.count(axis=1)
    ew = rets.mean(axis=1, skipna=True)
    ew[counts < min_k] = np.nan
    ew = ew.dropna()
    if ew.empty or ew.size < 8:
        return pd.Series(dtype=float)

    idx = (1.0 + ew).cumprod()
    idx = 100.0 * idx / idx.iloc[0]
    idx.name = "EW"
    return idx


def rs_ratio_momentum(x: pd.Series, b: pd.Series, n: int = 55, m: int = 13) -> pd.DataFrame:
    # Alinha e limpa
    x, b = x.align(b, join="inner")
    rs = (x / b) * 100.0

    # JdK RS-Ratio = 100 * RS / SMA_n(RS)
    rs_sma = rs.rolling(n, min_periods=n).mean()
    rr = 100.0 * (rs / rs_sma)

    # JdK RS-Momentum = 100 * RR / SMA_m(RR)   <-- este era o ponto que deixava “diagonal”
    rr_sma = rr.rolling(m, min_periods=m).mean()
    rm = 100.0 * (rr / rr_sma)

    out = pd.DataFrame({"RS_Ratio": rr, "RS_Momentum": rm})
    return out



def _slice_by_range(s: pd.Series, dt_ini: date, dt_fim: date, freq_key: str) -> pd.Series:
    """Pega barras suficientes pela série unificada e recorta no intervalo."""
    # estima barras mínimas
    # margem 40 dias úteis: cobre saltos e garante reamostragens
    days = max(5, (dt_fim - dt_ini).days + 40)
    bars_need = min(2000, int(bars_for_freq("D", days)))
    s_full = _get_price_series(s.name if s.name else BENCHMARK, bars=bars_need, freq_key="D")
    if s_full is None or s_full.dropna().empty:
        return pd.Series(dtype=float)
    # recorta e reamostra
    s_full = s_full.loc[(s_full.index.date >= dt_ini) & (s_full.index.date <= dt_fim)]
    if freq_key.upper() == "W":
        s_full = s_full.resample("W-FRI").last()
    else:
        s_full = s_full.asfreq("B").ffill()
    return s_full.dropna()

def _get_range_series(sym: str, dt_ini: date, dt_fim: date, freq_key: str) -> pd.Series:
    """Convenience para baixar por símbolo."""
    bars_est = max(5, (dt_fim - dt_ini).days + 40)
    s = _get_price_series(sym, bars=min(2000, int(bars_est)), freq_key="D")
    if s is None or s.dropna().empty:
        return pd.Series(dtype=float)
    s.name = sym
    return _slice_by_range(s, dt_ini, dt_fim, freq_key)

def _sanitize_naive_sp_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o índice seja DatetimeIndex (ou PeriodIndex convertido), sem tz (naïve),
    normalizado para dia, sem duplicados/fora de ordem e sem datas futuras.
    Retorna SEMPRE um DataFrame com índice higienizado.
    """
    # 1) Assegura que é datetime (ou converte PeriodIndex)
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()

    # 2) Remove timezone -> America/Sao_Paulo e volta a ser naïve
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/Sao_Paulo").tz_localize(None)

    # 3) Normaliza para dia, remove duplicados e ordena
    df.index = df.index.floor("D")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 4) Corta datas futuras usando hoje em SP (naïve)
    cutoff = pd.Timestamp.now(tz="America/Sao_Paulo").normalize().tz_localize(None)
    df = df.loc[df.index <= cutoff]

    return df


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Parâmetros")
    freq = st.radio("Frequência", ["Semanal","Diária"], index=0, horizontal=True, key="freq_global")
    freq_key = "W" if freq == "Semanal" else "D"
    lookback = st.slider("Lookback (dias úteis base D1)", 120, 1000, LOOKBACK, step=10)
    n_ratio = st.slider("Janela RS-Ratio (n)", 20, 180, N_RATIO, step=5)
    n_mom = st.slider("Janela RS-Momentum (m)", 10, 90, N_MOM, step=5)
    trail = st.slider("Tamanho da cauda", 4, 24, TRAIL, step=1)
    

if LOOKBACK != lookback:
    LOOKBACK = lookback

# =============================
# Pipeline
# =============================
with st.spinner("Conectando ao MetaTrader5..."):
    if not mt5_init():
        st.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
        st.stop()

# --- helper para padronizar índice (SP naive + diário, sem duplicatas)
def _sanitize_idx(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if not isinstance(x.index, pd.DatetimeIndex):
        x = x.copy()
        x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[x.index.notna()]
    if getattr(x.index, "tz", None) is not None:
        x = x.copy()
        x.index = x.index.tz_convert("America/Sao_Paulo").tz_localize(None)
    # padroniza em dia útil + ffill leve (séries já devem estar alinhadas, mas garantimos)
    x = x.copy()
    x.index = x.index.floor("D")
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x

# =============================
# Benchmark fixo (BOVA11) — forçar símbolo e limitar histórico
# =============================
bench_name = "BOVA11"
min_need = max(n_ratio, n_mom) + trail + 10  # folga p/ janelas e último trail
bars_need = bars_for_freq(freq_key, LOOKBACK + min_need)
bars_try = min(1000, int(bars_need))

bench_px = _get_price_series(bench_name, bars=bars_try, freq_key=freq_key)
if bench_px is None or bench_px.dropna().empty:
    st.error("Falha ao obter preços históricos do BOVA11 (via série unificada). Abra o BOVA11 no MT5 em D1, role histórico e recarregue.")
    st.stop()
bench_px = _sanitize_idx(bench_px.rename("BOVA11"))

# Índices equal-weight por setor
sector_indices: Dict[str, pd.Series] = {}
logs = []
with st.spinner("Calculando índices equal-weight por setor..."):
    bars_need = bars_for_freq(freq_key, LOOKBACK)
    for setor, tickers in SETORES_ATIVOS.items():
        logs.append(f"Baixando: {setor} -> {', '.join(tickers)} [bars={bars_need}]")
        idx = sector_equal_weight_index(tickers, freq=freq_key, bars=bars_need)
        if not idx.empty:
            idx = _sanitize_idx(idx.rename(setor))
            tmp = pd.concat([idx, bench_px], axis=1, join="inner").dropna()
            if not tmp.empty:
                sector_indices[setor] = tmp.iloc[:, 0].rename(setor)

if not sector_indices:
    st.warning("Nenhum índice setorial pôde ser montado. Verifique os tickers expostos no seu MT5.")
    st.stop()

rows_dbg = []
for setor, s_idx in sector_indices.items():
    common = s_idx.dropna().index.intersection(bench_px.dropna().index)
    rows_dbg.append({
        "setor": setor,
        "len_EW": int(s_idx.dropna().shape[0]),
        "len_common_vs_BOVA": int(len(common)),
        "ini_common": common.min().date() if len(common) else "—",
        "fim_common": common.max().date() if len(common) else "—",
    })

# =============================
# RS-Ratio / RS-Momentum
# =============================
# --- RS-Ratio / RS-Momentum (com checagens e log útil) ---
sector_curves: Dict[str, pd.DataFrame] = {}

for setor, s_idx in sector_indices.items():
    dfab = pd.concat([s_idx, bench_px], axis=1, join="inner").dropna()
    # --- normaliza/limpa índice ---
    if not isinstance(dfab.index, pd.DatetimeIndex):
        dfab.index = pd.to_datetime(dfab.index, errors="coerce")
    dfab = dfab[dfab.index.notna()]
    if getattr(dfab.index, "tz", None) is not None:
        dfab.index = dfab.index.tz_convert("America/Sao_Paulo").tz_localize(None)
    dfab = dfab[~dfab.index.duplicated(keep="last")].sort_index()

    # >>>> NOVO: se for semanal, padroniza tudo em W-FRI (fechamento de sexta)
    if str(freq_key).upper().startswith("W"):
        dfab = dfab.resample("W-FRI").last().dropna(how="all")

    # corta datas futuras (naive)
    today_naive_sp = pd.Timestamp.now(tz="America/Sao_Paulo").normalize().tz_localize(None)
    dfab = dfab.loc[dfab.index <= today_naive_sp]

    need = max(int(n_ratio), int(n_mom)) + max(5, int(trail))
    if len(dfab) < need:
        logs.append(f"[{setor}] curto p/ RRG: {len(dfab)} < {need}")
        continue

    rr = rs_ratio_momentum(dfab.iloc[:, 0], dfab.iloc[:, 1], n=n_ratio, m=n_mom)
    if rr is None or rr.dropna(how="all").empty:
        logs.append(f"[{setor}] rr vazio (após cálculo)")
        continue

    rr = rr.sort_index().ffill()
    sector_curves[setor] = rr


rows = []
for setor, dfc in sector_curves.items():
    if dfc is None or dfc.empty:
        continue
    tail_df = dfc.iloc[-trail:].copy()
    tail_df["Setor"] = setor
    tail_df["Date"] = tail_df.index
    rows.append(tail_df)

if not rows:
    st.warning("Sem dados suficientes para montar o RRG com as janelas atuais.")
    st.stop()

rrg = pd.concat(rows, axis=0)
rrg["Quadrante"] = np.select(
    [
        (rrg["RS_Ratio"] >= 100) & (rrg["RS_Momentum"] >= 100),
        (rrg["RS_Ratio"] >= 100) & (rrg["RS_Momentum"] < 100),
        (rrg["RS_Ratio"] < 100) & (rrg["RS_Momentum"] < 100),
        (rrg["RS_Ratio"] < 100) & (rrg["RS_Momentum"] >= 100),
    ],
    ["Leading", "Weakening", "Lagging", "Improving"],
    default="—"
)

TAB_NAMES = ["Visão", "Rotação", "Setor", "Classes", "Risco", "Orbital", "L&S", "Portfólio"]
tabs = st.tabs(TAB_NAMES)
tmap = {name: tabs[i] for i, name in enumerate(TAB_NAMES)}


# =============================
# Aba VISÃO — Panorama diário
# =============================
with tmap["Visão"]:
    st.subheader("Overview")

    # ----- Controles -----
    colv1, colv2 = st.columns([1,1])
    with colv1:
        freq_visao = st.radio("Base para cálculos", ["Diária"], index=0, horizontal=True, key="visao_freq")
        fk_visao = "D"  # por enquanto só diária
    with colv2:
        lookback_visao = st.slider("Prazo para download", 40, 200, 40, step=10, key="visao_lb")

    # ----- Universo (todos os tickers dos setores) -----
    def _universe(SETORES_ATIVOS: dict[str, list[str]]) -> list[str]:
        seen, out = set(), []
        for lst in SETORES_ATIVOS.values():
            for tk in lst:
                if tk not in seen:
                    seen.add(tk); out.append(tk)
        return out

    universe = _universe(SETORES_ATIVOS)
    if not universe:
        st.warning("Universo vazio.")
        st.stop()

    # ----- Baixa preços (D1 base) via série unificada -----
    bars_need = min(1000, int(bars_for_freq("D", lookback_visao + 5)))
    
    px_list = []
    sector_of = {tk: setor for setor, tks in SETORES_ATIVOS.items() for tk in tks}
    
    for tk in universe:
        s = _get_price_series(tk, bars=bars_need, freq_key="D")
        if s is None or s.dropna().empty:
            continue
        px_list.append(s.rename(tk))
    
    if not px_list:
        st.error("Não consegui montar preços para a visão. Abra os papéis no MT5 em D1 e role histórico.")
        st.stop()
    
    prices = pd.concat(px_list, axis=1)
    prices = prices.asfreq("B").ffill()                     # força dias úteis
    prices = prices.tail(max(45, lookback_visao + 2))       # janela de trabalho

    # ----- Métricas do dia -----
    rets = prices.pct_change()
    last_row = rets.tail(1).T.rename(columns=lambda _: "ret_dia")   # retorno de hoje
    last_row["setor"] = last_row.index.map(lambda tk: sector_of.get(tk, "—"))

    # Distância para MME20 (usando janela 'lookback_visao' quando >=20; senão 20)
    span = max(20, int(lookback_visao))
    ema20 = prices.ewm(span=20, adjust=False).mean()
    dist20 = (prices.iloc[-1] / ema20.iloc[-1] - 1.0)
    last_row["dist_MME20"] = dist20

    # Flags
    last_row["acima_MME20"] = (dist20 > 0).astype(int)
    last_row["positivo_hoje"] = (last_row["ret_dia"] > 0).astype(int)

    # ----- KPIs de amplitude -----
    df_kpi = last_row.dropna(subset=["ret_dia"])
    if df_kpi.empty:
        st.info("Sem dados suficientes para o dia.")
        st.stop()

    pct_pos = float(df_kpi["positivo_hoje"].mean() * 100.0)
    pct_abv = float(df_kpi["acima_MME20"].mean() * 100.0)
    mean_ret = float(df_kpi["ret_dia"].mean() * 100.0)

    k1, k2, k3 = st.columns(3)
    k1.metric("% de ativos positivos (D)", f"{pct_pos:.1f}%")
    k2.metric("% acima da MME20", f"{pct_abv:.1f}%")
    k3.metric("Retorno médio (D)", f"{mean_ret:.2f}%")

    st.markdown("---")

    # =============================
    # Visão Geral do Dia — Treemap (patch)
    # =============================
    st.markdown("---")
    st.subheader("Visão Geral do Dia")
    
    # --- controles
    colA, colB = st.columns([1.6, 1])
    with colA:
        winN = st.slider("Janela (dias) para métricas (MME / Vol / Correlação / Beta)",
                         min_value=10, max_value=120, value=20, step=5)
    with colB:
        pass
    
    # Benchmark: tenta a sua variável BENCHMARK; se vazar, tenta variações comuns do BOVA
    bench_syms = []
    try:
        if isinstance(BENCHMARK, str):
            bench_syms.append(BENCHMARK)
    except NameError:
        pass
    for alt in ("BOVA11.SA", "BOVA11"):
        if alt not in bench_syms:
            bench_syms.append(alt)
    
    # monta universo
    all_tickers = sorted({tk for lst in SETORES_ATIVOS.values() for tk in lst})
    need_bars = max(260, int(winN * 6))
    
    # baixa benchmark (primeiro que funcionar)
    bench = pd.Series(dtype=float)
    for bs in bench_syms:
        s = _get_price_series(bs, bars=need_bars, freq_key="D")
        if not s.empty:
            bench = s.rename(bs)
            BENCH_USED = bs
            break
    
    px_list = [_get_price_series(tk, bars=need_bars, freq_key="D") for tk in all_tickers]
    px_df = pd.concat([s for s in px_list if not s.empty], axis=1)
    
    if bench.empty or px_df.empty:
        st.error("Não consegui montar as séries para o treemap (benchmark ou papéis vazios).")
        st.stop()
    
    # alinhar
    px_df, bench = px_df.align(bench, join="inner", axis=0)
    if px_df.empty:
        st.error("Interseção de datas vazia após alinhamento.")
        st.stop()
    
    # --- métricas (obedecem a winN)
    last = px_df.iloc[-1]
    prev = px_df.shift(1).iloc[-1]
    var_dia = (last / prev - 1.0).rename("var_dia")
    mme = px_df.ewm(span=winN, adjust=False).mean().iloc[-1]
    d_mme = (last / mme - 1.0).rename("d_mme")
    rets = px_df.pct_change()
    vol_ann = (rets.rolling(winN).std(ddof=1) * np.sqrt(252)).iloc[-1].rename("vol_ann")
    rho_bova = rets.rolling(winN).corr(bench.pct_change()).iloc[-1].rename("rho_bova")
    
    sector_by_tk = {tk: setr for setr, lst in SETORES_ATIVOS.items() for tk in lst}
    rows = []
    for tk in px_df.columns:
        if tk not in sector_by_tk: 
            continue
        rows.append({
            "Setor": sector_by_tk[tk],
            "Ticker": tk,
            "var_dia": float(var_dia.get(tk, np.nan)),
            "d_mme": float(d_mme.get(tk, np.nan)),
            "vol_ann": float(vol_ann.get(tk, np.nan)),
            "rho_bova": float(rho_bova.get(tk, np.nan)),
        })
    df = pd.DataFrame(rows)
    
    def _fmt_pct(x, d=2): return "—" if pd.isna(x) else f"{x*100:+.{d}f}%"
    def _fmt_rho(x, d=2): return "—" if pd.isna(x) else f"{x:+.{d}f}"
    
    df["leaf_text"] = (
        "<b>" + df["Ticker"] + "</b><br>" +
        "Variação: " + df["var_dia"].apply(lambda x: _fmt_pct(x, 2)) + "<br>" +
        f"ΔMME{winN}: " + df["d_mme"].apply(lambda x: _fmt_pct(x, 2)) + "<br>" +
        f"σ{winN} (ann): " + df["vol_ann"].apply(lambda x: _fmt_pct(x, 2)) + "<br>" +
        f"ρ({BENCH_USED}): " + df["rho_bova"].apply(lambda x: _fmt_rho(x, 2))
    )
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # AQUI ESTÁ A CORREÇÃO DO customdata: passamos custom_data no px.treemap
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   
    
    # --- Treemap Mercado (visão geral) ---
    fig = px.treemap(
        df,                                  # df com colunas: Setor, Ticker, var_dia, d_mme, vol_ann, rho_bova
        path=["Setor", "Ticker"],
        color="var_dia",
        range_color=(-0.05, 0.05),
        custom_data=["var_dia", "d_mme", "vol_ann", "rho_bova"],
        hover_data={"Ticker": False, "Setor": False}
    )
    
    # Layout branco e escala centrada em 0 (vermelho → branco → verde)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=6, r=6, t=28, b=6),
        height=780,
        coloraxis_colorscale=[
            [0.00, "#b91c1c"],
            [0.45, "#fca5a5"],
            [0.50, "#ffffff"],
            [0.55, "#bbf7d0"],
            [1.00, "#15803d"],
        ],
        coloraxis_cmid=0,
        coloraxis_colorbar=dict(title="Variação (dia)", tickformat=".2%")
    )
    
    # Bordas e padding mínimos para ganhar área útil
    fig.update_traces(
        marker_line_width=0.8,
        marker_line_color="#e5e7eb",
        tiling=dict(pad=1),
        selector=dict(type="treemap")
    )
    
    # Substitui cores inválidas (NaN) por branco, evitando “cinza morto”
    fig.for_each_trace(
        lambda t: t.update(
            marker_colors=[
                "#ffffff" if (c is None or str(c).lower() == "nan") else c
                for c in t.marker.colors
            ]
        )
    )
    
    # ---------- TEXTO VISÍVEL: ticker + variação (apenas nas folhas) ----------
    # Formata a variação do dia (duas casas com sinal)
    df["_var_txt"] = df["var_dia"].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    leaf_map = dict(zip(df["Ticker"], df["_var_txt"]))
    
    # Monta 'text' nó-a-nó: setores (parent == "") recebem vazio; folhas recebem a variação
    for tr in fig.data:
        labels, parents, text = list(tr.labels), list(tr.parents), []
        for lbl, par in zip(labels, parents):
            if par == "" or par is None:     # nó de setor (não mostrar métricas)
                text.append("")
            else:                             # folha (papel)
                text.append(leaf_map.get(lbl, ""))
        tr.text = tuple(text)
    
    # Mostra SEMPRE o ticker; e, nas folhas, quebra de linha com % do dia
    fig.update_traces(
        textinfo="label+text",
        texttemplate="<b>%{label}</b><br>%{text}",
        textfont=dict(size=14, color="#111827"),
        selector=dict(type="treemap")
    )
    
    # Afrouxa a regra que escondia rótulos em blocos pequenos
    fig.update_layout(uniformtext_minsize=6, uniformtext_mode="show")
    
    # ---------- HOVER COMPLETO (aparece tanto na visão geral quanto ao clicar no setor) ----------
    winN = 20
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Variação: %{customdata[0]:.2%}<br>"
            f"ΔMME{winN}: %{{customdata[1]:.2%}}<br>"
            f"σ{winN} (ann): %{{customdata[2]:.2%}}<br>"
            f"ρ({BENCH_USED}): %{{customdata[3]:.2f}}<extra></extra>"
        ),
        selector=dict(type="treemap")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Retornos por ativo")

    px_daily = px_df.asfreq("B").ffill()
    
    # se você não usa o benchmark aqui, pode remover essas duas linhas:
    # bench_daily = bench.asfreq("B").ffill()
    # px_daily, bench_daily = px_daily.align(bench_daily, join="inner", axis=0)
    
    if not px_daily.empty:
        dt_last = px_daily.index.max()
    
        # Início do mês corrente
        month_start = dt_last.replace(day=1)
        # Início do ano corrente
        year_start  = dt_last.replace(month=1, day=1)
    
        # datas aproximadas para janelas móveis (1M, 3M, 6M, 9M, 12M, 24M)
        idx = px_daily.index
    
        def _start_n_months_back(n_meses: int) -> pd.Timestamp:
            # ~21 dias úteis por mês
            n_bdays = int(21 * n_meses)
            pos = max(0, len(idx) - n_bdays)
            return idx[pos]
    
        from_1m  = _start_n_months_back(1)
        from_3m  = _start_n_months_back(3)
        from_6m  = _start_n_months_back(6)
        from_9m  = _start_n_months_back(9)
        from_12m = _start_n_months_back(12)
        from_24m = _start_n_months_back(24)
    
        def _period_ret(series: pd.Series, start_dt, end_dt):
            s = series.loc[(series.index >= start_dt) & (series.index <= end_dt)]
            return float(s.iloc[-1] / s.iloc[0] - 1.0) if s.shape[0] >= 2 else np.nan
    
        rows = []
        for tk in px_daily.columns:
            s = px_daily[tk].dropna()
            if s.shape[0] < 2:
                continue
    
            # Retorno no mês (calendário corrente)
            r_mes  = _period_ret(s, month_start, dt_last)
            # Janelas móveis
            r_1m   = _period_ret(s, from_1m,  dt_last)
            r_3m   = _period_ret(s, from_3m,  dt_last)
            r_6m   = _period_ret(s, from_6m,  dt_last)
            r_9m   = _period_ret(s, from_9m,  dt_last)
            r_12m  = _period_ret(s, from_12m, dt_last)
            r_24m  = _period_ret(s, from_24m, dt_last)
            # YTD
            r_ytd  = _period_ret(s, year_start, dt_last)
    
            rows.append(
                [tk, r_mes, r_1m, r_3m, r_6m, r_9m, r_12m, r_24m, r_ytd]
            )
    
        tb_ret = pd.DataFrame(
            rows,
            columns=[
                "Ticker",
                "Mês",      # retorno no mês corrente
                "1M",       # últimos ~21 dias úteis
                "3M",
                "6M",
                "9M",
                "12M",
                "24M",
                "YTD",
            ],
        ).set_index("Ticker")
    
        fmt_ret = tb_ret.applymap(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )
        st.dataframe(fmt_ret, use_container_width=True)
    else:
        st.info("Sem base diária suficiente para o resumo por papel.")


    
    st.markdown("---")
    st.subheader("Performance Setores")
    
    cini, cfim = st.columns(2)
    with cini:
        dt_ini_set = st.date_input("Início", value=px_df.index.min().date())
    with cfim:
        dt_fim_set = st.date_input("Fim", value=px_df.index.max().date())
    
    # Índices EW por setor (usando os mesmos preços do treemap)
    rets_all = px_df.pct_change()
    ew_by_sector = {}
    for setor, tks in SETORES_ATIVOS.items():
        cols = [c for c in px_df.columns if c in tks]
        if len(cols) >= 2:
            ew_by_sector[setor] = (1 + rets_all[cols].mean(axis=1, skipna=True)).cumprod().rename(setor)
    
    if ew_by_sector:
        ew_df = pd.concat(ew_by_sector.values(), axis=1).dropna(how="all")
        # alinhar benchmark
        bench_al = bench.reindex(ew_df.index).dropna()
        common = ew_df.index.intersection(bench_al.index)
        ew_df, bench_al = ew_df.loc[common], bench_al.loc[common]
    
        # filtro de datas
        m = (ew_df.index.date >= dt_ini_set) & (ew_df.index.date <= dt_fim_set)
        ew_df, bench_al = ew_df.loc[m], bench_al.loc[m]
    
        if not ew_df.empty and not bench_al.empty:
            # base 100
            ew100 = 100 * ew_df / ew_df.iloc[0]
            bench100 = (100 * bench_al / bench_al.iloc[0]).rename(BENCH_USED)
            plot_df = pd.concat([ew100, bench100], axis=1).dropna(how="all")
    
            fig_ew = px.line(plot_df, x=plot_df.index, y=plot_df.columns,
                             labels={"value":"Base 100","index":"Data","variable":"Série"})
            fig_ew.update_layout(height=420, legend_title_text="Série")
            st.plotly_chart(fig_ew, use_container_width=True)
        else:
            st.info("Sem interseção de datas suficiente para montar o gráfico.")
    else:
        st.info("Não consegui montar o EW por setor (faltam papéis com histórico).")
            
    # =========================================
    # Matriz de Betas — janela winN (mesma do slider)
    # =========================================
    st.markdown("---")
    st.subheader("Matriz de Betas (β)")
    
    # retornos de todos os papéis + benchmark
    rets_all = px_df.pct_change().dropna(how="all")
    bench_ret = bench.pct_change().rename(BENCH_USED)
    
    rets_all, bench_ret = rets_all.align(bench_ret, join="inner", axis=0)
    rt_win = pd.concat([rets_all, bench_ret], axis=1).dropna(how="all").tail(winN)
    
    # remove ativos com pouquíssimas observações dentro da janela
    min_obs = max(5, int(winN * 0.2))   # ex: com winN=20 pede pelo menos 5 pontos
    good_cols = [c for c in rt_win.columns if rt_win[c].count() >= min_obs]
    rt_win = rt_win[good_cols]
    
    if rt_win.shape[0] < 5 or rt_win.shape[1] < 2:
        st.info("Dados insuficientes na janela selecionada para montar a matriz de betas.")
    else:
        # -------------------------------
        # Garante que o benchmark (ex: BOVA11) fique SEMPRE no início (canto sup. esq.)
        # -------------------------------
        cols_all = list(rt_win.columns)

        if BENCH_USED in cols_all:
            bench_label = BENCH_USED
        else:
            # fallback: procura algo que contenha "BOVA11" no nome
            bench_label = None
            for c in cols_all:
                if "BOVA11" in c:
                    bench_label = c
                    break

        if bench_label is not None and bench_label in cols_all:
            cols = [bench_label] + [c for c in cols_all if c != bench_label]
            rt_win = rt_win[cols]   # reordena dataframe
        else:
            cols = cols_all

        # matriz na MESMA ordem das colunas (benchmark já em primeiro)
        B = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
        # β_{i→j} = cov(R_i, R_j) / var(R_j)
        for i in cols:
            yi_full = rt_win[i]
            for j in cols:
                xj_full = rt_win[j]
                y, x = yi_full.align(xj_full, join="inner")
                if len(y) >= 5:
                    x_mean = x.mean()
                    y_mean = y.mean()
                    cov_xy = ((x - x_mean) * (y - y_mean)).sum() / (len(x) - 1)
                    var_x  = ((x - x_mean) ** 2).sum() / (len(x) - 1)
                    if var_x != 0 and not np.isnan(var_x):
                        B.loc[i, j] = float(cov_xy / var_x)
                    else:
                        B.loc[i, j] = np.nan
                else:
                    B.loc[i, j] = np.nan
    
        fig_beta = px.imshow(
            B.astype(float),
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            origin="upper",
            aspect="auto",
            title="Selecionar período acima",
        )
        fig_beta.update_layout(height=600, coloraxis_colorbar=dict(title="β i→j"))
        st.plotly_chart(fig_beta, use_container_width=True)



            
        
    # =============================
    # Plot RRG
    # =============================
    
    with tmap["Rotação"]:
        st.subheader("Rotação Setorial Equal Weight")
    
        # ---- Parâmetros ----
        trail_len = int(trail) if "trail" in locals() else 6
    
        # ---- (1) Monta últimos pontos (um por setor) a partir de sector_curves ----
        last_rows = []
        for setor, rr in sector_curves.items():
            if rr is None or rr.empty:
                continue
            rrv = rr.dropna(subset=["RS_Ratio", "RS_Momentum"])
            if rrv.empty:
                continue
    
            last = rrv.tail(1).iloc[0]
            rr_val = float(last["RS_Ratio"])
            rm_val = float(last["RS_Momentum"])
    
            # evite np.select sobre escalares -> use if/elif
            if rr_val >= 100 and rm_val >= 100:
                quad = "Leading"
            elif rr_val >= 100 and rm_val < 100:
                quad = "Weakening"
            elif rr_val < 100 and rm_val < 100:
                quad = "Lagging"
            else:
                # rr_val < 100 and rm_val >= 100
                quad = "Improving"
    
            last_rows.append({
                "Setor": setor,
                "Date": rrv.index[-1],             # Timestamp (hashable)
                "RS_Ratio": rr_val,                # float
                "RS_Momentum": rm_val,             # float
                "Quadrante": str(quad),            # garantir tipo string
            })
    
        last_pts_df = pd.DataFrame(last_rows, columns=["Setor","Date","RS_Ratio","RS_Momentum","Quadrante"])
        if last_pts_df.empty:
            st.warning("Sem pontos válidos para montar o RRG nesta aba.")
            st.stop()
    
        # ---- (2) Scatter base com os últimos pontos ----
        fig = px.scatter(
            last_pts_df,
            x="RS_Ratio",
            y="RS_Momentum",
            color="Quadrante",
            hover_name="Setor",
            hover_data={"RS_Ratio":":.2f","RS_Momentum":":.2f","Quadrante":True},
            size=[12]*len(last_pts_df),
            opacity=0.95,
        )
        fig.add_hline(y=100, line_dash="dash", line_width=1)
        fig.add_vline(x=100, line_dash="dash", line_width=1)
    
        # Cores de fundo dos quadrantes
        lead_color = "#22c55e"   # green-500
        weak_color = "#f59e0b"   # amber-500
        lagg_color = "#ef4444"   # red-500
        impr_color = "#60a5fa"   # blue-400
    
        # ---- (3) Trilhas suavizadas + marcadores nos nós ----
        def _smooth_and_densify(rr_df: pd.DataFrame, trail_len: int, alpha: float = 0.55, pps: int = 6):
            """
            rr_df: DataFrame com ['RS_Ratio','RS_Momentum'] indexado por datas.
            trail_len: quantos pontos finais usar (nós originais).
            alpha: EWM para suavizar (0..1).
            pps: pontos-intermediários por segmento (densidade).
            """
            g = rr_df.dropna(subset=["RS_Ratio", "RS_Momentum"]).tail(trail_len + 1).copy()
            if g.empty or g.shape[0] < 2:
                return None
        
            # suavização (evita “cotovelos”)
            g["x"] = g["RS_Ratio"].ewm(alpha=alpha, adjust=False).mean()
            g["y"] = g["RS_Momentum"].ewm(alpha=alpha, adjust=False).mean()
        
            # nós (pontos originais após suavização)
            knots_x = g["x"].to_numpy()
            knots_y = g["y"].to_numpy()
            knots_dt = [pd.to_datetime(d).date() for d in g.index]
        
            # densificação para a linha ficar contínua
            n = len(knots_x)
            xi = np.linspace(0, n - 1, (n - 1) * pps + 1)
            xs = np.interp(xi, np.arange(n), knots_x)
            ys = np.interp(xi, np.arange(n), knots_y)
        
            return xs, ys, knots_x, knots_y, knots_dt
        
        xs_all = [last_pts_df["RS_Ratio"].astype(float)]
        ys_all = [last_pts_df["RS_Momentum"].astype(float)]
        
        for setor, rr in sector_curves.items():
            if rr is None or rr.empty:
                continue
        
            out = _smooth_and_densify(rr, trail_len=trail_len, alpha=0.55, pps=6)
            if out is None:
                continue
            xs, ys, kx, ky, kdt = out
        
            # p/ ranges (robustos)
            xs_all.append(pd.Series(xs))
            ys_all.append(pd.Series(ys))
        
            # linha suavizada
            fig.add_scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=2),
                name=f"{setor} (trail)", showlegend=False,
                hoverinfo="skip", line_shape="spline", opacity=0.7
            )
        
            # marcadores nos nós (ex.: semana a semana)
            fig.add_scatter(
                x=kx, y=ky, mode="markers",
                marker=dict(size=5, symbol="circle", line=dict(width=0.5, color="#111")),
                name=f"{setor} (nós)", showlegend=False,
                hovertext=[f"{setor} — {d}" for d in kdt], hoverinfo="text"
            )
        
            # ponto final destacado
            fig.add_scatter(
                x=[kx[-1]], y=[ky[-1]], mode="markers",
                marker=dict(size=10, symbol="circle", line=dict(width=1, color="#111")),
                name=setor, showlegend=False,
                hovertext=f"{setor} — {kdt[-1]}", hoverinfo="text"
            )
        
        xs_all = pd.concat(xs_all) if xs_all else pd.Series(dtype=float)
        ys_all = pd.concat(ys_all) if ys_all else pd.Series(dtype=float)
        

    
        # ---- (4) Eixos centrados em 100 com faixa robusta ----
        if not xs_all.empty and not ys_all.empty:
            devx = (xs_all - 100).abs()
            devy = (ys_all - 100).abs()
        
            # usa P95 (robusto a outliers), com pisos mínimos
            dx = float(np.quantile(devx, 0.95))
            dy = float(np.quantile(devy, 0.95))
            dx = max(dx, 5.0)
            dy = max(dy, 5.0)
        
            padx = dx * 1.15
            pady = dy * 1.15
        
            # Depois de calcular padx/pady e fixar os ranges:
            fig.update_xaxes(range=[100 - padx, 100 + padx])
            fig.update_yaxes(range=[100 - pady, 100 + pady])
            
            x0, x1 = 100 - padx, 100 + padx
            y0, y1 = 100 - pady, 100 + pady
            
            # Limpa shapes antigos (opcional, se já houver)
            fig.layout.shapes = ()
            
            # Quadrantes: preencher TODO o eixo (usando coordenadas de dados)
            fig.add_shape(type="rect", x0=100, x1=x1, y0=100, y1=y1,
                          fillcolor=lead_color, opacity=0.12, line_width=0,
                          xref="x", yref="y", layer="below")
            fig.add_shape(type="rect", x0=100, x1=x1, y0=y0,  y1=100,
                          fillcolor=weak_color, opacity=0.12, line_width=0,
                          xref="x", yref="y", layer="below")
            fig.add_shape(type="rect", x0=x0,  x1=100, y0=y0,  y1=100,
                          fillcolor=lagg_color, opacity=0.12, line_width=0,
                          xref="x", yref="y", layer="below")
            fig.add_shape(type="rect", x0=x0,  x1=100, y0=100, y1=y1,
                          fillcolor=impr_color, opacity=0.12, line_width=0,
                          xref="x", yref="y", layer="below")

        
            fig.add_annotation(x=100 + dx * 0.55, y=100 + dy * 0.55, text="LEADING",   showarrow=False, font=dict(size=12, color="#6b7280"))
            fig.add_annotation(x=100 + dx * 0.55, y=100 - dy * 0.55, text="WEAKENING", showarrow=False, font=dict(size=12, color="#6b7280"))
            fig.add_annotation(x=100 - dx * 0.55, y=100 - dy * 0.55, text="LAGGING",   showarrow=False, font=dict(size=12, color="#6b7280"))
            fig.add_annotation(x=100 - dx * 0.55, y=100 + dy * 0.55, text="IMPROVING", showarrow=False, font=dict(size=12, color="#6b7280"))

        # ---- (5) Layout ----
        fig.update_layout(
            height=650,
            margin=dict(l=8, r=8, t=60, b=10),
            xaxis_title=f"RS-Ratio (força relativa vs {bench_name})",
            yaxis_title="RS-Momentum (aceleração da força relativa)"
            # , cliponaxis=True  # ative se quiser forçar o corte dentro dos eixos
        )
        st.plotly_chart(fig, use_container_width=True)
    
        
    


# --- Funções auxiliares específicas (fora de abas) ---


def rebase_100(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return (100 * s / s.iloc[0]) if not s.empty else s

def calc_dd(idx: pd.Series) -> float:
    if idx.empty:
        return float("nan")
    peak = idx.cummax()
    return float((idx/peak - 1.0).min())

def ann_factor(freq_key: str) -> int:
    return 252 if freq_key == "D" else 52

def _is_us_asset(sym: str) -> bool:
    return sym in {"^GSPC", "TLT", "DBC"}  # acrescente outros se quiser

def _is_fx(sym: str) -> bool:
    return sym.upper() in {"USDBRL=X"}     # acrescente pares FX se usar

def _yf_symbol(sym: str) -> str:
    """
    Converte ticker “cru” para o formato esperado no yfinance.
    Regras:
      - US/FX: retorna como está
      - B3: garante sufixo '.SA'
    """
    if _is_us_asset(sym) or _is_fx(sym):
        return sym
    return sym if sym.endswith(".SA") else f"{sym}.SA"

# ================================
# Helpers específicos da aba Classes
# ================================

def _period_pct(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1.0) if s.shape[0] >= 2 else float("nan")

def _rotation_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Tabela de rotação: retornos em 1m,3m,6m,12m,YTD,36m + ranks + rank médio."""
    if prices is None or prices.dropna(how="all").empty:
        return pd.DataFrame()
    # frequência “de calendário” aproximada via últimos N pontos (diário/semanais)
    is_daily = (prices.index.freqstr or "").startswith("B") or (prices.index.inferred_freq in ("B","D", "C"))
    N_1m, N_3m, N_6m, N_12m, N_36m = (21, 63, 126, 252, 756) if is_daily else (4, 13, 26, 52, 156)

    rows = []
    last = prices.index.max()
    ytd_start = last.replace(month=1, day=1)

    for col in prices.columns:
        s = prices[col].dropna()
        if s.shape[0] < 5: 
            continue
        def lastNret(N):
            ss = s.tail(N)
            return _period_pct(ss)
        r_1m  = lastNret(N_1m)
        r_3m  = lastNret(N_3m)
        r_6m  = lastNret(N_6m)
        r_12m = lastNret(N_12m)
        r_36m = lastNret(N_36m)
        # YTD
        s_ytd = s.loc[(s.index >= ytd_start) & (s.index <= last)]
        r_ytd = _period_pct(s_ytd) if s_ytd.shape[0] >= 2 else np.nan
        rows.append([col, r_1m, r_3m, r_6m, r_12m, r_ytd, r_36m])

    tb = pd.DataFrame(rows, columns=["Ativo","1m","3m","6m","12m","YTD","36m"]).set_index("Ativo")
    if tb.empty: 
        return tb

    # ranks por coluna de retorno (desc = melhor retorno)
    for c in ["1m","3m","6m","12m","YTD","36m"]:
        tb[f"Rank_{c}"] = tb[c].rank(ascending=False, method="min")
   
    tb = tb.sort_values("12m", ascending=False)
    return tb

def _fmt_pct(x): 
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


# --- Aba SETOR ---
with tmap["Setor"]:
    st.subheader("Análise por Setor")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        setor_sel = st.selectbox("Escolha um setor para análise detalhada:", list(SETORES_ATIVOS.keys()))
    with col_b:
        freq_setor = st.radio("Frequência", ["Diária","Semanal"], index=0, horizontal=True, key="freq_setor")
    freq_key_setor = "D" if freq_setor == "Diária" else "W"

    default_start = date.today() - timedelta(days=365)
    c1, c2 = st.columns(2)
    with c1:
        dt_ini = st.date_input("Data inicial", value=default_start)
    with c2:
        dt_fim = st.date_input("Data final", value=date.today())

    st.caption(f"Papéis (equal-weight): {', '.join(SETORES_ATIVOS[setor_sel])}")

    # 1) Preços do setor (EW)
    tickers = SETORES_ATIVOS[setor_sel]
    px_list_raw = [_get_range_series(tk, dt_ini, dt_fim, freq_key_setor) for tk in tickers]
    px_df = pd.concat([s.rename(tk) for s, tk in zip(px_list_raw, tickers) if s is not None and not s.dropna().empty], axis=1)
    px_df = px_df.dropna(how="all")

    min_obs = 30 if freq_key_setor == "D" else 15
    valid_cols = [c for c in px_df.columns if px_df[c].dropna().shape[0] >= min_obs]
    if len(valid_cols) < len(tickers):
        try:
            faltantes = [t for t in tickers if t not in valid_cols]
            logs.append(f"Aviso: {setor_sel} — removidos por poucas barras: {', '.join(faltantes)}")
        except Exception:
            pass
    px_df = px_df[valid_cols]

    # 2) Benchmark
    bench = _get_range_series(BENCHMARK, dt_ini, dt_fim, freq_key_setor)

    if px_df.empty or px_df.shape[1] == 0 or bench.empty:
        st.warning("Sem dados suficientes para o período/frequência escolhidos.")
    else:
        # Índice EW do setor
        rets = px_df.pct_change()
        ew = (1 + rets.mean(axis=1, skipna=True)).cumprod().rename("EW_Setor").dropna()

        # Alinha com benchmark
        bench = bench.reindex(ew.index).dropna()
        common = ew.index.intersection(bench.index)
        ew = ew.loc[common]
        bench = bench.loc[common]

        # Helpers
        def _rebase100(s: pd.Series) -> pd.Series:
            s = s.dropna()
            return 100 * s / s.iloc[0] if not s.empty else s

        def _period_return(s: pd.Series) -> float:
            s = s.dropna()
            return float(s.iloc[-1] / s.iloc[0] - 1.0) if s.shape[0] >= 2 else float("nan")

        # Cards
        ret_set = _period_return(ew)
        ret_ben = _period_return(bench)
        alpha   = ret_set - ret_ben if (not np.isnan(ret_set) and not np.isnan(ret_ben)) else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric("Retorno Setor (EW)", f"{ret_set*100:.2f}%")
        c2.metric("Retorno BOVA11",     f"{ret_ben*100:.2f}%")
        c3.metric("Alfa (Setor − BOVA11)", f"{alpha*100:.2f}%")

        # Base 100
        chart_df = pd.DataFrame({"Setor (EW)": _rebase100(ew), BENCHMARK: _rebase100(bench)}).dropna()
        fig_set = px.line(chart_df, x=chart_df.index, y=chart_df.columns,
                          labels={"value": "Base 100", "index": "time", "variable": "Série"})
        fig_set.update_layout(height=420, legend_title_text="Série")

        # vlines nas trocas de quadrante
        try:
            _rrg_set = (
                rrg.loc[rrg["Setor"].eq(setor_sel), ["Date", "RS_Ratio", "RS_Momentum"]]
                   .dropna()
                   .sort_values("Date")
                   .set_index("Date")
            )
            if not _rrg_set.empty:
                rsr = _rrg_set["RS_Ratio"].astype(float)
                rsm = _rrg_set["RS_Momentum"].astype(float)
                cross_ratio = (rsr > 100) != (rsr.shift(1) > 100)
                cross_mom   = (rsm > 100) != (rsm.shift(1) > 100)
                cross_dates = _rrg_set.index[(cross_ratio | cross_mom)]
                for d in cross_dates:
                    fig_set.add_vline(x=d, line_color="#f59e0b", line_dash="dot", opacity=0.75)
        except Exception:
            pass

        st.plotly_chart(fig_set, use_container_width=True)

        # Retorno por papel
        px_al = px_df.reindex(index=common)
        per_asset_returns = {}
        for col in px_al.columns:
            s = px_al[col].dropna()
            if s.empty:
                continue
            s = s.loc[s.index.min():s.index.max()]
            if s.shape[0] >= 2:
                per_asset_returns[col] = _period_return(s)

        if per_asset_returns:
            ser_ret = pd.Series(per_asset_returns).sort_values(ascending=False)
            n = ser_ret.shape[0]
            w = 1.0 / n
            ser_contrib = ser_ret * w

            colL, colR = st.columns(2)
            with colL:
                fig_bar_ret = px.bar(ser_ret.rename("Retorno"),
                                     labels={"index": "Ticker", "value": "Retorno (%)"})
                fig_bar_ret.update_traces(hovertemplate="%{x}: %{y:.2%}")
                fig_bar_ret.update_layout(height=420, yaxis_tickformat=".0%")
                st.subheader("Retorno por papel")
                st.plotly_chart(fig_bar_ret, use_container_width=True)

            with colR:
                fig_bar_contrib = px.bar(ser_contrib.rename("Contribuição"),
                                         labels={"index": "Ticker", "value": "Contribuição (%)"})
                fig_bar_contrib.update_traces(hovertemplate="%{x}: %{y:.2%}")
                fig_bar_contrib.update_layout(height=420, yaxis_tickformat=".0%")
                st.subheader("Contribuição por papel (EW)")
                st.plotly_chart(fig_bar_contrib, use_container_width=True)

            st.caption(
                f"Período: {common.min().date()} → {common.max().date()} · "
                f"Papéis considerados: {n} · Contribuição = peso igual × retorno do papel."
            )

            # Papel vs BOVA11
            st.markdown("---")
            st.subheader("Papel vs BOVA11")

            ticker_escolhido = st.selectbox("Escolha um papel do setor", options=tickers, index=0)
            tk_ser  = _get_range_series(ticker_escolhido, dt_ini, dt_fim, freq_key_setor)
            bench_t = _get_range_series(BENCHMARK,       dt_ini, dt_fim, freq_key_setor)
            
            if tk_ser is None or tk_ser.dropna().empty:
                st.warning(f"Sem histórico suficiente para {ticker_escolhido} no período selecionado.")
            elif bench_t is None or bench_t.dropna().empty:
                st.warning("Não foi possível obter o benchmark no período selecionado.")
            else:
                # garante interseção de datas
                tk_ser, bench_t = tk_ser.align(bench_t, join="inner")

                if tk_ser.empty or bench_t.empty:
                    st.warning("Interseção de datas vazia após alinhamento (verifique o período/frequência).")
                else:
                    ret_tk = _period_return(tk_ser)
                    ret_bv = _period_return(bench_t)
                    alfa_tk = ret_tk - ret_bv if (not np.isnan(ret_tk) and not np.isnan(ret_bv)) else float("nan")

                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Retorno {ticker_escolhido}", f"{ret_tk*100:.2f}%")
                    c2.metric("Retorno BOVA11", f"{ret_bv*100:.2f}%")
                    c3.metric("Alfa (Papel − BOVA11)", f"{alfa_tk*100:.2f}%")

                    base100_df = pd.DataFrame({
                        f"{ticker_escolhido}": _rebase100(tk_ser),
                        BENCHMARK: _rebase100(bench_t),
                    }).dropna()

                    fig_papel = px.line(
                        base100_df, x=base100_df.index, y=base100_df.columns,
                        labels={"value": "Base 100", "index": "Data", "variable": "Série"},
                    )
                    fig_papel.update_layout(height=420, legend_title_text="Série")
                    st.plotly_chart(fig_papel, use_container_width=True)
        else:
            st.info("Não foi possível calcular retornos por papel para o período selecionado.")

        # Série escolhida
        st.markdown("---")
        escolha_serie = st.radio(
            "Série para as análises abaixo:",
            [f"{setor_sel} (EW)", "Papel selecionado"],
            index=0, horizontal=True, key="who_obj_setor"
        )
        if escolha_serie == "Papel selecionado" and 'ticker_escolhido' in locals():
            serie_obj = px_df[ticker_escolhido].dropna()
            nome_obj  = ticker_escolhido
        else:
            serie_obj = ew.copy()
            nome_obj  = "Setor (EW)"

        # Diferença acumulada + Retorno acumulado móvel
        st.subheader("Diferença de retorno acumulada (obj − BOVA11)  +  Retorno acumulado móvel")
        base_acc = st.radio("Base de acumulação", ["Por barra", "Por dia útil"],
                            horizontal=True, key="base_acc_setor")
        win_acc = st.slider("Janela do retorno acumulado móvel",
                            min_value=5, max_value=120, value=22, step=1, key="win_acc_setor")

        obj_work   = serie_obj.copy()
        bench_work = bench.copy()
        if base_acc == "Por dia útil":
            obj_work   = obj_work.asfreq("B").ffill()
            bench_work = bench_work.asfreq("B").ffill()

        ret_o, ret_b = obj_work.pct_change().dropna(), bench_work.pct_change().dropna()
        ret_o, ret_b = ret_o.align(ret_b, join="inner")

        diff_acum = (ret_o - ret_b).cumsum().rename("Diferença acumulada (obj − BOVA11)")
        roll_acc = (
            (1.0 + ret_o)
            .rolling(win_acc, min_periods=win_acc)
            .apply(lambda x: np.prod(x) - 1.0, raw=True)
            .rename(f"Retorno acumulado {win_acc}")
        )

        plot_df = pd.concat([diff_acum, roll_acc], axis=1).dropna(how="all")

        # >>> Aqui entra a cor da linha do "Retorno acumulado {win_acc}"
        fig_diff = px.line(
            plot_df,
            x=plot_df.index,
            y=plot_df.columns,
            labels={"value": "Retorno (%)", "index": "Data", "variable": "Série"},
            color_discrete_map={
                f"Retorno acumulado {win_acc}": "#22c55e",                 # destaque (verde)
                "Diferença acumulada (obj − BOVA11)": "#0ea5e9",           # azul
            },
        )
        fig_diff.update_layout(height=360, yaxis_tickformat=".2%")
        fig_diff.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        st.plotly_chart(fig_diff, use_container_width=True)

        # Z-score de distância à média
        st.markdown("---")
        st.subheader(f"Z-score de distância à média — {nome_obj}")

        win_z = st.slider("Período da média (janelas em barras/dias úteis conforme a base acima)",
                          min_value=10, max_value=120, value=20, step=1, key="win_z_setor")

        z_base = obj_work.copy()
        ma = z_base.rolling(win_z, min_periods=win_z//2).mean()
        sd = z_base.rolling(win_z, min_periods=win_z//2).std(ddof=1)
        zscore_series = ((z_base - ma) / sd).rename("Z-score").dropna()

        fig_z = px.line(zscore_series, x=zscore_series.index, y="Z-score",
                        labels={"index": "Data", "Z-score": "Desvio-padrão"})
        fig_z.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_z.add_hline(y= 2, line_dash="dot",  line_color="#ef4444")
        fig_z.add_hline(y=-2, line_dash="dot",  line_color="#ef4444")
        fig_z.update_layout(height=320)
        st.plotly_chart(fig_z, use_container_width=True)


# ================================
# BLOCO TRADINGVIEW – CLASSES
# ================================
from dataclasses import dataclass
from enum import Enum
import logging
import os
import re
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

# === Substitua as strings abaixo pelo seu login do TradingView antes de rodar ===
TV_USERNAME_CONFIG = "riosasset"
TV_PASSWORD_CONFIG = "atgdm12newom9365%*"

try:  # tvdatafeed é opcional; tratamos ausência com mensagem amigável.
    from tvdatafeed import Interval as TvInterval  # type: ignore[import]
    from tvdatafeed import TvDatafeed  # type: ignore[import]
except Exception:
    try:
        from tvDatafeed import Interval as TvInterval  # type: ignore[import]
        from tvDatafeed import TvDatafeed  # type: ignore[import]
    except Exception:  # pragma: no cover - depende do ambiente do usuário
        TvInterval = None
        TvDatafeed = None  # type: ignore[assignment]


class Interval(Enum):
    """Pequena enum fallback caso o tvdatafeed não esteja disponível."""

    in_daily = "1d"
    in_weekly = "1w"


if TvInterval is not None:  # respeita a implementação oficial quando existir
    Interval = TvInterval  # type: ignore[misc]


_tv: TvDatafeed | None = None  # singleton
_TV_CREDENTIALS: dict[str, str | None] = {"user": None, "password": None}


def _empty_series(name: str | None = None) -> pd.Series:
    s = pd.Series(dtype=float)
    if name:
        s.name = name
    return s


def configure_tv_credentials(user: str | None, password: str | None) -> None:
    """Configura (e invalida se preciso) o singleton do tvdatafeed."""
    global _tv, _TV_CREDENTIALS
    user = (user or "").strip()
    password = (password or "").strip()
    if not user or not password:
        raise ValueError("Informe usuário e senha do TradingView para continuar.")
    if (
        _TV_CREDENTIALS.get("user") != user
        or _TV_CREDENTIALS.get("password") != password
    ):
        _tv = None
    _TV_CREDENTIALS = {"user": user, "password": password}


# tenta carregar as credenciais hardcoded logo no import; se não forem ajustadas, ignoramos.
try:
    configure_tv_credentials(TV_USERNAME_CONFIG, TV_PASSWORD_CONFIG)
except ValueError:
    pass


def _resolve_tv_credentials(
    user: str | None = None, password: str | None = None
) -> tuple[str, str]:
    env_user = os.getenv("TV_USERNAME") or os.getenv("TRADINGVIEW_USERNAME")
    env_pass = os.getenv("TV_PASSWORD") or os.getenv("TRADINGVIEW_PASSWORD")
    hardcoded_user = TV_USERNAME_CONFIG.strip()
    hardcoded_pass = TV_PASSWORD_CONFIG.strip()
    if "COLOQUE" in hardcoded_user.upper():
        hardcoded_user = ""
    if "COLOQUE" in hardcoded_pass.upper():
        hardcoded_pass = ""

    resolved_user = (
        user
        or _TV_CREDENTIALS.get("user")
        or env_user
        or hardcoded_user
        or ""
    ).strip()
    resolved_pass = (
        password
        or _TV_CREDENTIALS.get("password")
        or env_pass
        or hardcoded_pass
        or ""
    ).strip()
    if not resolved_user or not resolved_pass:
        raise RuntimeError(
            "Credenciais do TradingView são obrigatórias. "
            "Edite as constantes TV_USERNAME_CONFIG e TV_PASSWORD_CONFIG "
            "ou defina as variáveis de ambiente TV_USERNAME/TV_PASSWORD."
        )
    return resolved_user, resolved_pass


def _tv_client(user: str | None = None, password: str | None = None) -> TvDatafeed:
    if TvDatafeed is None:
        raise RuntimeError("tvdatafeed não está instalado; instale-o para usar TradingView.")
    resolved_user, resolved_pass = _resolve_tv_credentials(user, password)
    global _tv
    if _tv is None:
        _tv = TvDatafeed(username=resolved_user, password=resolved_pass)
    return _tv


def _tv_interval(freq_key: str) -> Interval:
    fk = (freq_key or "D").upper()
    if fk == "W":
        return Interval.in_weekly
    return Interval.in_daily


def _get_price_series_tv(symbol: str, exchange: str, bars: int, freq_key: str) -> pd.Series:
    """
    Puxa OHLC do TradingView e retorna Close como Series.
    symbol: 'USOIL', 'GOLD', 'US10Y', 'DXY', 'ZN1!' etc.
    exchange: 'TVC', 'CBOT', 'COMEX', 'NYMEX' etc.
    """
    try:
        tv = _tv_client()
    except RuntimeError as exc:
        logging.warning("TVDatafeed indisponível: %s", exc)
        return _empty_series(f"{exchange}:{symbol}")

    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=_tv_interval(freq_key),
            n_bars=int(bars),
        )
        if df is None or df.empty:
            return _empty_series(f"{exchange}:{symbol}")
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = f"{exchange}:{symbol}"
        return s.sort_index()
    except Exception as exc:  # pragma: no cover - depende de API externa
        logging.error("Falha ao buscar %s:%s no TradingView: %s", exchange, symbol, exc)
        return _empty_series(f"{exchange}:{symbol}")


def _get_price_series_unified(meta: dict, bars: int, freq_key: str) -> pd.Series:
    """
    meta pode conter:
      - 'source' == 'tv' e 'tv_symbol' + 'tv_exchange'
    """
    if meta.get("source") == "tv":
        sym = meta.get("tv_symbol", "")
        ex = meta.get("tv_exchange", "")
        s_tv = _get_price_series_tv(sym, ex, bars=bars, freq_key=freq_key)
        if s_tv is not None and not s_tv.dropna().empty:
            return s_tv

    logging.warning("Meta sem configuração de TradingView: %s", meta)
    return _empty_series()


def bars_for_freq_tv(freq_key: str, lookback: str | int) -> int:
    """Calcula quantidade aproximada de barras dadas frequência e janela."""
    fk = (freq_key or "D").upper()
    if isinstance(lookback, (int, float)):
        candidate = int(lookback)
    else:
        normalized = str(lookback).strip().upper().replace(" ", "")
        match = re.fullmatch(r"(\d+)([MY])", normalized)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            days = value * (21 if unit == "M" else 252)
        else:
            days = 252
        candidate = days if fk == "D" else max(1, days // 5)
    return max(10, min(candidate, 5000))


CLASSES_MAP = {
    # ============================
    # Indíces
    # ============================
    "BOVA11 (BR Ações Large)": {
        "source": "tv", "tv_symbol": "BOVA11",
        "tv_exchange": "BMFBOVESPA", "ccy": "BRL"
    },
    "SMAL11 (BR Ações Small)": {
        "source": "tv", "tv_symbol": "SMAL11",
        "tv_exchange": "BMFBOVESPA", "ccy": "BRL"
    },
    "IRFM11 (RF Prefixada)": {
        "source": "tv", "tv_symbol": "IRFM11",
        "tv_exchange": "BMFBOVESPA", "ccy": "BRL"
    },
    "IMAB11 (RF IPCA+)": {
        "source": "tv", "tv_symbol": "IMAB11",
        "tv_exchange": "BMFBOVESPA", "ccy": "BRL"
    },
    "IFIX (FII)": {
        "source": "tv", "tv_symbol": "IFIX",
        "tv_exchange": "BMFBOVESPA", "ccy": "BRL"
    },   
    "SP&500": {
        "source": "tv", "tv_symbol": "SPX",
        "tv_exchange": "TVC", "ccy": "USD"
    },
    "RUSSELL": {
        "source": "tv", "tv_symbol": "RUT",
        "tv_exchange": "TVC", "ccy": "USD"
    },
    "Dow Jones": {
        "source": "tv", "tv_symbol": "DJI",
        "tv_exchange": "TVC", "ccy": "USD"
    },    
    "CDI": {
        "source": "tv", "tv_symbol": "LFTS11",
        "tv_exchange": "BMFBOVESPA", "ccy": "BRL"
    },

    # ============================
    # COMMODITIES / SPOT — TradingView
    # ============================
    "Petróleo UK (USD) - Global": {
        "source": "tv", "tv_symbol": "UKOIL",
        "tv_exchange": "TVC", "ccy": "USD"
    },
    "DXY - Dollar Index": {
        "source": "tv", "tv_symbol": "DXY",
        "tv_exchange": "TVC", "ccy": "USD"
    },

    # ============================
    # FUTUROS — TradingView (continuous)
    # ============================
    "AW1! - BCOM": {
        "source": "tv", "tv_symbol": "AW1!",
        "tv_exchange": "CBOT", "ccy": "USD"
    },
    "CL1! - Texas Crude Oil": {
        "source": "tv", "tv_symbol": "CL1!",
        "tv_exchange": "NYMEX", "ccy": "USD"
    },
    "GC1! - Ouro": {
        "source": "tv", "tv_symbol": "GC1!",
        "tv_exchange": "COMEX", "ccy": "USD"
    },
    "HG1! - Copper": {
        "source": "tv", "tv_symbol": "HG1!",
        "tv_exchange": "COMEX", "ccy": "USD"
    },
    "SI1! - Silver": {
        "source": "tv", "tv_symbol": "SI1!",
        "tv_exchange": "COMEX", "ccy": "USD"
    },
    "TIO1! - Iron Ore": {
        "source": "tv", "tv_symbol": "TIO1!",
        "tv_exchange": "SGX", "ccy": "USD"
    },
    
    "ZS1! - Soybean": {
        "source": "tv", "tv_symbol": "ZS1!",
        "tv_exchange": "CBOT", "ccy": "USD"
    },
    "ZC1! - Corn": {
        "source": "tv", "tv_symbol": "ZC1!",
        "tv_exchange": "CBOT", "ccy": "USD"
    },

    # ============================
    # ÍNDICES DE VOL / RISCO
    # ============================
    "VIX": {
        "source": "tv", "tv_symbol": "VIX",
        "tv_exchange": "CBOE", "ccy": "%"
    },
    "MOVE": {
        "source": "tv", "tv_symbol": "MOVE",
        "tv_exchange": "TVC", "ccy": "%"
    },

    # ============================
    # YIELDS EUA (curva US)
    # ============================
    "US 3M Yield": {
        "source": "tv", "tv_symbol": "US03MY",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 3/12
    },
    "US 4M Yield": {
        "source": "tv", "tv_symbol": "US04MY",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 4/12
    },
    "US 6M Yield": {
        "source": "tv", "tv_symbol": "US06MY",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 6/12
    },
    "US 1Y Yield": {
        "source": "tv", "tv_symbol": "US01Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 1.0
    },
    "US 2Y Yield": {
        "source": "tv", "tv_symbol": "US02Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 2.0
    },
    "US 3Y Yield": {
        "source": "tv", "tv_symbol": "US03Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 3.0
    },
    "US 4Y Yield": {
        "source": "tv", "tv_symbol": "US04Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 4.0
    },
    "US 5Y Yield": {
        "source": "tv", "tv_symbol": "US05Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 5.0
    },
    "US 6Y Yield": {
        "source": "tv", "tv_symbol": "US06Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 6.0
    },
    "US 7Y Yield": {
        "source": "tv", "tv_symbol": "US07Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 7.0
    },
    "US 8Y Yield": {
        "source": "tv", "tv_symbol": "US08Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 8.0
    },
    "US 9Y Yield": {
        "source": "tv", "tv_symbol": "US09Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 9.0
    },
    "US 10Y Yield": {
        "source": "tv", "tv_symbol": "US10Y",
        "tv_exchange": "TVC", "ccy": "%", "tenor_yrs": 10.0
    },

    # ============================
    # Futuros de Juros BR (curva BR)
    # ============================
    "BR3M": {
        "source": "tv", "tv_symbol": "DI1H2026",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 3/12
    },
    "BR4M": {
        "source": "tv", "tv_symbol": "DI1J2026",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 4/12
    },
    "BR6M": {
        "source": "tv", "tv_symbol": "DI1M2026",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 6/12
    },

    "BR1Y": {
        "source": "tv", "tv_symbol": "DI1F2027",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 1.0
    },
    "BR2Y": {
        "source": "tv", "tv_symbol": "DI1F2028",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 2.0
    },
    "BR3Y": {
        "source": "tv", "tv_symbol": "DI1F2029",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 3.0
    },
    "BR4Y": {
        "source": "tv", "tv_symbol": "DI1F2030",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 4.0
    },
    "BR5Y": {
        "source": "tv", "tv_symbol": "DI1F2031",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 5.0
    },
    "BR6Y": {
        "source": "tv", "tv_symbol": "DI1F2032",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 6.0
    },
    "BR7Y": {
        "source": "tv", "tv_symbol": "DI1F2033",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 7.0
    },
    "BR8Y": {
        "source": "tv", "tv_symbol": "DI1F2034",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 8.0
    },
    "BR9Y": {
        "source": "tv", "tv_symbol": "DI1F2035",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 9.0
    },
    "BR10Y": {
        "source": "tv", "tv_symbol": "DI1F2036",
        "tv_exchange": "BMFBOVESPA", "ccy": "%", "tenor_yrs": 10.0
    },

    # ============================
    # FX — TradingView
    # ============================
    "USDGBP": {
        "source": "tv", "tv_symbol": "USDGBP",
        "tv_exchange": "FXCM", "ccy": "FX"
    },
    "BRLUSD": {
        "source": "tv", "tv_symbol": "BRLUSD",
        "tv_exchange": "FXCM", "ccy": "FX"
    },
    "USDJPY": {
        "source": "tv", "tv_symbol": "USDJPY",
        "tv_exchange": "FXCM", "ccy": "FX"
    },
    "EURUSD": {
        "source": "tv", "tv_symbol": "EURUSD",
        "tv_exchange": "FXCM", "ccy": "FX"
    },
    "USDMXN": {
        "source": "tv", "tv_symbol": "USDMXN",
        "tv_exchange": "FXCM", "ccy": "FX"
    },
    "USDCHF": {
        "source": "tv", "tv_symbol": "USDCHF",
        "tv_exchange": "FXCM", "ccy": "FX"
    },
    "USDCNY": {
        "source": "tv", "tv_symbol": "USDCNY",
        "tv_exchange": "ICE", "ccy": "FX"
    },
    "BRLCNY": {
        "source": "tv", "tv_symbol": "BRLCNY",
        "tv_exchange": "ICE", "ccy": "FX"
    },    
    "BTCUSD": {
        "source": "tv", "tv_symbol": "BTCUSD",
        "tv_exchange": "BITSTAMP", "ccy": "FX"
    },
}

# =========================================
# CACHE GLOBAL DE DADOS DE MERCADO
# =========================================
import streamlit as st

MAX_LOOKBACK_D = 1500   # barras base diária para universo B3
MAX_LOOKBACK_W = 400    # barras base semanal
MAX_LOOKBACK_TV = 1500  # barras TradingView (CLASSES)


@st.cache_data(show_spinner=False)
def load_universe_prices(freq_key: str, max_lookback: int = 1500) -> pd.DataFrame:
    """
    Carrega TODO o universo de papéis B3 (SETORES_ATIVOS + BENCHMARK)
    de uma vez, para uma frequência (D ou W), e guarda em cache.

    Depois as abas só fatiam essa base por lookback, sem chamar _get_price_series de novo.
    """
    freq_key = (freq_key or "D").upper()
    univ = sorted(set(_flatten_universe(SETORES_ATIVOS) + [BENCHMARK]))

    # barras máximas que vamos pedir na primeira vez
    base_lookback = max_lookback if isinstance(max_lookback, (int, float)) else MAX_LOOKBACK_D
    bars_need = min(2000, int(bars_for_freq(freq_key, base_lookback)))

    px_list = []
    for tk in univ:
        s = _get_price_series(tk, bars=bars_need, freq_key=freq_key)
        if s is not None and not s.dropna().empty:
            px_list.append(s.rename(tk))

    if not px_list:
        return pd.DataFrame()

    px = pd.concat(px_list, axis=1).sort_index()

    if freq_key == "D":
        px = px.asfreq("B").ffill()
    else:  # "W"
        px = px.resample("W-FRI").last().ffill()

    return px


@st.cache_data(show_spinner=False)
def load_all_classes_prices(freq_key: str = "D", max_lookback: int = 1500) -> ClassesResult:
    """
    Baixa TODAS as séries de CLASSES_MAP de uma vez via TradingView
    para a frequência desejada, cacheia, e depois as abas só selecionam
    as colunas necessárias.
    """
    freq_key = (freq_key or "D").upper()
    labels_all = list(CLASSES_MAP.keys())
    bars_need = min(1500, int(bars_for_freq_tv(freq_key, max_lookback)))

    px_list: list[pd.Series] = []
    dbg = {"sucessos": [], "falhas": []}

    for nome in labels_all:
        meta = CLASSES_MAP.get(nome)
        if not meta:
            continue
        s = _get_price_series_unified(meta, bars=bars_need, freq_key=freq_key)
        if s is not None and not s.dropna().empty:
            px_list.append(s.rename(nome))
            dbg["sucessos"].append(nome)
        else:
            dbg["falhas"].append(nome)

    if not px_list:
        return ClassesResult(pd.DataFrame(), dbg)

    px_classes = pd.concat(px_list, axis=1).sort_index()

    if freq_key == "D":
        px_classes = (
            px_classes.resample("B")
            .last()
            .ffill()
        )
    elif freq_key == "W":
        px_classes = (
            px_classes.resample("W-FRI")
            .last()
            .ffill()
        )
    else:
        px_classes = px_classes.ffill()

    return ClassesResult(px_classes, dbg)

# ============================
# ETTJ — helpers de curva de juros
# ============================

US_YIELD_LABELS = [
    "US 3M Yield", "US 4M Yield", "US 6M Yield",
    "US 1Y Yield", "US 2Y Yield", "US 3Y Yield", "US 4Y Yield",
    "US 5Y Yield", "US 6Y Yield", "US 7Y Yield", "US 8Y Yield",
    "US 9Y Yield", "US 10Y Yield",
]

BR_YIELD_LABELS = [
    "BR3M", "BR4M", "BR6M",
    "BR1Y", "BR2Y", "BR3Y", "BR4Y", "BR5Y",
    "BR6Y", "BR7Y", "BR8Y", "BR9Y", "BR10Y",
]


def _build_yield_curve_df(
    labels: list[str],
    freq_key: str = "D",
    lookback: str | int = "3Y",
    bars_cap: int = 1500,
) -> pd.DataFrame:
    """
    Usa build_classes_dataframe para buscar os vértices,
    pega o último yield de cada um e retorna:
      columns = ['tenor', 'yield', 'label']
    """
    # usa a função já existente que você tem para pegar as séries do TradingView
    res = build_classes_dataframe(labels, freq_key=freq_key, lookback=lookback, bars_cap=bars_cap)
    df_px = res.dataframe
    if df_px is None or df_px.dropna(how="all").empty:
        return pd.DataFrame(columns=["tenor", "yield", "label"])

    rows = []
    for label in labels:
        if label not in df_px.columns:
            continue
        s = df_px[label].dropna()
        if s.empty:
            continue
        meta = CLASSES_MAP.get(label, {})
        tenor = meta.get("tenor_yrs")
        if tenor is None:
            continue
        y_last = float(s.iloc[-1])
        rows.append((float(tenor), y_last, label))

    if not rows:
        return pd.DataFrame(columns=["tenor", "yield", "label"])

    curve = (
        pd.DataFrame(rows, columns=["tenor", "yield", "label"])
        .sort_values("tenor")
        .drop_duplicates(subset="tenor", keep="last")
        .reset_index(drop=True)
    )
    return curve

   
@dataclass
class ClassesResult:
    dataframe: pd.DataFrame
    debug: dict


def build_classes_dataframe(
    sel_labels: Sequence[str],
    freq_key: str,
    lookback: str | int,
    bars_cap: int = 1500,
) -> ClassesResult:
    """
    Agora usa load_all_classes_prices (cacheado) para pegar TODAS as classes
    e depois apenas seleciona as colunas desejadas.
    """
    all_res = load_all_classes_prices(freq_key=freq_key, max_lookback=lookback)
    px_all = all_res.dataframe

    if px_all is None or px_all.dropna(how="all").empty:
        return ClassesResult(pd.DataFrame(), all_res.debug)

    cols = [c for c in sel_labels if c in px_all.columns]
    if not cols:
        return ClassesResult(pd.DataFrame(), {"sucessos": [], "falhas": sel_labels})

    # Recorte das colunas selecionadas
    px_sel = px_all[cols].copy()

    # Ajusta debug para mostrar só o que o usuário escolheu
    dbg = {
        "sucessos": [c for c in cols if c in all_res.debug.get("sucessos", [])],
        "falhas": [c for c in cols if c in all_res.debug.get("falhas", [])],
    }

    return ClassesResult(px_sel, dbg)



def _default_selection(labels: Iterable[str]) -> list[str]:
    labels_list = list(labels)
    return labels_list[:4] if len(labels_list) >= 4 else labels_list


# ================================
# Helpers específicos da aba Classes
# ================================
def _period_pct(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1.0) if s.shape[0] >= 2 else float("nan")


def _rotation_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Tabela de rotação: retornos em 1m,3m,6m,12m,YTD,36m + ranks + rank médio."""
    if prices is None or prices.dropna(how="all").empty:
        return pd.DataFrame()
    # frequência “de calendário” aproximada via últimos N pontos (diário/semanais)
    is_daily = (prices.index.freqstr or "").startswith("B") or (
        prices.index.inferred_freq in ("B", "D", "C")
    )
    N_1m, N_3m, N_6m, N_12m, N_36m = (21, 63, 126, 252, 756) if is_daily else (4, 13, 26, 52, 156)

    rows = []
    last = prices.index.max()
    ytd_start = last.replace(month=1, day=1)

    for col in prices.columns:
        s = prices[col].dropna()
        if s.shape[0] < 5:
            continue

        def lastNret(N):
            ss = s.tail(N)
            return _period_pct(ss)

        r_1m = lastNret(N_1m)
        r_3m = lastNret(N_3m)
        r_6m = lastNret(N_6m)
        r_12m = lastNret(N_12m)
        r_36m = lastNret(N_36m)
        # YTD
        s_ytd = s.loc[(s.index >= ytd_start) & (s.index <= last)]
        r_ytd = _period_pct(s_ytd) if s_ytd.shape[0] >= 2 else np.nan
        rows.append([col, r_1m, r_3m, r_6m, r_12m, r_ytd, r_36m])

    tb = pd.DataFrame(rows, columns=["Ativo", "1m", "3m", "6m", "12m", "YTD", "36m"]).set_index("Ativo")
    if tb.empty:
        return tb

    # ranks por coluna de retorno (desc = melhor retorno)
    for c in ["1m", "3m", "6m", "12m", "YTD", "36m"]:
        tb[f"Rank_{c}"] = tb[c].rank(ascending=False, method="min")
    tb = tb.sort_values("12m", ascending=False)
    return tb


def _fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


def _ettj_from_vertices(
    vertices_labels: list[str],
    px_classes: pd.DataFrame,
    selic_or_fed_rate: float,
    country: str,
    ref_date: date,
    n_points: int = 120,
) -> Optional[pd.DataFrame]:
    """
    Monta ETTJ a partir de:
      - taxa overnight (SELIC ou Fed Funds), em % a.a.
      - vértices (labels do CLASSES_MAP com tenor_yrs)
      - px_classes: DataFrame com colunas = labels e valores = yields (nível) para cada data

    Retorna DataFrame com colunas:
      - 'tenor_yrs'
      - 'yield_pct'
      - 'country' (string 'BR' ou 'US')
    """
    if px_classes is None or px_classes.empty:
        return None

    # Lista de (tenor_em_anos, yield_%_a.a.) já ordenável
    T: list[float] = []
    Y: list[float] = []

    # Ponto t=0 (overnight)
    if selic_or_fed_rate is not None and not np.isnan(selic_or_fed_rate):
        T.append(0.0)
        Y.append(float(selic_or_fed_rate))

    # Demais vértices vindos do CLASSES_MAP
    for label in vertices_labels:
        meta = CLASSES_MAP.get(label, {})
        tenor = meta.get("tenor_yrs")
        if tenor is None:
            continue
        if label not in px_classes.columns:
            continue

        s = px_classes[label].dropna()
        if s.empty:
            continue

        # pega o valor da curva até a data de referência
        s_ref = s[s.index.date <= ref_date]
        if s_ref.empty:
            continue

        y_val = float(s_ref.iloc[-1])  # já vem em "% a.a." do TradingView
        T.append(float(tenor))
        Y.append(y_val)

    if len(T) < 2:
        return None

    # Ordena por tenor
    T_arr = np.array(T)
    Y_arr = np.array(Y)
    order = np.argsort(T_arr)
    T_sorted = T_arr[order]
    Y_sorted = Y_arr[order]

    # Grid contínuo de tenores (do 0 até o último vértice)
    T_min = T_sorted[0]
    T_max = T_sorted[-1]
    grid = np.linspace(T_min, T_max, n_points)

    # Interpolação linear em yield (em % a.a.)
    Y_grid = np.interp(grid, T_sorted, Y_sorted)

    df_curve = pd.DataFrame(
        {
            "tenor_yrs": grid,
            "yield_pct": Y_grid,
        }
    )
    df_curve["country"] = country
    return df_curve

# =============================
# Aba CLASSES — Rotação | ETTJ | Retorno | Correlação | Regressão
# =============================
with tmap["Classes"]:
    st.subheader("Classes de Ativos")

    # ----- Controles principais -----
    colu1, colu2, colu3 = st.columns([1.3, 1, 1])
    with colu1:
        labels_all = list(CLASSES_MAP.keys())
        sel_labels = st.multiselect(
            "Selecione as classes",
            labels_all,
            default=labels_all[:6],
        )
    with colu2:
        freq_classes = st.radio(
            "Frequência",
            ["Diária", "Semanal"],
            index=0,
            horizontal=True,
            key="freq_classes",
        )
        freq_key_classes = "D" if freq_classes == "Diária" else "W"
    with colu3:
        lb_classes = st.slider(
            "Período para download",
            120, 1200, 750, step=30,
        )

    if not sel_labels:
        st.info("Selecione ao menos uma classe para análise.")
        st.stop()

    # ----- Dados das classes (TradingView) -----
    with st.spinner("Baixando séries das classes..."):
        classes_res = build_classes_dataframe(
            sel_labels,
            freq_key=freq_key_classes,
            lookback=lb_classes,
            bars_cap=1500,
        )
        px_classes = classes_res.dataframe
        dbg = {
            "sucessos": classes_res.debug.get("sucessos", []),
            "falhas": classes_res.debug.get("falhas", []),
        }

    if px_classes is None or px_classes.dropna(how="all").empty:
        st.error("Não foi possível montar as séries das classes com as escolhas atuais.")
        with st.expander("Diagnóstico (Classes) — clique para ver detalhes", expanded=False):
            st.write("**Sucessos**:", dbg.get("sucessos", []))
            st.write("**Falhas**:", dbg.get("falhas", []))
        st.stop()

    # ---------- Rotação ----------
    st.markdown("### Rotação (tabela de retornos e ranks)")
    tb_rot = _rotation_table(px_classes)
    if tb_rot.empty:
        st.info("Sem dados suficientes para montar a tabela de rotação.")
    else:
        fmt = tb_rot.copy()
        for c in ["1m", "3m", "6m", "12m", "YTD", "36m"]:
            fmt[c] = fmt[c].map(_fmt_pct)
        st.dataframe(fmt, use_container_width=True)

    st.markdown("---")

    # ---------- ETTJ BR x EUA ----------
    st.markdown("### Curvas de Juros (ETTJ) — BR e USA")
    
    col_r0, col_r1, col_r2 = st.columns(3)
    with col_r0:
        selic_atual = st.number_input(
            "SELIC atual (% a.a.)",
            value=10.00,
            step=0.25,
            format="%.2f",
        )
    with col_r1:
        fed_atual = st.number_input(
            "Fed Funds atual (% a.a.)",
            value=5.00,
            step=0.25,
            format="%.2f",
        )
    with col_r2:
        btn_ettj = st.button("Montar curvas ETTJ (BR e US)", type="primary")
    
    if btn_ettj:
        with st.spinner("Montando curvas ETTJ (BR e US)..."):
            br_curve = _build_yield_curve_df(BR_YIELD_LABELS, freq_key="D", lookback="3Y")
            us_curve = _build_yield_curve_df(US_YIELD_LABELS, freq_key="D", lookback="3Y")
    
        # adiciona ponto t=0 com SELIC / Fed
        if not br_curve.empty and not np.isnan(selic_atual):
            br0 = pd.DataFrame(
                {"tenor": [0.0], "yield": [float(selic_atual)], "label": ["Selic (t=0)"]}
            )
            br_curve_plot = (
                pd.concat([br0, br_curve], ignore_index=True)
                .sort_values("tenor")
                .reset_index(drop=True)
            )
        else:
            br_curve_plot = pd.DataFrame(columns=["tenor", "yield", "label"])
    
        if not us_curve.empty and not np.isnan(fed_atual):
            us0 = pd.DataFrame(
                {"tenor": [0.0], "yield": [float(fed_atual)], "label": ["Fed Funds (t=0)"]}
            )
            us_curve_plot = (
                pd.concat([us0, us_curve], ignore_index=True)
                .sort_values("tenor")
                .reset_index(drop=True)
            )
        else:
            us_curve_plot = pd.DataFrame(columns=["tenor", "yield", "label"])
    
        if br_curve_plot.empty and us_curve_plot.empty:
            st.warning("Não foi possível montar nenhuma curva (BR ou US).")
        else:
            # helper para montar cada gráfico com zoom agressivo no eixo Y
            def _plot_curve(curve_df: pd.DataFrame, titulo: str):
                fig = go.Figure()
                x_vals = curve_df["tenor"].values
                y_vals = curve_df["yield"].values
    
                # interpolação suave
                x_grid = np.linspace(x_vals.min(), x_vals.max(), 120)
                y_grid = np.interp(x_grid, x_vals, y_vals)
    
                fig.add_trace(go.Scatter(
                    x=x_grid, y=y_grid,
                    mode="lines",
                    name="Curva (interp.)",
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="markers",
                    name="Vértices",
                ))
    
                # zoom no eixo Y
                y_min = float(np.nanmin(y_vals))
                y_max = float(np.nanmax(y_vals))
                span = y_max - y_min if y_max > y_min else 0.10
    
                # margem bem pequena pra destacar curvatura
                pad = max(0.02, span * 0.05)      # ~5% da amplitude ou 2 bps
                # ticks finos
                if span < 0.40:
                    dtick = 0.02
                elif span < 1.00:
                    dtick = 0.05
                else:
                    dtick = 0.10
    
                fig.update_layout(
                    title=titulo,
                    height=380,
                    xaxis_title="Prazo (anos)",
                    yaxis=dict(
                        title="Taxa (% a.a.)",
                        range=[y_min - pad, y_max + pad],
                        tickformat=".2f",
                        dtick=dtick,
                    ),
                    hovermode="x unified",
                )
                return fig
    
            col_br, col_us = st.columns(2)
    
            with col_br:
                if br_curve_plot.empty:
                    st.info("Sem dados suficientes para montar a curva BR.")
                else:
                    fig_br = _plot_curve(br_curve_plot, "Curva de Juros BR")
                    st.plotly_chart(fig_br, use_container_width=True)
    
            with col_us:
                if us_curve_plot.empty:
                    st.info("Sem dados suficientes para montar a curva US.")
                else:
                    fig_us = _plot_curve(us_curve_plot, "Curva de Juros US")
                    st.plotly_chart(fig_us, use_container_width=True)
    
    st.markdown("---")
    


    # -----------------------------
    # Retorno rolling (área) — 12M (usa as MESMAS sel_labels)
    # -----------------------------
    st.subheader("Retorno rolling de 12 meses")

    janela_meses = st.slider("Janela (meses)", 6, 36, 12, step=1)
    
    # px_classes JÁ está filtrado em build_classes_dataframe(sel_labels, ...)
    px_df_cls = px_classes.dropna(how="all")
    
    if px_df_cls.empty:
        st.info("Sem dados suficientes para calcular o retorno rolling.")
    else:
        # ~21 dias úteis por mês
        roll_days = int(janela_meses * 21)
        ret_roll = px_df_cls / px_df_cls.shift(roll_days) - 1.0
        ret_roll = ret_roll.dropna(how="all")
    
        if ret_roll.empty:
            st.info(
                f"A janela de {janela_meses} meses é maior que o histórico disponível. "
                "Diminua a janela ou troque as classes."
            )
        else:
            fig_roll_area = go.Figure()
            for c in ret_roll.columns:
                s = ret_roll[c].dropna()
                fig_roll_area.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        name=c,
                        mode="lines",
                        fill="tozeroy",
                        hovertemplate="%{x|%d-%m-%Y}<br>%{y:.2%}<extra>" + c + "</extra>",
                    )
                )
    
            fig_roll_area.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
            fig_roll_area.update_yaxes(tickformat=".1%")
            fig_roll_area.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                legend_title_text="Classe",
                xaxis_title="Data",
                yaxis_title=f"Retorno rolling {janela_meses}m",
            )
            st.plotly_chart(fig_roll_area, use_container_width=True)


    # ---------- Desempenho (período independente) ----------
    st.markdown("### Desempenho")
    cda, cdb = st.columns(2)
    with cda:
        dt_ini_c = st.date_input(
            "Início do período (Desempenho)",
            value=(px_classes.index.min().date() if not px_classes.empty else date.today() - timedelta(days=365)),
        )
    with cdb:
        dt_fim_c = st.date_input(
            "Fim do período (Desempenho)",
            value=(px_classes.index.max().date() if not px_classes.empty else date.today()),
        )

    px_perf = px_classes.loc[
        (px_classes.index.date >= dt_ini_c)
        & (px_classes.index.date <= dt_fim_c)
    ].dropna(how="all")

    if px_perf.dropna(how="all").empty:
        st.info("Sem dados para o período escolhido.")
    else:
        acc = (px_perf / px_perf.iloc[0]).dropna(how="all")
        fig_acc = px.line(
            acc,
            x=acc.index,
            y=acc.columns,
            labels={
                "value": "Índice (base=1)",
                "index": "Data",
                "variable": "Classe",
            },
        )
        fig_acc.update_layout(height=420, legend_title_text="Classe")
        st.plotly_chart(fig_acc, use_container_width=True)

        rets = px_perf.pct_change().dropna(how="all")
        rows_stats = []
        AF = 252 if freq_key_classes == "D" else 52
        for c in rets.columns:
            r = rets[c].dropna()
            if r.empty:
                continue
            ret_acum = float((1 + r).prod() - 1.0)
            vol_ann = float(r.std(ddof=1) * np.sqrt(AF))
            sharpe = ((r.mean() * AF) / vol_ann) if vol_ann and not np.isnan(vol_ann) else np.nan
            idx_ = (1 + r).cumprod()
            mdd = float((idx_ / idx_.cummax() - 1.0).min()) if not idx_.empty else np.nan
            rows_stats.append([c, ret_acum, vol_ann, sharpe, mdd])
        df_stats = pd.DataFrame(
            rows_stats,
            columns=["Classe", "Retorno acumulado", "Vol anual.", "Sharpe (rf=0)", "Max DD"],
        ).set_index("Classe")
        df_show = df_stats.copy()
        df_show["Retorno acumulado"] = df_show["Retorno acumulado"].map(_fmt_pct)
        df_show["Vol anual."] = df_show["Vol anual."].map(_fmt_pct)
        df_show["Sharpe (rf=0)"] = df_show["Sharpe (rf=0)"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )
        df_show["Max DD"] = df_show["Max DD"].map(_fmt_pct)
        st.dataframe(df_show, use_container_width=True)

    st.markdown("---")

        # -------------------------------------------
    # Desempenho por ano (cada ano inicia em 0%)
    # -------------------------------------------
    st.markdown("### Desempenho Sazonal")

    cols_yr = list(px_classes.columns)
    if not cols_yr:
        st.info("Sem séries carregadas para montar o desempenho anual.")
    else:
        classe_ano = st.selectbox(
            "Selecione uma classe para análise anual",
            cols_yr,
            key="classe_ano_perf"
        )

        s_cls = px_classes[classe_ano].dropna()
        if s_cls.empty:
            st.info("Sem dados suficientes para essa classe.")
        else:
            # retornos diários
            rets_d = s_cls.pct_change().dropna()
            df_yr = rets_d.to_frame("ret_diario")
            df_yr["Ano"] = df_yr.index.year
            df_yr["Mes"] = df_yr.index.month
            df_yr["dia_ano"] = df_yr.index.dayofyear

            # --------- GRÁFICO: curvas anuais sobrepostas ---------
            fig_year = go.Figure()
            base_year = 2000  # ano sintético só para o eixo X

            for ano in sorted(df_yr["Ano"].unique()):
                sub = df_yr[df_yr["Ano"] == ano].copy()
                if sub.empty:
                    continue
                cum = (1.0 + sub["ret_diario"]).cumprod() - 1.0

                # constrói um índice sintético "dia do ano" a partir de 2000-01-01
                dias = sub["dia_ano"].values
                x_idx = pd.to_datetime(
                    pd.Timestamp(base_year, 1, 1)
                ) + pd.to_timedelta(dias - 1, unit="D")

                fig_year.add_trace(
                    go.Scatter(
                        x=x_idx,
                        y=cum.values,
                        mode="lines",
                        name=str(ano),
                    )
                )

            fig_year.add_hline(y=0, line_dash="dash", line_color="black")
            fig_year.update_yaxes(title_text="Retorno acumulado no ano", tickformat=".0%")
            fig_year.update_xaxes(title_text="Data (dia do ano)")
            fig_year.update_layout(height=420, legend_title_text="Ano")

            st.plotly_chart(fig_year, use_container_width=True)

            # --------- TABELA: retornos mensais por ano ---------
            # retorno mensal = (1 + ret_diario)^n - 1 por ano/mês
            df_month = (
                df_yr
                .groupby(["Ano", "Mes"])["ret_diario"]
                .apply(lambda x: (1.0 + x).prod() - 1.0)
                .unstack("Mes")
                .sort_index()
            )
            
            # garante colunas 1..12
            for m in range(1, 13):
                if m not in df_month.columns:
                    df_month[m] = np.nan
            df_month = df_month.reindex(columns=range(1, 13))
            
            # -------- NOVO: coluna TOTAL (somatório anual) --------
            df_month["Total"] = df_month.sum(axis=1, skipna=True)
            
            # linha de média dos meses (para todas as colunas, inclusive Total)
            mean_row = df_month.mean(axis=0, skipna=True)
            
            df_month_with_mean = pd.concat(
                [df_month, mean_row.to_frame().T],
                axis=0
            )
            
            # renomeia índice (anos como str + "Média")
            idx_years = [str(i) for i in df_month.index]
            df_month_with_mean.index = idx_years + ["Média"]
            
            # nomes dos meses
            month_map = {
                1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
                5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
                9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
            }
            df_month_show = df_month_with_mean.rename(columns=month_map)
            
            # formatação em %
            df_month_fmt = df_month_show.applymap(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
            )
            
            st.markdown("#### Retornos por mês")
            st.dataframe(df_month_fmt, use_container_width=True)

            


    # ---------- Correlação ----------
    st.markdown("### Correlação")
    cec, cfd = st.columns(2)
    with cec:
        dt_ini_corr = st.date_input(
            "Início (Correlação)",
            value=(px_classes.index.min().date() if not px_classes.empty else date.today() - timedelta(days=365)),
        )
    with cfd:
        dt_fim_corr = st.date_input(
            "Fim (Correlação)",
            value=(px_classes.index.max().date() if not px_classes.empty else date.today()),
        )

    px_corr = px_classes.loc[
        (px_classes.index.date >= dt_ini_corr)
        & (px_classes.index.date <= dt_fim_corr)
    ].dropna(how="all")

    if px_corr.shape[1] >= 2:
        rt_corr = px_corr.pct_change().dropna(how="all")
        if rt_corr.dropna().shape[0] >= 2:   # agora aceita a partir de 2 observações
            cm = rt_corr.corr()
            figc = px.imshow(
                cm,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Matriz de Correlação (retornos)",
            )
            figc.update_layout(height=520)
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info("Dados insuficientes (precisa de ≥ 2 observações).")
    else:
        st.info("Selecione ao menos 2 classes para ver correlação.")

    st.markdown("---")

    # ---------- Regressão ----------
    st.markdown("### Regressão OLS — Sensibilidade entre classes")
    all_cols = list(px_classes.columns)
    if len(all_cols) < 2:
        st.info("Selecione pelo menos duas classes para regressão.")
    else:
        dep = st.selectbox("Dependente (Y)", all_cols, index=0, key="regY_cls")
        candX = [c for c in all_cols if c != dep]
        indep = st.multiselect("Independentes (X)", candX, default=candX[:2], key="regX_cls")

        colr1, colr2, colr3, colr4 = st.columns(4)
        with colr1:
            freq_reg = st.radio(
                "Frequência regressão",
                ["Diária", "Semanal"],
                index=0,
                horizontal=True,
                key="regfreq_cls",
            )
            fk_reg = "D" if freq_reg == "Diária" else "W"
        with colr2:
            n_days_reg = st.slider("Últimos N períodos", 30, 500, 60, step=5, key="regN_cls")
        with colr3:
            max_lag = st.slider("Lags nas X (meses/períodos)", 0, 12, 0, 1, key="regLag_cls")
        with colr4:
            use_hac = st.checkbox("HAC/Newey-West", value=True, key="regHAC_cls")

        if not indep:
            st.info("Selecione ao menos uma X.")
        else:
            def to_freq_any(df: pd.DataFrame, fk: str) -> pd.DataFrame:
                return df.asfreq("B").ffill() if fk == "D" else df.resample("W-FRI").last().ffill()

            pxR = to_freq_any(px_classes, fk_reg).dropna(how="all").tail(n_days_reg + 5)
            if pxR.shape[0] < n_days_reg + 2:
                st.info("Janela muito curta para a base disponível.")
            else:
                rt = pxR.pct_change().dropna(how="all").tail(n_days_reg)
                y = rt[[dep]].copy()
                X = rt[indep].copy()

                if max_lag > 0:
                    Xlag = {}
                    for c in X.columns:
                        for L in range(0, max_lag + 1):
                            Xlag[f"{c}_L{L}"] = X[c].shift(L)
                    X = pd.DataFrame(Xlag, index=X.index)

                df_ols = pd.concat([y, X], axis=1).dropna()
                if df_ols.shape[0] < 12 or df_ols.shape[1] < 2:
                    st.warning("Dados insuficientes para regressão (≥12 obs e ≥1 X).")
                else:
                    y_end = df_ols[dep]
                    X_ols = sm.add_constant(df_ols.drop(columns=[dep]), has_constant="add")
                    try:
                        model = (
                            sm.OLS(y_end, X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
                            if use_hac
                            else sm.OLS(y_end, X_ols).fit()
                        )
                        st.write(f"Obs.: {int(model.nobs)} | R²: {model.rsquared:.3f} | R² ajust.: {model.rsquared_adj:.3f}")
                        st.write(f"AIC: {model.aic:.1f} | BIC: {model.bic:.1f}")

                        params = model.params
                        conf = model.conf_int()
                        coefs = pd.DataFrame(
                            {
                                "coef": params,
                                "t": model.tvalues,
                                "p": model.pvalues,
                                "ci_low": conf[0],
                                "ci_high": conf[1],
                            }
                        ).rename_axis("variável").reset_index()
                        st.dataframe(
                            coefs.style.format(
                                {
                                    "coef": "{:.4f}",
                                    "t": "{:.2f}",
                                    "p": "{:.3f}",
                                    "ci_low": "{:.4f}",
                                    "ci_high": "{:.4f}",
                                }
                            ),
                            use_container_width=True,
                        )

                        sdY = y_end.std(ddof=1)
                        sdXs = df_ols.drop(columns=[dep]).std(ddof=1)
                        imp = (params.drop("const", errors="ignore") * (sdXs / sdY)).dropna()
                        if not imp.empty:
                            st.markdown("**Importância padronizada (β · σX / σY)**")
                            st.dataframe(
                                imp.sort_values(ascending=False).to_frame("β_std"),
                                use_container_width=True,
                            )

                        fitted = model.fittedvalues.reindex(y_end.index)
                        cumY = (1 + y_end).cumprod()
                        cumFit = (1 + fitted).cumprod()
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=cumY.index, y=cumY.values, mode="lines", name="Y acumulado"))
                        fig1.add_trace(go.Scatter(x=cumFit.index, y=cumFit.values, mode="lines", name="Ajustado acumulado"))
                        fig1.update_layout(height=360, title="Acumulado: Y vs Ajustado", yaxis_title="Índice (base=1)")
                        st.plotly_chart(fig1, use_container_width=True)

                        resid = model.resid.reindex(y_end.index)
                        fig2 = px.line(resid, x=resid.index, y=resid.values, labels={"x": "Data", "y": "Resíduo"})
                        fig2.add_hline(y=0, line_dash="dash")
                        fig2.update_layout(height=280, title="Resíduos (Y − Ajustado)")
                        st.plotly_chart(fig2, use_container_width=True)

                    except Exception as e:
                        st.error(f"Falha na regressão: {e}")



  
# --- Pré-condições mínimas usadas nas abas RISCO/ORBITAL/L&S ---
BENCHMARK = globals().get("BENCHMARK", "BOVA11.SA")

# helper: janela de retornos para betas (toda a base de papéis + bench)
def _returns_window_for_risk(px_df_: pd.DataFrame,
                             bench_: pd.Series,
                             win: int) -> pd.DataFrame:
    rets_all = px_df_.pct_change().dropna(how="all")
    bench_ret = bench_.pct_change().rename(BENCH_USED)
    rets_all, bench_ret = rets_all.align(bench_ret, join="inner", axis=0)
    rt_win = pd.concat([rets_all, bench_ret], axis=1).dropna()
    return rt_win.tail(win)

def _safe_get_bench(bars=260, fk="D"):
    s = _get_price_series(BENCHMARK, bars=bars, freq_key=fk)
    return s.rename(BENCHMARK) if s is not None else pd.Series(dtype=float, name=BENCHMARK)

def _ensure_risk_context():
    """
    Garante px_df, bench, ew e freq_key_setor quando a aba RISCO é acessada
    sem ter aberto a aba SETOR antes.
    """
    global px_df, bench, ew, freq_key_setor
    if "freq_key_setor" not in globals():
        freq_key_setor = "D"

    have_px = ("px_df" in globals()) and isinstance(px_df, pd.DataFrame) and not px_df.dropna(how="all").empty
    have_ben = ("bench" in globals()) and isinstance(bench, pd.Series) and not bench.dropna().empty

    if not have_px:
        # monta um setor padrão com 1 ano útil
        any_setor = next(iter(SETORES_ATIVOS.keys()))
        tks = SETORES_ATIVOS[any_setor]
        lst = [_get_price_series(tk, bars_for_freq("D", 260), "D").rename(tk) for tk in tks]
        px = [s for s in lst if s is not None and not s.dropna().empty]
        globals()["px_df"] = pd.concat(px, axis=1) if px else pd.DataFrame()

    if not have_ben:
        globals()["bench"] = _safe_get_bench(bars_for_freq("D", 260), "D")

    if ("ew" not in globals()) or not isinstance(ew, pd.Series) or ew.dropna().empty:
        if "px_df" in globals() and not px_df.dropna(how="all").empty:
            rets_tmp = px_df.pct_change()
            globals()["ew"] = (1 + rets_tmp.mean(axis=1, skipna=True)).cumprod().rename("EW_Setor").dropna()
        else:
            globals()["ew"] = pd.Series(dtype=float, name="EW_Setor")

_ensure_risk_context()

          
# --- Aba RISCO ---
with tmap["Risco"]:
    st.subheader("Painel de Risco — Setor ou Papel")

    # seleção de ativo (ou setor)
    tipo_sel = st.radio("Analisar:", ["Setor (EW)", "Papel individual"], horizontal=True)

    # retornos diários/semanais — dependem do que o usuário escolheu antes
    if 'ew' not in locals() or ew.empty:
        # recria índice equal-weight do setor
        rets_temp = px_df.pct_change()
        ew = (1 + rets_temp.mean(axis=1, skipna=True)).cumprod().rename("EW_Setor").dropna()
        
    ret_ben = bench.pct_change().dropna()  # benchmark sempre é série
    if tipo_sel == "Setor (EW)":
        obj_name = "Setor (EW)"
        ret_obj = ew.pct_change().dropna()
        base100_obj = rebase_100(ew)
    else:
        papel_sel = st.selectbox("Escolha um papel do setor:", list(px_df.columns))
        obj_name = papel_sel
        ret_obj = px_df[papel_sel].pct_change().dropna()
        base100_obj = rebase_100(px_df[papel_sel])

 

    # =========================
    # KPIs de risco + Retornos
    # =========================
    st.markdown("---")
    st.subheader("Indicadores de risco e retorno")
    
    # Periodicidade (p/ anualização e RF por período)
    af = 252 if (freq_key_setor == "D") else 52
    
    # Parâmetro: taxa livre de risco (a.a.)
    with st.expander("Parâmetros de risco", expanded=True):
        rf_annual = st.number_input(
            "Taxa livre de risco (a.a., %)", value=10.00, step=0.25, format="%.2f",
            help="Usada no Sharpe. Convertemos para o período (252 se diário; 52 se semanal)."
        )
    rf_per = (1.0 + rf_annual/100.0)**(1/af) - 1.0  # RF por período

    # **Janela que será usada TANTO no radar quanto no rolling beta**
    roll_win = st.slider(
        "Janela para Betas / Rolling (dias úteis)",
        20, 120, 60, step=5, key="rollriskwin"
    )
    
    # Séries de trabalho
    rets_obj = ret_obj.dropna()
    if rets_obj.empty:
        st.warning("Sem retornos suficientes para calcular os indicadores.")
    else:
        # --- VaR histórico (95%) por período ---
        raw_var_5pct = float(np.percentile(rets_obj.values, 5))
        var_95_loss  = -raw_var_5pct if raw_var_5pct < 0 else 0.0   # exibe como perda (positivo)

        # --- Máx Drawdown no período exibido ---
        idx_base = base100_obj.dropna()
        dd_series = idx_base / idx_base.cummax() - 1.0
        max_dd = float(dd_series.min()) if not dd_series.empty else float("nan")
    
        # --- Sharpe anualizado (excesso vs RF) ---
        exc = rets_obj - rf_per
        vol_p = exc.std(ddof=1)
        sharpe_ann = float((exc.mean()/vol_p) * np.sqrt(af)) if vol_p and not np.isnan(vol_p) else float("nan")
    
        # --- Retornos (YTD, 12m aprox., mês e dia) ---
        # Dia (último retorno disponível)
        ret_day = float(rets_obj.iloc[-1])
    
        # Mês corrente
        _idx = idx_base.index
        cur_month = _idx[-1].month
        sel_m = idx_base.loc[_idx.month == cur_month]
        ret_month = float(sel_m.iloc[-1]/sel_m.iloc[0] - 1.0) if len(sel_m) >= 2 else float("nan")
    
        # YTD
        cur_year = _idx[-1].year
        sel_ytd = idx_base.loc[_idx.year == cur_year]
        ret_ytd = float(sel_ytd.iloc[-1]/sel_ytd.iloc[0] - 1.0) if len(sel_ytd) >= 2 else float("nan")
    
        # 12 meses ~ af períodos (252 d.u. se diário; 52 se semanal)
        win12 = min(len(idx_base), af)
        sel_12m = idx_base.iloc[-win12:]
        ret_12m = float(sel_12m.iloc[-1]/sel_12m.iloc[0] - 1.0) if len(sel_12m) >= 2 else float("nan")
    
        # --- Cards ---
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("VaR 95% (1 período)", f"{var_95_loss*100:.2f}%", help="Percentil histórico de 5% (perda esperada).")
        k2.metric("Máx Drawdown", f"{max_dd*100:.2f}%")
        k3.metric("Sharpe (anualizado)", f"{sharpe_ann:.2f}", help=f"Excesso vs RF {rf_annual:.2f}% a.a.")
        k4.metric("Retorno YTD", f"{ret_ytd*100:.2f}%")
        k5.metric("Retorno 12m", f"{ret_12m*100:.2f}%")
        k6.metric("Retorno mês", f"{ret_month*100:.2f}%")
        st.caption(f"Retorno do dia: {ret_day*100:.2f}%")
    
        # ====================================
        # VaR (95%) + Radar de Betas (lado a lado)
        # ====================================
        st.markdown("---")
        
        rets_var = ret_obj.dropna().rename("Retorno")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Distribuição do VaR (95%)")
            if rets_var.empty:
                st.info("Sem retornos suficientes para estimar o VaR.")
            else:
                fig_var = px.histogram(
                    rets_var.to_frame(),
                    x="Retorno",
                    nbins=50,
                    opacity=0.85,
                    labels={"Retorno": "Retorno por período"},
                    title=None,
                )
                fig_var.update_layout(height=400, bargap=0.05, xaxis_tickformat=".2%")
                
                # Linha no P5 (NEGATIVO): vai para o lado esquerdo da distribuição
                fig_var.add_vline(
                    x=raw_var_5pct,
                    line_color="red",
                    line_dash="dash",
                    annotation_text="VaR 95%",
                    annotation_position="top left",
                )
                
                # (Opcional) sombrear a cauda de perda à esquerda do VaR:
                try:
                    x_min = float(rets_var.min())
                    fig_var.add_vrect(x0=x_min, x1=raw_var_5pct, fillcolor="red", opacity=0.08, line_width=0)
                except Exception:
                    pass
                
                fig_var.add_vline(x=0, line_color="#9ca3af", line_dash="dot")
                st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            st.subheader("Sensibilidade ao benchmark")
        
            # --- cálculo de betas na MESMA janela da matriz (winN) ---
            # (se por algum motivo winN não existir, cai no padrão 60)
            beta_win = int(winN) if "winN" in locals() else 60
        
            # retornos de todos os papéis + benchmark
            rets_all_risk = px_df.pct_change()
            bench_ret_risk = bench.pct_change().rename(BENCH_USED)
        
            rets_all_risk, bench_ret_risk = rets_all_risk.align(bench_ret_risk, join="inner", axis=0)
        
            rt_win_beta = pd.concat([rets_all_risk, bench_ret_risk], axis=1).dropna(how="all").tail(beta_win)
        
            min_obs_beta = max(5, int(beta_win * 0.4))
        
            if rt_win_beta.shape[0] < min_obs_beta:
                st.info("Dados insuficientes na janela selecionada para calcular os betas.")
            else:
                betas = []
                bench_col = bench_ret_risk.name
                # apenas colunas que realmente existem na janela
                cols_beta = [c for c in px_df.columns if c in rt_win_beta.columns]
        
                for tk in cols_beta:
                    y_full = rt_win_beta[tk]
                    x_full = rt_win_beta[bench_col]
        
                    y, x = y_full.align(x_full, join="inner")
                    if len(y) >= min_obs_beta:
                        x_mean = x.mean()
                        y_mean = y.mean()
                        cov_xy = ((x - x_mean) * (y - y_mean)).sum() / (len(x) - 1)
                        var_x = ((x - x_mean) ** 2).sum() / (len(x) - 1)
                        if var_x != 0 and not np.isnan(var_x):
                            betas.append((tk, float(cov_xy / var_x)))
        
                beta_df = pd.DataFrame(betas, columns=["Ticker", "Beta"]).sort_values("Ticker")
        
                if beta_df.empty:
                    st.info("Não foi possível calcular os betas para este setor.")
                else:
                    beta_med = float(beta_df["Beta"].mean())
                    min_beta = float(beta_df["Beta"].min())
                    max_beta = float(beta_df["Beta"].max())
                    span = max(abs(min_beta), abs(max_beta))
        
                    # seletor de visualização
                    tipo_graf = st.radio(
                        "Tipo de gráfico de sensibilidade:",
                        ["Radar", "Lollipop"],
                        horizontal=True,
                        key="tipo_beta_plot",
                    )
        
                    # =========================
                    # 1) RADAR AJUSTADO
                    # =========================
                    if tipo_graf == "Radar (ajustado)":
                        offset = -min_beta if min_beta < 0 else 0.0
                        beta_df["r_plot"] = beta_df["Beta"] + offset
                        max_r = float(beta_df["r_plot"].max()) * 1.1
        
                        bruto = np.linspace(-span, span, 9)
                        ticks_orig = [v for v in bruto if (min_beta - 1e-9) <= v <= (max_beta + 1e-9)]
                        if 0.0 not in ticks_orig:
                            ticks_orig.append(0.0)
                        ticks_orig = sorted(ticks_orig)
                        ticks_vals = [v + offset for v in ticks_orig]
                        ticks_text = [f"{v:.1f}" for v in ticks_orig]
        
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=beta_df["r_plot"],
                            theta=beta_df["Ticker"],
                            fill="toself",
                            mode="lines+markers",
                            line=dict(color="#3b82f6"),
                            marker=dict(size=6),
                            name="β",
                            customdata=np.c_[beta_df["Beta"]],
                            hovertemplate="%{theta}: β = %{customdata[0]:.2f}<extra></extra>",
                        ))
        
                        fig_radar.update_layout(
                            title="Sensibilidade ao benchmark",
                            height=420,
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max_r],
                                    tickvals=ticks_vals,
                                    ticktext=ticks_text,
                                    gridcolor="#e5e7eb",
                                ),
                                angularaxis=dict(showgrid=False),
                            ),
                            showlegend=False,
                        )
        
                        fig_radar.add_annotation(
                            x=0.5, y=0.5, xref="paper", yref="paper",
                            text=f"β médio = {beta_med:.2f}",
                            showarrow=False, font=dict(size=12, color="#374151")
                        )
        
                        st.plotly_chart(fig_radar, use_container_width=True)
        
                    # =========================
                    # 2) LOLLIPOP
                    # =========================
                    else:
                        beta_df_ord = beta_df.sort_values("Beta")
                        max_abs = span * 1.1 if span > 0 else 1.0
        
                        x_line, y_line = [], []
                        for _, row in beta_df_ord.iterrows():
                            tk = row["Ticker"]
                            b = row["Beta"]
                            x_line.extend([tk, tk, None])
                            y_line.extend([0.0, b, None])
        
                        fig_lolli = go.Figure()
        
                        fig_lolli.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="#93c5fd", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ))
        
                        cores = np.where(beta_df_ord["Beta"] >= 0, "#2563eb", "#dc2626")
                        fig_lolli.add_trace(go.Scatter(
                            x=beta_df_ord["Ticker"],
                            y=beta_df_ord["Beta"],
                            mode="markers",
                            marker=dict(size=9, color=cores),
                            name="β",
                            hovertemplate="%{x}: β = %{y:.2f}<extra></extra>",
                        ))
        
                        fig_lolli.add_hline(y=0.0, line_dash="dash", line_color="#9ca3af")
        
                        fig_lolli.update_layout(
                            title="Sensibilidade ao benchmark",
                            height=420,
                            yaxis=dict(
                                title="β",
                                range=[-max_abs, max_abs],
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor="#9ca3af",
                            ),
                            xaxis=dict(
                                title="Ticker",
                                tickangle=-45,
                            ),
                            margin=dict(l=40, r=10, t=60, b=80),
                        )
        
                        fig_lolli.add_annotation(
                            x=0.01, y=1.08, xref="paper", yref="paper",
                            text=f"β médio = {beta_med:.2f}",
                            showarrow=False, font=dict(size=12, color="#374151"),
                            align="left"
                        )
        
                        st.plotly_chart(fig_lolli, use_container_width=True)


            

    # --- rolling beta, correlação e retorno excedente ---
    st.markdown("---")
    st.subheader("Rolling Beta, Correlação e Retorno Excedente")

    # usa a MESMA janela roll_win definida acima

    # alinha retornos do objeto vs benchmark
    ret_obj, ret_ben = ret_obj.align(ret_ben, join="inner")
    ret_obj = ret_obj.dropna()
    ret_ben = ret_ben.dropna()

    if len(ret_obj) < roll_win + 5:
        st.info("Dados insuficientes para calcular o rolling beta na janela selecionada.")
    else:
        # cálculo consistente com regressão: beta = cov / var; alpha = E[a] - beta * E[b]
        ra = ret_obj
        rb = ret_ben

        ma = ra.rolling(roll_win).mean()
        mb = rb.rolling(roll_win).mean()
        cov_ab = ra.rolling(roll_win).cov(rb)
        var_b  = rb.rolling(roll_win).var()

        beta_roll = cov_ab / var_b
        beta_roll.name = "Beta (rolling)"

        alpha_roll = ma - beta_roll * mb
        alpha_roll.name = "Alpha (rolling)"

        corr_roll = ra.rolling(roll_win).corr(rb).rename("Correlação (rolling)")

        # alinhar tudo no mesmo índice do benchmark
        beta_roll = beta_roll.reindex(rb.index)
        alpha_roll = alpha_roll.reindex(rb.index)
        corr_roll = corr_roll.reindex(rb.index)

        # ancoragem do primeiro valor válido (repete para trás)
        def anchor_from_first_valid(s: pd.Series, base_idx: pd.DatetimeIndex):
            s2 = s.reindex(base_idx)
            if s2.first_valid_index() is None:
                return s2
            first_ix = s2.index.get_loc(s2.first_valid_index())
            s2.iloc[:first_ix] = s2.iloc[first_ix]
            return s2

        base_index = rb.index
        beta_roll = anchor_from_first_valid(beta_roll, base_index)
        corr_roll = anchor_from_first_valid(corr_roll, base_index)

        # retorno excedente: ancorar em zero antes do 1º ponto válido
        exp_ret = alpha_roll + beta_roll * rb
        resid_roll = (ra.reindex(base_index) - exp_ret).rename("Retorno Excedente (rolling)")
        if resid_roll.first_valid_index() is not None:
            first_ix = resid_roll.index.get_loc(resid_roll.first_valid_index())
            resid_roll.iloc[:first_ix] = 0.0

        fig_roll = px.line(
            pd.concat([beta_roll, corr_roll], axis=1),
            x=base_index,
            y=["Beta (rolling)", "Correlação (rolling)"],
            labels={"value": "Valor", "index": "Data", "variable": "Métrica"},
        )
        fig_roll.update_traces(connectgaps=True)
        fig_roll.update_layout(height=350, xaxis_range=[base_index.min(), base_index.max()])
        st.plotly_chart(fig_roll, use_container_width=True)
        
        fig_resid = px.line(
            resid_roll,
            x=base_index,
            y="Retorno Excedente (rolling)",
            labels={"value": "Retorno Excedente (rolling)", "index": "Data"},
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_resid.update_traces(connectgaps=True)
        fig_resid.update_layout(height=300, yaxis_tickformat=".2%", xaxis_range=[base_index.min(), base_index.max()])
        st.plotly_chart(fig_resid, use_container_width=True)


    # --- drawdown preenchido ---
    st.markdown("---")
    st.subheader("Drawdown Comparativo")

    dd_obj = base100_obj / base100_obj.cummax() - 1
    dd_ben = rebase_100(bench) / rebase_100(bench).cummax() - 1
    dd_obj = dd_obj.reindex(dd_ben.index).interpolate().ffill()
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_obj.index, y=dd_obj,
        name=f"DD {obj_name}",
        line=dict(color="rgba(59,130,246,0.9)"), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.25)",
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd_ben.index, y=dd_ben,
        name="DD BOVA11",
        line=dict(color="rgba(239,68,68,0.9)"), fill="tozeroy",
        fillcolor="rgba(239,68,68,0.25)",
    ))
    fig_dd.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig_dd.update_layout(height=300, yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

    
    # --- matriz de correlação (setor/papéis + BOVA11) ---
    st.markdown("---")
    st.subheader("Matriz de Correlação (BOVA11 como referência)")
    
    # montar DataFrame de preços e calcular retornos
    corr_df = pd.concat([bench.rename(BENCHMARK), px_df], axis=1).pct_change()
    
    # garantir que BOVA11 seja o primeiro
    if BENCHMARK in corr_df.columns:
        cols = [BENCHMARK] + [c for c in corr_df.columns if c != BENCHMARK]
        corr_df = corr_df[cols]
    
    # calcular matriz de correlação
    corr_mat = corr_df.corr()
    
    # plotar heatmap
    fig_corr = px.imshow(
        corr_mat,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlação entre papéis e BOVA11",
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)


   

    # --- regressão e resíduos com subperíodo por dias úteis ---
    st.markdown("---")
    st.subheader("Regressão e Resíduos Padronizados")

    n_days = st.slider("Últimos N dias úteis para regressão", 30, 250, 60)
    ret_obj_reg = ret_obj.tail(n_days)
    ret_ben_reg = ret_ben.tail(n_days)

    sc_df = pd.DataFrame({"BOVA11": ret_ben_reg, obj_name: ret_obj_reg}).dropna()
    if sc_df.shape[0] < 5:
        st.info("Janela curta para regressão.")
    else:
        X = sc_df["BOVA11"].values
        Y = sc_df[obj_name].values
        beta_sub, alpha_sub = np.polyfit(X, Y, 1)

        fig_sc = px.scatter(
            sc_df, x="BOVA11", y=obj_name,
            labels={"BOVA11": "Retorno BOVA11", obj_name: f"Retorno {obj_name}"},
        )
        x_line = np.linspace(X.min(), X.max(), 50)
        y_line = beta_sub * x_line + alpha_sub
        fig_sc.add_scatter(
            x=x_line, y=y_line,
            mode="lines", name="Regressão", line=dict(color="red", width=2)
        )
        fig_sc.update_layout(height=350, xaxis_tickformat=".2%", yaxis_tickformat=".2%")
        st.plotly_chart(fig_sc, use_container_width=True)

        # resíduos padronizados (z-score)
        resid = Y - (beta_sub * X + alpha_sub)
        z = (resid - resid.mean()) / resid.std(ddof=1)
        res_df = pd.Series(z, index=sc_df.index, name="Resíduo padronizado").to_frame()

        fig_res = px.line(
            res_df, x=res_df.index, y="Resíduo padronizado",
            labels={"index": "Data", "Resíduo padronizado": "z-score"},
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_res.add_hline(y=2, line_dash="dot", line_color="#ef4444")
        fig_res.add_hline(y=-2, line_dash="dot", line_color="#ef4444")
        fig_res.update_layout(height=300)
        st.plotly_chart(fig_res, use_container_width=True)

        st.caption(
            f"Últimos {n_days} dias úteis · y = {beta_sub:.2f}·x + {alpha_sub:.4f} · "
            )

# =============================
# Aba ORBITAL
# =============================
# ATENÇÃO: ajuste o índice conforme sua ordem de tabs (abaixo assume: 0=Rotação,1=Setor,2=Risco,3=Orbital,4=Avisos)
with tmap["Orbital"]:
    st.subheader("Orbital")

    # -----------------------------
    # Controles de universo / janela
    # -----------------------------
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        setor_orb = st.selectbox("Setor para o Orbital:", list(SETORES_ATIVOS.keys()), key="orb_setor")
    with col2:
        freq_orb = st.radio("Frequência", ["Semanal", "Diária"], index=0, horizontal=True, key="orb_freq")
    with col3:
        lookback_orb = st.slider("Lookback (dias úteis base D1)", 120, 1000, 252, step=10, key="orb_lb")

    freq_key_orb = "W" if freq_orb == "Semanal" else "D"
    AF = 52 if freq_key_orb == "W" else 252

    # Modo de visualização
    modo_orb = st.radio("Modo do Orbital", ["IR × ΔIR", "Retorno × Volatilidade (z-score)"], index=1, horizontal=True, key="orb_modo")

    # Parâmetros dos modos
    colw1, colw2, colw3 = st.columns(3)
    with colw1:
        trail_orb = st.slider("Tamanho da cauda", 4, 24, 8, step=1, key="orb_trail")
    with colw2:
        W_ir = st.slider("Janela IR (rolling)", 20, 180, 90, step=5, key="orb_W_ir")
    with colw3:
        M_delta = st.slider("ΔIR (períodos)", 5, 60, 30, step=5, key="orb_M_delta")

    # Parâmetros da visão Retorno×Vol
    colrv1, colrv2, colrv3 = st.columns(3)
    with colrv1:
        W_rv = st.slider("Janela Ret/Vol (rolling)", 20, 180, 60, step=5, key="orb_W_rv")
    with colrv2:
        zscore_on = st.checkbox("Padronizar (z-score) ret/vol", value=True, key="orb_z")
    with colrv3:
        usar_excesso_bench = st.checkbox("Usar excesso vs BOVA11 (apenas na visão IR×ΔIR)", value=True, key="orb_excesso")

    tickers_orb = SETORES_ATIVOS[setor_orb]

    # -----------------------------
    # Dados: benchmark (já carregado anteriormente como bench_px) e preços dos papéis
    # Para robustez, recarregamos para o lookback específico do Orbital
    # -----------------------------
    # benchmark
    # --- Usa universo cacheado ---
    px_univ_orb = load_universe_prices(freq_key=freq_key_orb, max_lookback=lookback_orb)
    
    if px_univ_orb.dropna(how="all").empty:
        st.error("Orbital: não consegui obter histórico de preços do universo.")
        st.stop()
    
    if BENCHMARK not in px_univ_orb.columns:
        st.error(f"Orbital: {BENCHMARK} não encontrado no universo carregado.")
        st.stop()
    
    bench_px_orb = px_univ_orb[BENCHMARK].dropna().rename(BENCHMARK)
    
    # papéis do setor
    px_df_orb = px_univ_orb[[c for c in tickers_orb if c in px_univ_orb.columns]].dropna(how="all")

    # cobertura mínima (mantém sua regra)
    min_rows = 20 if freq_key_orb == "W" else 60
    good_cols = [c for c in px_df_orb.columns if px_df_orb[c].dropna().shape[0] >= min_rows]
    px_df_orb = px_df_orb[good_cols]

    if px_df_orb.empty or bench_px_orb.empty:
        st.warning("Orbital: sem dados suficientes para montar a visão.")
        st.stop()

    # Alinhar com benchmark
    px_df_orb, bench_px_orb = px_df_orb.align(bench_px_orb, join="inner", axis=0)
    if px_df_orb.empty:
        st.warning("Orbital: interseção de datas vazia após alinhamento.")
        st.stop()

    # -----------------------------
    # Cálculos dos dois modos
    # -----------------------------
    def _last_trail_paths(df_points: pd.DataFrame, trail: int) -> pd.DataFrame:
        """Recebe pontos por ticker (index datetime, columns=['x','y','Ticker']); devolve últimas trilhas concatenadas."""
        rows = []
        for tk, g in df_points.groupby("Ticker"):
            g = g.sort_values("Date").tail(trail).copy()
            rows.append(g)
        return pd.concat(rows, axis=0) if rows else pd.DataFrame()

    # ---------- MODO 1: IR × ΔIR ----------
    # IR: razão do excesso médio / desvio do excesso (rolling W_ir)
    # ΔIR: IR_t − IR_{t−M_delta}
    df_ir = None
    if modo_orb == "IR × ΔIR":
        rets = px_df_orb.pct_change()
        ret_b = bench_px_orb.pct_change()
        rets, ret_b = rets.align(ret_b, join="inner", axis=0)

        if usar_excesso_bench:
            exc = rets.sub(ret_b, axis=0)
        else:
            exc = rets.copy()

        # IR por papel
        roll_mu = exc.rolling(W_ir).mean()
        roll_sd = exc.rolling(W_ir).std(ddof=1)
        ir = roll_mu / (roll_sd.replace(0, np.nan) + 1e-12)
        
        # ΔIR
        delta_ir = ir - ir.shift(M_delta)

        # Monta dataframe longo para trails
        frames = []
        for c in ir.columns:
            tmp = pd.DataFrame({
                "Date": ir.index,
                "x": ir[c].values,
                "y": delta_ir[c].values,
                "Ticker": c
            }).dropna()
            frames.append(tmp)
        df_ir = pd.concat(frames, axis=0) if frames else pd.DataFrame()
        df_trails = _last_trail_paths(df_ir, trail_orb)

        title_mode = "IR × ΔIR (excesso vs BOVA11)" if usar_excesso_bench else "IR × ΔIR (retornos absolutos)"

    # ---------- MODO 2: Retorno × Vol (z-score por padrão) ----------
    df_rv = None
    if modo_orb == "Retorno × Volatilidade (z-score)":
        rets = px_df_orb.pct_change()
        # rolling retorno anualizado simples e vol anualizada
        ret_ann = rets.rolling(W_rv).mean()*AF
        vol_ann = rets.rolling(W_rv).std(ddof=1)*np.sqrt(AF)

        frames = []
        for c in px_df_orb.columns:
            x = ret_ann[c].copy()
            y = vol_ann[c].copy()
            if zscore_on:
                # z-score em janelas rolling (centrado no regime da própria janela)
                mu_x = x.rolling(W_rv).mean()
                sd_x = x.rolling(W_rv).std(ddof=1)
                mu_y = y.rolling(W_rv).mean()
                sd_y = y.rolling(W_rv).std(ddof=1)
                x = (x - mu_x) / (sd_x + 1e-12)
                y = (y - mu_y) / (sd_y + 1e-12)
            tmp = pd.DataFrame({
                "Date": ret_ann.index,
                "x": x.values,
                "y": y.values,
                "Ticker": c
            }).dropna()
            frames.append(tmp)
        df_rv = pd.concat(frames, axis=0) if frames else pd.DataFrame()
        df_trails = _last_trail_paths(df_rv, trail_orb)

        title_mode = "Retorno × Vol (z-score ON)" if zscore_on else "Retorno × Vol (níveis)"

    # -----------------------------
    # Plot Orbital (quadrantes)
    # -----------------------------
    if df_trails is None or df_trails.empty:
        st.warning("Orbital: sem pontos suficientes com as janelas atuais.")
    else:
        # Ponto atual (último de cada ticker)
        last_pts = df_trails.sort_values("Date").groupby("Ticker").tail(1)

        fig_orb = px.scatter(
            last_pts, x="x", y="y", color="Ticker",
            hover_name="Ticker",
            hover_data={"x":":.2f","y":":.2f","Ticker":True},
            title=title_mode
        )
        fig_orb.update_traces(marker=dict(size=12), selector=dict(mode="markers"))
        # Trails
        for tk, g in df_trails.groupby("Ticker"):
            g = g.sort_values("Date")
            fig_orb.add_scatter(
                x=g["x"], y=g["y"], mode="lines+markers",
                name=f"{tk} (trail)", showlegend=False, opacity=0.5
            )

        # Eixos centrados (z-score já centraliza em 0; IR×ΔIR também costuma oscilar em torno de 0)
        xs = df_trails["x"].astype(float); ys = df_trails["y"].astype(float)
        dx = float(max(abs(xs.min()), abs(xs.max()), 1.0))
        dy = float(max(abs(ys.min()), abs(ys.max()), 1.0))
        padx = max(1.0, dx*1.15); pady = max(1.0, dy*1.15)
        fig_orb.update_xaxes(range=[-padx, padx], zeroline=True, zerolinewidth=1, zerolinecolor="#9ca3af")
        fig_orb.update_yaxes(range=[-pady, pady], zeroline=True, zerolinewidth=1, zerolinecolor="#9ca3af")
        fig_orb.add_vline(x=0, line_dash="dash", line_width=1)
        fig_orb.add_hline(y=0, line_dash="dash", line_width=1)

        # Backgrounds suaves dos quadrantes
        lead_color="#22c55e"; weak_color="#f59e0b"; lagg_color="#ef4444"; impr_color="#60a5fa"
        x0, x1 = -padx, padx; y0, y1 = -pady, pady
        fig_orb.add_shape(type="rect", x0=0, x1=x1, y0=0,  y1=y1, fillcolor=lead_color, opacity=0.08, line_width=0)
        fig_orb.add_shape(type="rect", x0=0, x1=x1, y0=y0, y1=0,  fillcolor=weak_color, opacity=0.08, line_width=0)
        fig_orb.add_shape(type="rect", x0=x0, x1=0,  y0=y0, y1=0,  fillcolor=lagg_color, opacity=0.08, line_width=0)
        fig_orb.add_shape(type="rect", x0=x0, x1=0,  y0=0,  y1=y1, fillcolor=impr_color, opacity=0.08, line_width=0)

        # Rótulos
        fig_orb.add_annotation(x=+dx*0.55, y=+dy*0.55, text="Q1", showarrow=False, font=dict(size=12, color="#6b7280"))
        fig_orb.add_annotation(x=+dx*0.55, y=-dy*0.55, text="Q2", showarrow=False, font=dict(size=12, color="#6b7280"))
        fig_orb.add_annotation(x=-dx*0.55, y=-dy*0.55, text="Q3", showarrow=False, font=dict(size=12, color="#6b7280"))
        fig_orb.add_annotation(x=-dx*0.55, y=+dy*0.55, text="Q4", showarrow=False, font=dict(size=12, color="#6b7280"))

        fig_orb.update_layout(height=620, xaxis_title="Eixo X (conforme modo)", yaxis_title="Eixo Y (conforme modo)")
        st.plotly_chart(fig_orb, use_container_width=True)

    # =========================================================
    # TABELA + BARRAS — Retorno relativo (ativo − BOVA11): 1D, MTD, YTD, 12M
    # =========================================================
    st.markdown("---")
    st.subheader("Retorno relativo (ativo − BOVA11): 1D, MTD, YTD, 12M")

    # Para cortes de calendário, melhor base diária de negócios
    px_daily = px_df_orb.asfreq("B").ffill()
    bench_daily = bench_px_orb.asfreq("B").ffill()
    px_daily, bench_daily = px_daily.align(bench_daily, join="inner", axis=0)

    if px_daily.empty or bench_daily.empty:
        st.info("Sem base diária suficiente para calcular 1D/MTD/YTD/12M.")
        st.stop()
    else:
        dt_last = px_daily.index.max()
        # inícios de MTD e YTD (calendário)
        month_start = dt_last.replace(day=1)
        year_start  = dt_last.replace(month=1, day=1)
        # 12M (aprox. 252 dias úteis atrás)
        from_12m = px_daily.index[ max(0, len(px_daily.index)-252) ]

        def _period_ret(series: pd.Series, start_dt, end_dt) -> float:
            s = series.loc[(series.index >= start_dt) & (series.index <= end_dt)]
            if s.shape[0] < 2: return np.nan
            return float(s.iloc[-1] / s.iloc[0] - 1.0)

        rows = []
        for tk in px_daily.columns:
            s_tk = px_daily[tk].dropna()
            # 1D: último retorno
            r1_tk = s_tk.pct_change().iloc[-1] if s_tk.shape[0] >= 2 else np.nan
            r1_bv = bench_daily.pct_change().iloc[-1] if bench_daily.shape[0] >= 2 else np.nan

            rel_1d  = r1_tk - r1_bv
            rel_mtd = _period_ret(s_tk, month_start, dt_last) - _period_ret(bench_daily, month_start, dt_last)
            rel_ytd = _period_ret(s_tk, year_start,  dt_last) - _period_ret(bench_daily, year_start,  dt_last)
            rel_12m = _period_ret(s_tk, from_12m,    dt_last) - _period_ret(bench_daily, from_12m,    dt_last)

            rows.append([tk, rel_1d, rel_mtd, rel_ytd, rel_12m])

        tb = pd.DataFrame(rows, columns=["Ticker","1D","MTD","YTD","12M"]).set_index("Ticker")
        # Exibe tabela formatada (%)
        st.dataframe(tb.applymap(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"), use_container_width=True)
        
        
        
        # --- Barras agrupadas: retornos relativos por período (1D, MTD, YTD, 12M) ---
        st.subheader("Retorno relativo vs. BOVA11 por período")

        # Prepara tabela em formato "longo"
        ret_tbl = tb.reset_index()  # tb tinha o índice "Ticker"
        ord_periodos = ["1D", "MTD", "YTD", "12M"]

        # Ordena os tickers por YTD (opcional, para leitura melhor)
        if "YTD" in ret_tbl.columns:
            order_tickers = ret_tbl.sort_values("YTD", ascending=False)["Ticker"].tolist()
        else:
            order_tickers = ret_tbl["Ticker"].tolist()

        ret_long = ret_tbl.melt(
            id_vars="Ticker",
            value_vars=ord_periodos,
            var_name="Período",
            value_name="Retorno"
        ).dropna()

        ret_long["Período"] = pd.Categorical(ret_long["Período"], categories=ord_periodos, ordered=True)

        fig_barras = px.bar(
            ret_long,
            x="Ticker",
            y="Retorno",
            color="Período",
            barmode="group",
            category_orders={"Ticker": order_tickers, "Período": ord_periodos},
            labels={"Retorno": "Excesso vs BOVA11", "Ticker": "Ativo"}
        )
        fig_barras.update_layout(height=520, xaxis_tickangle=-45, legend_title_text="Período")
        fig_barras.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_barras, use_container_width=True)

    
# =============================
# Aba L&S — Scanner + Par (EG/Kalman)
# =============================

DEFAULT_WINDOWS = [30, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250]


# ===== Helpers estatísticos e de modelo =====

def _beta_eg(x: pd.Series, y: pd.Series) -> float:
    x, y = x.align(y, join="inner")
    if len(x) < 5 or len(y) < 5: return np.nan
    x = x - x.mean(); y = y - y.mean()
    denom = np.sum(x**2)
    if denom == 0: return np.nan
    return float(np.sum(x * y) / denom)

def _spread_from_beta(y: pd.Series, x: pd.Series, beta) -> pd.Series:
    y, x = y.align(x, join="inner")
    if isinstance(beta, (float, int, np.floating)):
        return (y - beta * x).dropna()
    elif isinstance(beta, pd.Series):
        beta = beta.reindex(y.index).ffill()
        return (y - beta * x).dropna()
    return pd.Series(dtype=float, name="spread")

def _adf_pvalue(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 10: return np.nan
    try: return float(adfuller(spread, maxlag=1, autolag=None)[1])
    except Exception: return np.nan

def _half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 10: return np.nan
    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    spread_lag, delta = spread_lag.align(delta, join="inner")
    X = np.vstack([spread_lag, np.ones(len(spread_lag))]).T
    try:
        beta = np.linalg.lstsq(X, delta, rcond=None)[0][0]
        hl = -np.log(2) / beta if beta != 0 else np.nan
        return float(hl) if hl > 0 else np.nan
    except Exception:
        return np.nan

def _zscore(spread: pd.Series, window: int = 20) -> float:
    if len(spread.dropna()) < window: return np.nan
    s = spread.dropna().iloc[-window:]
    return float((s.iloc[-1] - s.mean()) / s.std(ddof=1))

def _kalman_beta(y: pd.Series, x: pd.Series, q=1e-6, r=1e-3) -> pd.Series:
    y, x = y.align(x, join="inner")
    n = len(y)
    if n < 5: return pd.Series(dtype=float, name="beta_kf")
    beta = np.zeros(n); P = np.zeros(n)
    beta[0] = 1.0; P[0] = 1.0
    for t in range(1, n):
        beta_pred = beta[t-1]; P_pred = P[t-1] + q
        H = x.iloc[t]
        if np.isnan(H) or np.isnan(y.iloc[t]):
            beta[t] = beta_pred; P[t] = P_pred; continue
        K = P_pred * H / (H**2 * P_pred + r)
        beta[t] = beta_pred + K * (y.iloc[t] - H * beta_pred)
        P[t] = (1 - K * H) * P_pred
    return pd.Series(beta, index=y.index, name="beta_kf")

with tmap["L&S"]:
    # estados seguros
    st.session_state.setdefault("ls_scan_result", None)
    st.session_state.setdefault("ls_scan_params", None)
    st.session_state.setdefault("ls_last_scan_ts", None)
    st.session_state.setdefault("ls_pair_A", "")
    st.session_state.setdefault("ls_pair_B", "")

    st.subheader("Long & Short — Cointegração")

    # -----------------------------
    # CONTROLES DO SCANNER (Form para evitar rerun contínuo)
    # -----------------------------
    with st.form("ls_scan_form", clear_on_submit=False):
        colU, colM, colF = st.columns([2,1.2,1.2])
        with colU:
            universo_mode = st.radio(
                "Universo do scanner",
                ["Intra-setor", "Todos vs BOVA11"],
                index=0, horizontal=False
            )
            setor_scan = st.selectbox("Setor:", list(SETORES_ATIVOS.keys()))
        with colM:
            st.caption("Frequência: **Diária**")
            freq_key_ls = "D"
            lookback_ls = st.slider("Lookback (base D1)", 120, 1000, 252, step=10)

        with colF:
            windows_ui = st.multiselect("Janelas para ADF", DEFAULT_WINDOWS, default=DEFAULT_WINDOWS)
            windows = sorted(set(int(x) for x in (windows_ui if windows_ui else DEFAULT_WINDOWS)))
            need_wins = st.number_input("Mín. janelas 'vencedoras'",
                                        min_value=1, max_value=len(DEFAULT_WINDOWS),
                                        value=min(5, len(windows)), step=1)
        run_scan = st.form_submit_button("Rodar scanner", use_container_width=True)

    # -----------------------------
    # EXECUÇÃO DO SCANNER
    # -----------------------------
    if run_scan:
        with st.spinner("Executando scanner..."):
            px_univ_ls = load_universe_prices(freq_key=freq_key_ls, max_lookback=lookback_ls)
            if px_univ_ls.dropna(how="all").empty or BENCHMARK not in px_univ_ls.columns:
                st.error("Não consegui obter BOVA11 / universo de papéis para o scanner.")
            else:
                bench_s = px_univ_ls[BENCHMARK].dropna()
                needed = (max(windows) + 5) if windows else (max(DEFAULT_WINDOWS) + 5)
    
                # <<< inicializa aqui, antes dos dois modos >>>
                out_rows = []
    
                if universo_mode == "Todos vs BOVA11":
                    universe = _flatten_universe(SETORES_ATIVOS)
                    # usa somente os tickers do universo que existem no cache
                    px_df_all = px_univ_ls[[c for c in universe if c in px_univ_ls.columns]].dropna(how="all")
                    if px_df_all.empty:
                        st.warning("Sem dados suficientes no universo de papéis.")
                    else:
                        px_df_all, bench_s2 = px_df_all.align(bench_s, join="inner", axis=0)
                        for tk in px_df_all.columns:
                            sY = px_df_all[tk].dropna()
                            sX = bench_s2.dropna()
                            sX, sY = sX.align(sY, join="inner")
                            if len(sX) < needed or len(sY) < needed:
                                continue
    
                            pv_list, beta_list, hl_list = [], [], []
                            for W in windows:
                                swX = sX.tail(W)
                                swY = sY.tail(W)
                                beta = _beta_eg(swX, swY)  # Y ~ beta*X
                                if np.isnan(beta):
                                    pv_list += [np.nan]
                                    beta_list += [np.nan]
                                    hl_list += [np.nan]
                                    continue
                                spread = _spread_from_beta(swY, swX, beta)
                                pv = _adf_pvalue(spread)
                                hl = _half_life(spread)
                                pv_list.append(pv)
                                beta_list.append(beta)
                                hl_list.append(hl)
    
                            pv_arr = np.array([p for p in pv_list if pd.notna(p)])
                            beta_arr = np.array([b for b in beta_list if pd.notna(b)])
                            hl_arr = np.array([h for h in hl_list if pd.notna(h)])
                            wins_windows = [
                                int(W) for (W, pv) in zip(windows, pv_list)
                                if pd.notna(pv) and pv < 0.05
                            ]
                            wins = len(wins_windows)
    
                            beta_used = (
                                float(np.nanmedian(beta_arr))
                                if beta_arr.size
                                else (float(beta_list[-1]) if beta_list else np.nan)
                            )
                            z_now = (
                                _zscore(_spread_from_beta(sY, sX, beta_used), window=20)
                                if pd.notna(beta_used) else np.nan
                            )
    
                            out_rows.append(
                                [
                                    tk,
                                    wins,
                                    float(np.nanmedian(hl_arr)) if hl_arr.size else np.nan,
                                    z_now,
                                    ", ".join(map(str, wins_windows)),
                                ]
                            )
    
                else:
                    tickers = SETORES_ATIVOS[setor_scan]
                    px_df_all = px_univ_ls[[c for c in tickers if c in px_univ_ls.columns]].dropna(how="all")
                    if px_df_all.empty:
                        st.warning("Sem dados suficientes para o setor selecionado.")
                    else:
                        for a, b in itertools.combinations(px_df_all.columns, 2):
                            sA = px_df_all[a].dropna()
                            sB = px_df_all[b].dropna()
                            sA, sB = sA.align(sB, join="inner")
                            if len(sA) < needed or len(sB) < needed:
                                continue
    
                            pv_list, beta_list, hl_list = [], [], []
                            for W in windows:
                                swA = sA.tail(W)
                                swB = sB.tail(W)
                                beta = _beta_eg(swB, swA)  # A ~ beta*B
                                if np.isnan(beta):
                                    pv_list += [np.nan]
                                    beta_list += [np.nan]
                                    hl_list += [np.nan]
                                    continue
                                spread = _spread_from_beta(swA, swB, beta)
                                pv = _adf_pvalue(spread)
                                hl = _half_life(spread)
                                pv_list.append(pv)
                                beta_list.append(beta)
                                hl_list.append(hl)
    
                            pv_arr = np.array([p for p in pv_list if pd.notna(p)])
                            beta_arr = np.array([b for b in beta_list if pd.notna(b)])
                            hl_arr = np.array([h for h in hl_list if pd.notna(h)])
                            wins_windows = [
                                int(W) for (W, pv) in zip(windows, pv_list)
                                if pd.notna(pv) and pv < 0.05
                            ]
                            wins = len(wins_windows)
    
                            beta_used = (
                                float(np.nanmedian(beta_arr))
                                if beta_arr.size
                                else (float(beta_list[-1]) if beta_list else np.nan)
                            )
                            z_now = (
                                _zscore(_spread_from_beta(sA, sB, beta_used), window=20)
                                if pd.notna(beta_used) else np.nan
                            )
    
                            out_rows.append(
                                [
                                    f"{a}/{b}",
                                    wins,
                                    float(np.nanmedian(hl_arr)) if hl_arr.size else np.nan,
                                    z_now,
                                    ", ".join(map(str, wins_windows)),
                                ]
                            )
    
                if out_rows:
                    df_scan = (
                        pd.DataFrame(
                            out_rows,
                            columns=[
                                "Ativo/Par",
                                "Wins(ADF<0.05)",
                                "Half-life(med)",
                                "z-score(20)",
                                "Janelas_OK",
                            ],
                        )
                        .sort_values(
                            ["Wins(ADF<0.05)", "Half-life(med)"],
                            ascending=[False, True],
                        )
                        .reset_index(drop=True)
                    )
    
                    df_scan_filtrado = df_scan.loc[
                        df_scan["Wins(ADF<0.05)"] >= int(need_wins)
                    ].reset_index(drop=True)
                    st.session_state["ls_scan_result"] = df_scan_filtrado
                    st.session_state["ls_scan_params"] = dict(
                        universo_mode=universo_mode,
                        setor=setor_scan,
                        freq_key=freq_key_ls,
                        lookback=lookback_ls,
                        windows=windows,
                        need_wins=int(need_wins),
                    )
                    st.session_state["ls_last_scan_ts"] = datetime.now()
                else:
                    st.session_state["ls_scan_result"] = None
                    st.info("Scanner: nenhum par passou nos critérios atuais.")
    
    
    # -----------------------------
    # MOSTRAR RESULTADOS DO SCANNER (SE EXISTIREM)
    # -----------------------------
    colL, colR = st.columns([2,1])
    with colL:
        st.markdown("### Resultados do Scanner")
        if st.session_state["ls_scan_result"] is not None and not st.session_state["ls_scan_result"].empty:
            st.dataframe(st.session_state["ls_scan_result"], use_container_width=True)
        else:
            st.info("Rode o scanner para ver resultados.")
    with colR:
        st.markdown("### Estado")
        if st.session_state["ls_last_scan_ts"] is not None:
            st.success(f"Resultados salvos em: {st.session_state['ls_last_scan_ts'].strftime('%H:%M:%S')}")
        if st.session_state["ls_scan_params"] is not None:
            p = st.session_state["ls_scan_params"]
            st.caption(
                f"Universo: {p['universo_mode']} · Setor: {p['setor']} · "
                f"Freq: Diária · "
                f"Lookback: {p['lookback']} · Janelas: {p['windows']} · "
                f"Mín. wins: {p['need_wins']}"
            )



    st.markdown("---")

    # -----------------------------
    # ANÁLISE DO PAR — seleção manual e motor (EG x Kalman)
    # -----------------------------
    st.markdown("## Análise detalhada do par")

    universe_all = sorted(set(_flatten_universe(SETORES_ATIVOS) + [BENCHMARK]))
    if not st.session_state["ls_pair_A"]:
        st.session_state["ls_pair_A"] = (universe_all[0] if universe_all else BENCHMARK)
    if not st.session_state["ls_pair_B"]:
        st.session_state["ls_pair_B"] = (universe_all[1] if len(universe_all) > 1 else BENCHMARK)

    csel1, csel2, csel3, csel4 = st.columns([1.4,1.4,1.0,1.2])
    with csel1:
        st.session_state["ls_pair_A"] = st.selectbox(
            "Ativo A (Y)", options=universe_all,
            index=universe_all.index(st.session_state["ls_pair_A"]) if st.session_state["ls_pair_A"] in universe_all else 0,
            key="ls_pair_A_select"
        )
    with csel2:
        st.session_state["ls_pair_B"] = st.selectbox(
            "Ativo B (X)", options=universe_all,
            index=universe_all.index(st.session_state["ls_pair_B"]) if st.session_state["ls_pair_B"] in universe_all else min(1, len(universe_all)-1),
            key="ls_pair_B_select"
        )
    with csel3:
        hedge_mode = st.radio("Hedge", ["EG (fixo)","Kalman (dinâmico)"], index=0, horizontal=False)
    with csel4:
        z_win = st.slider("Janela z-score", 10, 250, 20, step=1)

    cpar1, cpar2, cpar3 = st.columns(3)
    with cpar1:
        st.caption("Frequência (par): **Diária**")
        freq_key_pair = "D"
    with cpar2:
        lookback_pair = st.slider("Lookback (par, base D1)", 120, 1500, 750, step=30)
    with cpar3:
        use_bench_overlay = st.checkbox("Mostrar BOVA11 normalizado no gráfico", value=False)

    # Dados do par
    px_univ_pair = load_universe_prices(freq_key=freq_key_pair, max_lookback=lookback_pair)

    symA = st.session_state["ls_pair_A"]
    symB = st.session_state["ls_pair_B"]
    
    if symA not in px_univ_pair.columns or symB not in px_univ_pair.columns:
        st.warning("Sem dados suficientes para o par selecionado."); st.stop()
    
    sA = px_univ_pair[symA].dropna()
    sB = px_univ_pair[symB].dropna()
    
    if sA.empty or sB.empty:
        st.warning("Sem dados suficientes para o par selecionado."); st.stop()

    # Hedge e spread
    if hedge_mode.startswith("EG"):
        beta = _beta_eg(sB, sA)  # A = beta * B
        spread = _spread_from_beta(sA, sB, beta)
        beta_last = beta; beta_series = None
    else:
        beta_series = _kalman_beta(sA, sB, q=1e-6, r=1e-4)
        beta_last = float(beta_series.dropna().iloc[-1]) if beta_series is not None and not beta_series.dropna().empty else np.nan
        spread = _spread_from_beta(sA, sB, beta_series if beta_series is not None else beta_last)

    # Métricas do spread
    # === Métricas do spread (usando a mesma janela do slider z_win) ===
    # z-score rolling robusto (mesma lógica do gráfico)
    ma_win = spread.rolling(int(z_win), min_periods=max(5, int(z_win)//2)).mean()
    sd_win = spread.rolling(int(z_win), min_periods=max(5, int(z_win)//2)).std(ddof=1)
    z_roll = (spread - ma_win) / sd_win
    
    # z-score atual (mostra sempre que houver dados suficientes)
    z_now = float(z_roll.dropna().iloc[-1]) if z_roll.notna().any() else np.nan
    
    # half-life + pvalue como já fazia
    pv = _adf_pvalue(spread)
    hl = _half_life(spread)
    
    # Volatilidade do par na MESMA janela:
    # usamos a variação do spread (Δspread), que é o que gera P&L do par,
    
    rA = sA.pct_change()
    rB = sB.pct_change()
    
    # pesos dollar-neutral no t0 (igualando notionals das pernas)
    # wA = +1 "unidade" de A; wB compensa notional com o hedge beta
    if pd.notna(beta_last):
        wA = 1.0
        wB = float(beta_last) * (sA.iloc[0] / sB.iloc[0])  # equaliza R$ nas pernas
    else:
        wA, wB = 1.0, (sA.iloc[0] / sB.iloc[0])  # fallback simples
    
    # retorno do par (dimensionalmente consistente)
    r_pair = (wA * rA - wB * rB)
    
    # mesma janela da régua do z-score
    win = int(z_win)
    r_pair_win = r_pair.rolling(win, min_periods=max(5, win//2))
    
    vol_par_ann = float(r_pair_win.std(ddof=1).iloc[-1] * np.sqrt(252)) if r_pair_win.count().iloc[-1] >= max(5, win//2) else np.nan


    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("p-value (ADF)", f"{pv:.3f}" if pd.notna(pv) else "—")
    cc2.metric("Half-life (barras)", f"{hl:.1f}" if pd.notna(hl) else "—")
    cc3.metric("β (último)", f"{beta_last:.3f}" if pd.notna(beta_last) else "—")
    cc4.metric(f"z-score ({int(z_win)})", f"{z_now:.2f}" if pd.notna(z_now) else "—",
               help="z-score do spread usando média e desvio rolling na janela selecionada.")
    cc5.metric(f"Vol do par ({int(z_win)})",
           f"{vol_par_ann*100:.2f}%" if pd.notna(vol_par_ann) else "—",
           help="Vol anualizada do retorno do par (long A, short β·B) dollar-neutral, na janela selecionada.")

    # ---- Gráfico 1: Preço normalizado (Base 100) + BOVA opcional
    df_ab = pd.concat([sA.rename(st.session_state["ls_pair_A"]),
                       sB.rename(st.session_state["ls_pair_B"])], axis=1).dropna()
    base100 = 100 * df_ab / df_ab.iloc[0]  # normalização conjunta (mesmo ponto-base)
    fig_p = px.line(base100, x=base100.index, y=base100.columns,
                    labels={"value":"Base 100","index":"Data","variable":"Preço normalizado"})
    fig_p.update_layout(height=380, legend_title_text="Preço normalizado")
    if use_bench_overlay:
        sBench = _get_price_series(BENCHMARK, bars_pair, freq_key_pair)
        if not sBench.empty:
            sBench = sBench.reindex(base100.index).dropna()
            if not sBench.empty:
                sBenchN = 100 * sBench / sBench.iloc[0]
                fig_p.add_scatter(x=sBenchN.index, y=sBenchN.values, name=BENCHMARK, line=dict(width=1, dash="dot"))
    st.plotly_chart(fig_p, use_container_width=True)

    # ---- Diferença de retorno acumulada (A − B) com janela selecionável
    st.markdown("### Diferença de retorno acumulada (A − B)")
    ret_win = st.slider("Janela do retorno acumulado (dias/barras)", 5, 252, 22, 1, key="ret_win_pair")
    retA = sA.pct_change().dropna()
    retB = sB.pct_change().dropna()
    retA, retB = retA.align(retB, join="inner")
    accA = (1.0 + retA).rolling(ret_win).apply(np.prod, raw=True) - 1.0
    accB = (1.0 + retB).rolling(ret_win).apply(np.prod, raw=True) - 1.0
    diff_win = (accA - accB).rename(f"Diferença acumulada (A − B) — janela {ret_win}")
    if diff_win.dropna().empty:
        st.info("Sem dados suficientes para calcular a diferença acumulada (verifique janela/lookback).")
    else:
        fig_diff = px.line(diff_win.dropna(), x=diff_win.dropna().index, y=diff_win.dropna().name,
                           labels={"value":"Diferença acumulada", "index":"Data"})
        fig_diff.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_diff.update_layout(height=360)
        st.plotly_chart(fig_diff, use_container_width=True)

    # ---- Z-score do spread + média móvel do z (lookback inteiro)
    st.markdown("### Z-score do spread")
    zser = z_roll
    z_ma_win = st.slider("Janela da média móvel do z-score", 5, 120, 10, 1, key="z_ma_win_pair")
    z_ma_ser = zser.rolling(int(z_ma_win)).mean()
    z_df = pd.concat([zser.rename(f"z-score ({z_win})"),
                      z_ma_ser.rename(f"MM z ({z_ma_win})")], axis=1).dropna(how="all")
    if z_df.dropna().empty:
        st.info("Sem dados suficientes para plotar o z-score.")
    else:
        fig_z = px.line(z_df, x=z_df.index, y=z_df.columns,
                        labels={"value":"z-score", "index":"Data", "variable":"Série"})
        # força cor vermelha na linha da média móvel
        fig_z.for_each_trace(lambda tr: tr.update(line=dict(color="#ef4444", width=3))
                             if tr.name.startswith("MM z") else None)

        fig_z.add_hline(y=0,  line_dash="dash", line_color="#9ca3af")
        fig_z.add_hline(y= 2, line_dash="dot",  line_color="#ef4444")
        fig_z.add_hline(y=-2, line_dash="dot",  line_color="#ef4444")
        fig_z.update_layout(height=560, legend_title_text="Série")
        st.plotly_chart(fig_z, use_container_width=True)

    # ---- Kalman: β_t
    if beta_series is not None and not beta_series.dropna().empty:
        fig_b = px.line(beta_series, x=beta_series.index, y="beta_kf",
                        labels={"value":"β_t (Kalman)","index":"Data"})
        fig_b.update_layout(height=260, title="Hedge dinâmico (Filtro de Kalman)")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### Backtest (P&L do spread por sinal de z-score)")
    colSig1, colSig2, colSig3 = st.columns(3)
    with colSig1: z_open  = st.slider("Entrada |z| ≥", 1.0, 3.0, 2.0, 0.1)
    with colSig2: z_close = st.slider("Saída |z| ≤",   0.2, 1.5, 0.8, 0.1)
    with colSig3: z_stop  = st.slider("Stop |z| ≥",    2.0, 5.0, 3.0, 0.1)
    
    z = z_roll.copy().dropna()  # usa o z-score rolling que você já calculou
    if z.empty:
        st.info("Sem z-score suficiente para rodar o backtest (aumente o lookback / reduza a janela).")
    else:
        # alinha spread ao z
        spread_z = spread.reindex(z.index).dropna()
        spread_z, z = spread_z.align(z, join="inner")
    
        pos = 0
        pnl = [0.0]
    
        for i in range(1, len(z)):
            zt = z.iloc[i]
            # lógica de posição
            if pos == 0:
                if zt >= z_open:
                    pos = -1  # spread alto: vende spread (short A, long B*beta)
                elif zt <= -z_open:
                    pos = +1  # spread baixo: compra spread (long A, short B*beta)
            else:
                if abs(zt) >= z_stop or abs(zt) <= z_close:
                    pos = 0
    
            dS = float(spread_z.iloc[i] - spread_z.iloc[i - 1])
            pnl.append(pnl[-1] + pos * dS)
    
        pnl_ser = pd.Series(pnl, index=z.index, name="P&L (proxy)")
        fig_pnl = px.line(pnl_ser, x=pnl_ser.index, y="P&L (proxy)")
        fig_pnl.update_layout(height=260)
        st.plotly_chart(fig_pnl, use_container_width=True)
        st.caption("Observação: P&L do spread é uma **proxy** (não inclui custos, fricções, tamanhos por leg, nem execução).")
    
# =============================
# Helpers Portfólio
# =============================

def _rebase_100_from_returns(r: pd.Series) -> pd.Series:
    """Transforma uma série de retornos em índice base 100."""
    r = r.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    idx = (1.0 + r).cumprod()
    return 100.0 * idx / idx.iloc[0]


def _rebase_100_from_prices(p: pd.Series) -> pd.Series:
    """Transforma uma série de preços em índice base 100."""
    p = p.dropna()
    if p.empty:
        return pd.Series(dtype=float)
    return 100.0 * p / p.iloc[0]


def _portfolio_kpis(idx_port: pd.Series,
                    idx_bench: pd.Series | None,
                    freq_key: str,
                    rf_annual: float = 0.0) -> dict:
    """
    Calcula KPIs básicos do portfólio:
    - retornos (dia, mês, YTD, 12m, total)
    - vol anualizada, Sharpe, MaxDD
    - beta e correlação vs benchmark (se disponível)
    - VaR histórico 95% (1 período)
    """
    out = {}
    idx_port = idx_port.dropna()
    if idx_port.empty:
        return out

    # frequência
    af = 252 if freq_key.upper() == "D" else 52

    # retornos discretos
    r = idx_port.pct_change().dropna()
    if r.empty:
        return out

    # RF por período
    rf_per = (1.0 + rf_annual / 100.0) ** (1.0 / af) - 1.0

    # retorno do dia (último)
    out["ret_day"] = float(r.iloc[-1])

    # Índices para cortes de calendário (usa datas do índice)
    _idx = idx_port.index
    last_dt = _idx[-1]
    cur_month = last_dt.month
    cur_year = last_dt.year

    # Mês corrente
    sel_m = idx_port[_idx.month == cur_month]
    out["ret_month"] = float(sel_m.iloc[-1] / sel_m.iloc[0] - 1.0) if len(sel_m) >= 2 else float("nan")

    # YTD
    sel_ytd = idx_port[_idx.year == cur_year]
    out["ret_ytd"] = float(sel_ytd.iloc[-1] / sel_ytd.iloc[0] - 1.0) if len(sel_ytd) >= 2 else float("nan")

    # 12m (~af períodos)
    win12 = min(len(idx_port), af)
    sel_12m = idx_port.iloc[-win12:]
    out["ret_12m"] = float(sel_12m.iloc[-1] / sel_12m.iloc[0] - 1.0) if len(sel_12m) >= 2 else float("nan")

    # total (desde o início da série)
    out["ret_total"] = float(idx_port.iloc[-1] / idx_port.iloc[0] - 1.0)

    # volatilidade e Sharpe
    exc = r - rf_per
    vol_p = float(exc.std(ddof=1))
    out["vol_ann"] = float(vol_p * np.sqrt(af)) if not np.isnan(vol_p) else float("nan")

    if vol_p and not np.isnan(vol_p):
        sharpe = (float(exc.mean()) / vol_p) * np.sqrt(af)
    else:
        sharpe = float("nan")
    out["sharpe"] = float(sharpe)

    # Max Drawdown
    dd_series = idx_port / idx_port.cummax() - 1.0
    out["max_dd"] = float(dd_series.min()) if not dd_series.empty else float("nan")

    # === NOVO: VaR 95% (1 período) ===
    try:
        raw_var_5 = float(np.percentile(r.values, 5))   # P5 da distribuição de retornos
        out["var_95"] = -raw_var_5 if raw_var_5 < 0 else 0.0  # exibe como perda positiva
    except Exception:
        out["var_95"] = float("nan")

    # Beta e correlação vs benchmark
    out["beta"] = float("nan")
    out["corr"] = float("nan")
    if idx_bench is not None:
        idx_bench = idx_bench.dropna()
        if not idx_bench.empty:
            rp = idx_port.pct_change().dropna()
            rb = idx_bench.pct_change().dropna()
            rp, rb = rp.align(rb, join="inner")
            if len(rp) >= 20:
                cov = np.cov(rb.values, rp.values)[0, 1]
                varb = np.var(rb.values)
                out["beta"] = float(cov / varb) if varb != 0 else float("nan")
                out["corr"] = float(np.corrcoef(rb.values, rp.values)[0, 1])

    return out



def _format_pct(x: float) -> str:
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


def _format_num(x: float, nd=2) -> str:
    return f"{x:.{nd}f}" if pd.notna(x) else "—"
            
# =============================
# Aba PORTFÓLIO
# =============================
with tmap["Portfólio"]:
    st.subheader("Portfólio — Visão Institucional")

    # -----------------------------
    # Controles principais
    # -----------------------------
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        freq_port = st.radio(
            "Frequência",
            ["Diária", "Semanal"],
            index=0,
            horizontal=True,
            key="freq_port",
        )
        freq_key_port = "D" if freq_port == "Diária" else "W"

    with colp2:
        lookback_port = st.slider(
            "Lookback (base D1 / lógica interna)",
            120, 1200, 750, step=30,
            help="Usado para montar o histórico de preços dos ativos da carteira."
        )

    with colp3:
        incluir_ls = st.checkbox(
            "Incluir estratégia L&S atual como ativo",
            value=False,
            help="Usa o par configurado na aba L&S como um ativo sintético na carteira."
        )

    # Taxa livre de risco para Sharpe
    with st.expander("Parâmetros de risco do Portfólio", expanded=True):
        rf_annual_port = st.number_input(
            "Taxa livre de risco (a.a., %)",
            value=10.00,
            step=0.25,
            format="%.2f",
            help="Usada para calcular o Sharpe do portfólio."
        )

    st.markdown("---")

    # -----------------------------
    # 1) Carregar universo de preços
    # -----------------------------
    with st.spinner("Carregando universo de preços para o Portfólio..."):
        # Reaproveita o mesmo helper usado no scanner L&S
        px_univ_port = load_universe_prices(
            freq_key=freq_key_port,
            max_lookback=lookback_port,
        )

    if px_univ_port is None or px_univ_port.dropna(how="all").empty:
        st.error("Não consegui montar o universo de preços para o Portfólio.")
        st.stop()

    # Garante que o benchmark esteja presente
    if BENCHMARK not in px_univ_port.columns:
        st.warning(f"O benchmark {BENCHMARK} não está no universo carregado.")
        bench_port = None
    else:
        bench_port = px_univ_port[BENCHMARK].dropna()

    # universo de ativos para o usuário escolher (exceto benchmark)
    universe_all = [c for c in px_univ_port.columns if c != BENCHMARK]
    universe_all = sorted(universe_all)

    if not universe_all:
        st.error("Universo de ativos vazio para montar a carteira.")
        st.stop()

    # -----------------------------
    # 2) Definição da carteira (pesos)
    # -----------------------------
    st.markdown("### Composição da carteira")

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        sel_assets = st.multiselect(
            "Selecione os ativos da carteira",
            options=universe_all,
            default=universe_all[:5] if len(universe_all) >= 5 else universe_all,
            help="Esses ativos (ou classes) serão usados para compor o portfólio."
        )

    ls_label = None
    s_ls_index = None

    # Ativo sintético L&S (opcional)
    if incluir_ls:
        # tenta aproveitar o par configurado na aba L&S
        pair_A = st.session_state.get("ls_pair_A")
        pair_B = st.session_state.get("ls_pair_B")
        hedge_mode = st.session_state.get("ls_hedge_mode", "EG")  # se quiser salvar isso depois

        if pair_A and pair_B:
            try:
                # mesmo lookback/frequência do portfólio
                bars_pair = min(2000, int(bars_for_freq(freq_key_port, lookback_port)))
                sA = _get_price_series(pair_A, bars_pair, freq_key_port)
                sB = _get_price_series(pair_B, bars_pair, freq_key_port)
                if not sA.empty and not sB.empty:
                    # hedge do par (igual lógica da aba L&S, versão simplificada EG)
                    beta_ls = _beta_eg(sB, sA)
                    spread_ls = _spread_from_beta(sA, sB, beta_ls)
                    # retorno do par = variação do spread (P&L do spread)
                    r_ls = spread_ls.diff().dropna()
                    if not r_ls.empty:
                        s_ls_index = _rebase_100_from_returns(r_ls).rename("LS_pair")
                        ls_label = f"L&S: {pair_A}/{pair_B}"
            except Exception as e:
                st.warning(f"Não foi possível montar o ativo L&S para o portfólio: {e}")

    if incluir_ls and ls_label is not None:
        st.caption(f"Estratégia L&S adicionada como ativo sintético: **{ls_label}**")
        # usuário pode incluir o LS como mais uma linha da carteira
        if ls_label not in sel_assets:
            sel_assets = sel_assets + [ls_label]

    if not sel_assets:
        st.info("Selecione ao menos um ativo para montar a carteira.")
        st.stop()

    # Pesos por ativo (em %)
    with col_sel2:
        st.markdown("#### Pesos aproximados (%)")
        weights_input = {}
        for tk in sel_assets:
            default_w = 100.0 / len(sel_assets)
            w = st.number_input(
                f"Peso de {tk}",
                min_value=0.0,
                max_value=100.0,
                value=float(default_w),
                step=1.0,
                key=f"peso_{tk}",
            )
            weights_input[tk] = w

        total_w = sum(weights_input.values())
        if total_w <= 0:
            st.warning("A soma dos pesos é zero. Ajuste os pesos da carteira.")
            st.stop()
        # normaliza pesos para 100%
        weights_norm = {k: v / total_w for k, v in weights_input.items()}
        st.caption(f"Soma dos pesos informados: {total_w:.2f}%. Usaremos pesos **normalizados** (somando 100%).")

    # -----------------------------
    # 3) Construção da série do portfólio
    # -----------------------------
    # DataFrame com os ativos da carteira
    px_sel = pd.DataFrame(index=px_univ_port.index)

    for tk in sel_assets:
        if tk == ls_label and s_ls_index is not None:
            # ativo sintético L&S
            px_sel = px_sel.join(s_ls_index, how="outer")
            px_sel.rename(columns={"LS_pair": ls_label}, inplace=True)
        else:
            if tk in px_univ_port.columns:
                px_sel[tk] = px_univ_port[tk]

    px_sel = px_sel.dropna(how="all")
    if px_sel.empty:
        st.error("Não foi possível montar as séries de preços dos ativos da carteira.")
        st.stop()

    # converte preços para retornos
    rets_sel = px_sel.pct_change().dropna(how="all")
    # alinha com benchmark para depois calcular KPIs
    if bench_port is not None:
        bench_aligned = bench_port.reindex(rets_sel.index).dropna()
    else:
        bench_aligned = None

    # remove colunas totalmente vazias após %change
    rets_sel = rets_sel.dropna(how="all", axis=1)
    # filtra weights para colunas realmente presentes
    weights_use = {k: w for k, w in weights_norm.items() if k in rets_sel.columns}

    if not weights_use:
        st.error("Nenhum dos ativos com peso possui histórico de retornos suficiente.")
        st.stop()

    # garante que pesos somem 1 no subconjunto final
    s_w = sum(weights_use.values())
    weights_use = {k: v / s_w for k, v in weights_use.items()}

    # monta vetor de pesos na ordem das colunas
    w_vec = np.array([weights_use.get(c, 0.0) for c in rets_sel.columns])

    # retorno do portfólio: combinação linear dos retornos
    ret_port = (rets_sel * w_vec).sum(axis=1)
    idx_port = _rebase_100_from_returns(ret_port)
    
    # -----------------------------
    # Matrizes para decomposição de risco
    # -----------------------------
    # Covariância dos retornos dos ativos da carteira
    cov_mat = rets_sel.cov()
    
    # Série de pesos do portfólio (somando 1) na mesma ordem de rets_sel
    w_port = pd.Series(
        [weights_use.get(c, 0.0) for c in rets_sel.columns],
        index=rets_sel.columns,
        name="Peso"
    )
    
    # Retorno do portfólio alinhado com rets_sel (para tail-risk)
    ret_port_aligned = (rets_sel[w_port.index] * w_port.values).sum(axis=1).dropna()


    # índice do benchmark (base 100) se existir
    if bench_aligned is not None and not bench_aligned.empty:
        idx_bench = _rebase_100_from_prices(bench_aligned)
        idx_bench = idx_bench.reindex(idx_port.index).dropna()
    else:
        idx_bench = None

    # -----------------------------
    # 3) Construção dos KPIs
    # -----------------------------
    # DataFrame com os ativos da carteira
    st.markdown("### Indicadores do Portfólio")

    kpis = _portfolio_kpis(idx_port, idx_bench, freq_key=freq_key_port, rf_annual=rf_annual_port)
    
    if not kpis:
        st.info("Não foi possível calcular os indicadores do portfólio.")
    else:
    
        def _fmt_pct(x: float) -> str:
            return f"{x*100:.2f}%" if pd.notna(x) else "—"
    
        def _fmt_num(x: float, nd: int = 2) -> str:
            return f"{x:.{nd}f}" if pd.notna(x) else "—"
    
        # =======================
        # LINHA 1 — RETORNOS
        # =======================
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Retorno do dia", _fmt_pct(kpis.get("ret_day", float("nan"))))
        c2.metric("Retorno mês (MTD)", _fmt_pct(kpis.get("ret_month", float("nan"))))
        c3.metric("Retorno ano (YTD)", _fmt_pct(kpis.get("ret_ytd", float("nan"))))
        c4.metric("Retorno 12m", _fmt_pct(kpis.get("ret_12m", float("nan"))))
        c5.metric("Retorno Total", _fmt_pct(kpis.get("ret_total", float("nan"))))
    
        # =======================
        # LINHA 2 — RISCO
        # =======================
        r1, r2, r3, r4, r5, r6 = st.columns(6)
    
        r1.metric("Vol anualizada", _fmt_pct(kpis.get("vol_ann", float("nan"))))
        r2.metric("Sharpe", _fmt_num(kpis.get("sharpe", float("nan")), nd=2))
        r3.metric("Máx Drawdown", _fmt_pct(kpis.get("max_dd", float("nan"))))
    
        beta_val = kpis.get("beta", float("nan"))
        corr_val = kpis.get("corr", float("nan"))
        var95_val = kpis.get("var_95", float("nan"))
    
        r4.metric("Beta vs BOVA11", _fmt_num(beta_val, nd=2))
        r5.metric("Correlação vs BOVA11", _fmt_num(corr_val, nd=2))
        r6.metric("VaR 95% (1 período)", _fmt_pct(var95_val))




    # -----------------------------
    # 5) Gráficos: evolução e composição
    # -----------------------------
    st.markdown("---")
    st.markdown("### Evolução do Portfólio vs Benchmark")

    df_idx_plot = pd.DataFrame({"Portfólio": idx_port})
    if idx_bench is not None and not idx_bench.empty:
        df_idx_plot[BENCHMARK] = idx_bench.reindex(idx_port.index)

    fig_idx = px.line(
        df_idx_plot,
        x=df_idx_plot.index,
        y=df_idx_plot.columns,
        labels={"value": "Índice (base=100)", "index": "Data", "variable": "Série"},
    )
    fig_idx.update_layout(height=380, legend_title_text="Série")
    st.plotly_chart(fig_idx, use_container_width=True)

        # --- Composição por pesos ---
    st.subheader("Composição da carteira — Peso × Contribuição de risco")

    # Série de pesos (pode ter long/short)
    w = w_port[w_port != 0].dropna()
    if w.empty:
        st.info("Nenhum ativo com peso diferente de zero na carteira.")
    else:
        # restringe cov_mat e retornos ao subconjunto com peso ≠ 0
        Sigma = cov_mat.loc[w.index, w.index]
        
        # variância total do portfólio (por período)
        port_var = float(w.values @ Sigma.values @ w.values)
        if not np.isfinite(port_var) or port_var <= 0:
            st.info("Não foi possível decompor o risco do portfólio (variância não positiva).")
        else:
            port_vol = float(np.sqrt(port_var))

            # marginal risk (Σ w)
            mrc_vec = Sigma.values @ w.values

            # === 1) Contribuição de risco "clássica" (variância) ===
            rc_var = w * mrc_vec / port_var            # soma ≈ 1, pode ter sinal
            rc_var_pct = rc_var * 100.0                # em %

            # === 2) Contribuição para a VOLATILIDADE (Barra-style) ===
            # soma ≈ σ_portfolio (valor de vol). Em % dá o mesmo perfil da variância.
            rc_vol = w * mrc_vec / port_vol            # contrib. em vol (não %)
            rc_vol_pct = (rc_vol / port_vol) * 100.0   # % da vol (≈ rc_var_pct)

            # === 3) Tail-Risk Contribution (a partir dos piores 5% dias) ===
            tail_rc_pct = pd.Series(index=w.index, dtype=float)

            if ret_port_aligned.shape[0] >= 60:
                thr = float(np.nanpercentile(ret_port_aligned.dropna(), 5))
                mask_tail = ret_port_aligned <= thr
                if mask_tail.sum() >= max(10, len(w) + 2):
                    rets_tail = rets_sel.loc[mask_tail, w.index].dropna(how="all")
                    if rets_tail.shape[0] >= len(w) + 2:
                        Sigma_tail = rets_tail.cov()
                        tail_var = float(w.values @ Sigma_tail.values @ w.values)
                        if np.isfinite(tail_var) and tail_var > 0:
                            mrc_tail = Sigma_tail.values @ w.values
                            tail_rc = w * mrc_tail / tail_var
                            tail_rc_pct = tail_rc * 100.0

            # === 4) Beta e Beta Contribution (se benchmark disponível) ===
            beta_ser = pd.Series(index=w.index, dtype=float)
            beta_ctr_pct = pd.Series(index=w.index, dtype=float)

            if idx_bench is not None and not idx_bench.empty:
                rb = idx_bench.pct_change().dropna()
                # alinha com rets_sel
                rb = rb.reindex(rets_sel.index).dropna()

                for tk in w.index:
                    r_i = rets_sel[tk].dropna()
                    r_i, rb_i = r_i.align(rb, join="inner")
                    if len(rb_i) >= 20:
                        cov_ib = np.cov(rb_i.values, r_i.values)[0, 1]
                        var_b = np.var(rb_i.values)
                        beta_i = cov_ib / var_b if var_b > 0 else np.nan
                        beta_ser.loc[tk] = beta_i

                beta_port = float((beta_ser * w).sum())
                if np.isfinite(beta_port) and beta_port != 0:
                    beta_ctr = beta_ser * w
                    beta_ctr_pct = (beta_ctr / beta_port) * 100.0

            # === Tabela consolidada de risco por ativo ===
            peso_pct = (w / w.sum()) * 100.0 if w.sum() != 0 else w * 0.0
            df_risk = pd.DataFrame({
                "Peso_%": peso_pct,
                "RC_var_%": rc_var_pct,            # contribuição % da variância total
                "RC_vol_%": rc_vol_pct,            # ≈ igual à RC_var_%
                "Tail_RC_%": tail_rc_pct,          # contribuição % na cauda (5%)
                "Beta": beta_ser,
                "Beta_ctr_%": beta_ctr_pct,        # contribuição para o beta do portfólio
            })
            df_risk["Direção"] = np.where(w.values >= 0, "Long", "Short")
            df_risk = df_risk.sort_index()

            st.markdown("#### Tabela de risco por ativo")
            st.dataframe(
                df_risk.style.format({
                    "Peso_%": "{:.2f}",
                    "RC_var_%": "{:.2f}",
                    "RC_vol_%": "{:.2f}",
                    "Tail_RC_%": "{:.2f}",
                    "Beta": "{:.2f}",
                    "Beta_ctr_%": "{:.2f}",
                }),
                use_container_width=True,
            )

            # -------- Eficiência de risco (RC / Peso) --------
            st.markdown("#### Eficiência de risco (RC / Peso)")
            df_eff = df_risk.copy()
            df_eff["RC/Peso"] = df_eff["RC_var_%"] / df_eff["Peso_%"]
            st.dataframe(
                df_eff[["Peso_%", "RC_var_%", "RC/Peso"]].style.format(
                    {"Peso_%": "{:.2f}", "RC_var_%": "{:.2f}", "RC/Peso": "{:.2f}"}
                ),
                use_container_width=True,
            )

            # -------- Barras: Peso × RC lado a lado --------
            st.markdown("#### Peso vs. Contribuição de risco (variância)")

            df_long = df_risk.reset_index(names="Ativo").melt(
                id_vars=["Ativo"],
                value_vars=["Peso_%", "RC_var_%"],
                var_name="Métrica",
                value_name="Valor",
            )

            fig_bar_rc = px.bar(
                df_long,
                x="Ativo",
                y="Valor",
                color="Métrica",
                barmode="group",
                labels={"Valor": "%", "Ativo": "Ativo"},
            )
            fig_bar_rc.update_layout(height=420, xaxis_tickangle=-45, legend_title_text="Métrica")
            st.plotly_chart(fig_bar_rc, use_container_width=True)

                        # -------- Risk Decomposition estilo MSCI Barra --------
            st.markdown("#### Risk Decomposition (estilo MSCI Barra)")

            df_barra = df_risk.copy()
            df_barra["Bloco"] = "Risco total"

            # reset_index e renomeia a coluna do índice para 'Ativo'
            df_barra_plot = df_barra.reset_index().rename(columns={"index": "Ativo"})

            fig_barra = px.bar(
                df_barra_plot,
                x="RC_var_%",
                y="Bloco",
                color="Ativo",
                orientation="h",
                barmode="stack",
                labels={"RC_var_%": "Contribuição de risco (%)", "Bloco": ""},
            )
            fig_barra.update_layout(height=260, showlegend=True, legend_title_text="Ativo")
            st.plotly_chart(fig_barra, use_container_width=True)


            # -------- Treemap: Peso × RC (com RC podendo ser negativo) --------
            st.markdown("#### Mapa de calor por peso e contribuição de risco")

            df_tree = pd.DataFrame({
                "Ativo": df_risk.index,
                "Peso_plot": w.abs().values,        # tamanho do retângulo
                "Peso_%": df_risk["Peso_%"].values,
                "RC_%": df_risk["RC_var_%"].values,  # aqui pode ser negativo
                "Direção": df_risk["Direção"].values,
            })

            fig_tree = px.treemap(
                df_tree,
                path=["Direção", "Ativo"],
                values="Peso_plot",
                color="RC_%",                         # cor = contribuição de risco (%)
                color_continuous_scale="RdYlGn_r",
                color_continuous_midpoint=0.0,        # 0 = hedge / neutro
                hover_data={
                    "Peso_%": ":.2f",
                    "RC_%": ":.2f",
                    "Direção": True,
                    "Peso_plot": False,
                },
            )

            fig_tree.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br>" +
                    "Direção: %{customdata[2]}<br>" +
                    "Peso: %{customdata[0]:.2f}%<br>" +
                    "Contrib. risco: %{customdata[1]:.2f}%<extra></extra>"
                )
            )
            fig_tree.update_layout(
                height=520,
                coloraxis_colorbar=dict(title="RC (variância) %"),
            )

            st.plotly_chart(fig_tree, use_container_width=True)


    # -----------------------------
    # 6) Correlação entre ativos da carteira
    # -----------------------------
    st.markdown("---")
    st.markdown("### Correlação entre ativos da carteira")

    # matriz de correlação dos retornos dos ativos (não do portfólio)
    if rets_sel.shape[1] >= 2:
        corr_mat = rets_sel.corr()
        fig_corr = px.imshow(
            corr_mat,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Matriz de correlação (retornos dos ativos da carteira)",
        )
        fig_corr.update_layout(height=520)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("É necessário pelo menos 2 ativos na carteira para montar a matriz de correlação.")

    # -----------------------------
    # 7) Espaço para futuras simulações de cenário
    # -----------------------------
    st.markdown("---")
    st.markdown("### Simulação de Cenários (em desenvolvimento)")
    st.caption(
        "Nesta seção vamos acoplar, em uma próxima etapa, choques em juros, bolsa, câmbio "
        "e commodities, usando betas estimados para projetar o impacto no portfólio."
    )
