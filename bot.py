"""
Top 6 Crypto Auto-Analysis Bot
Sends technical analysis twice daily (09:00 & 21:00 UTC)
"""

import os
import logging
import asyncio
from datetime import datetime, timezone
from matplotlib.patches import Rectangle

import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegram import Bot

BOT_TOKEN = os.getenv('BOT_TOKEN')
CHANNEL_ID = os.getenv('CHANNEL_ID')

TARGET_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT',
    'SOL/USDT', 'XRP/USDT', 'DOGE/USDT'
]

PAIR_NAMES = {
    'BTC/USDT': 'Bitcoin', 'ETH/USDT': 'Ethereum', 'BNB/USDT': 'BNB',
    'SOL/USDT': 'Solana', 'XRP/USDT': 'XRP', 'DOGE/USDT': 'Dogecoin'
}

COLORS = {
    'bg': '#000000',
    'grid': '#1a1a1a',
    'text': '#888888',
    'green': '#00e676',
    'red': '#ff1744',
    'blue': '#448aff',
    'purple': '#e040fb',
    'orange': '#ffab00',
    'btc': '#f7931a',
}

DELAY_BETWEEN = 600

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def format_price(price):
    if price >= 1000: return f"{price:,.2f}"
    elif price >= 1: return f"{price:,.3f}"
    elif price >= 0.01: return f"{price:,.4f}"
    else: return f"{price:,.6f}"


class CryptoAnalyzer:
    def __init__(self):
        self.exchange = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

    def get_ohlcv(self, symbol, timeframe='1d', limit=300):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def calculate_indicators(self, df):
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=(14-1), min_periods=14).mean()
        avg_loss = loss.ewm(com=(14-1), min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        k_raw = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_k'] = k_raw.rolling(window=3).mean()
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()

        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        return df

    def analyze(self, symbol, timeframe='1d'):
        df = self.get_ohlcv(symbol, timeframe=timeframe, limit=300)
        df = self.calculate_indicators(df)
        df.dropna(inplace=True)

        last = df.iloc[-1]
        tf_label = 'Daily' if timeframe == '1d' else '4H'
        name = PAIR_NAMES.get(symbol, symbol)
        support = df.tail(50)['low'].min()
        resistance = df.tail(50)['high'].max()

        buy_count = 0
        if last['close'] > last['ema_50']: buy_count += 1
        if 45 < last['rsi'] < 70: buy_count += 1
        if last['macd'] > last['macd_signal']: buy_count += 1
        if last['close'] > last['bb_middle']: buy_count += 1
        if last['stoch_k'] > last['stoch_d']: buy_count += 1
        if last['volume'] > last['volume_sma']: buy_count += 1
        sell_count = 6 - buy_count

        message = self._build_human_message(name, tf_label, last, support, resistance, buy_count, sell_count)
        chart_path = self._create_chart(symbol, df, timeframe)
        return message, chart_path

    def _build_human_message(self, name, tf, last, support, resistance, buy_c, sell_c):
        ema20_str = f"${format_price(last['ema_20'])}"
        ema50_str = f"${format_price(last['ema_50'])}"
        h_str = f"{last['macd_hist']:,.2f}"
        res_str = f"${format_price(resistance)}"
        sup_str = f"${format_price(support)}"

        is_bullish, is_bearish = buy_c >= 4, sell_c >= 4

        if is_bullish:
            p1 = f"📊 The {tf} timeframe for {name} is showing clear bullish momentum. Price is holding strong above the key dynamic supports at the **EMA 20 ({ema20_str})** and **EMA 50 ({ema50_str})**, confirming a solid upward structure."
        elif is_bearish:
            p1 = f"📊 The {tf} chart for {name} is leaning heavily bearish as price struggles below critical moving averages. We are trading beneath both the EMA 20 and EMA 50, which suggests the sellers are in full control of the current trend."
        else:
            p1 = f"📊 The {tf} timeframe for {name} is currently caught in a consolidation phase. Price is chopping between dynamic levels, and we are waiting for a clear break of structure."

        if is_bullish:
            p2 = f"📈 Looking at momentum, the MACD line is trading above the signal line with a positive histogram (**{h_str}**), adding weight to the upside. RSI is sitting healthy at **{last['rsi']:.1f}**, showing strong buying pressure with plenty of room to run."
        elif is_bearish:
            p2 = f"📉 Momentum is shifting downwards as the MACD line crosses below the signal line, accompanied by a negative histogram (**{h_str}**). RSI at **{last['rsi']:.1f}** shows increasing selling pressure."
        else:
            p2 = f"⚖️ Momentum indicators are flattening out. MACD is hovering around the zero line, and RSI is sitting at **{last['rsi']:.1f}**, reflecting indecision among buyers and sellers."

        vol_ratio = last['volume'] / last['volume_sma'] if last['volume_sma'] > 0 else 1
        vol_status = "surging" if vol_ratio > 1.3 else ("drying up" if vol_ratio < 0.8 else "supporting the move")
        stoch_status = "overbought" if last['stoch_k'] > 80 else ("oversold" if last['stoch_k'] < 20 else "neutral")
        p3 = f"From a volatility standpoint, price is trading within the Bollinger Bands. Volume is currently **{vol_status} ({vol_ratio:.1f}x average)**, {'which validates the current move' if vol_ratio > 1.2 else 'so keep an eye out for a sudden volatility spike'}. The Stochastic is at **{last['stoch_k']:.1f}** ({stoch_status})."

        if is_bullish:
            p4 = f"🎯 **Key Levels:**\nImmediate resistance at **{res_str}**. A break above could trigger a quick sweep. Downside **{sup_str}** is the line in the sand.\n\n✅ **Bias: Bullish** ({buy_c}/6)"
        elif is_bearish:
            p4 = f"🎯 **Key Levels:**\nEyeing a move to **{sup_str}** demand zone. Supply wall at **{res_str}** must be defended by sellers.\n\n🔴 **Bias: Bearish** ({sell_c}/6)"
        else:
            p4 = f"🎯 **Key Levels:**\nBreak above **{res_str}** ignites upside. Losing **{sup_str}** opens doors for correction.\n\n🟡 **Bias: Neutral** (Buy {buy_c}/6 | Sell {sell_c}/6)"

        return f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}\n\nTechnical Analysis Not Financial Advice.\nBy: @Dr_Python_bot"

    def _create_chart(self, symbol, df, timeframe):
        df_d = df.tail(60).copy()
        n, x = len(df_d), list(range(len(df_d)))
        fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.15)

        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(COLORS['bg'])
        self._draw_candles(ax1, df_d)
        ax1.plot(x, df_d['ema_20'].values, color=COLORS['blue'], lw=1, alpha=0.8, label='EMA 20')
        ax1.plot(x, df_d['ema_50'].values, color=COLORS['purple'], lw=1, alpha=0.8, label='EMA 50')
        ax1.plot(x, df_d['ema_200'].values, color=COLORS['orange'], lw=1, alpha=0.8, label='EMA 200')
        ax1.plot(x, df_d['bb_upper'].values, color=COLORS['green'], lw=0.7, ls='--', alpha=0.5)
        ax1.plot(x, df_d['bb_middle'].values, color='#444444', lw=0.7, ls='--', alpha=0.5)
        ax1.plot(x, df_d['bb_lower'].values, color=COLORS['red'], lw=0.7, ls='--', alpha=0.5)
        ax1.fill_between(x, df_d['bb_upper'].values, df_d['bb_lower'].values, alpha=0.06, color=COLORS['blue'])
        ax1.set_title(f'{symbol}  -  {timeframe.upper()}', color=COLORS['btc'], fontsize=15, fontweight='bold', pad=12)
        ax1.legend(loc='upper left', fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
        self._style_axis(ax1, n, df_d)

        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(COLORS['bg'])
        ax2.plot(x, df_d['rsi'].values, color=COLORS['purple'], lw=1.2)
        ax2.axhline(70, color=COLORS['red'], ls='--', lw=0.8, alpha=0.7)
        ax2.axhline(30, color=COLORS['green'], ls='--', lw=0.8, alpha=0.7)
        ax2.fill_between(x, 70, df_d['rsi'].values, where=df_d['rsi'].values >= 70, alpha=0.25, color=COLORS['red'])
        ax2.fill_between(x, 30, df_d['rsi'].values, where=df_d['rsi'].values <= 30, alpha=0.25, color=COLORS['green'])
        ax2.set_ylabel('RSI', color=COLORS['text'], fontsize=9)
        ax2.set_ylim(0, 100)
        self._style_axis(ax2, n, df_d)

        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(COLORS['bg'])
        ax3.plot(x, df_d['macd'].values, color=COLORS['blue'], lw=1, label='MACD')
        ax3.plot(x, df_d['macd_signal'].values, color=COLORS['red'], lw=1, label='Signal')
        hist = df_d['macd_hist'].values
        ax3.bar(x, hist, color=[COLORS['green'] if v >= 0 else COLORS['red'] for v in hist], alpha=0.6, width=0.7)
        ax3.set_ylabel('MACD', color=COLORS['text'], fontsize=9)
        ax3.legend(loc='upper left', fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
        self._style_axis(ax3, n, df_d)

        path = f"chart_{symbol.replace('/', '_')}_{timeframe}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'], edgecolor='none')
        plt.close('all')
        return path

    def _draw_candles(self, ax, df):
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        for i in range(len(df)):
            color = COLORS['green'] if c[i] >= o[i] else COLORS['red']
            ax.plot([i, i], [l[i], h[i]], color=color, lw=0.8)
            ax.add_patch(Rectangle((i - 0.3, min(o[i], c[i])), 0.6, max(abs(c[i] - o[i]), 0.01), facecolor=color, edgecolor=color, lw=0.5))
        ax.set_xlim(-0.5, len(df) - 0.5)

    def _style_axis(self, ax, n, df):
        step = max(1, n // 8)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([df.index[i].strftime('%m/%d %H:%M') for i in ticks], fontsize=7)
        ax.tick_params(colors=COLORS['text'], labelsize=8)
        ax.grid(True, color=COLORS['grid'], ls='--', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])


async def send_analysis(bot, symbol, timeframe):
    try:
        logger.info(f"Analyzing {symbol} ({timeframe})...")
        analyzer = CryptoAnalyzer()
        message, chart_path = analyzer.analyze(symbol, timeframe)
        with open(chart_path, 'rb') as photo:
            await bot.send_photo(chat_id=CHANNEL_ID, photo=photo, caption=message, parse_mode='Markdown')
        logger.info(f"Successfully sent {symbol}")
        if os.path.exists(chart_path):
            os.remove(chart_path)
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")


def get_scheduled_tasks():
    now = datetime.now(timezone.utc)
    h = now.hour
    tasks = []
    if h == 9:
        for s in TARGET_PAIRS:
            tasks.append((s, '1d'))
    elif h == 21:
        for s in TARGET_PAIRS:
            tasks.append((s, '4h'))
    else:
        if 9 <= h < 21:
            for s in TARGET_PAIRS:
                tasks.append((s, '1d'))
        else:
            for s in TARGET_PAIRS:
                tasks.append((s, '4h'))
    return tasks


async def run():
    if not BOT_TOKEN or not CHANNEL_ID:
        logger.error("BOT_TOKEN or CHANNEL_ID missing from GitHub Secrets!")
        return

    bot = Bot(token=BOT_TOKEN)
    tasks = get_scheduled_tasks()

    if not tasks:
        logger.info("No scheduled task for this hour. Exiting cleanly.")
        return

    logger.info(f"Found {len(tasks)} analysis tasks for this hour")

    for i, (symbol, timeframe) in enumerate(tasks):
        await send_analysis(bot, symbol, timeframe)
        if i < len(tasks) - 1:
            logger.info(f"Waiting {DELAY_BETWEEN // 60} minutes before next analysis...")
            await asyncio.sleep(DELAY_BETWEEN)


if __name__ == '__main__':
    asyncio.run(run())
