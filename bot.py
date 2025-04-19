from keep_alive import keep_alive  # 🟢 Start web server to keep Replit alive

keep_alive()

import os
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import numpy as np

BOT_TOKEN = os.environ['BOT_TOKEN']

ASK_COIN, SHOW_DETAILS, ASK_ADVICE = range(3)

# ---- Helper Functions ----

def get_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}USDT"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except:
        return None


def get_historical_prices(symbol, interval='1m', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}USDT&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [float(entry[4]) for entry in data]  # Closing prices
    except:
        return []


def calculate_indicators(prices):
    np_prices = np.array(prices)
    indicators = {}

    delta = np.diff(np_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0.01
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0.01
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.01
    rsi = 100 - (100 / (1 + rs))
    indicators['RSI'] = rsi

    ema12 = np_prices[-12:].mean() if len(prices) >= 12 else np_prices[-1]
    ema26 = np_prices[-26:].mean() if len(prices) >= 26 else np_prices[-1]
    macd = ema12 - ema26
    indicators['MACD'] = macd

    sma20 = np_prices[-20:].mean()
    std = np_prices[-20:].std()
    indicators['Bollinger_Upper'] = sma20 + 2 * std
    indicators['Bollinger_Lower'] = sma20 - 2 * std

    indicators['EMA_20'] = np_prices[-20:].mean()
    indicators['SMA_50'] = np_prices[-50:].mean()

    tr = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    indicators['ATR'] = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0

    indicators['Stochastic'] = (np_prices[-1] - np.min(np_prices[-14:])) / (
        np.max(np_prices[-14:]) - np.min(np_prices[-14:])) * 100
    indicators['Ichimoku'] = (np_prices[-9:].mean() +
                              np_prices[-26:].mean()) / 2
    indicators['ADX'] = abs(np.mean(delta[-14:]))
    indicators['Parabolic_SAR'] = np.min(np_prices[-14:])
    indicators['Pivot'] = (np.max(np_prices[-1:]) + np.min(np_prices[-1:]) +
                           np_prices[-1]) / 3
    indicators['VWAP'] = np.mean(np_prices)
    indicators['CCI'] = (np_prices[-1] - sma20) / (0.015 * std)
    indicators['OBV'] = np.sum(delta)

    return indicators


def strong_support_resistance(prices):
    sorted_prices = sorted(prices)
    support = sorted_prices[int(len(prices) * 0.15)]
    resistance = sorted_prices[int(len(prices) * 0.85)]
    return support, resistance


# ---- Ping Handler ----

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is alive and working fine! ✅")

# ---- Handlers ----

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome To Live Crypto Price Alert Created By *Nasir Memon*",
        parse_mode="Markdown")
    await update.message.reply_text(
        "Which coin are you interested in? (e.g., BTC, ETH)")
    return ASK_COIN


async def ask_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.upper()
    context.user_data['symbol'] = symbol

    prices = get_historical_prices(symbol)

    if len(prices) < 50:
        await update.message.reply_text(
            "Couldn't fetch enough data. Try another coin or try later.")
        return ConversationHandler.END

    indicators = calculate_indicators(prices)
    support, resistance = strong_support_resistance(prices)
    current_price = prices[-1]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    context.user_data.update({
        'indicators': indicators,
        'support': support,
        'resistance': resistance,
        'current_price': current_price
    })

    def fmt(value):
        return f"{value:.8f}" if current_price < 0.1 else f"{value:.2f}"

    detail = f"""
📊 Coin: {symbol}
🕒 Time: {now}
💰 Current Price: ${fmt(current_price)}
📉 Support: ${fmt(support)}
📈 Resistance: ${fmt(resistance)}

📈 RSI: {indicators['RSI']:.2f}
📉 MACD: {indicators['MACD']:.4f}
🎯 Bollinger Bands: {fmt(indicators['Bollinger_Lower'])} - {fmt(indicators['Bollinger_Upper'])}
📊 EMA(20): ${fmt(indicators['EMA_20'])}, SMA(50): ${fmt(indicators['SMA_50'])}
⚡ ATR: {indicators['ATR']:.4f}, ADX: {indicators['ADX']:.4f}
☁️ Ichimoku Avg: {fmt(indicators['Ichimoku'])}
🎯 Pivot: {fmt(indicators['Pivot'])}, VWAP: {fmt(indicators['VWAP'])}
🌀 CCI: {indicators['CCI']:.2f}, OBV: {indicators['OBV']:.2f}
"""
    await update.message.reply_text(detail)
    await update.message.reply_text(
        "Do you want me to advise you for short term trade? (yes/no)")
    return ASK_ADVICE


async def handle_advice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text.lower() != 'yes':
        await update.message.reply_text("Okay, I'm here if you need anything!")
        return ConversationHandler.END

    data = context.user_data
    indicators = data['indicators']
    price = data['current_price']
    support = data['support']
    resistance = data['resistance']

    score = 0
    reasons = []

    if indicators['RSI'] < 45:
        score += 2
        reasons.append("RSI below 45 - potential bullish setup")
    elif indicators['RSI'] > 65:
        score -= 2
        reasons.append("RSI above 65 - potential bearish trend")

    if indicators['MACD'] > -0.5:
        score += 2
        reasons.append("MACD near or above zero - mild bullish sentiment")
    elif indicators['MACD'] < -0.5:
        score -= 2
        reasons.append("MACD strongly negative - bearish crossover")

    if price <= indicators['Bollinger_Lower'] * 1.01:
        score += 1.5
        reasons.append("Price near lower Bollinger Band - likely oversold")
    elif price >= indicators['Bollinger_Upper'] * 0.99:
        score -= 1.5
        reasons.append("Price near upper Bollinger Band - likely overbought")

    if price > indicators['EMA_20'] and price > indicators['SMA_50']:
        score += 2
        reasons.append("Price above EMA & SMA - uptrend")
    elif price < indicators['EMA_20'] and price < indicators['SMA_50']:
        score -= 2
        reasons.append("Price below EMA & SMA - downtrend")

    if indicators['CCI'] > 100:
        score += 1
        reasons.append("CCI > 100 - bullish")
    elif indicators['CCI'] < -100:
        score -= 1
        reasons.append("CCI < -100 - bearish")

    if indicators['Stochastic'] < 35:
        score += 1
        reasons.append("Stochastic oversold")
    elif indicators['Stochastic'] > 75:
        score -= 1
        reasons.append("Stochastic overbought")

    if indicators['ADX'] > 25:
        score += 1
        reasons.append("ADX > 25 - strong trend")

    momentum = price - support if price < (
        support + resistance) / 2 else resistance - price
    if momentum < indicators['ATR']:
        score -= 0.5
        reasons.append("Low momentum - caution")

    if score >= 1:
        signal = "LONG"
    elif score <= -1:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    tp = resistance * 1.015 if signal == "LONG" else (
        support * 0.985 if signal == "SHORT" else None)
    sl = support * 0.97 if signal == "LONG" else (
        resistance * 1.03 if signal == "SHORT" else None)

    fmt = lambda x: f"{x:.8f}" if price < 0.1 else f"{x:.2f}"
    reason_text = '\n'.join([f"- {r}" for r in reasons])

    advice = f"""
🎯 Trade Suggestion: *{signal}*
📊 Score: {score:.2f}

📌 Reasoning:
{reason_text}

📉 Price: ${fmt(price)}, Support: ${fmt(support)}, Resistance: ${fmt(resistance)}
"""

    if signal != "NEUTRAL":
        advice += f"""🛑 Stop Loss: ${fmt(sl)}
🎯 Take Profit: ${fmt(tp)}"""

    advice += "\n\n⚠️ *Disclaimer:* This is a highly accurate AI-based suggestion using 15+ top indicators. Not financial advice. Please DYOR."

    await update.message.reply_text(advice, parse_mode="Markdown")
    return ConversationHandler.END


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Type /start to begin.")


# ---- Main Setup ----

app = ApplicationBuilder().token(BOT_TOKEN).build()

conv_handler = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        ASK_COIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_coin)],
        ASK_ADVICE:
        [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_advice)],
    },
    fallbacks=[])

# Adding ping handler
app.add_handler(CommandHandler("ping", ping))

app.add_handler(conv_handler)
app.add_handler(MessageHandler(filters.ALL, unknown_message))

print("🤖 Bot is running...")
app.run_polling()
