import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Tickers con Ethereum incluido
# -------------------------------
tickers = ["TSLA", "ARKK", "AAPL", "PG", "JNJ", "ETH-USD", "BIL", "SHV", "GLD", "NOC"]

# -------------------------------
# 2. Descargar precios diarios (cierre del mercado)
# -------------------------------
data = yf.download(
    tickers,
    start="2018-01-01",
    end="2024-12-31",
    interval="1d",
    auto_adjust=True
)["Close"]

# -------------------------------
# 3. Gráfico de precios diarios
# -------------------------------
plt.figure(figsize=(14, 6))
for ticker in tickers:
    plt.plot(data[ticker], label=ticker)
plt.title("Precios ajustados diarios (2018-2024)")
plt.xlabel("Fecha")
plt.ylabel("Precio ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Retornos logarítmicos diarios
# -------------------------------
returns = np.log(data / data.shift(1)).dropna()

# -------------------------------
# 5. Matriz de correlaciones de retornos
# -------------------------------
correlation_matrix = returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de correlaciones (retornos logarítmicos diarios)")
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Cálculo de VaR y CVaR al 95%
# -------------------------------

def calculate_var_cvar(returns_series, alpha=0.95):
    sorted_returns = np.sort(returns_series)
    index = int((1 - alpha) * len(sorted_returns))
    var = sorted_returns[index]
    cvar = sorted_returns[:index].mean()
    return var, cvar

results = {
    "Ticker": [],
    "VaR_95 (%)": [],
    "CVaR_95 (%)": []
}

for col in returns.columns:
    var, cvar = calculate_var_cvar(returns[col])
    results["Ticker"].append(col)
    results["VaR_95 (%)"].append(round(var * 100, 2))
    results["CVaR_95 (%)"].append(round(cvar * 100, 2))

df_var_cvar = pd.DataFrame(results)
print("\n📉 VaR y CVaR al 95% de cada activo (diario):\n")
print(df_var_cvar)