import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import acf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def extract_features(data):
    """
    Extrae características clave de series temporales financieras de forma eficiente
    sin depender de multiprocessing.
    
    Args:
        data (pd.DataFrame): DataFrame con precios de cierre de los activos
        
    Returns:
        pd.DataFrame: DataFrame con características por activo
    """
    # Calcular retornos diarios
    returns = data.pct_change().dropna()
    
    features = {}
    
    for ticker in returns.columns:
        # Obtener la serie de retornos para este activo
        series = returns[ticker].dropna()
        
        # Características básicas
        mean = series.mean()
        std = series.std()
        skew = series.skew()
        kurtosis = stats.kurtosis(series)
        
        # Autocorrelación - captura dependencia temporal
        acf_values = acf(series, nlags=5, fft=True)
        acf1 = acf_values[1]  # Autocorrelación con lag 1
        
        # Volatilidad
        rolling_std_30 = series.rolling(window=30).std().mean()
        
        # Tendencia - pendiente de la línea de tendencia
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        # Ratio de días positivos/negativos
        positive_days = (series > 0).sum() / len(series)
        
        # Características adicionales
        # Drawdown máximo
        cumulative_returns = (1 + series).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Asimetría de volatilidad
        neg_returns = series[series < 0]
        pos_returns = series[series > 0]
        vol_asymmetry = neg_returns.std() / pos_returns.std() if len(pos_returns) > 0 else np.nan
        
        # Almacenar características
        features[ticker] = {
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurtosis': kurtosis,
            'acf1': acf1,
            'rolling_std_30': rolling_std_30,
            'slope': slope,
            'positive_days_ratio': positive_days,
            'max_drawdown': max_drawdown,
            'vol_asymmetry': vol_asymmetry
        }
    
    # Convertir a DataFrame
    return pd.DataFrame.from_dict(features, orient='index')

if __name__ == '__main__':
    # 1. Configuración
    tickers = ["TSLA", "ARKK", "AAPL", "PG", "JNJ", "ETH-USD", "BIL", "SHV", "GLD", "NOC"]
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    
    # 2. Descarga precios ajustados
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    
    # 3. Extraer características
    features_df = extract_features(data)
    print(features_df)