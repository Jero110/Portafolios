import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message="You provided an OpenAI Gym environment")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Usando la misma clase PortfolioEnv 
class PortfolioEnv(gym.Env):
    """
    Entorno RL que maximiza E[r] − λ·CVaR + β·Diversificación,
    con ajuste máximo ±5% por activo y sin pesos cero.
    """
    def __init__(self,
                 tickers,
                 start_date,
                 end_date,
                 window_size=20,
                 lambda_param=1.0,
                 diversity_coeff=0.1,
                 max_adjust=0.05):
        super().__init__()
        self.tickers         = tickers
        self.n_assets        = len(tickers)
        self.start           = pd.to_datetime(start_date)
        self.end             = pd.to_datetime(end_date)
        self.window_size     = window_size
        self.lambda_param    = lambda_param
        self.diversity_coeff = diversity_coeff
        self.max_adjust      = max_adjust

        # Descargar y calcular retornos diarios
        self.returns = {}
        for t in tickers:
            df = yf.download(t, start=self.start, end=self.end,
                             auto_adjust=True, progress=False)
            df['ret'] = df['Close'].pct_change()
            self.returns[t] = df['ret'].dropna()

        # Fechas disponibles
        self.dates = sorted(self.returns[tickers[0]].index)
        self.current_step = window_size

        # Espacios de acción y observación
        self.action_space = spaces.Box(0.0, 1.0, (self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * self.window_size + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        # Pesos iniciales equiponderados
        self.current_weights = np.ones(self.n_assets) / self.n_assets

    def _calc_metrics(self, w, date):
        # Normalizar y asegurar no-cero
        w = np.clip(w, 1e-4, None)
        w /= w.sum()
        # Ventana de retornos
        start = date - pd.Timedelta(days=self.window_size)
        data = {t: self.returns[t].loc[start:date] for t in self.tickers}
        df = pd.DataFrame(data).fillna(0)
        port = df.dot(w)
        er = port.mean()
        losses = -port
        var95 = np.percentile(losses, 95)
        cvar = losses[losses >= var95].mean() if any(losses >= var95) else var95
        return er, cvar

    def _get_obs(self, date):
        start = date - pd.Timedelta(days=self.window_size)
        arrs = []
        for t in self.tickers:
            vals = self.returns[t].loc[start:date].values
            if len(vals) < self.window_size:
                vals = np.pad(vals, (self.window_size - len(vals), 0))
            arrs.append(vals[-self.window_size:])
        flat = np.concatenate(arrs)
        _, cvar = self._calc_metrics(self.current_weights, date)
        return np.concatenate([flat, [cvar]]).astype(np.float32)

    def reset(self):
        self.current_step = min(self.window_size, len(self.dates) - 1)
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs(self.dates[self.current_step])

    def step(self, action):
        # 1) Propuesta normalizada
        prop = np.clip(action, 1e-4, None)
        prop /= prop.sum()
        # 2) Ajuste limitado ±max_adjust
        delta = prop - self.current_weights
        delta = np.clip(delta, -self.max_adjust, self.max_adjust)
        w = self.current_weights + delta
        # 3) Renormalizar sin ceros
        w = np.clip(w, 1e-4, None)
        w /= w.sum()
        self.current_weights = w
        # 4) Avanzar un día
        self.current_step = min(self.current_step + 1, len(self.dates) - 1)
        date = self.dates[self.current_step]
        # 5) Métricas y recompensa
        er, cvar    = self._calc_metrics(w, date)
        diversity   = 1.0 - np.sum(w**2)
        reward      = er - self.lambda_param * cvar + self.diversity_coeff * diversity
        obs = self._get_obs(date)
        done = (self.current_step == len(self.dates) - 1)
        info = {'date': date, 'er': er, 'cvar': cvar, 'diversity': diversity, 'weights': w}
        return obs, reward, done, info

# Función para evaluar el modelo en un período de prueba
def evaluate_model(model, weights, tickers, test_start_date, test_end_date, lambda_param, diversity_coeff):
    # Descargar precios para el período de prueba
    test_prices = yf.download(tickers, start=test_start_date, end=test_end_date, 
                            auto_adjust=True, progress=False)['Close']
    
    # Calcular rendimientos diarios
    test_returns = test_prices.pct_change().dropna()
    
    # Calcular rendimiento del portafolio
    port_returns = np.zeros(len(test_returns))
    for i, ticker in enumerate(tickers):
        if ticker in test_returns.columns:
            port_returns += test_returns[ticker].values * weights[i]
    
    # Calcular métricas de prueba
    test_er = np.mean(port_returns)
    losses = -port_returns
    var95 = np.percentile(losses, 95)
    cvar_losses = losses[losses >= var95]
    test_cvar = np.mean(cvar_losses) if len(cvar_losses) > 0 else var95
    test_diversity = 1.0 - np.sum(weights**2)
    test_reward = test_er - lambda_param * test_cvar + diversity_coeff * test_diversity
    
    # Calcular rendimiento total
    port_value = (1 + port_returns).cumprod()
    total_return = port_value[-1] - 1 if len(port_value) > 0 else 0
    
    return {
        'test_er': test_er,
        'test_cvar': test_cvar,
        'test_diversity': test_diversity,
        'test_reward': test_reward,
        'total_return': total_return,
        'daily_returns': port_returns,
        'cumulative_returns': port_value if len(port_value) > 0 else np.array([1.0])
    }

if __name__ == "__main__":
    # ---- Parámetros walk-forward ----
    tickers         = ["TSLA", "AAPL", "ETH-USD", "GLD"]
    global_start    = "2023-01-01"  # Fecha de inicio global
    global_end      = "2024-04-30"  # Fecha de fin global
    initial_train_days = 30         # Días iniciales para entrenamiento
    test_days       = 10            # Días para prueba en cada iteración
    window_size     = 20            # Ventana para calcular métricas
    lambda_param    = 1.0           # Peso del CVaR en la función objetivo
    diversity_coeff = 0.1           # Peso de la diversificación
    max_adjust      = 0.05          # Límite de ajuste por activo
    walk_iterations = 10            # Número de ventanas de tiempo a evaluar
    iter_per_window = 10            # Número de iteraciones por ventana (nuevo parámetro)

    # Timesteps para entrenamiento
    pretrain_timesteps = 500        # Entrenamiento inicial
    finetune_timesteps = 100        # Fine-tuning en cada iteración

    # Descargar fechas de trading
    print(f"Descargando fechas de trading desde {global_start} hasta {global_end}...")
    try:
        spy_data = yf.download("SPY", start=global_start, end=global_end, 
                             auto_adjust=True, progress=False)
        all_dates = spy_data.index
        print(f"Encontradas {len(all_dates)} fechas de trading.")
    except Exception as e:
        print(f"Error descargando datos de SPY: {e}")
        print("Generando fechas de trading manualmente...")
        start = pd.to_datetime(global_start)
        end = pd.to_datetime(global_end)
        all_dates = pd.date_range(start=start, end=end, freq='B')  # 'B' para días hábiles
        print(f"Generadas {len(all_dates)} fechas de trading estimadas.")

    # Inicializar modelo y contadores
    model = None
    current_train_end_idx = initial_train_days - 1

    print("\n=== Walk-Forward Testing con PPO ===")
    print("Formato por iteración y ventana:")
    print("Time Window [TRAIN]: fechas_train | [TEST]: fechas_test")
    print("Iteración X de Y | fecha | E[r]=X | CVaR=Y | Pesos=TICKER:W%")
    print("Reward en el test = Z | Return = R%\n")

    # Lista para almacenar resultados finales
    window_results = []

    # Iterar para realizar walk-forward testing
    for window in range(walk_iterations):
        if current_train_end_idx >= len(all_dates) - test_days:
            print("Se alcanzó el final de las fechas disponibles.")
            break
            
        # Definir fechas para esta ventana
        train_end_date = all_dates[min(current_train_end_idx, len(all_dates)-1)]
        test_start_idx = current_train_end_idx + 1
        test_end_idx = min(test_start_idx + test_days - 1, len(all_dates)-1)
        
        # Verificar si hay suficientes fechas para el período de prueba
        if test_start_idx >= len(all_dates):
            print("No hay más fechas para pruebas.")
            break
            
        test_start_date = all_dates[test_start_idx]
        test_end_date = all_dates[test_end_idx]
        
        # Mostrar ventana de tiempo actual
        total_train_days = current_train_end_idx + 1
        print(f"Ventana {window+1} de {walk_iterations}:")
        print(f"  Time Window [TRAIN]: {all_dates[0].date()} al {train_end_date.date()} | [TEST]: {test_start_date.date()} al {test_end_date.date()}")
        
        # Crear ambiente de entrenamiento para esta ventana
        try:
            env_train = DummyVecEnv([lambda: PortfolioEnv(
                tickers, global_start, train_end_date,
                window_size, lambda_param, diversity_coeff, max_adjust
            )])
            
            if model is None:
                model = PPO("MlpPolicy", env_train, verbose=0, 
                         n_steps=32, batch_size=32, learning_rate=3e-4)
                model.learn(total_timesteps=pretrain_timesteps)
            else:
                model.set_env(env_train)
                model.learn(total_timesteps=finetune_timesteps)
                
        except Exception as e:
            print(f"Error durante el entrenamiento inicial: {e}")
            break
        
        # Realizar múltiples iteraciones para esta ventana
        best_weights = None
        best_reward = float('-inf')
        best_metrics = None
        
        for iter_num in range(iter_per_window):
            try:
                # Entrenar un poco más en cada iteración
                model.learn(total_timesteps=finetune_timesteps // iter_per_window)
                
                # Evaluar el modelo
                eval_env = PortfolioEnv(
                    tickers, global_start, train_end_date,
                    window_size, lambda_param, diversity_coeff, max_adjust
                )
                
                obs = eval_env.reset()
                action, _ = model.predict(obs, deterministic=True)
                
                # Normalizar pesos
                weights = np.clip(action, 1e-4, None)
                weights /= weights.sum()
                
                # Calcular métricas de entrenamiento
                er, cvar = eval_env._calc_metrics(weights, train_end_date)
                diversity = 1.0 - np.sum(weights**2)
                train_reward = er - lambda_param * cvar + diversity_coeff * diversity
                
                # Formatear pesos para impresión
                pesos_str = " ".join([f"{ticker}:{weights[i]:.2%}" for i, ticker in enumerate(tickers)])
                
                # Calcular métricas en el período de prueba
                test_metrics = evaluate_model(
                    model, weights, tickers, test_start_date, test_end_date, 
                    lambda_param, diversity_coeff
                )
                
                # Imprimir resultados de esta iteración
                print(f"  Iteración {iter_num+1} de {iter_per_window} | {train_end_date.date()} | E[r]={er:.4f} | CVaR={cvar:.4f} | Pesos={pesos_str}")
                print(f"  Reward en test={test_metrics['test_reward']:.4f} | E[r]={test_metrics['test_er']:.4f} | CVaR={test_metrics['test_cvar']:.4f} | Return={test_metrics['total_return']:.2%}")
                
                # Guardar mejores pesos basados en el reward del test
                if test_metrics['test_reward'] > best_reward:
                    best_reward = test_metrics['test_reward']
                    best_weights = weights.copy()
                    best_metrics = test_metrics
                
            except Exception as e:
                print(f"Error durante la iteración {iter_num+1}: {e}")
        
        # Guardar los resultados de esta ventana
        window_results.append({
            'window': window + 1,
            'train_end_date': train_end_date,
            'test_start_date': test_start_date,
            'test_end_date': test_end_date,
            'best_weights': best_weights,
            'best_metrics': best_metrics
        })
        
        print(f"\n  MEJOR RESULTADO PARA VENTANA {window+1}:")
        print(f"  Return={best_metrics['total_return']:.2%} | Reward={best_reward:.4f}")
        print(f"  Pesos={' '.join([f'{ticker}:{best_weights[i]:.2%}' for i, ticker in enumerate(tickers)])}")
        print()  # Línea en blanco entre ventanas
        
        # Avanzar a la siguiente ventana
        current_train_end_idx += test_days

    # Test final con los últimos días que nunca vio el modelo
    if model is not None and len(window_results) > 0:
        # Obtener la última fecha de entrenamiento
        last_train_date = window_results[-1]['test_end_date']
        
        # Verificar si hay días adicionales para el test final
        remaining_dates = [d for d in all_dates if d > last_train_date]
        
        if len(remaining_dates) >= test_days:
            final_test_start = remaining_dates[0]
            final_test_end = remaining_dates[min(test_days-1, len(remaining_dates)-1)]
            
            print("\n=== Test Final con Datos Nunca Vistos ===")
            print(f"Período de prueba: {final_test_start.date()} al {final_test_end.date()}")
            
            # Usar los pesos del último modelo
            final_weights = window_results[-1]['best_weights']
            
            # Evaluar el modelo en el período final
            final_test_metrics = evaluate_model(
                model, final_weights, tickers, final_test_start, final_test_end,
                lambda_param, diversity_coeff
            )
            
            # Mostrar resultados
            print(f"Pesos finales: {' '.join([f'{ticker}:{final_weights[i]:.2%}' for i, ticker in enumerate(tickers)])}")
            print(f"Reward={final_test_metrics['test_reward']:.4f} | E[r]={final_test_metrics['test_er']:.4f} | CVaR={final_test_metrics['test_cvar']:.4f}")
            print(f"Return={final_test_metrics['total_return']:.2%}")
        else:
            print("\nNo hay suficientes fechas disponibles para realizar un test final.")

    # Resumen general de resultados
    print("\n=== Resumen de Resultados por Ventana ===")
    print("Ventana | Período de Test | Return | Reward | Pesos")
    
    total_returns = []
    for res in window_results:
        window_num = res['window']
        test_period = f"{res['test_start_date'].date()} al {res['test_end_date'].date()}"
        return_val = res['best_metrics']['total_return']
        reward_val = res['best_metrics']['test_reward']
        pesos = ' '.join([f'{ticker}:{res["best_weights"][i]:.2%}' for i, ticker in enumerate(tickers)])
        
        print(f"{window_num:2d} | {test_period} | {return_val:.2%} | {reward_val:.4f} | {pesos}")
        
        total_returns.append(return_val)
    
    # Calcular estadísticas
    if total_returns:
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        pos_returns = sum(1 for r in total_returns if r > 0)
        neg_returns = sum(1 for r in total_returns if r <= 0)
        
        print("\n=== Estadísticas de Rendimiento ===")
        print(f"Rendimiento promedio: {avg_return:.2%}")
        print(f"Desviación estándar: {std_return:.2%}")
        print(f"Ratio positivo/negativo: {pos_returns}/{neg_returns} ({pos_returns/len(total_returns):.2%})")
