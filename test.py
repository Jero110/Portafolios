import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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

if __name__ == "__main__":
    # ---- Parámetros walk-forward ----
    tickers         = ["TSLA", "AAPL", "ETH-USD", "GLD"]
    global_start    = "2024-01-01"
    global_end      = "2024-12-31"
    train_days      = 30
    test_days       = 10
    window_size     = 20
    lambda_param    = 1.0
    diversity_coeff = 0.1
    max_adjust      = 0.05

    # Timesteps reducidos para pruebas
    pretrain_timesteps = 500
    finetune_timesteps = 100

    # Serie de fechas de retorno
    raw = yf.download(tickers[0], start=global_start, end=global_end,
                      auto_adjust=True, progress=False)
    raw['ret'] = raw['Close'].pct_change()
    dates = raw['ret'].dropna().index

    train_end_idx = train_days - 1
    model = None

    print("=== Walk-Forward con PPO ===")
    while train_end_idx + test_days < len(dates):
        # Definir fechas
        train_end_date = dates[train_end_idx]
        test_dates     = dates[train_end_idx+1 : train_end_idx+1+test_days]

        # 1) Entrenar con expanding window hasta train_end_date
        env_train = DummyVecEnv([lambda: PortfolioEnv(
            tickers, global_start, train_end_date,
            window_size, lambda_param, diversity_coeff, max_adjust
        )])
        if model is None:
            model = PPO("MlpPolicy", env_train,
                        verbose=0, n_steps=32, batch_size=32, learning_rate=3e-4)
            model.learn(total_timesteps=pretrain_timesteps)
        else:
            model.set_env(env_train)
            model.learn(total_timesteps=finetune_timesteps)

        # 2) Evaluar en cada día de test_dates
        print(f"\nEvaluación de {train_end_date.date()+pd.Timedelta(days=1)} "
              f"hasta {test_dates[-1].date()}:")
        for date in test_dates:
            eval_env = PortfolioEnv(
                tickers, global_start, date,
                window_size, lambda_param, diversity_coeff, max_adjust
            )
            obs = eval_env.reset()
            action, _ = model.predict(obs, deterministic=True)
            w = np.clip(action, 1e-4, None); w /= w.sum()
            er, cvar = eval_env._calc_metrics(w, date)
            n_train = train_end_idx + 1
            pesos_str = " ".join(f"{t}:{p:.2%}" for t,p in zip(tickers, w))
            print(f"{date.date()} | Train días={n_train:>3d} | E[r]={er:.4f} | "
                  f"CVaR={cvar:.4f} | Pesos={pesos_str}")

        # 3) Avanzar ventana de entrenamiento
        train_end_idx += test_days

    # ---- Test final comprando 1000 acciones totales ----
    final_weights = w  # pesos del último día
    total_shares   = 1000
    shares_per_asset = (final_weights * total_shares).astype(int)

    # Simular rendimiento desde el primer test hasta el último test
    test_start = dates[train_days]                 # empieza justo tras el primer bloque
    test_end   = dates[train_end_idx]              # hasta el último test

    prices = yf.download(tickers, start=test_start, end=test_end,
                         auto_adjust=True, progress=False)["Close"]
    portfolio_value = prices.mul(shares_per_asset, axis=1).sum(axis=1)

    init_val = portfolio_value.iloc[0]
    final_val= portfolio_value.iloc[-1]
    total_ret = (final_val / init_val - 1) * 100

    print(f"\n--- Test final comprando {total_shares} acciones totales ---")
    print(f"Valor inicial portafolio: ${init_val:,.2f}")
    print(f"Valor final   portafolio: ${final_val:,.2f}")
    print(f"Retorno total: {total_ret:.2f}%")
