import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
from features import extract_features
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Silenciar warnings innecesarios
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message="You provided an OpenAI Gym environment")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)

# --------------------------------------------------
# 1) Definición del entorno RL
# --------------------------------------------------
class PortfolioEnv(gym.Env):
    def __init__(self, tickers, start_date, end_date,
                 window_size=15, lambda_param=0.8,
                 diversity_coeff=0.2, max_adjust=0.05,
                 static_features=None):
        super().__init__()
        self.tickers        = tickers
        self.n_assets       = len(tickers)
        self.start          = pd.to_datetime(start_date)
        self.end            = pd.to_datetime(end_date)
        self.window_size    = window_size
        self.lambda_param   = lambda_param
        self.diversity_coeff= diversity_coeff
        self.max_adjust     = max_adjust
        # features estáticas (o ceros)
        self.static_features = (static_features.values.flatten()
                                 if static_features is not None
                                 else np.zeros(self.n_assets * 10))

        # Descargar y calcular retornos diarios
        self.returns = {}
        for t in tickers:
            df = yf.download(t, start=self.start, end=self.end,
                             auto_adjust=True, progress=False)
            df['ret'] = df['Close'].pct_change()
            self.returns[t] = df['ret'].dropna()

        # Fechas disponibles y step inicial
        self.dates = sorted(self.returns[tickers[0]].index)
        self.current_step = window_size

        # Espacios de acción y observación
        self.action_space = spaces.Box(0.0, 1.0, (self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * self.window_size + 1 + self.n_assets * 10
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        # Pesos iniciales
        self.current_weights = np.ones(self.n_assets) / self.n_assets

    def _calc_metrics(self, w, date):
        w = np.clip(w, 1e-4, None); w /= w.sum()
        start = date - pd.Timedelta(days=self.window_size)
        data = {t: self.returns[t].loc[start:date] for t in self.tickers}
        df = pd.DataFrame(data).fillna(0)
        port = df.dot(w)
        er = port.mean()
        losses = -port
        var95 = np.percentile(losses, 95)
        cvar = losses[losses >= var95].mean() if any(losses >= var95) else var95
        diversity = 1.0 - np.sum(w**2)
        return er, cvar, diversity

    def _get_obs(self, date):
        start = date - pd.Timedelta(days=self.window_size)
        arrs = []
        for t in self.tickers:
            vals = self.returns[t].loc[start:date].values
            if len(vals) < self.window_size:
                vals = np.pad(vals, (self.window_size - len(vals), 0))
            arrs.append(vals[-self.window_size:])
        flat = np.concatenate(arrs)
        _, cvar, _ = self._calc_metrics(self.current_weights, date)
        return np.concatenate([flat, [cvar], self.static_features]).astype(np.float32)

    def reset(self):
        self.current_step = min(self.window_size, len(self.dates)-1)
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs(self.dates[self.current_step])

    def step(self, action):
        prop = np.clip(action, 1e-4, None); prop /= prop.sum()
        delta = np.clip(prop - self.current_weights,
                        -self.max_adjust, self.max_adjust)
        w = self.current_weights + delta
        w = np.clip(w, 1e-4, None); w /= w.sum()
        self.current_weights = w

        self.current_step = min(self.current_step+1, len(self.dates)-1)
        date = self.dates[self.current_step]
        er, cvar, diversity = self._calc_metrics(w, date)
        reward = er - self.lambda_param * cvar + self.diversity_coeff * diversity
        obs = self._get_obs(date)
        done = (self.current_step == len(self.dates)-1)
        info = {'date': date, 'er': er, 'cvar': cvar,
                'diversity': diversity, 'weights': w}
        return obs, reward, done, info

# --------------------------------------------------
# 2) Función para evaluar métricas en test
# --------------------------------------------------
def evaluate_model(weights, tickers, start_date, end_date,
                   lambda_param, diversity_coeff):
    prices = yf.download(tickers, start=start_date, end=end_date,
                         auto_adjust=True, progress=False)['Close']
    rets = prices.pct_change().dropna()
    if rets.empty:
        return None
    port = np.zeros(len(rets))
    for i, t in enumerate(tickers):
        if t in rets.columns:
            port += rets[t].values * weights[i]
    er = port.mean()
    losses = -port
    var95 = np.percentile(losses, 95)
    cvar = losses[losses>=var95].mean() if any(losses>=var95) else var95
    diversity = 1.0 - np.sum(weights**2)
    reward = er - lambda_param * cvar + diversity_coeff * diversity
    total_ret = (1+port).cumprod()[-1] - 1
    return {'er': er, 'cvar': cvar, 'diversity': diversity,
            'reward': reward, 'total_return': total_ret}

# --------------------------------------------------
# 3) Walk-forward con expanding window + test hasta mayo/25
# --------------------------------------------------
if __name__ == "__main__":
    # Parámetros globales
    tickers          = ["TSLA","ARKK","AAPL","PG","JNJ",
                        "ETH-USD","BIL","SHV","GLD","NOC"]
    global_start     = "2018-01-01"
    global_end       = "2024-04-30"
    window_size      = 15
    lambda_param     = 0.8        # mayor peso al CVaR
    diversity_coeff  = 0.2
    max_adjust       = 0.05

    # Aumentamos timesteps y tamaño de red para mejorar aprendizaje
    pretrain_timesteps = 1000
    finetune_timesteps = 800
    iter_per_window    = 10
    policy_kwargs      = dict(net_arch=[128, 128])

    # Definir los cortes de test en bloques 80/20 secuenciales 2018–2024
    spy = yf.download("SPY", start=global_start, end=global_end,
                      auto_adjust=True, progress=False)
    dates = spy.index
    N, K = len(dates), 10
    B = N // K
    window_defs = []
    for i in range(K):
        s = i*B
        e = (i+1)*B if i<K-1 else N
        cut = s + int(0.8*(e-s))
        window_defs.append({
            'train_end' : dates[cut-1],
            'test_start': dates[cut],
            'test_end'  : dates[e-1]
        })

    model = None
    last_weights = None

    # Loop principal: expanding window de entrenamiento
    for idx, w in enumerate(window_defs):
        print(f"\n=== Ventana {idx+1} | Train {global_start}→{w['train_end'].date()} | "
              f"Test {w['test_start'].date()}→{w['test_end'].date()} ===")

        # extraer features hasta train_end
        feat_prices = yf.download(tickers, start=global_start,
                                  end=w['train_end'], auto_adjust=True, progress=False)['Close']
        static_feats = extract_features(feat_prices)

        env_train = DummyVecEnv([lambda: PortfolioEnv(
            tickers, global_start, w['train_end'], window_size,
            lambda_param, diversity_coeff, max_adjust,
            static_features=static_feats
        )])

        # Pretrain o fine-tune
        if model is None:
            model = PPO("MlpPolicy", env_train, verbose=0,
                        learning_rate=3e-5, n_steps=256,
                        batch_size=64, gamma=0.99,
                        gae_lambda=0.95, clip_range=0.15,
                        ent_coef=0.0005,
                        policy_kwargs=policy_kwargs)
            model.learn(total_timesteps=pretrain_timesteps, reset_num_timesteps=False)
        else:
            model.set_env(env_train)
            model.learn(total_timesteps=finetune_timesteps//iter_per_window,
                        reset_num_timesteps=False)

        # Iteraciones adicionales por ventana
        for it in range(iter_per_window):
            model.learn(total_timesteps=finetune_timesteps//iter_per_window,
                        reset_num_timesteps=False)

            # evaluar en train_end
            eval_env = PortfolioEnv(
                tickers, global_start, w['train_end'], window_size,
                lambda_param, diversity_coeff, max_adjust,
                static_features=static_feats
            )
            obs = eval_env.reset()
            action, _ = model.predict(obs, deterministic=True)
            weights = np.clip(action, 1e-4, None); weights /= weights.sum()
            er, cvar, div = eval_env._calc_metrics(weights, w['train_end'])

            # evaluar en test
            res_test = evaluate_model(weights, tickers,
                                      w['test_start'], w['test_end'],
                                      lambda_param, diversity_coeff) or {}
            ret = res_test.get('total_return', np.nan)
            rew = res_test.get('reward', np.nan)

            print(f" Iter {it+1}/{iter_per_window} | "
                  f"E[r]={er:.4f} | CVaR={cvar:.4f} | Div={div:.4f}")
            print(f"  Pesos: {', '.join(f'{t}:{weights[i]:.2%}' for i,t in enumerate(tickers))}")
            print(f"  → Test Return={ret:.2%} | Reward={rew:.4f}")

            last_weights = weights.copy()

    # --------------------------------------------------
    # 4) Test final en ventanas de 15 días hasta mayo/25
    # --------------------------------------------------
    print("\n=== Test Final 2025 (hasta 2025-05-31) con λ=0.5 ===")
    all_2025 = pd.date_range("2025-01-01", "2025-05-31", freq="B")
    window_days = 15
    for i in range(0, len(all_2025), window_days):
        end_idx = i + window_days - 1
        if end_idx >= len(all_2025):
            break
        s25, e25 = all_2025[i], all_2025[end_idx]
        res = evaluate_model(last_weights, tickers,
                             s25, e25, lambda_param, diversity_coeff)
        if res is None:
            print(f"Ventana2025-{i//window_days+1}: {s25.date()}→{e25.date()} — no hay datos.")
            continue
        print(f"\nVentana2025-{i//window_days+1}: {s25.date()}→{e25.date()}")
        print(f"  E[r]       = {res['er']:.4f}")
        print(f"  CVaR      = {res['cvar']:.4f}")
        print(f"  Diversidad = {res['diversity']:.4f}")
        print("  Pesos     =", ", ".join(f"{t}:{last_weights[j]:.2%}"
                                       for j,t in enumerate(tickers)))
        print(f"  Return    = {res['total_return']:.2%}")
        print(f"  Reward    = {res['reward']:.4f}")
