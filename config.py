"""Todos los parametros del problema y del entrenamiento."""
from pathlib import Path

# ---- rutas (relativas a este archivo) ----
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# ---- el problema ----
LEAD_TIME = 3            # dias que tarda en llegar un pedido
DEMAND_MEAN = 20         # demanda diaria ~ Poisson(20)
HOLDING_COST = 1.0       # por unidad que sobra al final del dia
LOST_SALE_COST = 9.0     # por unidad de demanda que no pudimos atender
FIXED_ORDER_COST = 10.0  # solo si se pide mas de 0
UNIT_ORDER_COST = 2.0    # por unidad pedida
MAX_ORDER = 50           # pedido maximo por dia
N_ACTIONS = 11           # modo discreto: pedir {0, 5, ..., 50}
EPISODE_LEN = 100        # dias por episodio
GAMMA = 0.99
INIT_INVENTORY = 40
INIT_PIPELINE = 20       # unidades en cada posicion del pipeline al inicio

# ---- observacion y recompensa ----
OBS_SCALE = 50.0         # se divide el estado entre esto
OBS_CLIP = 3.0           # y se recorta a [0, OBS_CLIP]
REWARD_SCALE = 10.0      # recompensa = -costo / REWARD_SCALE

# ---- semillas (train, validacion y test nunca se mezclan) ----
TRAIN_SEEDS = [0, 1, 2]
SS_TUNE_SEEDS = list(range(2000, 2030))
VAL_SEEDS = list(range(500, 520))
TEST_SEEDS = list(range(1000, 1200))

# ---- busqueda (s, S) ----
SS_S_RANGE = range(0, 81, 4)
SS_BIG_S_RANGE = range(20, 141, 4)

# ---- entrenamiento ----
NET_ARCH = [64, 64]
EVAL_FREQ = 10_000
N_VAL_EPISODES = 20
DQN_STEPS = 150_000
PPO_STEPS = 300_000
QUICK_FACTOR = 0.1       # en --quick los pasos se multiplican por esto
DQN_PARAMS = dict(
    learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
    batch_size=64, target_update_interval=500, train_freq=4,
    exploration_fraction=0.3, exploration_final_eps=0.05, gamma=GAMMA,
)
PPO_PARAMS = dict(
    learning_rate=3e-4, n_steps=1000, batch_size=100, n_epochs=10,
    clip_range=0.2, gae_lambda=0.95, ent_coef=0.01, gamma=GAMMA,
)
