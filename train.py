"""Entrena DQN (acciones discretas) y PPO (acciones continuas) con Stable-Baselines3."""
import argparse
import shutil
import time

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import config as C
from inventory_env import InventoryEnv


class ValidationEnv(InventoryEnv):
    """Entorno de validacion: cada episodio usa la siguiente semilla de VAL_SEEDS."""

    def __init__(self, continuous):
        super().__init__(continuous)
        self.k = 0

    def reset(self, seed=None, options=None):
        # ignoramos la semilla pedida: siempre las mismas 20 de validacion
        s = C.VAL_SEEDS[self.k % len(C.VAL_SEEDS)]
        self.k += 1
        return super().reset(seed=s, options=options)


def train_one(algo, seed, quick):
    """Entrena un modelo y deja el mejor (segun validacion) en models/."""
    continuous = algo == "ppo"
    steps = C.PPO_STEPS if continuous else C.DQN_STEPS
    if quick:
        steps = int(steps * C.QUICK_FACTOR)

    set_random_seed(seed)  # python, numpy y torch
    env = Monitor(InventoryEnv(continuous))
    val_env = Monitor(ValidationEnv(continuous))

    log_dir = C.LOGS_DIR / f"{algo}_seed{seed}"
    callback = EvalCallback(
        val_env, n_eval_episodes=C.N_VAL_EPISODES, eval_freq=C.EVAL_FREQ,
        deterministic=True, best_model_save_path=str(log_dir),
        log_path=str(log_dir), verbose=0)

    cls = PPO if continuous else DQN
    params = C.PPO_PARAMS if continuous else C.DQN_PARAMS
    if quick and not continuous:
        # con pocos pasos hay que empezar a aprender antes
        params = dict(params, learning_starts=500)
    model = cls("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=C.NET_ARCH), **params)

    print(f"[{algo.upper()}] semilla {seed}: entrenando {steps} pasos...", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callback)
    print(f"[{algo.upper()}] semilla {seed}: listo en {time.time() - t0:.0f} s", flush=True)

    # nos quedamos con el checkpoint de mejor recompensa en validacion
    C.MODELS_DIR.mkdir(exist_ok=True)
    best = log_dir / "best_model.zip"
    if best.exists():
        shutil.copy(best, C.MODELS_DIR / f"{algo}_seed{seed}.zip")
    else:
        model.save(C.MODELS_DIR / f"{algo}_seed{seed}")


def main():
    ap = argparse.ArgumentParser(description="Entrena DQN y PPO")
    ap.add_argument("--algo", choices=["dqn", "ppo", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="prueba rapida (pocos pasos)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)  # redes chicas: un hilo es mas rapido
    seeds = args.seeds if args.seeds is not None else ([0] if args.quick else C.TRAIN_SEEDS)
    algos = ["dqn", "ppo"] if args.algo == "all" else [args.algo]
    t0 = time.time()
    for algo in algos:
        for seed in seeds:
            train_one(algo, seed, args.quick)
    print(f"Entrenamiento terminado en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
