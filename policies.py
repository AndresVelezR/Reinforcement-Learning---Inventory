"""Politicas de referencia (ingenua y (s, S)) y utilidades para simular episodios."""
import argparse
import json
import time

import numpy as np

import config as C
from inventory_env import InventoryEnv


def units_to_action(units):
    """Pedido en unidades -> accion del entorno continuo (inversa de _to_order)."""
    return np.array([units / (C.MAX_ORDER / 2) - 1.0], dtype=np.float32)


class NaivePolicy:
    """Ingenua: pide siempre la demanda media."""

    def order(self, inventory, pipeline):
        return C.DEMAND_MEAN


class SSPolicy:
    """(s, S): si la posicion de inventario baja de s, pide hasta llegar a S."""

    def __init__(self, s, big_s):
        self.s = s
        self.big_s = big_s

    def order(self, inventory, pipeline):
        # posicion de inventario = lo que tengo + lo que viene en camino
        ip = inventory + sum(pipeline)
        if ip <= self.s:
            return int(np.clip(round(self.big_s - ip), 0, C.MAX_ORDER))
        return 0


def run_episode(env, choose, seed):
    """Simula un episodio completo; choose(obs, env) devuelve la accion."""
    obs, _ = env.reset(seed=seed)
    rec = {"inventory": [], "demand": [], "order": [], "lost_sales": [],
           "cost": [], "sales": [], "pipeline": []}
    done = False
    while not done:
        rec["pipeline"].append(list(env.pipeline))
        obs, _, term, trunc, info = env.step(choose(obs, env))
        done = term or trunc
        for k in ("inventory", "demand", "order", "lost_sales", "cost", "sales"):
            rec[k].append(info[k])
    return {k: np.array(v) for k, v in rec.items()}


def baseline_chooser(policy):
    """Adapta una politica de referencia (que usa el estado crudo) al entorno continuo."""
    def choose(obs, env):
        inv, pipe = env.raw_state()
        return units_to_action(policy.order(inv, pipe))
    return choose


def mean_cost(policy, seeds):
    """Costo promedio por episodio de una politica de referencia."""
    env = InventoryEnv(continuous=True)
    choose = baseline_chooser(policy)
    return float(np.mean([run_episode(env, choose, s)["cost"].sum() for s in seeds]))


def grid_search_ss():
    """Busqueda exhaustiva (grid search) de s y S con las semillas de ajuste."""
    best = None
    for s in C.SS_S_RANGE:
        for big_s in C.SS_BIG_S_RANGE:
            if big_s <= s:
                continue
            cost = mean_cost(SSPolicy(s, big_s), C.SS_TUNE_SEEDS)
            if best is None or cost < best["cost"]:
                best = {"s": s, "S": big_s, "cost": cost}
    return best


def load_ss():
    """Lee s y S guardados por la busqueda."""
    p = json.loads((C.RESULTS_DIR / "ss_params.json").read_text())
    return SSPolicy(p["s"], p["S"])


if __name__ == "__main__":
    argparse.ArgumentParser(description="Busca s y S con grid search").parse_args()
    t0 = time.time()
    print("Buscando los mejores s y S (grid search)...")
    best = grid_search_ss()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "ss_params.json").write_text(json.dumps(best, indent=2))
    print(f"Mejor: s={best['s']}, S={best['S']}, costo promedio={best['cost']:.1f} "
          f"({time.time() - t0:.0f} s)")
