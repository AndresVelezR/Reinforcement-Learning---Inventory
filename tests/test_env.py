"""Pruebas basicas del entorno de inventario."""
import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from inventory_env import InventoryEnv


def test_lead_time():
    # un pedido del dia 0 aparece en el inventario al inicio del dia 3, no antes
    env = InventoryEnv()
    env.reset(seed=0)
    env._sample_demand = lambda: 0
    env.inventory, env.pipeline = 0, [0, 0]
    env.step(2)  # pide 10 el dia 0
    assert env.inventory == 0   # inicio dia 1
    env.step(0)
    assert env.inventory == 0   # inicio dia 2
    env.step(0)
    assert env.inventory == 10  # inicio dia 3


def test_costs_and_lost_sales():
    env = InventoryEnv()
    env.reset(seed=0)
    env._sample_demand = lambda: 30
    # inventario 40 -> demanda 30: ventas 30, sobran 10, sin perdidas
    _, r, _, _, info = env.step(2)  # pide 10
    # costo = 1*10 + 9*0 + 10 + 2*10 = 40
    assert info["cost"] == 40 and info["lost_sales"] == 0
    assert np.isclose(r, -40 / C.REWARD_SCALE)
    # ahora inventario = 10 + 20 = 30; demanda 50 -> pierde 20
    env._sample_demand = lambda: 50
    _, _, _, _, info = env.step(0)
    # costo = 0 + 9*20 + 0 + 0 = 180
    assert info["cost"] == 180 and info["lost_sales"] == 20


def test_same_seed_same_trajectory():
    def run():
        env = InventoryEnv()
        env.reset(seed=7)
        return [env.step(3)[4]["demand"] for _ in range(20)]
    assert run() == run()


def test_shapes_and_truncation():
    d = InventoryEnv(continuous=False)
    obs, _ = d.reset(seed=0)
    assert obs.shape == (C.LEAD_TIME,) and obs.dtype == np.float32
    assert d.action_space.n == 11
    c = InventoryEnv(continuous=True)
    c.reset(seed=0)
    assert c.action_space.shape == (1,)
    # al dia 100 se trunca, nunca termina
    for i in range(C.EPISODE_LEN):
        _, _, term, trunc, _ = d.step(0)
    assert trunc and not term


def test_check_env():
    check_env(InventoryEnv(continuous=False))
    check_env(InventoryEnv(continuous=True))


def fixed_demand(env, value):
    # reemplaza el sorteo de la demanda por un valor fijo
    env._sample_demand = lambda: value


def test_lead_time_with_nonzero_pipeline():
    # el pipeline [7, 9] llega en orden (el mas viejo primero) y el pedido nuevo al tercer dia
    env = InventoryEnv()
    env.reset(seed=0)
    fixed_demand(env, 0)
    env.inventory, env.pipeline = 0, [7, 9]
    env.step(2)  # pide 10
    assert env.inventory == 7 and env.pipeline == [9, 10]    # inicio dia 1
    env.step(0)
    assert env.inventory == 16 and env.pipeline == [10, 0]   # inicio dia 2
    env.step(0)
    assert env.inventory == 26 and env.pipeline == [0, 0]    # inicio dia 3: llega el pedido


def test_lost_sales_when_demand_equals_inventory():
    # demanda igual al inventario: se vende todo y no se pierde nada
    env = InventoryEnv()
    env.reset(seed=0)
    fixed_demand(env, 40)  # el inventario inicial es 40
    _, _, _, _, info = env.step(0)
    assert info["sales"] == 40 and info["lost_sales"] == 0 and info["inventory"] == 0
    assert info["cost"] == 0  # sin sobrante, sin ventas perdidas, sin pedido
    # una unidad mas de demanda: se pierde exactamente una
    env.reset(seed=0)
    fixed_demand(env, 41)
    _, _, _, _, info = env.step(0)
    assert info["lost_sales"] == 1 and info["cost"] == C.LOST_SALE_COST


def test_truncated_only_at_last_day():
    env = InventoryEnv()
    env.reset(seed=0)
    for day in range(1, C.EPISODE_LEN + 1):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert truncated == (day == C.EPISODE_LEN)  # solo el dia 100


def test_continuous_action_is_rounded_not_truncated():
    env = InventoryEnv(continuous=True)

    def order_for(x):
        env.reset(seed=0)
        fixed_demand(env, 0)
        return env.step(np.array([x], dtype=np.float32))[4]["order"]

    assert order_for(-0.496) == 13  # (x + 1) / 2 * 50 = 12.6 -> 13 (truncar daria 12)
    assert order_for(0.404) == 35   # 35.1 -> 35
    assert order_for(0.0) == 25
    assert order_for(-1.0) == 0 and order_for(1.0) == 50
