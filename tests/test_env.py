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
