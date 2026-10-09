"""Simulador de una tienda con un solo producto (estilo Gymnasium).

Orden de eventos de cada dia t:
  1. El agente ve el estado: inventario I_t (ya con la llegada de hoy) y el
     pipeline [q_{t-L+1}, ..., q_{t-1}] (pedidos en camino, el mas viejo primero).
  2. El agente elige el pedido q_t.
  3. Se sortea la demanda D_t. ventas = min(I_t, D_t),
     ventas perdidas = D_t - ventas, inventario final = I_t - ventas.
  4. costo = h*inv_final + p*ventas_perdidas + K*(q_t > 0) + c*q_t
  5. cola = pipeline + [q_t]; el primero de la cola llega:
     I_{t+1} = inv_final + llegada, y el pipeline nuevo es el resto.
  6. Resultado: lo pedido el dia t llega al inicio del dia t + L.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C


class InventoryEnv(gym.Env):
    """Inventario con demanda aleatoria, lead time y ventas perdidas."""

    def __init__(self, continuous=False):
        super().__init__()
        self.continuous = continuous
        self.n_pipe = C.LEAD_TIME - 1  # pedidos "en camino" que ve el agente

        # IDEA CLAVE: el estado incluye el pipeline, no solo el inventario.
        # Si el agente no viera lo que ya pidio, pediria de nuevo lo mismo
        # sin saber que viene en camino, y el problema dejaria de ser markoviano
        # (el estado no bastaria para decidir bien).
        self.observation_space = spaces.Box(
            0.0, C.OBS_CLIP, shape=(1 + self.n_pipe,), dtype=np.float32)

        if continuous:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(C.N_ACTIONS)

        self.rng = np.random.default_rng()
        self.inventory = 0
        self.pipeline = []
        self.day = 0

    # ---- utilidades ----
    def _to_order(self, action):
        """Convierte la accion del agente en unidades a pedir."""
        if self.continuous:
            # [-1, 1] -> [0, MAX_ORDER], redondeado a entero
            x = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            return int(round((x + 1.0) / 2.0 * C.MAX_ORDER))
        step = C.MAX_ORDER // (C.N_ACTIONS - 1)
        return int(action) * step

    def _sample_demand(self):
        return int(self.rng.poisson(C.DEMAND_MEAN))

    def raw_state(self):
        """Estado sin normalizar: (inventario, lista del pipeline)."""
        return self.inventory, list(self.pipeline)

    def _obs(self):
        raw = np.array([self.inventory] + self.pipeline, dtype=np.float32)
        return np.clip(raw / C.OBS_SCALE, 0.0, C.OBS_CLIP).astype(np.float32)

    # ---- API de gymnasium ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.inventory = C.INIT_INVENTORY
        self.pipeline = [C.INIT_PIPELINE] * self.n_pipe
        self.day = 0
        return self._obs(), {}

    def step(self, action):
        order = self._to_order(action)
        demand = self._sample_demand()

        # si el inventario no alcanza, lo que falta se pierde (venta perdida)
        sales = min(self.inventory, demand)
        lost_sales = demand - sales
        end_inventory = self.inventory - sales

        cost = (C.HOLDING_COST * end_inventory
                + C.LOST_SALE_COST * lost_sales
                + C.FIXED_ORDER_COST * (order > 0)
                + C.UNIT_ORDER_COST * order)

        # el pedido que hice hace L dias llega hoy
        queue = self.pipeline + [order]
        arrival = queue.pop(0)
        self.inventory = end_inventory + arrival
        self.pipeline = queue

        self.day += 1
        truncated = self.day >= C.EPISODE_LEN  # el episodio termina solo por tiempo
        info = {"cost": float(cost), "lost_sales": lost_sales, "demand": demand,
                "order": order, "sales": sales, "inventory": end_inventory}
        return self._obs(), -float(cost) / C.REWARD_SCALE, False, truncated, info
