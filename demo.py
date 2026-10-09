"""Imprime un episodio dia por dia para ver que decide cada politica."""
import argparse

import config as C
from policies import make_agent


def main():
    ap = argparse.ArgumentParser(description="Demo dia por dia")
    ap.add_argument("--policy", choices=["ingenua", "ss", "dqn", "ppo"], default="ss")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000, help="semilla del episodio")
    ap.add_argument("--model-seed", type=int, default=0, help="semilla de entrenamiento del modelo")
    args = ap.parse_args()

    env, choose = make_agent(args.policy, args.model_seed)
    obs, _ = env.reset(seed=args.seed)
    print(f"Politica: {args.policy} | lead time = {C.LEAD_TIME} dias | demanda ~ Poisson({C.DEMAND_MEAN})")
    print("(inventario = en bodega al inicio del dia; pipeline = pedidos en camino, el mas viejo primero)\n")
    print(f"{'dia':>4} {'inventario':>11} {'pipeline':>10} {'pedido':>7} {'demanda':>8} {'ventas perdidas':>16} {'costo':>7}")

    total = 0
    for day in range(min(args.days, C.EPISODE_LEN)):
        inv, pipe = env.raw_state()  # lo que ve el agente antes de decidir
        obs, _, _, _, info = env.step(choose(obs, env))
        total += info["cost"]
        print(f"{day:>4} {inv:>11} {str(pipe):>10} {info['order']:>7} {info['demand']:>8} "
              f"{info['lost_sales']:>16} {info['cost']:>7.0f}")
    print(f"\nCosto acumulado: {total:.0f}")


if __name__ == "__main__":
    main()
