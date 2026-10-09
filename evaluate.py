"""Compara las 4 politicas en episodios de test y genera tablas y figuras."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from policies import NaivePolicy, load_ss, make_agent, run_episode

COLORS = {"Ingenua": "#7f7f7f", "(s, S)": "#0072B2", "DQN": "#D55E00", "PPO": "#009E73"}
plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.3})


def summarize(env, choose, seeds):
    """Corre un episodio por semilla y devuelve costo, servicio e inventario de cada uno."""
    eps = [run_episode(env, choose, s) for s in seeds]
    return {
        "cost": np.array([e["cost"].sum() for e in eps]),
        "service": np.array([e["sales"].sum() / e["demand"].sum() for e in eps]),
        "inventory": np.array([e["inventory"].mean() for e in eps]),
    }


def main():
    ap = argparse.ArgumentParser(description="Evalua politicas y genera figuras")
    ap.add_argument("--quick", action="store_true", help="solo 20 episodios de test")
    args = ap.parse_args()
    test_seeds = C.TEST_SEEDS[:20] if args.quick else C.TEST_SEEDS
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. evaluar todo con las mismas semillas (mismas demandas) ----
    print("Evaluando politicas en test...")
    results = {}      # nombre -> lista de resultados (uno por semilla de entrenamiento)
    val_cost = {}     # (algo, semilla) -> costo en validacion, para elegir el mejor
    for name, policy in [("Ingenua", NaivePolicy()), ("(s, S)", load_ss())]:
        env, choose = make_agent("ingenua" if name == "Ingenua" else "ss")
        results[name] = [summarize(env, choose, test_seeds)]
        val_cost[name] = summarize(env, choose, C.VAL_SEEDS)["cost"].mean()
    for algo in ("dqn", "ppo"):
        results[algo.upper()] = []
        for seed in C.TRAIN_SEEDS:
            if not (C.MODELS_DIR / f"{algo}_seed{seed}.zip").exists():
                continue
            env, choose = make_agent(algo, seed)
            results[algo.upper()].append(summarize(env, choose, test_seeds))
            val_cost[(algo, seed)] = summarize(env, choose, C.VAL_SEEDS)["cost"].mean()

    # ---- 2. tabla ----
    rows = []
    for name, runs in results.items():
        seed_means = [r["cost"].mean() for r in runs]
        rows.append({
            "Politica": name,
            "Costo promedio": np.mean(seed_means),
            "Desv. entre episodios": np.mean([r["cost"].std() for r in runs]),
            "Desv. entre semillas": np.std(seed_means) if len(runs) > 1 else np.nan,
            "Nivel de servicio (%)": 100 * np.mean([r["service"].mean() for r in runs]),
            "Inventario promedio": np.mean([r["inventory"].mean() for r in runs]),
        })
    df = pd.DataFrame(rows).round(1)
    df.to_csv(C.RESULTS_DIR / "metrics.csv", index=False)
    md = ["| Politica | Costo promedio | Desv. entre episodios | Desv. entre semillas "
          "| Nivel de servicio (%) | Inventario promedio |", "|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        sd = "-" if np.isnan(r[3]) else f"{r[3]:.1f}"
        md.append(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {sd} | {r[4]:.1f} | {r[5]:.1f} |")
    (C.RESULTS_DIR / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- 3. figura: curvas de aprendizaje ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ("dqn", "ppo"):
        curves = []
        for seed in C.TRAIN_SEEDS:
            f = C.LOGS_DIR / f"{algo}_seed{seed}" / "evaluations.npz"
            if f.exists():
                d = np.load(f)
                steps = d["timesteps"]
                curves.append(-d["results"].mean(axis=1) * C.REWARD_SCALE)  # recompensa -> costo
        if curves:
            m, sd = np.mean(curves, axis=0), np.std(curves, axis=0)
            ax.plot(steps, m, label=algo.upper(), color=COLORS[algo.upper()], lw=2)
            ax.fill_between(steps, m - sd, m + sd, color=COLORS[algo.upper()], alpha=0.2)
    for name in ("Ingenua", "(s, S)"):
        ax.axhline(val_cost[name], color=COLORS[name], ls="--", lw=2, label=name)
    ax.set_ylim(top=val_cost["Ingenua"] * 1.3)
    ax.set_xlabel("Pasos de entrenamiento")
    ax.set_ylabel("Costo por episodio (validacion)")
    ax.set_title("Curvas de aprendizaje (menor es mejor)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "curvas_aprendizaje.png", dpi=200)
    plt.close(fig)

    # ---- 4. figura: comparacion de costos ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(df["Politica"], df["Costo promedio"], yerr=df["Desv. entre episodios"],
           color=[COLORS[p] for p in df["Politica"]], capsize=6)
    for i, v in enumerate(df["Costo promedio"]):
        ax.text(i, v / 2, f"{v:.0f}", ha="center", color="white", fontweight="bold")
    ax.set_ylabel("Costo promedio por episodio")
    ax.set_title(f"Costo en test ({len(test_seeds)} episodios, menor es mejor)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "comparacion_costos.png", dpi=200)
    plt.close(fig)

    # ---- 5. mejor semilla de cada algoritmo, elegida con validacion (no con test) ----
    best = {}
    for algo in ("dqn", "ppo"):
        cands = {k[1]: v for k, v in val_cost.items() if isinstance(k, tuple) and k[0] == algo}
        if cands:
            best[algo] = min(cands, key=cands.get)
    overall = min(best, key=lambda a: val_cost[(a, best[a])])
    (C.RESULTS_DIR / "best_seeds.json").write_text(
        json.dumps({"por_algoritmo": best, "mejor_global": overall}, indent=2))

    # ---- 6. figura: trayectoria de un episodio ----
    ss_env, ss_choose = make_agent("ss")
    rl_env, rl_choose = make_agent(overall, best[overall])
    seed, days = test_seeds[0], 60
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, name, env, ch in [(axes[0], "(s, S)", ss_env, ss_choose),
                              (axes[1], overall.upper(), rl_env, rl_choose)]:
        ep = run_episode(env, ch, seed)
        x = np.arange(days)
        ax.bar(x, ep["order"][:days], color="#E69F00", alpha=0.6, label="Pedido")
        ax.plot(x, ep["inventory"][:days], color=COLORS[name], lw=2, label="Inventario al final del dia")
        ax.plot(x, ep["demand"][:days], color="black", lw=1, ls=":", label="Demanda")
        out = x[ep["lost_sales"][:days] > 0]
        ax.scatter(out, np.zeros(len(out)), marker="x", color="red", s=80, zorder=5,
                   label="Quiebre de stock")
        ax.set_title(f"{name}: costo del episodio = {ep['cost'].sum():.0f}")
        ax.set_ylabel("Unidades")
    axes[1].set_xlabel("Dia")
    axes[0].legend(ncol=4, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "trayectoria.png", dpi=200)
    plt.close(fig)

    # ---- 7. figura: politica aprendida ----
    ss = load_ss()
    ips = np.arange(0, 141)
    fig, axes = plt.subplots(1, len(best), figsize=(6 * len(best), 5.5), sharey=True, squeeze=False)
    for ax, algo in zip(axes[0], best):
        env, _ = make_agent(algo, best[algo])
        from stable_baselines3 import DQN, PPO
        model = (DQN if algo == "dqn" else PPO).load(
            C.MODELS_DIR / f"{algo}_seed{best[algo]}.zip", device="cpu")
        orders = []
        for ip in ips:
            # el pipeline se reparte en partes iguales (inventario y 2 pedidos en camino)
            obs = np.clip(np.full(C.LEAD_TIME, ip / C.LEAD_TIME) / C.OBS_SCALE,
                          0, C.OBS_CLIP).astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            orders.append(env._to_order(action))
        ax.step(ips, orders, where="post", color=COLORS[algo.upper()], lw=2.5, label=algo.upper())
        ax.step(ips, [ss.order(ip, []) for ip in ips], where="post", color=COLORS["(s, S)"],
                lw=2, ls="--", label="(s, S)")
        ax.set_title(f"{algo.upper()} (semilla {best[algo]})")
        ax.set_xlabel("Posicion de inventario (bodega + en camino)")
        ax.legend()
    axes[0][0].set_ylabel("Unidades que pide")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "politica_aprendida.png", dpi=200)
    plt.close(fig)
    print(f"Listo: tabla en {C.RESULTS_DIR.name}/metrics.md y figuras en {C.FIGURES_DIR.name}/")


if __name__ == "__main__":
    main()
