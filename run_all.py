"""Corre todo en orden: tests, busqueda (s, S), entrenamiento y evaluacion."""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("Corriendo tests", ["-m", "pytest", "-q"]),
    ("Buscando s y S (grid search)", ["policies.py"]),
    ("Entrenando DQN y PPO", ["train.py"]),
    ("Evaluando y generando figuras", ["evaluate.py"]),
]


def main():
    ap = argparse.ArgumentParser(description="Corre todo el proyecto")
    ap.add_argument("--quick", action="store_true", help="version rapida para verificar")
    args = ap.parse_args()

    t0 = time.time()
    for i, (name, cmd) in enumerate(STEPS, 1):
        if args.quick and cmd[0] in ("train.py", "evaluate.py"):
            cmd = cmd + ["--quick"]
        print(f"\n=== Paso {i}/{len(STEPS)}: {name} ===", flush=True)
        t = time.time()
        subprocess.run([sys.executable] + cmd, check=True)
        print(f"--- listo en {time.time() - t:.0f} s", flush=True)
    print(f"\nTodo terminado en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
