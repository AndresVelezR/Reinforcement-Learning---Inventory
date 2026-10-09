# inventario-rl: aprendizaje por refuerzo para controlar un inventario

Ejemplo en Python de **aprendizaje por refuerzo (RL)** aplicado a un problema de inventarios. Hecho para la exposición del curso Aprendizaje Automático (EAFIT, 2026-2).
Compara dos algoritmos de RL (DQN y PPO) contra dos reglas clásicas de referencia.

> Si no sabes nada de RL, empieza por [`docs/CONCEPTOS.md`](docs/CONCEPTOS.md): son 5 minutos y explica todo desde cero.

## El problema

**Control de inventarios con demanda estocástica y lead time** (*lost-sales inventory problem with lead time*). Una tienda vende un producto y cada día decide cuánto pedir. La demanda es aleatoria (Poisson, media 20) y lo que se pide llega 3 días después. Si falta inventario, la venta se pierde. Se pagan costos por almacenar, por perder ventas y por pedir. Hay que decidir sin conocer la demanda y con retraso, y la recompensa es diferida, por eso es un caso de estudio de RL.

## Resultados

Costo por episodio de 100 días en 200 episodios de test (mismas demandas para todas las políticas). Menor es mejor. DQN y PPO se entrenaron con 3 semillas.

| Politica | Costo promedio | Desv. entre episodios | Desv. entre semillas | Nivel de servicio (%) | Inventario promedio |
|---|---|---|---|---|---|
| Ingenua | 8015.7 | 1486.3 | - | 99.1 | 28.5 |
| (s, S) | 6254.2 | 183.8 | - | 96.1 | 7.5 |
| DQN | 6229.3 | 185.0 | 6.7 | 95.9 | 7.1 |
| PPO | 6227.4 | 178.8 | 29.5 | 96.6 | 7.8 |

![Comparacion de costos](results/figures/comparacion_costos.png)

**Lectura honesta:** RL mejora claramente a la política ingenua (~22 % menos costo), pero **empata con (s, S)**: la diferencia (~25 de ~6250) es menor que la variación entre episodios y entre semillas. Es esperable: (s, S) afinada con grid search ya es muy buena para este problema con un solo producto, así que hay poco margen. El valor de RL aquí es que llega a ese nivel sin que nadie le diga la regla. Más figuras en `results/figures/`.

## Cómo correrlo

Requisitos: Python 3.10-3.12 (probado con 3.11.9; librerías exactas en `requirements.txt`).

```bash
# Linux / Mac
python3 -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
python -m venv .venv; .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python run_all.py --quick   # verificacion rapida (~1 min)
python run_all.py           # completo (~15 min en CPU)
python demo.py --policy ppo --days 10   # ingenua | ss | dqn | ppo
pytest
```

Dónde queda cada cosa: modelos en `models/`, tabla en `results/metrics.md` y `results/metrics.csv`, s y S en `results/ss_params.json`, figuras en `results/figures/`, curvas de entrenamiento en `results/logs/` (no se versiona). Ojo: `run_all.py` sobrescribe `models/` y `results/`, y la versión `--quick` deja resultados de prueba.

## Estructura

- `config.py`: todos los parámetros del problema y del entrenamiento.
- `inventory_env.py`: el simulador de la tienda (estilo Gymnasium).
- `policies.py`: políticas ingenua y (s, S), y la búsqueda de s y S.
- `train.py`: entrena DQN y PPO con Stable-Baselines3.
- `evaluate.py`: compara las 4 políticas y genera tabla y figuras.
- `demo.py`: imprime un episodio día por día.
- `run_all.py`: corre todo en orden.
- `docs/CONCEPTOS.md`: RL explicado desde cero.
- `tests/test_env.py`: pruebas del entorno.

## Cambiar parámetros

Todo está en `config.py`: lead time, demanda, costos, largo del episodio, hiperparámetros, semillas y rangos de la búsqueda. Si cambias el problema, vuelve a correr `python run_all.py`.

## Reproducibilidad

Las semillas están fijas (entrenamiento 0-2, validación 500-519, test 1000-1199). Aun así, los resultados exactos pueden variar un poco entre sistemas operativos y versiones de librerías.

## Lecturas recomendadas

- Mnih et al. (2015). *Human-level control through deep reinforcement learning* (DQN). *Nature*.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Gijsbrechts et al. (2022). *Can Deep Reinforcement Learning Improve Inventory Management?* *Manufacturing & Service Operations Management*.
- Sutton y Barto. *Reinforcement Learning: An Introduction* (libro gratuito en línea).
