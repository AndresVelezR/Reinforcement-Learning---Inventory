# Guía de uso

Cómo instalar, reproducir, usar y modificar el proyecto. Para entender qué es RL, mira [CONCEPTOS.md](CONCEPTOS.md). Para saber por qué está hecho así, mira [DECISIONES.md](DECISIONES.md).

Nota sobre las mediciones: los tiempos y cifras de esta guía son **mediciones puntuales** en un portátil con CPU (16 hilos, sin GPU), Linux, Python 3.11.9. En otro equipo variarán. Cuando algo no se pudo probar, se dice.

## 1. Instalación

Requisitos: Python 3.10, 3.11 o 3.12. Se probó con **3.11.9**. Python 3.13 y 3.14 no se probaron con este repo.

Linux o Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell). Estos comandos no se pudieron probar en este equipo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Si el Python de tu sistema no es una versión soportada** (por ejemplo, tienes 3.13), instala una 3.10-3.12 y crea el entorno con ella. Con pyenv, esto funcionó en el equipo de pruebas (Python 3.11.9 ya instalado en pyenv):

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv .venv
source .venv/bin/activate
python --version        # debe decir 3.11.9
pip install -r requirements.txt
```

Para comprobar que quedó bien: `pytest` debe terminar con `5 passed`.

Cada vez que abras una terminal nueva hay que volver a activar el entorno (`source .venv/bin/activate`).

## 2. Reproducir los resultados

`run_all.py` corre cuatro pasos en orden. Tiempos medidos (CPU, 1 hilo para las redes):

| Paso | Qué hace | Genera | `--quick` | Completo |
|---|---|---|---|---|
| 1. Tests | `pytest -q` | nada | menos de 1 s | menos de 1 s |
| 2. Grid search | busca `s` y `S` | `results/ss_params.json` | 36 s | 36 s (no cambia con `--quick`) |
| 3. Entrenamiento | DQN y PPO | `models/*.zip`, `results/logs/` | 31 s (1 semilla) | 824 s (3 semillas x 2 algoritmos) |
| 4. Evaluación | tabla y figuras | `results/metrics.*`, `results/figures/`, `results/best_seeds.json` | 7-10 s | 32 s |
| **Total** | | | **72-80 s** | **892 s (~15 min)** |

Por algoritmo y semilla en la corrida completa: DQN ~93 s, PPO ~183 s.

Comandos:

```bash
python run_all.py --quick   # verificación rápida: NO es representativa del rendimiento
python run_all.py           # corrida completa, la que genera los números del README
```

Qué esperar en la consola: una línea `=== Paso N/4: ... ===` por paso, el avance del entrenamiento en español y al final `Todo terminado en N s`. Si algún paso falla, `run_all.py` se detiene ahí.

Cómo comprobar que funcionó: terminó con `Todo terminado`, existen los archivos de la columna "Genera" y `results/metrics.md` tiene la tabla de 4 políticas. Los números deben parecerse a los del README (no ser idénticos, ver "Resultados ligeramente distintos" en la sección 8). Los resultados de `--quick` **no sirven como evidencia de rendimiento** (se entrena con una décima parte de los pasos y una sola semilla, y la evaluación usa 20 episodios).

### Cuidado: `run_all.py` sobrescribe archivos versionados

Reescribe `models/` y `results/`, que están versionados en git. Para volver al estado original:

```bash
git checkout -- models results
```

Con eso se restauran los archivos versionados. `results/logs/` no está versionado, así que `git checkout` no lo toca; si quieres dejarlo limpio, borra esa carpeta (`rm -rf results/logs`). Esto se probó en un clon del repo: tras correr `run_all.py --quick` y hacer `git checkout -- models results`, `git status` quedó limpio.

## 3. Cada script y sus opciones

| Comando | Qué hace |
|---|---|
| `python policies.py` | Grid search de `s` y `S` (30 episodios por combinación). Escribe `results/ss_params.json`. Sin opciones. |
| `python train.py` | Entrena DQN y PPO. Guarda en `models/{algo}_seed{n}.zip` el **mejor checkpoint según validación** (se mide cada 10 000 pasos), que puede no ser el del último paso, y las curvas en `results/logs/`. |
| `python evaluate.py` | Evalúa las 4 políticas en 200 episodios de test. Genera tabla y 4 figuras. |
| `python demo.py` | Imprime un episodio día por día. |
| `python run_all.py` | Corre todo en orden. |
| `pytest` | Corre los tests del entorno. |

Opciones:

- `train.py --algo {dqn,ppo,all}` (por defecto `all`), `--quick` (una décima parte de los pasos), `--seeds 0 1 2` (lista de semillas; por defecto `0 1 2`, o solo `0` con `--quick`).
- `evaluate.py --quick`: solo 20 episodios de test.
- `demo.py --policy {ingenua,ss,dqn,ppo}` (por defecto `ss`), `--days N` (por defecto 10), `--seed N` (semilla del episodio, por defecto 1000), `--model-seed N` (qué modelo guardado cargar, por defecto 0).
- `run_all.py --quick`.

Ejemplos que se ejecutaron sin error:

```bash
python train.py --algo dqn --seeds 0 --quick
python demo.py --policy ppo --days 10 --model-seed 2
python demo.py --policy ss --days 5 --seed 1003
```

Salida real de `python demo.py --policy ppo --days 10 --model-seed 2` (los modelos de `models/`):

```
 dia  inventario   pipeline  pedido  demanda  ventas perdidas   costo
   0          40   [20, 20]      11       20                0      52
   1          40   [20, 11]      21       20                0      72
   2          40   [11, 21]      17       20                0      64
   3          31   [21, 17]      18       17                0      60
```

(se muestran las 4 primeras filas). Fíjate en que el pedido de 11 unidades del día 0 aparece en el inventario el día 3: 40 - 20 de ventas = 20, más 11 que llegan = 31.

Notas: `demo.py --policy ss` necesita `results/ss_params.json` (viene en el repo; si lo borras, corre `python policies.py`). `demo.py` con `dqn` o `ppo` necesita el modelo `models/{algo}_seed{n}.zip` correspondiente.

## 4. Cómo leer la tabla y las figuras

### La tabla (`results/metrics.md`)

Un detalle importante: las filas de DQN y PPO usan el **mejor checkpoint según validación** de cada semilla, no el modelo del último paso de entrenamiento. El modelo final rinde peor (en una medición puntual de una revisión independiente, semilla 0, no versionada en `results/`: PPO 6530.7 y DQN 6582.2 contra 6220.9 y 6226.3 con el checkpoint elegido). La tabla muestra lo que se logra **con parada temprana**.

| Columna | Qué significa | Cómo leerla |
|---|---|---|
| Costo promedio | Costo total de un episodio de 100 días, promedio de los episodios de test. Para DQN y PPO, además, promedio de las semillas de entrenamiento. | Menor es mejor. Es la medida principal. |
| Desv. entre episodios | Qué tanto varía el costo de un episodio a otro (por la suerte con la demanda). | Una política con valor alto es menos predecible. No es un error de la media. |
| Desv. entre semillas | Qué tanto varía el costo promedio entre entrenamientos con distinta semilla. Solo DQN y PPO. | Mide estabilidad del entrenamiento. Con 3 semillas es una estimación poco fiable. |
| Nivel de servicio (%) | Porcentaje de la demanda que se atendió. | Más alto es mejor para el cliente, pero no es gratis: la ingenua tiene 99.1 % porque acumula inventario, y por eso su costo es alto. |
| Inventario promedio | Unidades en bodega al final del día, en promedio. | Más inventario significa más costo de almacenar. |

Qué permite concluir: la ingenua es claramente peor que las otras tres. Qué **no** permite concluir: que DQN sea mejor o peor que PPO (diferencia dentro del ruido), ni que RL supere a (s, S): la diferencia (~0.4 % a favor de RL) es pequeña e inconsistente entre semillas, y con 3 semillas no es concluyente. Tampoco que RL "siempre" iguale a una heurística (es un solo problema con una sola configuración).

### Las figuras (`results/figures/`)

| Figura | Qué muestra | Qué buscar | Qué permite concluir / qué no |
|---|---|---|---|
| `curvas_aprendizaje.png` | Costo de validación (20 episodios, semillas 500-519) de DQN y PPO cada 10 000 pasos de entrenamiento: promedio de las 3 semillas y banda de desviación entre semillas. Las líneas horizontales son ingenua y (s, S), medidas en validación. | El eje y es el costo por episodio de 100 días, sin escalar, igual que las líneas horizontales. Mira que las curvas bajan rápido al principio y luego **oscilan**: los mejores puntos (PPO cerca de los 80 000 pasos, DQN cerca de los 110 000) quedan cerca de la línea de (s, S) (~6218), y los últimos puntos quedan por encima (DQN termina cerca de 6700; PPO anda entre ~6300 y ~6500 durante buena parte del entrenamiento y termina cerca de 6280; valores leídos de la figura). | Muestra que el entrenamiento es inestable, y por eso se usa el mejor checkpoint según validación (ver arriba, "Un detalle importante"). No muestra el costo de test. Solo existe con los logs de `results/logs/`, que se generan al entrenar. |
| `comparacion_costos.png` | Barras del costo promedio en test, con barras de error (desviación entre episodios). | Que las barras de (s, S), DQN y PPO queden casi a la misma altura. | Muestra que RL queda a la par de (s, S) (equivalencia práctica) y supera a la ingenua. Las barras de error son la variación entre episodios, no un intervalo de confianza de la media. |
| `trayectoria.png` | 60 días del mismo episodio con (s, S) y con el mejor agente de RL: inventario, demanda, pedidos y quiebres de stock (cruces rojas). | Cómo reacciona cada política a los picos de demanda y cuándo se queda sin stock. | Es un solo episodio: sirve para entender el comportamiento, no para comparar rendimiento. |
| `politica_aprendida.png` | El pedido que decide cada agente según la posición de inventario (con el pipeline repartido en partes iguales), junto a la regla (s, S). | Que la forma sea parecida: pide mucho cuando hay poco y deja de pedir cuando hay suficiente. | Sirve para explicar qué aprendió. En posiciones bajas el DQN difiere de (s, S): son estados que casi no ocurren (ver la FAQ de DECISIONES.md). |

## 5. Recetario de experimentos

Antes de cambiar algo, ten en cuenta tres reglas:

1. **Los modelos de `models/` corresponden a la configuración original.** Si cambias el problema (lead time, costos, demanda), hay que **reentrenar**. Comparar un modelo viejo en un problema nuevo no tiene sentido.
2. **Limpia antes de reentrenar.** `train.py --quick` solo reentrena la semilla 0; los modelos de las semillas 1 y 2 y los logs de la corrida anterior quedan en disco, y `evaluate.py` los usa y puede fallar o mezclar resultados. Borra primero: `rm -rf models results/logs`. Una corrida completa (`run_all.py`) reentrena las 3 semillas.
3. **`run_all.py` se detiene si los tests fallan.** Si cambiaste un parámetro que rompe un test, corre los pasos a mano: `python policies.py`, `python train.py`, `python evaluate.py`.

Todos estos cambios se probaron en una copia temporal fuera del repo, nunca en el repo real.

| Cambio | Dónde | Qué re-ejecutar | Qué esperar | Advertencias |
|---|---|---|---|---|
| Lead time de 3 a 2 | `config.py`: `LEAD_TIME` | `rm -rf models results/logs` y `python run_all.py` | La observación pasa de 3 a 2 números. Corrida `--quick` completa sin errores (80 s). | Los modelos de `models/` fallan con `ValueError: Unexpected observation shape (2,) for Box environment, please use (3,)`. Los tests **siguen pasando**, pero `test_lead_time` fija el pipeline en `[0, 0]`, así que no verifica realmente L=2. `ss_params.json` queda del problema anterior hasta que corras `policies.py`. |
| Costo de venta perdida de 9 a 19 | `config.py`: `LOST_SALE_COST` | `rm -rf models results/logs`, luego `python policies.py`, `python train.py`, `python evaluate.py` | La grid search elige s=80, S=92 (costo de ajuste 6686.8, medición puntual). Los tres pasos corren sin errores con `--quick`. | **Falla `test_costs_and_lost_sales`** (`assert (380.0 == 180)`): el test tiene el costo calculado a mano con 9. Por eso `run_all.py` se detiene en el paso 1. Además s=80 es el **máximo del rango** de la búsqueda (`SS_S_RANGE`): la búsqueda quedó en el borde y el resultado puede no ser el mejor posible. |
| Demanda media de 20 a 40 | `config.py`: `DEMAND_MEAN` | `rm -rf models results/logs` y `python run_all.py`; antes ensancha los rangos (ver abajo) | Los tests pasan. Con los rangos originales la búsqueda queda en el borde (s=80, S=108, costo de ajuste 18694.7) y esa (s, S) es peor que la ingenua en test (18665.6 contra 12784.5, 50 episodios). Con rangos ampliados (`SS_S_RANGE = range(0, 201, 4)`, `SS_BIG_S_RANGE = range(20, 261, 4)`) la búsqueda da s=148, S=168 (costo de ajuste 11113.1, test 11076.1 en 50 episodios) y tarda 156 s. | Moraleja: un resultado "(s, S) es mala" puede ser solo un rango de búsqueda insuficiente. Con demanda 40, el pedido máximo (50) deja poco margen y la observación se recorta a 150 unidades (`OBS_SCALE` x `OBS_CLIP`); esto último no se midió. El estado inicial (40 y 20) también queda bajo para esa demanda. |
| Más o menos pasos de entrenamiento | `config.py`: `DQN_STEPS`, `PPO_STEPS` | `python train.py --algo dqn --seeds 0` y luego `python evaluate.py` | DQN con 30 000 pasos: 24 s. Con 300 000: 209 s. Ver los costos en la FAQ de DECISIONES.md: no cambian claramente (una semilla). | `--quick` multiplica los pasos por `QUICK_FACTOR` (0.1). Los pasos de una corrida no se mezclan con logs de otra (ver regla 2). |
| Cambiar el tamaño de la red | `config.py`: `NET_ARCH` | `python train.py --algo dqn --seeds 0` | Con `[128, 128]` el DQN entrenó en 112 s. Los modelos viejos `[64, 64]` siguen cargándose, porque Stable-Baselines3 guarda la arquitectura dentro del `.zip`. | Una tabla con modelos de arquitecturas distintas mezcla experimentos distintos. Reentrena las 3 semillas de ambos algoritmos si quieres compararla con el README. |
| Más semillas de entrenamiento | `config.py`: `TRAIN_SEEDS` | `python train.py --seeds 0 1 2 3 4` y `python evaluate.py` | `train.py --seeds 5` crea `models/dqn_seed5.zip`, y `demo.py --model-seed 5` lo usa. | **`evaluate.py` solo mira `TRAIN_SEEDS` de `config.py`**: una semilla extra pasada solo a `train.py` se ignora. Si agregas la semilla a `TRAIN_SEEDS` pero sus logs vienen de una corrida distinta (`--quick`) a la de las otras, `evaluate.py` falla con `ValueError: setting an array element with a sequence`. La solución es `rm -rf results/logs` o reentrenar todas con el mismo modo. |

Si el error `ValueError: setting an array element with a sequence` aparece, la causa son logs de entrenamiento de largos distintos. Se comprobó que `rm -rf results/logs` seguido de `python evaluate.py --quick` lo resuelve.

## 6. Usar un modelo entrenado desde tu código

Guarda esto como `usar_modelo.py` **en la raíz del repo** (la ruta del modelo es relativa) y corre `python usar_modelo.py`:

```python
from stable_baselines3 import PPO
from inventory_env import InventoryEnv

model = PPO.load("models/ppo_seed2.zip", device="cpu")
env = InventoryEnv(continuous=True)   # PPO se entrenó con acciones continuas
obs, _ = env.reset(seed=123)

total = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total += info["cost"]
    done = terminated or truncated
print("costo del episodio:", total)
```

Salida al probarlo: `costo del episodio: 6337.0`. Para un modelo DQN usa `DQN.load(...)` y `InventoryEnv(continuous=False)`: el modo del entorno tiene que coincidir con el del algoritmo.

## 7. Extender el proyecto

Esta sección es una guía. El repo no se modificó. Las ediciones descritas se probaron en una copia temporal.

### Agregar una política nueva

Una política es una clase con un método `order(inventory, pipeline)` que devuelve cuántas unidades pedir (como `NaivePolicy` en `policies.py`). Por ejemplo, una política de "nivel fijo" (base-stock):

```python
import config as C
from policies import mean_cost


class BaseStockPolicy:
    """Pide lo necesario para dejar la posicion de inventario en un nivel fijo."""

    def __init__(self, level):
        self.level = level

    def order(self, inventory, pipeline):
        ip = inventory + sum(pipeline)  # posicion de inventario
        return max(0, min(C.MAX_ORDER, self.level - ip))


for level in (70, 80, 90):
    print(level, round(mean_cost(BaseStockPolicy(level), C.VAL_SEEDS), 1))
```

Se probó (guardado como un script en la raíz del repo). Salida, en las semillas de **validación**: nivel 70, 7171.9; nivel 80, 6326.7; nivel 90, 6318.4. Es un ejemplo de uso de `mean_cost`, no un resultado del proyecto.

Para que `demo.py` y `evaluate.py` la conozcan hay que editar tres archivos (esto se probó y funcionó con `demo.py` y `evaluate.py --quick`):

1. `policies.py`: agregar la clase y un nombre nuevo en `make_agent` (dentro del `if name in ("ingenua", "ss")`).
2. `evaluate.py`: agregar la política a la lista de referencias del inicio de `main` y una entrada en `COLORS`. Ojo: ese bucle crea el agente con `make_agent(...)` por nombre.
3. `demo.py`: agregar el nombre a `choices` de `--policy`.

### Agregar otro algoritmo de Stable-Baselines3 (por ejemplo A2C)

Archivos a tocar, con lo que se hizo en la prueba (A2C usa acciones continuas igual que PPO):

1. `config.py`: agregar un diccionario de hiperparámetros (en la prueba, `A2C_PARAMS = dict(learning_rate=7e-4, n_steps=5, gamma=GAMMA)`; son valores de ejemplo, no ajustados).
2. `train.py`: importar `A2C`, ampliar `choices` de `--algo`, y en `train_one` decidir `continuous`, la clase y los parámetros según el algoritmo.
3. `policies.py` (`make_agent`): cargar la clase correcta y devolver `InventoryEnv(continuous=...)` según el algoritmo.
4. `demo.py`: ampliar `choices`.
5. `evaluate.py`: ampliar los bucles que recorren `("dqn", "ppo")`, el `COLORS` y la carga del modelo en la figura de política aprendida.

Con esas ediciones, `python train.py --algo a2c --seeds 0 --quick` y `python demo.py --policy a2c` funcionaron, y `evaluate.py --quick` generó la tabla con la fila A2C. El rendimiento de esa A2C de prueba (entrenada en modo rápido) no es representativo y no se usa como evidencia de nada.

## 8. Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (u otro) | El entorno virtual no está activado, o `pip install` no terminó. | Activa `.venv` y corre `pip install -r requirements.txt` de nuevo, mirando el final de la salida. |
| La instalación de `torch` tarda o pesa mucho | En Linux, `pip` descarga una versión grande con soporte para GPU (cientos de MB). | Es normal. Solo se usa la CPU. Espera a que termine. |
| Versión de Python incorrecta | Usas una versión fuera de 3.10-3.12. | Crea el entorno con una versión soportada (sección 1). |
| `ValueError: setting an array element with a sequence` al correr `evaluate.py` | Hay logs en `results/logs/` de corridas distintas (por ejemplo `--quick` y completa). | `rm -rf results/logs` y repite, o reentrena todas las semillas con el mismo modo. |
| `ValueError: Unexpected observation shape ...` | Un modelo guardado se entrenó con otra forma de observación (por ejemplo, otro `LEAD_TIME`). | Reentrena: `rm -rf models results/logs` y `python run_all.py`. |
| `run_all.py` se detiene en el paso 1 | Un test falla (por ejemplo, tras cambiar un costo). | Corre `pytest` para ver cuál, y los pasos 2 a 4 a mano. |
| `FileNotFoundError` con `ss_params.json` | Borraste `results/ss_params.json`. | `python policies.py` lo regenera, o `git checkout -- results`. |
| `curvas_aprendizaje.png` aparece sin curvas de DQN y PPO | No hay logs en `results/logs/` (no se versionan; solo existen si entrenas en tu equipo). Ojo: correr `evaluate.py` sin logs **sobrescribe** la figura versionada con una sin curvas. | Entrena (`python train.py`) y vuelve a correr `evaluate.py`, o restaura la figura con `git checkout -- results`. |
| Los números no coinciden exactamente con el README | Diferencias entre sistemas operativos y versiones de librerías. | Es esperado. Las semillas están fijas, pero el resultado exacto puede variar un poco. Compara el orden de las políticas y el orden de magnitud, no los decimales. |
| Los resultados de `--quick` se ven peores que los del README | `--quick` usa una décima parte de los pasos y una sola semilla. | Es esperado. `--quick` solo verifica que todo corre. |
