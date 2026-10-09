# Registro de decisiones

Este documento explica **por qué** el proyecto está hecho como está y no de otra forma. Cada decisión tiene el mismo formato: qué se eligió, qué alternativas había, por qué, qué se sacrifica y dónde está en el código.

Una aclaración honesta: no hay evidencia en el repo de que se haya hecho una búsqueda sistemática de valores (costos, hiperparámetros, tamaños). Cuando eso aplica, se dice "valor de partida razonable", no "valor óptimo". Lo que no se pudo justificar con evidencia está marcado como **(por confirmar)**.

Si no conoces los términos, lee primero [CONCEPTOS.md](CONCEPTOS.md). Para correr o modificar cosas, mira [GUIA_USO.md](GUIA_USO.md).

---

## 1. El problema: inventario con ventas perdidas y lead time

- **Qué se eligió:** una tienda, un producto, demanda aleatoria, pedidos que tardan 3 días y ventas que se pierden si falta inventario.
- **Alternativas:** inventario con pedidos pendientes (*backorders*, la venta se atiende después), varios productos o bodegas, otros problemas clásicos de RL (juegos, control).
- **Por qué:** es un problema con las tres características que hacen que RL tenga sentido: decisiones secuenciales, recompensa diferida y ningún dato etiquetado. Además se explica en una frase y se simula rápido. Que el lead time haga el problema más difícil de resolver con una regla simple es el motivo para usarlo como caso de estudio (ver también el artículo de Gijsbrechts et al. 2022 en las lecturas del README).
- **Qué se sacrifica:** realismo. Un negocio real tiene muchos productos, demanda con tendencia y temporadas, y proveedores con tiempos variables.
- **Dónde:** `inventory_env.py` (clase `InventoryEnv`, método `step`), `LEAD_TIME` en `config.py`.

## 2. Demanda Poisson y valores de costos

- **Qué se eligió:** demanda diaria Poisson con media 20. Costos: almacenar 1 por unidad, venta perdida 9 por unidad, costo fijo por pedido 10, costo por unidad pedida 2.
- **Alternativas:** demanda normal, demanda con estacionalidad, datos reales; otros costos.
- **Por qué:** Poisson genera números enteros no negativos con un solo parámetro, que encaja con "unidades vendidas por día". Los costos tienen la forma típica de estos problemas: perder una venta (9) cuesta mucho más que guardar una unidad un día (1), y pedir tiene una parte fija (10) que castiga hacer pedidos pequeños y frecuentes. **(por confirmar)** si los números concretos se escogieron por alguna razón especial; en el repo son valores de partida.
- **Qué se sacrifica:** en Poisson la varianza es igual a la media, lo que fija cuánta incertidumbre hay. No hay tendencias ni temporadas. Los resultados dependen de estos costos: con otros valores cambian los pedidos óptimos y puede cambiar el ranking entre políticas.
- **Dónde:** `config.py`: `DEMAND_MEAN`, `HOLDING_COST`, `LOST_SALE_COST`, `FIXED_ORDER_COST`, `UNIT_ORDER_COST`; el sorteo está en `InventoryEnv._sample_demand`.

Otras constantes del problema que tampoco tienen justificación escrita **(por confirmar)**: episodio de 100 días (`EPISODE_LEN`), `GAMMA` = 0.99, estado inicial de 40 en bodega y 20 en cada posición del pipeline (`INIT_INVENTORY`, `INIT_PIPELINE`), pedido máximo 50 (`MAX_ORDER`). Un dato aritmético, no una justificación: con gamma 0.99 el "horizonte efectivo" es 1/(1-0.99) = 100 días, que coincide con el largo del episodio.

## 3. El estado incluye el pipeline

- **Qué se eligió:** el agente ve `[inventario, pedido de hace 2 días, pedido de ayer]`, es decir, la bodega más los `L-1` pedidos en camino.
- **Alternativas:** ver solo el inventario; ver el inventario y la suma de lo que viene en camino (la posición de inventario).
- **Por qué:** si solo viera la bodega, dos situaciones muy distintas (con o sin pedidos en camino) se verían igual y el agente no podría decidir bien: el estado dejaría de resumir lo necesario (propiedad de Markov, ver CONCEPTOS.md). Incluir cada pedido por separado deja que la red descubra sola cuánto importa cada uno.
- **Qué se sacrifica:** el estado crece con el lead time (un número más por cada día de retraso). La alternativa de pasar solo la posición de inventario sería más corta, pero pierde el orden de llegada.
- **Dónde:** `InventoryEnv.__init__` (`observation_space`), `InventoryEnv._obs`, `InventoryEnv.raw_state`. Hay un comentario con esta idea en `inventory_env.py`.

## 4. Acciones discretas para DQN y continuas para PPO

- **Qué se eligió:** el entorno tiene dos modos. Discreto: 11 acciones que significan pedir {0, 5, ..., 50}. Continuo: un número en [-1, 1] que se mapea a [0, 50] y se redondea a entero.
- **Alternativas:** usar el mismo tipo de acción para los dos algoritmos; acciones discretas con paso 1 (51 acciones).
- **Por qué:** DQN calcula un valor Q por acción y elige el máximo, así que necesita acciones discretas. PPO produce directamente la acción y admite continuas. El rango simétrico [-1, 1] es lo que recomienda Stable-Baselines3 para PPO.
- **Qué se sacrifica:** DQN solo puede pedir múltiplos de 5, y PPO puede pedir cualquier entero. Los dos algoritmos no tienen exactamente el mismo conjunto de acciones, lo que hace la comparación entre ellos menos pareja. Con paso 1 el DQN tendría más resolución, pero 51 salidas en la red hacen el aprendizaje más difícil **(esperado, no medido)**.
- **Dónde:** `config.py`: `N_ACTIONS`, `MAX_ORDER`; `InventoryEnv.__init__` (`continuous`), `InventoryEnv._to_order`; `train.py`: `continuous = algo == "ppo"`.

## 5. Por qué DQN y PPO, y no otros métodos

- **Qué se eligió:** DQN (valor, acciones discretas) y PPO (política, acciones continuas), dos familias distintas y muy conocidas. Ambos están en la bibliografía del README (Mnih 2015, Schulman 2017).
- **Alternativas y por qué no (explicadas con cautela):**
  - *Q-learning tabular:* guarda una tabla con un valor por cada combinación (estado, acción). Aquí el estado son 3 números enteros que pueden ir de 0 a más de 100, así que la tabla sería enorme y la mayoría de sus casillas casi nunca se visitaría. Una red neuronal generaliza entre estados parecidos; la tabla no.
  - *A2C:* es un método actor-crítico como PPO y más simple. No se incluyó para mantener el alcance corto. En una prueba de humo (copia temporal, 5 ediciones de código) entrena sin problema; no se evaluó su desempeño. Ver GUIA_USO.md, sección 7.
  - *DDPG, TD3, SAC:* métodos para acciones continuas que aprenden de un buffer de experiencias. Son alternativas válidas a PPO. No se incluyeron por alcance. No hay evidencia en el repo de que se hayan comparado.
  - *Programación dinámica exacta:* si se conoce la distribución de la demanda, se puede calcular la política óptima recorriendo todos los estados. Con un estado de pocos números enteros es posible en principio; no se implementó. Tiene una ventaja que RL no tiene: daría el **óptimo verdadero** contra el cual medir. Y una limitación: exige conocer el modelo (la distribución de la demanda); RL aprende solo de simulaciones.
- **Qué se sacrifica:** no sabemos si otro algoritmo lo haría mejor, y no tenemos el óptimo exacto como regla de medir.
- **Dónde:** `train.py` (`train_one`), `config.py` (`DQN_PARAMS`, `PPO_PARAMS`).

## 6. Referencias: ingenua y (s, S), con grid search

- **Qué se eligió:** dos políticas simples para comparar. La ingenua pide siempre 20 (la demanda media). La (s, S) pide hasta llegar a `S` cuando la posición de inventario baja de `s`. Los valores `s` y `S` se buscan con grid search, probando todas las combinaciones de una cuadrícula con 30 episodios de ajuste (semillas 2000-2029).
- **Alternativas:** comparar solo contra la ingenua; usar una metaheurística (algoritmo genético, recocido simulado) para buscar `s` y `S`; no comparar contra nada.
- **Por qué:** sin referencias, "RL aprendió" no significa nada; hay que saber contra qué. (s, S) es una heurística clásica de inventarios y es una referencia fuerte. Grid search es suficiente porque solo hay dos números que buscar; una metaheurística añadiría complejidad sin necesidad (y el proyecto fija que no se usa ninguna).
- **Qué se sacrifica:** la cuadrícula tiene paso 4 (`range(0, 81, 4)` para `s`, `range(20, 141, 4)` para `S`), así que la (s, S) encontrada puede no ser la mejor posible con pasos de 1. Además el rango es fijo: si cambias el problema, la búsqueda puede quedar pegada al borde (ver GUIA_USO.md, sección 5).
- **Dónde:** `policies.py`: `NaivePolicy`, `SSPolicy`, `grid_search_ss`; `config.py`: `SS_S_RANGE`, `SS_BIG_S_RANGE`, `SS_TUNE_SEEDS`; resultado en `results/ss_params.json` (s=76, S=84).

## 7. Escalado de la observación y de la recompensa

- **Qué se eligió:** la observación se divide entre 50 y se recorta a [0, 3] (equivale a recortar valores crudos a [0, 150]). La recompensa es `-costo / 10`. El costo real sin escalar va en `info["cost"]` y es el que se reporta.
- **Alternativas:** no escalar; normalizar con la media y desviación observadas; otros factores.
- **Por qué:** las redes neuronales entrenan mejor con entradas y recompensas de tamaño cercano a 1 que con números de cientos o miles. 50 es el pedido máximo, una escala natural del problema. **(por confirmar)** por qué 10 y no otro factor; no se comparó contra alternativas en el repo.
- **Qué se sacrifica:** valores sobre 150 unidades se recortan y el agente no los distingue. Si cambias la demanda o el pedido máximo, hay que revisar estas constantes.
- **Dónde:** `config.py`: `OBS_SCALE`, `OBS_CLIP`, `REWARD_SCALE`; `InventoryEnv._obs`, `InventoryEnv.step`.

## 8. Red [64, 64] e hiperparámetros

- **Qué se eligió:** red de 2 capas ocultas de 64 neuronas para ambos algoritmos. DQN: 150 000 pasos, tasa de aprendizaje 1e-3, buffer 50 000, etc. PPO: 300 000 pasos, tasa 3e-4, `n_steps` 1000, `ent_coef` 0.01, etc. (lista completa en `config.py`).
- **Alternativas:** redes más grandes o profundas; búsqueda sistemática de hiperparámetros.
- **Por qué:** son **valores de partida razonables, no optimizados**. No hay búsqueda sistemática en el repo, y no se ajustaron mirando los resultados de test. Funcionaron al primer intento: ambos algoritmos aprenden a un costo cercano al de (s, S). Pruebas puntuales hechas al documentar (una semilla, DQN, ver FAQ) sugieren que una red de 128x128 o más o menos pasos no cambian mucho el resultado en este problema.
- **Qué se sacrifica:** no sabemos si hay una configuración notablemente mejor. Con una búsqueda de hiperparámetros los resultados de RL podrían moverse.
- **Dónde:** `config.py`: `NET_ARCH`, `DQN_STEPS`, `PPO_STEPS`, `DQN_PARAMS`, `PPO_PARAMS`.

## 9. Tres semillas de entrenamiento

- **Qué se eligió:** entrenar cada algoritmo con 3 semillas (0, 1, 2) y reportar el promedio y la variación entre semillas.
- **Alternativas:** una sola semilla; 5, 10 o más.
- **Por qué:** el entrenamiento de RL tiene azar (inicialización de la red, exploración). Una sola semilla puede ser una casualidad buena o mala; con tres ya se ve si el resultado es estable. Más semillas cuestan más tiempo (la corrida completa tarda unos 15 minutos con 3).
- **Qué se sacrifica:** la desviación entre semillas calculada con 3 números es poco fiable. En la tabla, PPO tiene 29.5 y DQN 6.7: con 3 semillas no se puede afirmar que PPO sea realmente más inestable que DQN. Cada semilla individual tampoco es un intervalo de confianza.
- **Dónde:** `config.py`: `TRAIN_SEEDS`; `train.py` (`--seeds`); `evaluate.py` (columna "Desv. entre semillas").

## 10. Entrenamiento, validación y test separados

- **Qué se eligió:** tres grupos de semillas que nunca se mezclan: entrenamiento (0, 1, 2 para la inicialización; el entorno de entrenamiento usa el azar propio de cada corrida), validación (500-519, 20 episodios, para elegir el mejor checkpoint) y test (1000-1199, 200 episodios, solo para la tabla final). La (s, S) se ajusta con otras semillas (2000-2029).
- **Alternativas:** evaluar con los mismos episodios con que se entrenó; usar un solo grupo para elegir y para reportar.
- **Por qué:** si eliges el modelo mirando el test, el resultado reportado es optimista. Con validación y test separados, el número del test es una estimación honesta. Durante el entrenamiento, `EvalCallback` evalúa cada 10 000 pasos en validación y guarda el mejor checkpoint; ese es el modelo que queda en `models/` y el que se evalúa en test. Se reporta el mejor checkpoint, no el del último paso. Es parada temprana (*early stopping*), una práctica estándar, y no contamina el test porque la selección usa otras semillas.
- **Qué se sacrifica:** 20 episodios de validación son pocos, así que el "mejor checkpoint" tiene algo de ruido. Y como esos mismos 20 episodios se usan en muchas evaluaciones, el costo de validación es algo optimista; el de test no lo es.
- **Evidencia de que hace falta (medida al documentar):** al leer `model.num_timesteps` de los modelos de `models/` salen 100 000, 140 000 y 120 000 pasos en DQN (de 150 000) y 80 000, 300 000 y 300 000 en PPO (de 300 000). En `curvas_aprendizaje.png` las curvas de validación suben (empeoran) hacia el final. Medición puntual de una revisión independiente, no versionada en `results/` (semilla 0, 200 episodios de test, entrenando sin selección de checkpoint): modelo del último paso PPO 6530.7 y DQN 6582.2, frente a 6220.9 y 6226.3 con el checkpoint elegido y 6254.2 de (s, S). Es una sola semilla, pero el modelo final queda un 4-5 % peor que (s, S). La conclusión honesta: RL llega al nivel de (s, S) **con parada temprana**, no necesariamente en el último paso de entrenamiento.
- **Dónde:** `config.py`: `VAL_SEEDS`, `TEST_SEEDS`, `SS_TUNE_SEEDS`, `EVAL_FREQ`, `N_VAL_EPISODES`; `train.py`: `ValidationEnv`, `EvalCallback`; `evaluate.py`: selección de la mejor semilla con validación (`best_seeds.json`).

## 11. Las mismas demandas para todas las políticas

- **Qué se eligió:** las 4 políticas se evalúan con las mismas 200 semillas de test. Como en este entorno la demanda se sortea sin depender de la acción, cada política enfrenta exactamente la misma secuencia de demandas en cada episodio.
- **Alternativas:** semillas distintas por política.
- **Por qué:** con demanda aleatoria, una política puede parecer mejor solo porque le tocaron días fáciles. Usando los mismos días, las diferencias entre políticas vienen de sus decisiones y no de la suerte. Además permite comparar episodio por episodio (comparación pareada), que es más precisa.
- **Qué se sacrifica:** nada importante; solo que los 200 episodios son una muestra y no cubren todos los casos posibles.
- **Dónde:** `evaluate.py` (`summarize`, `test_seeds`), `policies.py` (`run_episode`).

---

## Limitaciones

- **Un solo producto y una sola bodega.** No se sabe si los resultados se mantienen en problemas más grandes.
- **Tres semillas de entrenamiento.** Alcanza para ver tendencias, no para afirmar diferencias finas entre algoritmos.
- **Un solo problema, con costos fijos.** No se probaron otros valores de costos, demanda o lead time de forma sistemática.
- **Sin ajuste sistemático de hiperparámetros.** Los valores son de partida; los resultados de RL podrían mejorar o empeorar con otros.
- **La referencia (s, S) tiene resolución 4** (paso de la cuadrícula). Medición puntual de una revisión independiente (no versionada en `results/`, la reprodujimos): con paso 1 sale s=78, S=86, el costo de (s, S) baja un poco (6250.8 en test) y la diferencia con RL pasa a -21.6 a -23.5, que sigue siendo distinguible de cero.
- **El intervalo de confianza de la comparación solo cubre la variabilidad de la demanda**, no la del entrenamiento (3 semillas).
- **No hay óptimo exacto** contra el cual comparar.

## Preguntas frecuentes

### ¿RL le gana a (s, S) o empata?

Respuesta corta: **equivalencia práctica**, sin afirmar que RL gane. En la tabla del README el costo promedio es 6254.2 para (s, S), 6229.3 para DQN y 6227.4 para PPO: RL queda unos 25 puntos (~0.4 %) más barato. Es una diferencia pequeña e inconsistente entre semillas, y con 3 semillas de entrenamiento no se puede afirmar que RL supere a (s, S).

Como todas las políticas enfrentan las mismas demandas, se puede mirar la diferencia episodio por episodio. Medición propia al documentar (copia temporal, 200 episodios de test con semillas 1000-1199, diferencia = RL menos (s, S); negativo significa RL más barato; intervalo de 95 % = media ± 1.96 × error estándar). Coincide con la de una revisión independiente:

| Modelo | Diferencia media | Intervalo 95 % | Mejor que (s, S) en |
|---|---|---|---|
| DQN semilla 0 | -27.9 | [-41.1, -14.8] | 60 % de episodios |
| DQN semilla 1 | -31.2 | [-43.5, -19.0] | 62 % |
| DQN semilla 2 | -15.8 | [-30.9, -0.6] | 53 % |
| PPO semilla 0 | -33.4 | [-48.1, -18.6] | 62 % |
| PPO semilla 1 | +12.1 | [+2.8, +21.4] | 44 % |
| PPO semilla 2 | -59.2 | [-72.4, -46.1] | 74 % |
| DQN, promedio de 3 semillas | -25.0 | [-37.0, -12.9] | 55 % |
| PPO, promedio de 3 semillas | -26.8 | [-38.1, -15.6] | 63 % |
| Los 6 modelos juntos | -25.9 | [-36.0, -15.8] | 62 % |

Cómo leerlo:

- En 5 de 6 modelos RL queda más barato y en uno (PPO semilla 1) queda más caro. El signo no es consistente.
- **El intervalo mide la variabilidad de la demanda, no la del entrenamiento.** Dice que ese modelo concreto fue algo más barato en esos 200 episodios; no dice que el algoritmo lo sea en general. La variación entre semillas (6.7 en DQN y 29.5 en PPO, ver GUIA_USO.md) es del mismo orden que la diferencia.
- La referencia (s, S) usa una cuadrícula de paso 4 (s=76, S=84). Con una cuadrícula de paso 1, buscada con las semillas de ajuste 2000-2029, sale s=78, S=86 y su costo de test baja a 6250.8 (medición puntual, la reprodujimos y coincide con la revisión independiente). La diferencia con RL pasa a -21.6 (DQN, intervalo [-33.5, -9.6]) y -23.5 (PPO, intervalo [-34.6, -12.3]): sigue siendo distinguible de cero, pero es un poco menor.
- Los modelos de RL se evalúan con el mejor checkpoint según validación (ver decisión 10). El modelo del último paso rinde peor.

Formulación recomendada: RL iguala a una heurística clásica bien ajustada (diferencia pequeña, inconsistente entre semillas), con parada temprana, sin haber recibido la estructura de la política. Evita decir "RL gana" y también "empata" sin este matiz.

Por qué podría pasar (hipótesis, no verificadas): (s, S) con `s` y `S` bien elegidos ya es una regla muy buena para un solo producto con demanda estable; casi no hay margen. Además esta medición es de un solo problema y 3 semillas.

### ¿Cuándo sí valdría la pena RL?

No lo medimos aquí. Intuiciones razonables, a tomar como hipótesis: cuando el problema tenga muchas piezas que interactúan (varios productos, varias bodegas, restricciones compartidas) y no exista una heurística clásica bien probada. Para una discusión seria, leer Gijsbrechts et al. (2022), listado en el README. Para un problema de un solo producto como este, una heurística bien afinada es una gran referencia y RL tiene que justificar su costo (entrenamiento, ajuste, falta de garantías).

### ¿Por qué no entrenar más tiempo?

Se hizo una prueba puntual (copia temporal, solo DQN, semilla 0, mismos hiperparámetros salvo los pasos). Costo promedio sobre las semillas de validación y de test:

| Pasos de DQN | Validación (20 episodios) | Test (200 episodios) | Tiempo |
|---|---|---|---|
| 30 000 | 6128.9 | 6210.3 | 24 s |
| 150 000 (valor del repo) | 6145.2 | 6226.3 | ~94 s |
| 300 000 | 6128.3 | 6198.4 | 209 s |

Las diferencias (decenas de puntos sobre ~6200) son del orden del ruido de una sola semilla; por esta prueba, más pasos no mejoran claramente. Es una medición puntual con una semilla: no es una conclusión general. Ojo con la lectura de la tabla: son costos del **mejor checkpoint según validación** de cada entrenamiento, no del último paso. En `curvas_aprendizaje.png` (promedio de 3 semillas, costo por episodio en validación) las curvas bajan rápido y luego oscilan: los mejores puntos quedan cerca de la línea de (s, S) y los últimos puntos quedan por encima (DQN termina cerca de 6700 y PPO cerca de 6280, valores leídos de la figura). Entrenar más no ayuda por eso: lo que se gana es sobre todo elegir bien el momento de parar (ver decisión 10). (La fila de 150 000 pasos de la tabla de arriba viene del modelo guardado en `models/`, que es su mejor checkpoint, de 100 000 pasos; las otras dos son entrenamientos nuevos de la prueba, y también se guardó su mejor checkpoint.)

Una prueba similar con una red de 128x128 (150 000 pasos, semilla 0) dio 6131.8 en validación y 6209.2 en test (112 s), también sin diferencia clara.

### ¿Por qué el DQN se ve distinto a (s, S) cuando hay poco inventario en politica_aprendida.png?

Mirando la figura: a la izquierda (posición de inventario bajo ~40) el DQN pide 25-30 unidades mientras (s, S) pide 50, pero del lado derecho las dos curvas coinciden bastante.

Explicación: son estados que casi no ocurren. Se midió en los 200 episodios de test: la posición de inventario mínima fue 45 con (s, S), 40 con el DQN (semilla 0) y 52 con el PPO (semilla 2); en ningún día la posición bajó de 40. Si el agente casi nunca visita esos estados, casi no tuvo datos para aprender qué hacer en ellos y la red extrapola de forma arbitraria. Eso no afecta el costo, porque esos estados no se visitan. Además, la figura es una situación artificial: reparte la posición de inventario en partes iguales entre bodega y pedidos en camino (ver `evaluate.py`), una combinación que rara vez ocurre tal cual.

Lo que sí se aprende bien es la zona donde vive el agente (posición de inventario entre ~50 y ~85), donde las políticas se parecen.
