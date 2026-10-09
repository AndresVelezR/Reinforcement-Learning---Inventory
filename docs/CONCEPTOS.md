# Aprendizaje por refuerzo desde cero (5 minutos)

Esta guía explica, sin suponer conocimientos previos, todo lo que necesitas para entender el código de este repo.

## 1. El problema

Una tienda vende **un solo producto**. Cada día decide cuántas unidades pedirle al proveedor. Dos cosas lo complican:

- **La demanda es aleatoria.** No sabes cuántos clientes vendrán mañana (aquí: en promedio 20 unidades, con variación).
- **Los pedidos tardan en llegar** (*lead time*). Lo que pides hoy llega 3 días después.

```
lunes      martes     miercoles  jueves
  |          |           |          |
pido 20 --- en camino -- en camino -> llegan 20 (ya los puedo vender)
```

Si un día la demanda supera lo que hay en bodega, **esas ventas se pierden** (el cliente se va). Si pides de más, pagas por almacenar. Si pides de menos, pierdes ventas. Y como hay retraso, un quiebre de stock hoy es consecuencia de lo que pediste (o no) hace 3 días.

Este problema se llama **control de inventarios con demanda estocástica y lead time**. En la literatura: *lost-sales inventory problem with lead time*. Es difícil porque hay que decidir sin conocer la demanda y con retraso; para esta variante no se conoce una regla óptima simple.

### Un día del simulador, paso a paso

Números reales de `python demo.py --policy ppo --days 10 --model-seed 2` (el modelo PPO guardado en `models/`, episodio de semilla 1000). Mira el **día 3**:

```
 dia  inventario   pipeline  pedido  demanda  ventas perdidas   costo
   3          31   [21, 17]      18       17                0      60
   4          35   [17, 18]      ...
```

1. **Estado.** Hay 31 unidades en bodega. En camino hay dos pedidos: uno de 21 (el más viejo, llega al final de hoy) y uno de 17.
2. **Acción.** El agente decide pedir 18 unidades.
3. **Demanda.** Hoy llegan clientes que quieren 17 unidades (el simulador la sortea; el agente no la conocía).
4. **Ventas.** Se venden min(31, 17) = 17. No faltó nada: ventas perdidas = 0. Sobran 31 - 17 = 14.
5. **Costo.** 1 x 14 (almacenar) + 9 x 0 (ventas perdidas) + 10 (hay pedido) + 2 x 18 (unidades pedidas) = **60**. La recompensa del paso es -60/10 = -6.
6. **Nuevo estado.** La cola queda [21, 17, 18]. Llegan los 21 más viejos: inventario = 14 + 21 = **35**, y el pipeline pasa a **[17, 18]**. Es exactamente la fila del día 4.

**El pedido tarda 3 días.** Las 18 unidades que se piden hoy (día 3) aparecen en la bodega al inicio del día 6: el día 5 empieza con 31, la demanda es 24, quedan 7; al terminar ese día llegan los 18 y el día 6 empieza con 7 + 18 = **25**. Lo mismo pasó con el pedido de 11 unidades del día 0: se ve en el inventario de 31 del día 3.

**Un día con quiebre de stock (día 8 de la misma corrida):** inventario 25, pipeline [21, 20], pide 18, la demanda es 35. Se venden 25 y se **pierden 10**. Costo = 0 + 9 x 10 + 10 + 2 x 18 = **136**.

## 2. Qué es RL en 5 líneas

1. Hay un **agente** que toma decisiones y un **entorno** donde esas decisiones tienen efecto.
2. En cada paso el agente ve el **estado**, elige una **acción** y recibe una **recompensa** (un número: mayor es mejor).
3. El entorno pasa a un nuevo estado y el ciclo se repite.
4. Nadie le dice al agente cuál es la acción correcta: aprende por **ensayo y error**.
5. Su objetivo es juntar la mayor recompensa total posible.

```
        accion
  +-------------------+
  |                   v
AGENTE            ENTORNO
  ^                   |
  +-------------------+
   estado + recompensa
```

| Término de RL | En nuestro problema |
|---|---|
| Agente | quien decide cuánto pedir |
| Entorno | el simulador de la tienda (`inventory_env.py`) |
| Estado | inventario en bodega + pedidos en camino |
| Acción | cuántas unidades pedir hoy |
| Recompensa | menos el costo del día (almacenar + ventas perdidas + pedir) |
| Política | la regla "dado este estado, pido tanto" |

Por qué el estado incluye los pedidos en camino: si el agente solo viera la bodega, no sabría que ya pidió y volvería a pedir lo mismo. Con el pipeline en el estado, lo que ve basta para decidir (propiedad de Markov).

## 3. Por qué esto es RL y no aprendizaje supervisado

1. **Decisiones secuenciales:** lo que pides hoy cambia el estado de mañana.
2. **Recompensa diferida:** el efecto de un pedido se ve días después.
3. **No hay respuestas etiquetadas:** nadie tiene una tabla con "el pedido correcto" para cada situación. El agente lo descubre simulando muchos días.

## 4. Retorno y gamma

El **retorno** es la suma de las recompensas de todo un **episodio** (aquí, 100 días). Lo que importa no es el costo de hoy, sino el total a largo plazo.

Con el factor de descuento **gamma** (0.99) lo lejano vale un poco menos: la recompensa de dentro de 10 días cuenta 0.99¹⁰ ≈ 0.90 de lo que contaría hoy.

## 5. La ecuación de Bellman en simple

El valor de una decisión es **lo que ganas ahora + el valor de lo mejor que puedas hacer después**:

```
Q(estado, accion) = recompensa_ahora + gamma * max Q(estado_siguiente, otra_accion)
```

**Q** (función Q) dice qué tan buena es una acción en un estado. Ejemplo con números inventados:

- Pedir 20: recompensa ahora = -5; el mejor valor desde el estado siguiente = -60.
  Q = -5 + 0.99 · (-60) = **-64.4**
- Pedir 0: recompensa ahora = -2 (no pagas el pedido), pero mañana habrá quiebre y el mejor valor siguiente = -75.
  Q = -2 + 0.99 · (-75) = **-76.3**

Pedir 20 tiene mejor Q: ahorrar hoy sale más caro después. Si conoces Q, la política es simple: elegir la acción con mayor Q.

## 6. DQN

*Deep Q-Network.* Una red neuronal estima Q para cada acción posible; la política es elegir la acción con mayor Q. Ingredientes:

- **Epsilon-greedy:** con probabilidad epsilon prueba una acción al azar (explorar); si no, la mejor conocida. Epsilon baja con el tiempo.
- **Replay buffer:** guarda experiencias pasadas y entrena con muestras al azar de ellas, para no aprender solo de lo último.
- **Red objetivo:** una copia de la red que se actualiza cada tanto y da el objetivo de Bellman; así el objetivo no se mueve en cada paso.

Necesita **acciones discretas** porque calcula un Q por cada acción y toma el máximo. Por eso en DQN el pedido es uno de {0, 5, 10, ..., 50}.

## 7. PPO

*Proximal Policy Optimization.* Aprende la **política directamente** (el **actor**) con ayuda de un **crítico** que estima qué tan bueno es cada estado.

- **Ventaja:** qué tan mejor (o peor) salió una acción de lo que el crítico esperaba. Se refuerzan las acciones con ventaja positiva.
- **Clipping:** limita cuánto puede cambiar la política en cada actualización, para que un mal paso no la arruine.

Como el actor produce directamente la acción, admite **acciones continuas**. Aquí el pedido va de 0 a 50 y se redondea.

## 8. Políticas de referencia

Para saber si RL aporta algo hay que compararlo con reglas simples (*baselines*):

- **Ingenua:** pedir siempre la demanda media (20). Es el piso de comparación.
- **(s, S):** una **heurística** clásica de inventarios (una regla simple basada en la experiencia). Se calcula la *posición de inventario* (bodega + en camino). Si baja de `s`, se pide lo necesario para llegar a `S`; si no, no se pide.

Los valores de `s` y `S` se buscan con **grid search**: se prueban muchas combinaciones simulando y se queda la mejor. Es una búsqueda exhaustiva, **no una metaheurística**. En este proyecto no se usa ninguna metaheurística.

## 9. Política final

La política final es la **red entrenada** que, dado el estado, decide el pedido. Se entrena con varias semillas; se elige la de mejor resultado en **validación** (episodios distintos a los de entrenamiento) y se compara contra las referencias en **test** (episodios que nunca se usaron para decidir nada). Las 4 políticas enfrentan exactamente las mismas demandas.

## 10. Qué NO es esto

- **No es aprendizaje supervisado.** Nadie le da al agente ejemplos de "en esta situación, el pedido correcto es X". Aprende solo con la recompensa (el costo) que obtiene al simular.
- **No es optimización exacta.** Los métodos exactos (por ejemplo, programación dinámica) calculan la mejor política con garantías, pero necesitan conocer el modelo, como la distribución de la demanda. Aquí no se calcula ningún óptimo: el agente aprende una buena política aproximada, sin garantía de que sea la mejor posible.
- **No garantiza ser mejor que una heurística.** En este proyecto RL queda prácticamente empatado con la regla (s, S) bien afinada: una diferencia de apenas ~0.4 %, que ni siquiera es igual en todas las semillas. Que un método sea más sofisticado no significa que gane.
- **No es un sistema listo para usar en una tienda real.** Es un ejemplo académico: un solo producto, demanda sintética y costos de partida.

## 11. Glosario

- **Agente:** quien toma decisiones.
- **Entorno:** el mundo simulado donde actúa el agente.
- **Estado:** lo que el agente ve para decidir.
- **Acción:** lo que el agente elige (cuánto pedir).
- **Recompensa:** número que dice qué tan bien salió el paso (aquí, menos el costo).
- **Política:** regla que asigna una acción a cada estado.
- **Episodio:** una corrida completa (100 días).
- **Retorno:** suma de recompensas de un episodio.
- **Gamma:** factor de descuento; lo lejano vale un poco menos.
- **Función Q:** valor esperado de tomar una acción en un estado.
- **Bellman:** relación entre el valor de hoy y el valor del estado siguiente.
- **Epsilon-greedy:** explorar al azar con probabilidad epsilon, si no elegir la mejor.
- **Replay buffer:** memoria de experiencias para entrenar con muestras al azar.
- **Red objetivo:** copia lenta de la red que estabiliza el aprendizaje en DQN.
- **Actor-crítico:** el actor elige acciones; el crítico evalúa estados.
- **Ventaja:** qué tan mejor salió una acción respecto de lo esperado.
- **Clipping:** límite al cambio de la política en cada actualización (PPO).
- **Lead time:** días entre pedir y recibir.
- **Demanda estocástica:** demanda aleatoria, desconocida de antemano.
- **Venta perdida:** demanda que no se atiende por falta de inventario; no se recupera.
- **Posición de inventario:** bodega + pedidos en camino.
- **Heurística:** regla simple basada en la experiencia, sin garantía de ser óptima.
- **Metaheurística:** método general de búsqueda (p. ej. algoritmos genéticos). No se usa aquí.
- **Baseline:** método simple de referencia contra el que se compara.
- **Semilla:** número que fija el azar para poder repetir resultados.
- **MDP:** proceso de decisión de Markov; el modelo formal (estados, acciones, transiciones, recompensas) de un problema de RL.

## 12. Para seguir

- [DECISIONES.md](DECISIONES.md): por qué el proyecto está hecho así y no de otra forma (alternativas, límites y preguntas difíciles).
- [GUIA_USO.md](GUIA_USO.md): instalar, reproducir, probar variaciones y extender el código.
