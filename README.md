# 🛒 Predicción de Ventas Minoristas con Machine Learning

> Modelo predictivo de regresión para pronosticar las ventas diarias de una tienda minorista a partir de variables de calendario y promociones, con el fin de optimizar la gestión de inventario, las promociones y la programación de personal.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Computation-013243?logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Viz-11557C">
  <img alt="seaborn" src="https://img.shields.io/badge/seaborn-Viz-4C72B0">
</p>

> 🇬🇧 *English version available at [README_EN.md](README_EN.md)*

---

## 📑 Tabla de Contenidos

1. [Resumen Ejecutivo](#-resumen-ejecutivo)
2. [Planteamiento del Problema](#-planteamiento-del-problema)
3. [Conjunto de Datos](#-conjunto-de-datos)
4. [Metodología](#-metodología)
5. [Análisis Exploratorio de Datos (EDA)](#-análisis-exploratorio-de-datos-eda)
6. [Modelado y Resultados](#-modelado-y-resultados)
7. [Conclusiones](#-conclusiones)
8. [Recomendaciones para el Negocio](#-recomendaciones-para-el-negocio)
9. [Limitaciones](#-limitaciones)
10. [Tecnologías Utilizadas](#-tecnologías-utilizadas)
11. [Cómo Reproducir el Proyecto](#-cómo-reproducir-el-proyecto)
12. [Créditos](#-créditos)

---

## 🎯 Resumen Ejecutivo

Una tienda minorista recopiló durante un año (365 días) datos de sus ventas diarias y desea **pronosticar las ventas del próximo mes** para tomar decisiones informadas sobre inventario, promociones y personal.

Se construyó y comparó un conjunto de tres modelos de regresión —**Regresión Lineal, Árbol de Decisión y Bosque Aleatorio**— para predecir el total de ventas a partir del día de la semana, las promociones y los días festivos.

**Resultado clave:** los tres modelos alcanzaron un rendimiento prácticamente idéntico (**R² ≈ 0.992**), lo que demuestra que la relación entre las variables y las ventas es esencialmente **lineal y simple**. Se seleccionó la **Regresión Lineal** como modelo final por ofrecer el mismo rendimiento con la mayor interpretabilidad y el menor costo computacional. El error promedio del modelo final es de apenas **≈ 67 unidades de venta (MAE)** sobre un ticket promedio de ~3,000.

---

## ❓ Planteamiento del Problema

> **¿Puede la tienda anticipar sus ventas diarias del próximo mes utilizando los datos históricos del año anterior?**

La tienda necesita un modelo que prediga el total de ventas para apoyar tres decisiones operativas críticas:

- 📦 **Gestión de inventario** — cuánto producto abastecer y cuándo.
- 🏷️ **Promociones** — evaluar si la estrategia de promociones actual genera valor.
- 👥 **Programación de personal** — asignar más empleados en los días de mayor demanda.

---

## 📊 Conjunto de Datos

**Archivo:** `Ventas.csv` — 365 registros (todo el año 2022), sin valores faltantes.

| Variable | Tipo | Descripción |
|---|---|---|
| `Fecha` | fecha | Fecha del registro (diario, 2022-01-01 a 2022-12-31) |
| `DíaDeLaSemana` | entero (1–7) | Día de la semana (1 = Lunes … 7 = Domingo) |
| `Promociones` | binaria (0/1) | Indica si ese día hubo promoción |
| `Festivo` | binaria (0/1) | Indica si ese día fue festivo |
| `Ventas` | entero | **Variable objetivo** — total de ventas del día |

**Estadísticas descriptivas de la variable objetivo (`Ventas`):**

| Métrica | Valor |
|---|---|
| Conteo | 365 |
| Media | 2,997.22 |
| Desviación estándar | 942.10 |
| Mínimo | 1,305 |
| Mediana (50%) | 3,074 |
| Máximo | 4,404 |

- Días con **promoción:** 73 (20% del año).
- Días **festivos:** 52 (14.2% del año).

---

## 🔬 Metodología

El proyecto siguió un flujo de trabajo estándar de ciencia de datos:

```
1. Preparación de Datos  →  2. EDA  →  3. Selección de Modelo  →  4. Entrenamiento y Evaluación  →  5. Conclusiones
```

1. **Preparación de datos** — carga del CSV, verificación de nulos (0 encontrados) y conversión de la columna `Fecha` de tipo texto a `datetime`.
2. **Análisis Exploratorio de Datos (EDA)** — distribución de ventas, ventas a lo largo del tiempo, ventas por día de la semana, festivo vs. no festivo, impacto de promociones, matriz de correlación y análisis de interacciones.
3. **Selección de modelo** — comparación de tres algoritmos de regresión mediante R² sobre un conjunto de prueba.
4. **Entrenamiento y evaluación** — partición *train/test* del **80/20** (`random_state=42`) y visualización de ventas reales vs. predichas.
5. **Conclusiones y recomendaciones de negocio.**

**Variables predictoras (X):** `DíaDeLaSemana`, `Promociones`, `Festivo`
**Variable objetivo (y):** `Ventas`

---

## 📈 Análisis Exploratorio de Datos (EDA)

### Ventas promedio por día de la semana

Las ventas siguen un patrón **creciente y muy consistente** a lo largo de la semana:

| Día | Lun | Mar | Mié | Jue | Vie | Sáb | Dom |
|---|---|---|---|---|---|---|---|
| **Ventas promedio** | 1,502 | 2,106 | 2,525 | 3,070 | 3,481 | 4,072 | 4,205 |

### Festivo vs. No Festivo y Promociones

| Segmento | Ventas promedio |
|---|---|
| Día **no festivo** | 2,796.5 |
| Día **festivo** | 4,205.2 |
| **Sin** promoción | 2,973.8 |
| **Con** promoción | 3,090.7 |

### Correlación con las Ventas

| Variable | Correlación con `Ventas` |
|---|---|
| `DíaDeLaSemana` | **0.987** 🟢 |
| `Festivo` | 0.523 🟡 |
| `Promociones` | 0.050 🔴 |

**Hallazgo del EDA:** el **día de la semana** domina por completo la variación de las ventas. Un detalle importante de estructura en los datos es que **todos los días festivos caen en domingo** (el día de mayor venta) y las **promociones solo ocurren en martes, jueves y sábado** — lo que explica por qué el efecto aparente de "festivo" se confunde con el del fin de semana.

---

## 🤖 Modelado y Resultados

Se entrenaron y evaluaron tres modelos sobre el mismo conjunto de prueba (20%):

| Modelo | R² (test) | MAE | RMSE |
|---|---|---|---|
| **Regresión Lineal** ✅ | **0.99242** | **67.43** | 86.24 |
| Árbol de Decisión | 0.99235 | 65.94 | 86.66 |
| Bosque Aleatorio | 0.99226 | 66.19 | 87.17 |

> Los tres modelos obtuvieron un rendimiento **casi idéntico (~99.2% R²)**. Esto confirma que la relación entre las variables y las ventas es fundamentalmente **lineal**, por lo que un modelo complejo no aporta ninguna ventaja.

### Modelo final seleccionado: Regresión Lineal

Se eligió la **Regresión Lineal** por igualar el rendimiento de los modelos más complejos siendo la opción más **interpretable, ligera y fácil de implementar**. La ecuación aprendida es:

```
Ventas ≈ 1,028.67 + 492.95 · DíaDeLaSemana + 181.07 · Promociones − 278.27 · Festivo
```

- Cada día que avanza la semana suma **≈ 493 ventas** — confirmando que el día de la semana es el motor principal.
- Una vez controlado el día de la semana, las **promociones aportan ≈ +181** y la variable festivo deja de ser positiva, evidenciando que su efecto bruto provenía de coincidir con el domingo.

La gráfica de **Ventas Reales vs. Ventas Predichas** muestra los puntos alineados muy cerca de la diagonal ideal, confirmando visualmente la alta precisión del modelo.

---

## ✅ Conclusiones

1. **El día de la semana es el predictor más fuerte de las ventas.** Las ventas son más bajas al inicio de la semana (lunes ~1,500) y aumentan progresivamente hacia el fin de semana (sábado/domingo ~4,100+). Este único factor explica la mayor parte de la variación.

2. **Las promociones tienen un impacto bajo.** Aunque los días con promoción muestran ventas ligeramente diferentes, el efecto es marginal comparado con el día de la semana. Solo el 20% de los días tuvieron promociones, y la venta durante esas promociones fue prácticamente igual.

3. **Los días festivos alteran las ventas, pero de forma confundida.** En crudo, los festivos venden más, pero esto se debe a que **todos caen en domingo** (el día de mayor venta), no a un efecto propio del festivo.

4. **La estructura de los datos confirma el análisis.** En los días festivos no hay promociones; las promociones solo se hacen en martes, jueves y sábado; y los festivos solo caen en domingo.

5. **Los tres modelos rinden casi idéntico (~99.2% R²).** La relación es esencialmente lineal y simple: no se necesitan modelos complejos.

6. **La Regresión Lineal es el modelo más adecuado** porque ofrece el mismo rendimiento que los modelos más complejos, pero es más fácil de interpretar e implementar.

---

## 💡 Recomendaciones para el Negocio

| Área | Recomendación |
|---|---|
| 👥 **Personal** | Programar más empleados los **viernes, sábados y domingos**, los días de mayor venta, o ampliar el horario de atención en esos días. |
| 📦 **Inventario** | Abastecer más producto hacia el **fin de semana**. Evitar el sobre-stock los lunes y martes para reducir costos de almacenamiento. |
| 🏷️ **Promociones** | **Reevaluar la estrategia actual**, ya que no genera impacto medible. Considerar reorientarla a los días de baja venta (lunes–miércoles) para estimularlos, o diseñar promociones especiales de fin de semana. |
| 🔮 **Análisis predictivo** | Usar el modelo de Regresión Lineal para **pronosticar las ventas del próximo mes** y planificar inventario y personal con anticipación. |

---

## ⚠️ Limitaciones

- **Solo 3 variables predictoras.** En la realidad, factores como el clima, la competencia, la ubicación y la estacionalidad anual también influyen en las ventas.
- **Es un conjunto de datos de práctica.** Un R² de 0.99 es inusualmente alto; en escenarios reales los modelos suelen tener menor precisión. Sería indispensable **validar el modelo con datos nuevos** antes de tomar decisiones críticas.
- **Variables confundidas por diseño** (festivo = domingo, promociones solo ciertos días), lo que limita la capacidad de aislar el efecto causal de cada una.

---

## 🛠️ Tecnologías Utilizadas

| Herramienta | Uso |
|---|---|
| **Python 3** | Lenguaje base |
| **pandas** | Carga y manipulación de datos |
| **NumPy** | Cálculo numérico |
| **Matplotlib / seaborn** | Visualización de datos |
| **scikit-learn** | Modelado (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`), partición *train/test* y métricas |
| **Jupyter Notebook** | Entorno de desarrollo interactivo |

---

## ▶️ Cómo Reproducir el Proyecto

```bash
# 1. Instalar dependencias
pip install jupyter pandas numpy matplotlib seaborn scikit-learn

# 2. Abrir el notebook
jupyter notebook "07 Proyecto del dia 11.ipynb"

# 3. Ejecutar todas las celdas (Cell → Run All)
```

**Estructura del proyecto:**

```
Dia 11/
├── 07 Proyecto del dia 11.ipynb   # Notebook con el análisis completo
├── Ventas.csv                      # Conjunto de datos (365 registros)
└── README.md                       # Este documento
```

---

## 🎓 Créditos

Este ejercicio de Machine Learning fue desarrollado a partir de un ejercicio de un curso de **Machine Learning de Udemy impartido por Federico Garay**.

---

<p align="center">
  <em>Proyecto educativo · Día 11 — Python para Data Science y Machine Learning</em>
</p>
