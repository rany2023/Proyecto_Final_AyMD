# Análisis del Comercio Exterior de México
### Predicción del Balance Comercial y Segmentación de Socios Comerciales

> Proyecto Final — Almacenes y Minería de Datos  
> Facultad de Ciencias, UNAM — Junio 2026

---

## Equipo

| Integrante | Usuario GitHub |
|---|---|
| [Solano Marcial Irany] | [@rany2023](https://github.com/rany2023) |
| [Jiménez Ruíz Brenda] | [@BrendaJimenezRuiz](https://github.com/BrendaJimenezRuiz) |

**Profesora:** Jessica Santizo Galicia  
**Ayudante de Teoría:** Diego Antonio Villalba González  
**Ayudante de Laboratorio:** Ares Gael Castro Romero

---

## Descripción del proyecto

Este proyecto analiza los flujos mensuales de importaciones y exportaciones de México
con sus socios comerciales durante el periodo 1993–2025. A partir de datos oficiales
del Banco de México se desarrollan dos análisis principales:

- **Modelo supervisado:** clasificación binaria para predecir si el balance comercial
  mensual de México con un país dado será superávit o déficit.
- **Agrupamiento:** segmentación de los países socios según su perfil de intercambio
  comercial con México (volumen, tendencia y tipo de flujo predominante).

---

## Dataset

| Atributo | Detalle |
|---|---|
| **Fuente** | [datamx.io — Comercio Exterior de México por País 1993–2025](https://datamx.io/es/dataset/comercio-exterior-de-mexico-por-pais-1993-2025) |
| **Origen** | Banco de México / Secretaría de Economía |
| **Cobertura** | Enero 1993 – 2025 (mensual) |
| **Registros** | 184,392 filas |
| **Variables** | 8 columnas (PERIODO, CONTINENTE, ID_CONTINENTE, PAIS, ID_PAIS, FLUJO, ID_FLUJO, VALOR_USD) |
| **Licencia** | Creative Commons CCZero (dominio público) |

---

## Estructura del repositorio

```
Proyecto_Final_AyMD/
├── data/               # Dataset crudo y procesado (o enlace a fuente)
├── notebooks/          # Notebooks de EDA, modelado y clustering
├── src/                # Código fuente Python (clases y módulos)
├── models/             # Modelos serializados (.pkl / .joblib)
├── reports/            # Reporte PDF y slides de presentación
├── diagrams/           # Diagrama UML de la arquitectura
├── docs/               # Sitio Quarto compilado (GitHub Pages)
├── _quarto.yml         # Configuración de Quarto
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/rany2023/Proyecto_Final_AyMD.git
cd Proyecto_Final_AyMD
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Orden de ejecución

Ejecutar desde la **raíz del repositorio** con el entorno virtual activo:

### Paso 1 — EDA: calidad de datos
```bash
PYTHONPATH=src python src/eda.py
```
Genera: `data/comercio_exterior_clean.csv` + gráficas en `reports/`

### Paso 2 — Descargar tipo de cambio Banxico
```bash
PYTHONPATH=src python src/data_loader.py
```
Genera: `data/comercio_exterior_enriquecido.csv`

### Paso 3 — EDA: visualizaciones
```bash
PYTHONPATH=src python src/eda_visualizaciones.py
```
Genera: histogramas, serie de tiempo, correlaciones en `reports/`

### Paso 4 — Preprocesamiento
```bash
PYTHONPATH=src python src/preprocessor.py
```
Genera: `models/scaler.joblib`, `models/le_*.joblib`

### Paso 5 — Modelo Random Forest
```bash
PYTHONPATH=src python src/model_trainer.py
```
Genera: `models/random_forest.joblib` + gráficas en `reports/`

### Paso 6 — Modelo XGBoost
```bash
PYTHONPATH=src python src/model_trainer_xgboost.py
```
Genera: `models/xgboost.joblib` + gráficas comparativas

### Paso 7 — Clustering K-Means
```bash
PYTHONPATH=src python src/clustering.py
```
Genera: `models/kmeans.joblib` + visualización PCA

### Paso 8 — Diagrama UML
```bash
PYTHONPATH=src python src/generar_uml.py
```
Genera: `diagrams/uml_arquitectura.png`

### Paso 9 — Demo en vivo (presentación)
```bash
PYTHONPATH=src python src/demo_prediccion.py
```
Predice balance comercial para los 3 clusters principales sin re-entrenar

### Alternativa — Pipeline completo
```bash
# Ejecutar todo desde el inicio
PYTHONPATH=src python src/pipeline.py

# Ejecutar desde una etapa específica (ej: desde preprocesamiento)
PYTHONPATH=src python src/pipeline.py --desde 4
```

---

## Ejecutar los notebooks

```bash
jupyter notebook notebooks/
```

Orden sugerido:
1. `01_EDA.ipynb` — Análisis exploratorio de datos
2. `02_modelo_supervisado.ipynb` — Entrenamiento y evaluación XGBoost
3. `02b_modelo_random_forest.ipynb` — Entrenamiento y evaluación Random Forest
4. `03_clustering.ipynb` — Agrupamiento de socios comerciales

---

## Sitio de documentación (Quarto)

🔗 **[Sitio del proyecto](https://rany2023.github.io/Proyecto_Final_AyMD/)**

---

## Presentación (slides)

🎥 **[Enlace de la Presentación](https://canva.link/95q9rrxvpgfs1x8)**

---

## Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![pandas](https://img.shields.io/badge/pandas-2.2-150458)
![Quarto](https://img.shields.io/badge/Quarto-1.4-75AADB)

---

## Licencia académica

Proyecto desarrollado con fines académicos para el curso de Almacenes y Minería de Datos,
Facultad de Ciencias, UNAM. Los datos utilizados son de dominio público.
