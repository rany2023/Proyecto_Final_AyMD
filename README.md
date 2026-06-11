# Análisis del Comercio Exterior de México
### Predicción del Balance Comercial y Segmentación de Socios Comerciales

> Proyecto Final — Almacenes y Minería de Datos  
> Facultad de Ciencias, UNAM — Junio 2025

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
| **Registros** | ~138,000 filas |
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

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/rany2023/Proyecto_Final_AyMD.git
cd Proyecto_Final_AyMD
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Descargar el dataset

Descargar el archivo CSV desde la fuente oficial y colocarlo en `data/`:

```
https://datamx.io/dataset/c6648e87-aa94-49ec-8439-56dd15e6edea/resource/b158871b-0b9f-4310-9f54-b01b308e3d40/download/data.csv
```

### 4. Ejecutar los notebooks

```bash
jupyter notebook notebooks/
```

Orden sugerido:
1. `01_EDA.ipynb` — Análisis exploratorio de datos
2. `02_modelo_supervisado.ipynb` — Entrenamiento y evaluación del modelo
3. `03_clustering.ipynb` — Agrupamiento de socios comerciales

### 5. Cargar el modelo entrenado

```python
import joblib

modelo = joblib.load("models/modelo_final.joblib")
prediccion = modelo.predict(nuevos_datos)
```

---

## Sitio de documentación (Quarto)

🔗 **[Sitio del proyecto](https://rany2023.github.io/Proyecto_Final_AyMD/)**

---

## Video de exposición

🎥 **[Pendiente — enlace al video en YouTube/Drive]**

---

## Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![pandas](https://img.shields.io/badge/pandas-2.2-150458)
![Quarto](https://img.shields.io/badge/Quarto-1.4-75AADB)

---

## Licencia académica

Proyecto desarrollado con fines académicos para el curso de Almacenes y Minería de Datos,
Facultad de Ciencias, UNAM. Los datos utilizados son de dominio público.
