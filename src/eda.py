"""
eda.py
Análisis Exploratorio de Datos — Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM

Uso:
    python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Semilla aleatoria fija (OBLIGATORIO para reproducibilidad)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Estilo de gráficas
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


class EDA:
    """
    Clase que encapsula el Análisis Exploratorio de Datos
    para el dataset de Comercio Exterior de México.

    Responsabilidades:
        - Cargar y describir el dataset
        - Evaluar calidad de datos (faltantes, duplicados)
        - Calcular estadísticas descriptivas
        - Detectar y visualizar valores atípicos
        - Guardar gráficas y dataset limpio
    """

    URL_DATASET = (
        'https://datamx.io/dataset/c6648e87-aa94-49ec-8439-56dd15e6edea/'
        'resource/b158871b-0b9f-4310-9f54-b01b308e3d40/download/data.csv'
    )

    def __init__(self, ruta_datos: str = 'data/comercio_exterior_raw.csv',
                 ruta_reportes: str = 'reports/'):
        """
        Parámetros
        ----------
        ruta_datos    : ruta al CSV local; si no existe, descarga desde la URL
        ruta_reportes : carpeta donde se guardan las gráficas generadas
        """
        self.ruta_datos    = ruta_datos
        self.ruta_reportes = ruta_reportes
        self.df            = None

        os.makedirs(self.ruta_reportes, exist_ok=True)
        os.makedirs('data', exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. CARGA DE DATOS
    # ──────────────────────────────────────────────────────────────────────────
    def cargar_datos(self) -> pd.DataFrame:
        """Carga el dataset desde disco o lo descarga si no existe."""
        if os.path.exists(self.ruta_datos):
            print(f'Cargando datos desde {self.ruta_datos}...')
            self.df = pd.read_csv(self.ruta_datos)
        else:
            print('Archivo local no encontrado. Descargando desde datamx.io...')
            self.df = pd.read_csv(self.URL_DATASET)
            self.df.to_csv(self.ruta_datos, index=False)
            print(f'Dataset guardado en {self.ruta_datos} ✓')

        # Conversión de tipos y variables derivadas
        self.df['PERIODO'] = pd.to_datetime(self.df['PERIODO'])
        self.df['ANIO']    = self.df['PERIODO'].dt.year
        self.df['MES']     = self.df['PERIODO'].dt.month

        print(f'Dataset cargado: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas ✓')
        return self.df

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DESCRIPCIÓN GENERAL
    # ──────────────────────────────────────────────────────────────────────────
    def descripcion_general(self):
        """Imprime dimensiones, tipos de datos y resumen del dataset."""
        print('\n' + '=' * 60)
        print('  DESCRIPCIÓN GENERAL DEL DATASET')
        print('=' * 60)
        print(f'  Filas:        {self.df.shape[0]:,}')
        print(f'  Columnas:     {self.df.shape[1]}')
        print(f'  Periodo:      {self.df["PERIODO"].min().date()} → '
              f'{self.df["PERIODO"].max().date()}')
        print(f'  Países únicos:{self.df["PAIS"].nunique()}')
        print(f'  Continentes:  {self.df["CONTINENTE"].nunique()}')
        print(f'  Tipos de flujo: {list(self.df["FLUJO"].unique())}')
        print('\nTipos de datos por columna:')
        print(self.df.dtypes.to_string())
        print('\nPrimeros 5 registros:')
        print(self.df.head().to_string())

    # ──────────────────────────────────────────────────────────────────────────
    # 3. CALIDAD DE DATOS
    # ──────────────────────────────────────────────────────────────────────────
    def calidad_datos(self):
        """Analiza valores faltantes y registros duplicados."""
        print('\n' + '=' * 60)
        print('  CALIDAD DE DATOS')
        print('=' * 60)

        # Valores faltantes
        faltantes = pd.DataFrame({
            'Faltantes':  self.df.isnull().sum(),
            'Porcentaje': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        cols_con_faltantes = faltantes[faltantes['Faltantes'] > 0]

        if cols_con_faltantes.empty:
            print('✓ No hay valores faltantes en el dataset.')
        else:
            print('Columnas con valores faltantes:')
            print(cols_con_faltantes.to_string())

        # Duplicados
        n_duplicados = self.df.duplicated().sum()
        print(f'\nRegistros duplicados: {n_duplicados} '
              f'({n_duplicados / len(self.df) * 100:.2f}%)')

        if n_duplicados > 0:
            self.df = self.df.drop_duplicates()
            print(f'✓ Duplicados eliminados. Nuevas dimensiones: {self.df.shape}')

        # Heatmap de faltantes
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False,
                    cmap='viridis', ax=ax)
        ax.set_title('Mapa de valores faltantes')
        ax.set_xlabel('Columnas')
        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'eda_faltantes.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Gráfica guardada en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ESTADÍSTICAS DESCRIPTIVAS
    # ──────────────────────────────────────────────────────────────────────────
    def estadisticas_descriptivas(self):
        """Calcula estadísticas para variables numéricas y categóricas."""
        print('\n' + '=' * 60)
        print('  ESTADÍSTICAS DESCRIPTIVAS')
        print('=' * 60)

        # Variables numéricas
        print('\n── Variables numéricas ──')
        desc = self.df[['VALOR_USD', 'ANIO', 'MES']].describe().T
        desc['CV (%)'] = (desc['std'] / desc['mean'] * 100).round(2)
        print(desc.round(2).to_string())

        # Variables categóricas
        print('\n── Variables categóricas ──')
        for col in ['CONTINENTE', 'FLUJO']:
            print(f'\n  {col}:')
            freq  = self.df[col].value_counts()
            pct   = (freq / len(self.df) * 100).round(2)
            tabla = pd.DataFrame({'Frecuencia': freq, 'Porcentaje (%)': pct})
            print(tabla.to_string())
            print(f'  Moda: {self.df[col].mode()[0]}')

    # ──────────────────────────────────────────────────────────────────────────
    # 5. DETECCIÓN DE OUTLIERS
    # ──────────────────────────────────────────────────────────────────────────
    def detectar_outliers(self):
        """Detecta outliers en VALOR_USD usando regla IQR y boxplots."""
        print('\n' + '=' * 60)
        print('  DETECCIÓN DE VALORES ATÍPICOS — REGLA IQR')
        print('=' * 60)

        Q1  = self.df['VALOR_USD'].quantile(0.25)
        Q3  = self.df['VALOR_USD'].quantile(0.75)
        IQR = Q3 - Q1

        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR

        outliers = self.df[
            (self.df['VALOR_USD'] < lim_inf) |
            (self.df['VALOR_USD'] > lim_sup)
        ]

        print(f'  Q1:          ${Q1:>15,.2f}')
        print(f'  Q3:          ${Q3:>15,.2f}')
        print(f'  IQR:         ${IQR:>15,.2f}')
        print(f'  Límite inf:  ${lim_inf:>15,.2f}')
        print(f'  Límite sup:  ${lim_sup:>15,.2f}')
        print(f'  Outliers:    {len(outliers):,} ({len(outliers)/len(self.df)*100:.2f}%)')

        print('\nTop 10 valores más altos:')
        print(self.df.nlargest(10, 'VALOR_USD')
              [['PERIODO', 'PAIS', 'FLUJO', 'VALOR_USD']].to_string())

        # Boxplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(y=self.df['VALOR_USD'], ax=axes[0], color='#5B9BD5')
        axes[0].set_title('Boxplot — VALOR_USD (escala original)')
        axes[0].set_ylabel('Valor (USD)')

        sns.boxplot(y=np.log1p(self.df['VALOR_USD']), ax=axes[1], color='#70AD47')
        axes[1].set_title('Boxplot — VALOR_USD (escala logarítmica)')
        axes[1].set_ylabel('log(VALOR_USD + 1)')

        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'eda_boxplots.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Boxplots guardados en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # 6. GUARDAR DATASET LIMPIO
    # ──────────────────────────────────────────────────────────────────────────
    def guardar_limpio(self, ruta: str = 'data/comercio_exterior_clean.csv'):
        """Guarda el dataset procesado en disco."""
        self.df.to_csv(ruta, index=False)
        print(f'\n✓ Dataset limpio guardado en {ruta}')
        print(f'  Dimensiones finales: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas')

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — ejecuta todo el EDA
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self):
        """Ejecuta el pipeline completo del EDA."""
        self.cargar_datos()
        self.descripcion_general()
        self.calidad_datos()
        self.estadisticas_descriptivas()
        self.detectar_outliers()
        self.guardar_limpio()
        print('\n' + '=' * 60)
        print('  EDA Día 2 completado ✓')
        print('  Siguiente: ejecutar eda_visualizaciones.py')
        print('=' * 60)


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    eda = EDA(
        ruta_datos='data/comercio_exterior_raw.csv',
        ruta_reportes='reports/'
    )
    eda.ejecutar()
