"""
eda_visualizaciones.py
EDA Parte 2 — Histogramas, serie de tiempo y análisis categórico
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM

Uso:
    python src/eda_visualizaciones.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize']  = 13
plt.rcParams['axes.labelsize']  = 11


class EDAVisualizaciones:
    """
    Clase que encapsula las visualizaciones del EDA para el dataset
    de Comercio Exterior de México.

    Responsabilidades (Rany):
        - Histogramas de variables numéricas
        - Serie de tiempo exportaciones vs importaciones 1993-2025

    Responsabilidades (Brenda):
        - Matriz de correlaciones + heatmap
        - Gráficas de barras por continente y tipo de flujo
    """

    def __init__(self, ruta_datos: str = 'data/comercio_exterior_clean.csv',
                 ruta_reportes: str = 'reports/'):
        self.ruta_datos    = ruta_datos
        self.ruta_reportes = ruta_reportes
        self.df            = None
        os.makedirs(self.ruta_reportes, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # CARGA
    # ──────────────────────────────────────────────────────────────────────────
    def cargar_datos(self) -> pd.DataFrame:
        """Carga el dataset limpio generado por eda.py"""
        self.df = pd.read_csv(self.ruta_datos, parse_dates=['PERIODO'])
        print(f'Dataset cargado: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas ✓')
        return self.df

    # ──────────────────────────────────────────────────────────────────────────
    # RANY — Histogramas
    # ──────────────────────────────────────────────────────────────────────────
    def histogramas(self):
        """
        Histogramas de VALOR_USD en escala original y logarítmica,
        y distribución de registros por año y por mes.
        """
        print('\n── Generando histogramas...')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribución de variables numéricas — Comercio Exterior México',
                     fontsize=14, fontweight='bold', y=1.01)

        # 1. VALOR_USD escala original
        axes[0, 0].hist(self.df['VALOR_USD'], bins=60, color='#5B9BD5',
                        edgecolor='white', linewidth=0.5)
        axes[0, 0].set_title('VALOR_USD — escala original')
        axes[0, 0].set_xlabel('Valor (USD)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${x/1e9:.0f}B'))

        # 2. VALOR_USD escala logarítmica
        log_vals = np.log1p(self.df['VALOR_USD'])
        axes[0, 1].hist(log_vals, bins=60, color='#70AD47',
                        edgecolor='white', linewidth=0.5)
        axes[0, 1].set_title('VALOR_USD — escala logarítmica (log1p)')
        axes[0, 1].set_xlabel('log(VALOR_USD + 1)')
        axes[0, 1].set_ylabel('Frecuencia')

        # 3. Registros por año
        conteo_anio = self.df['ANIO'].value_counts().sort_index()
        axes[1, 0].bar(conteo_anio.index, conteo_anio.values,
                       color='#ED7D31', edgecolor='white', linewidth=0.5)
        axes[1, 0].set_title('Número de registros por año')
        axes[1, 0].set_xlabel('Año')
        axes[1, 0].set_ylabel('Registros')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Registros por mes
        meses = ['Ene','Feb','Mar','Abr','May','Jun',
                 'Jul','Ago','Sep','Oct','Nov','Dic']
        conteo_mes = self.df['MES'].value_counts().sort_index()
        axes[1, 1].bar(meses, conteo_mes.values,
                       color='#A77DC2', edgecolor='white', linewidth=0.5)
        axes[1, 1].set_title('Número de registros por mes')
        axes[1, 1].set_xlabel('Mes')
        axes[1, 1].set_ylabel('Registros')

        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'eda_histogramas.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Histogramas guardados en {ruta}')

        # Estadísticas para interpretar
        print(f'\n  VALOR_USD — estadísticas clave:')
        print(f'  Media:    ${self.df["VALOR_USD"].mean():>15,.0f}')
        print(f'  Mediana:  ${self.df["VALOR_USD"].median():>15,.0f}')
        print(f'  Máximo:   ${self.df["VALOR_USD"].max():>15,.0f}')
        print(f'  Registros con VALOR_USD = 0: '
              f'{(self.df["VALOR_USD"] == 0).sum():,} '
              f'({(self.df["VALOR_USD"] == 0).mean()*100:.1f}%)')

    # ──────────────────────────────────────────────────────────────────────────
    # RANY — Serie de tiempo
    # ──────────────────────────────────────────────────────────────────────────
    def serie_tiempo(self):
        """
        Serie de tiempo mensual de exportaciones vs importaciones
        y balance comercial 1993-2025.
        """
        print('\n── Generando serie de tiempo...')

        # Agregar por mes y tipo de flujo
        serie = (self.df.groupby(['PERIODO', 'FLUJO'])['VALOR_USD']
                 .sum()
                 .reset_index())

        exportaciones = serie[serie['FLUJO'] == 'Exportaciones'].set_index('PERIODO')
        importaciones = serie[serie['FLUJO'] == 'Importaciones'].set_index('PERIODO')

        # Balance comercial mensual
        balance = (exportaciones['VALOR_USD']
                   .subtract(importaciones['VALOR_USD'], fill_value=0)
                   .reset_index())
        balance.columns = ['PERIODO', 'BALANCE']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Comercio Exterior de México 1993–2025',
                     fontsize=14, fontweight='bold')

        # Gráfica 1: Exportaciones vs Importaciones
        axes[0].plot(exportaciones.index, exportaciones['VALOR_USD'] / 1e9,
                     label='Exportaciones', color='#70AD47', linewidth=1.2)
        axes[0].plot(importaciones.index, importaciones['VALOR_USD'] / 1e9,
                     label='Importaciones', color='#ED7D31', linewidth=1.2,
                     linestyle='--')
        axes[0].set_title('Exportaciones vs Importaciones mensuales')
        axes[0].set_ylabel('Miles de millones USD')
        axes[0].legend()

        # Marcar eventos históricos clave
        eventos = {
            '1994-01': 'TLCAN',
            '2009-01': 'Crisis 2008',
            '2020-04': 'COVID-19',
            '2020-07': 'T-MEC'
        }
        for fecha, etiqueta in eventos.items():
            axes[0].axvline(pd.to_datetime(fecha), color='gray',
                            linestyle=':', linewidth=1, alpha=0.7)
            axes[0].text(pd.to_datetime(fecha), axes[0].get_ylim()[1] * 0.85,
                         etiqueta, fontsize=8, color='gray', rotation=90,
                         va='top', ha='right')

        # Gráfica 2: Balance comercial
        colores = ['#70AD47' if b >= 0 else '#ED7D31'
                   for b in balance['BALANCE']]
        axes[1].bar(balance['PERIODO'], balance['BALANCE'] / 1e9,
                    color=colores, width=20, alpha=0.8)
        axes[1].axhline(0, color='black', linewidth=0.8)
        axes[1].set_title('Balance comercial mensual (verde = superávit, naranja = déficit)')
        axes[1].set_ylabel('Miles de millones USD')
        axes[1].set_xlabel('Año')

        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'eda_serie_tiempo.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Serie de tiempo guardada en {ruta}')

        # Estadísticas del balance
        superavit = (balance['BALANCE'] >= 0).sum()
        deficit   = (balance['BALANCE'] <  0).sum()
        print(f'\n  Balance comercial:')
        print(f'  Meses con superávit: {superavit} ({superavit/len(balance)*100:.1f}%)')
        print(f'  Meses con déficit:   {deficit} ({deficit/len(balance)*100:.1f}%)')
        print(f'  Balance promedio:    ${balance["BALANCE"].mean():,.0f} USD')

    # ──────────────────────────────────────────────────────────────────────────
    # BRENDA — Matriz de correlaciones (placeholder)
    # ──────────────────────────────────────────────────────────────────────────
    def correlaciones(self):
        """Matriz de correlaciones — implementada por Brenda."""
        print('\n── [Brenda] Matriz de correlaciones — pendiente')
        # TODO: Brenda implementa aquí

    # ──────────────────────────────────────────────────────────────────────────
    # BRENDA — Gráficas categóricas (placeholder)
    # ──────────────────────────────────────────────────────────────────────────
    def graficas_categoricas(self):
        """Gráficas de barras categóricas — implementadas por Brenda."""
        print('\n── [Brenda] Gráficas categóricas — pendiente')
        # TODO: Brenda implementa aquí

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self):
        """Ejecuta todas las visualizaciones del EDA."""
        self.cargar_datos()
        self.histogramas()
        self.serie_tiempo()
        self.correlaciones()
        self.graficas_categoricas()
        print('\n' + '=' * 60)
        print('  EDA Día 3 completado ✓')
        print('  Gráficas guardadas en reports/')
        print('=' * 60)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    eda_viz = EDAVisualizaciones(
        ruta_datos='data/comercio_exterior_clean.csv',
        ruta_reportes='reports/'
    )
    eda_viz.ejecutar()
