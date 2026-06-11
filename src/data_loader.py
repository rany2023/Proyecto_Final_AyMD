"""
data_loader.py
Descarga y fusión del tipo de cambio MXN/USD desde la API de Banxico
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM
"""

import pandas as pd
import numpy as np
from sie_banxico import SIEBanxico
import os
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Token de Banxico ──────────────────────────────────────────────────────────

TOKEN_BANXICO = '753738904ddce34be4e03f24eb9fa0cb3581cb7520b1daaa2c05bcc0e48e7986'


class DataLoader:
    """
    Clase que encapsula la carga y enriquecimiento del dataset
    de Comercio Exterior con el tipo de cambio MXN/USD de Banxico.

    Responsabilidades:
        - Cargar el dataset limpio de comercio exterior
        - Descargar el tipo de cambio mensual MXN/USD (Serie SF43718)
        - Fusionar ambas fuentes por fecha
        - Guardar el dataset enriquecido en disco
    """

    # Serie de Banxico: tipo de cambio MXN/USD interbancario a 17hrs
    SERIE_TC = 'SF43718'

    def __init__(self, ruta_datos: str = 'data/comercio_exterior_clean.csv',
                 ruta_salida: str = 'data/comercio_exterior_enriquecido.csv'):
        self.ruta_datos  = ruta_datos
        self.ruta_salida = ruta_salida
        self.df          = None
        self.df_tc       = None
        self.df_final    = None

    # ──────────────────────────────────────────────────────────────────────────
    # 1. CARGAR DATASET BASE
    # ──────────────────────────────────────────────────────────────────────────
    def cargar_datos(self) -> pd.DataFrame:
        """Se carga el dataset limpio de comercio exterior."""
        self.df = pd.read_csv(self.ruta_datos, parse_dates=['PERIODO'])
        print(f'Dataset base cargado: {self.df.shape[0]:,} filas ✓')
        return self.df

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DESCARGAR TIPO DE CAMBIO DE BANXICO
    # ──────────────────────────────────────────────────────────────────────────
    def descargar_tipo_cambio(self) -> pd.DataFrame:
        """
        Descarga el tipo de cambio MXN/USD mensual desde la API de Banxico.
        Serie SF43718: tipo de cambio interbancario a 17hrs (Fix).
        Cobertura: 1993-2025.
        """
        print('\n── Descargando tipo de cambio MXN/USD desde Banxico...')

        api = SIEBanxico(token=TOKEN_BANXICO, id_series=[self.SERIE_TC])

        # Obtener datos mensuales 1993-2025
        datos = api.get_timeseries_range(
            init_date='1993-01-01',
            end_date='2025-12-31'
        )

        # Convertir a DataFrame
        registros = datos['bmx']['series'][0]['datos']
        df_tc = pd.DataFrame(registros)
        df_tc.columns = ['FECHA', 'TIPO_CAMBIO']
        df_tc['TIPO_CAMBIO'] = pd.to_numeric(df_tc['TIPO_CAMBIO'],
                                              errors='coerce')
        df_tc['FECHA'] = pd.to_datetime(df_tc['FECHA'], dayfirst=True)

        # Agregar a nivel mensual (promedio del mes)
        df_tc['PERIODO'] = df_tc['FECHA'].dt.to_period('M').dt.to_timestamp()
        df_tc = (df_tc.groupby('PERIODO')['TIPO_CAMBIO']
                 .mean()
                 .reset_index()
                 .round(4))

        self.df_tc = df_tc

        print(f'  Registros descargados: {len(df_tc):,}')
        print(f'  Periodo: {df_tc["PERIODO"].min().date()} → '
              f'{df_tc["PERIODO"].max().date()}')
        print(f'  TC mínimo:  ${df_tc["TIPO_CAMBIO"].min():.2f} MXN/USD')
        print(f'  TC máximo:  ${df_tc["TIPO_CAMBIO"].max():.2f} MXN/USD')
        print(f'  TC promedio: ${df_tc["TIPO_CAMBIO"].mean():.2f} MXN/USD')

        return self.df_tc

    # ──────────────────────────────────────────────────────────────────────────
    # 3. FUSIONAR DATASETS
    # ──────────────────────────────────────────────────────────────────────────
    def fusionar(self) -> pd.DataFrame:
        """
        Fusionamos el dataset de comercio exterior con el tipo de cambio
        usando PERIODO como llave de unión (left join).
        """
        print('\n── Fusionando datasets...')

        self.df_final = pd.merge(
            self.df,
            self.df_tc,
            on='PERIODO',
            how='left'
        )

        # Verificar faltantes tras el merge
        faltantes_tc = self.df_final['TIPO_CAMBIO'].isnull().sum()
        if faltantes_tc > 0:
            print(f'  ⚠️ {faltantes_tc} registros sin tipo de cambio '
                  f'— se imputa con la mediana')
            mediana_tc = self.df_final['TIPO_CAMBIO'].median()
            self.df_final['TIPO_CAMBIO'] = (self.df_final['TIPO_CAMBIO']
                                            .fillna(mediana_tc))
        else:
            print('  ✓ Sin valores faltantes tras el merge')

        # Variable derivada: valor comercial en pesos mexicanos
        self.df_final['VALOR_MXN'] = (self.df_final['VALOR_USD'] *
                                      self.df_final['TIPO_CAMBIO'])

        print(f'  Dataset enriquecido: {self.df_final.shape[0]:,} filas × '
              f'{self.df_final.shape[1]} columnas')
        print(f'  Columnas nuevas: TIPO_CAMBIO, VALOR_MXN')

        return self.df_final

    # ──────────────────────────────────────────────────────────────────────────
    # 4. GUARDAR
    # ──────────────────────────────────────────────────────────────────────────
    def guardar(self):
        """Guarda el dataset enriquecido en disco."""
        self.df_final.to_csv(self.ruta_salida, index=False)
        print(f'\n✓ Dataset enriquecido guardado en {self.ruta_salida}')

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de carga y enriquecimiento."""
        self.cargar_datos()
        self.descargar_tipo_cambio()
        self.fusionar()
        self.guardar()

        print('\n' + '=' * 60)
        print('  DataLoader completado ✓')
        print('  Dataset listo: data/comercio_exterior_enriquecido.csv')
        print('  Siguiente: actualizar preprocessor.py para usar')
        print('             el dataset enriquecido')
        print('=' * 60)

        return self.df_final


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    loader = DataLoader(
        ruta_datos='data/comercio_exterior_clean.csv',
        ruta_salida='data/comercio_exterior_enriquecido.csv'
    )
    loader.ejecutar()
