"""
preprocessor.py
Preprocesamiento de datos para el modelo supervisado
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM

Uso:
    python src/preprocessor.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class Preprocessor:
    """
    Clase que encapsula el preprocesamiento de datos para el modelo supervisado.

    Responsabilidades:
        - Crear la variable objetivo (superávit=1 / déficit=0)
        - Excluir columnas no relevantes
        - Codificar variables categóricas
        - Normalizar variables numéricas
        - Dividir en conjuntos de entrenamiento y prueba (80/20)
        - Guardar encoders y scaler para reutilización
    """

    def __init__(self, ruta_datos: str = 'data/comercio_exterior_enriquecido.csv',
                 ruta_modelos: str = 'models/'):
        self.ruta_datos   = ruta_datos
        self.ruta_modelos = ruta_modelos
        self.df           = None
        self.df_modelo    = None
        self.scaler       = StandardScaler()
        self.le_continente = LabelEncoder()
        self.le_pais       = LabelEncoder()
        self.le_flujo      = LabelEncoder()

        os.makedirs(self.ruta_modelos, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. CARGA
    # ──────────────────────────────────────────────────────────────────────────
    def cargar_datos(self) -> pd.DataFrame:
        """Carga el dataset limpio."""
        self.df = pd.read_csv(self.ruta_datos, parse_dates=['PERIODO'])
        print(f'Dataset cargado: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas ✓')
        return self.df

    # ──────────────────────────────────────────────────────────────────────────
    # 2. VARIABLE OBJETIVO
    # ──────────────────────────────────────────────────────────────────────────
    def crear_variable_objetivo(self) -> pd.DataFrame:
        """
        Crea la variable objetivo SUPERAVIT a nivel país-mes:
            1 = superávit  (exportaciones > importaciones)
            0 = déficit    (exportaciones <= importaciones)
        """
        print('\n── Creando variable objetivo...')

        # Pivot: una fila por país-mes con exportaciones e importaciones
        pivot = self.df.pivot_table(
            index=['PERIODO', 'PAIS', 'CONTINENTE', 'ID_PAIS', 'ANIO', 'MES'],
            columns='FLUJO',
            values='VALOR_USD',
            aggfunc='sum'
        ).reset_index()

        # Limpiar nombres de columnas
        pivot.columns.name = None

        # Manejar columnas faltantes si algún país solo tiene un flujo
        for col in ['Exportaciones', 'Importaciones']:
            if col not in pivot.columns:
                pivot[col] = 0

        pivot = pivot.fillna(0)

        # Variable objetivo
        pivot['BALANCE']   = pivot['Exportaciones'] - pivot['Importaciones']
        pivot['SUPERAVIT'] = (pivot['BALANCE'] >= 0).astype(int)

        self.df_modelo = pivot

        superavit = pivot['SUPERAVIT'].sum()
        deficit   = len(pivot) - superavit
        print(f'  Registros país-mes generados: {len(pivot):,}')
        print(f'  Superávit (1): {superavit:,} ({superavit/len(pivot)*100:.1f}%)')
        print(f'  Déficit   (0): {deficit:,} ({deficit/len(pivot)*100:.1f}%)')

        return self.df_modelo

    # ──────────────────────────────────────────────────────────────────────────
    # 3. FEATURES Y ENCODING
    # ──────────────────────────────────────────────────────────────────────────
    def preparar_features(self) -> tuple:
        """
        Selecciona y transforma las variables de entrada:
        - Excluye: PERIODO, BALANCE (causa data leakage), ID_CONTINENTE
        - Codifica: CONTINENTE, PAIS
        - Normaliza: Exportaciones, Importaciones, ID_PAIS, ANIO, MES
        """
        print('\n── Preparando features...')

        df = self.df_modelo.copy()


        # Agregamos TIPO_CAMBIO al df_modelo desde el dataset original
        tc = self.df.groupby('PERIODO')['TIPO_CAMBIO'].mean().reset_index()
        df = pd.merge(df, tc, on='PERIODO', how='left')

        # Codificación de variables categóricas
        df['CONTINENTE_ENC'] = self.le_continente.fit_transform(df['CONTINENTE'])
        df['PAIS_ENC']       = self.le_pais.fit_transform(df['PAIS'])

        # Features finales
        features = [
            'Exportaciones',
            'Importaciones',
            'TIPO_CAMBIO',
            'CONTINENTE_ENC',
            'PAIS_ENC',
            'ANIO',
            'MES'
        ]

        X = df[features].copy()
        y = df['SUPERAVIT'].copy()

        # Normalización
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        print(f'  Features seleccionadas: {features}')
        print(f'  Shape de X: {X_scaled.shape}')
        print(f'  Shape de y: {y.shape}')

        return X_scaled, y

    # ──────────────────────────────────────────────────────────────────────────
    # 4. DIVISIÓN 80/20
    # ──────────────────────────────────────────────────────────────────────────
    def dividir_datos(self, X: pd.DataFrame,
                      y: pd.Series) -> tuple:
        """
        Divide los datos en entrenamiento (80%) y prueba (20%).
        Se usa random_state fija para reproducibilidad.
        """
        print('\n── Dividiendo datos 80/20...')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y  # mantiene proporción superávit/déficit en ambos sets
        )

        print(f'  Entrenamiento: {X_train.shape[0]:,} registros')
        print(f'  Prueba:        {X_test.shape[0]:,} registros')
        print(f'  Proporción superávit en entrenamiento: '
              f'{y_train.mean()*100:.1f}%')
        print(f'  Proporción superávit en prueba:        '
              f'{y_test.mean()*100:.1f}%')

        return X_train, X_test, y_train, y_test

    # ──────────────────────────────────────────────────────────────────────────
    # 5. GUARDAR ENCODERS Y SCALER
    # ──────────────────────────────────────────────────────────────────────────
    def guardar_transformadores(self):
        """Guarda scaler y encoders para reutilización en predicciones nuevas."""
        joblib.dump(self.scaler,
                    os.path.join(self.ruta_modelos, 'scaler.joblib'))
        joblib.dump(self.le_continente,
                    os.path.join(self.ruta_modelos, 'le_continente.joblib'))
        joblib.dump(self.le_pais,
                    os.path.join(self.ruta_modelos, 'le_pais.joblib'))
        print(f'\n✓ Transformadores guardados en {self.ruta_modelos}')

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self) -> tuple:
        """Ejecuta el pipeline completo de preprocesamiento."""
        self.cargar_datos()
        self.crear_variable_objetivo()
        X, y = self.preparar_features()
        X_train, X_test, y_train, y_test = self.dividir_datos(X, y)
        self.guardar_transformadores()

        print('\n' + '=' * 60)
        print('  Preprocesamiento completado ✓')
        print('  Siguiente: ejecutar model_trainer.py')
        print('=' * 60)

        return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    prep = Preprocessor(
        ruta_datos='data/comercio_exterior_enriquecido.csv',
        ruta_modelos='models/'
    )
    X_train, X_test, y_train, y_test = prep.ejecutar()
