"""
pipeline.py
Patrón de diseño Pipeline — integra todo el flujo del proyecto
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM

Patrón implementado: Pipeline
Justificación: el procesamiento de datos ocurre en etapas secuenciales
bien definidas y dependientes entre sí:
    1. Carga y enriquecimiento  (DataLoader)
    2. EDA                      (EDA)
    3. Visualizaciones          (EDAVisualizaciones)
    4. Preprocesamiento         (Preprocessor)
    5. Modelo supervisado       (ModelTrainer)
    6. Clustering               (Clustering)
"""

import argparse
import time
import os
import sys

# Agregar src/ al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader          import DataLoader
from eda                  import EDA
from eda_visualizaciones  import EDAVisualizaciones
from preprocessor         import Preprocessor
from model_trainer        import ModelTrainer
from clustering           import Clustering

RANDOM_STATE = 42


class Stage:
    """
    Representa una etapa individual del pipeline.
    Encapsula nombre, descripción y función ejecutable.
    """

    def __init__(self, numero: int, nombre: str,
                 descripcion: str, funcion):
        self.numero      = numero
        self.nombre      = nombre
        self.descripcion = descripcion
        self.funcion     = funcion
        self.completada  = False
        self.duracion    = 0.0

    def ejecutar(self, contexto: dict) -> dict:
        """Ejecuta la etapa y actualiza el contexto compartido."""
        print(f'\n{"="*60}')
        print(f'  ETAPA {self.numero}: {self.nombre}')
        print(f'  {self.descripcion}')
        print(f'{"="*60}')

        inicio = time.time()
        contexto = self.funcion(contexto)
        self.duracion   = time.time() - inicio
        self.completada = True

        print(f'\n  ✓ Etapa {self.numero} completada en '
              f'{self.duracion:.1f}s')
        return contexto


class Pipeline:
    """
    Clase principal que implementa el patrón de diseño Pipeline.

    Orquesta la ejecución secuencial de todas las etapas del
    proyecto de minería de datos, pasando el contexto (datos
    intermedios) de una etapa a la siguiente.

    Patrón Pipeline:
        - Cada Stage es independiente y tiene una responsabilidad clara
        - El contexto es el objeto compartido entre etapas
        - Las etapas pueden ejecutarse desde cualquier punto
        - Facilita la depuración y el mantenimiento
    """

    def __init__(self, ruta_raw:         str = 'data/comercio_exterior_raw.csv',
                 ruta_clean:       str = 'data/comercio_exterior_clean.csv',
                 ruta_enriquecido: str = 'data/comercio_exterior_enriquecido.csv',
                 ruta_modelos:     str = 'models/',
                 ruta_reportes:    str = 'reports/'):

        self.ruta_raw         = ruta_raw
        self.ruta_clean       = ruta_clean
        self.ruta_enriquecido = ruta_enriquecido
        self.ruta_modelos     = ruta_modelos
        self.ruta_reportes    = ruta_reportes

        # Contexto compartido entre etapas
        self.contexto = {
            'X_train': None, 'X_test':  None,
            'y_train': None, 'y_test':  None,
            'modelo_rf': None, 'modelo_clustering': None
        }

        # Definir etapas del pipeline
        self.stages = self._definir_stages()

    # ──────────────────────────────────────────────────────────────────────────
    # DEFINICIÓN DE ETAPAS
    # ──────────────────────────────────────────────────────────────────────────
    def _definir_stages(self) -> list:
        """Define las etapas del pipeline en orden secuencial."""
        return [
            Stage(1, 'Carga y Enriquecimiento',
                  'Descarga tipo de cambio Banxico y fusiona con dataset',
                  self._etapa_carga),

            Stage(2, 'EDA — Calidad de Datos',
                  'Analiza faltantes, duplicados y estadísticas descriptivas',
                  self._etapa_eda),

            Stage(3, 'EDA — Visualizaciones',
                  'Genera histogramas, serie de tiempo y correlaciones',
                  self._etapa_visualizaciones),

            Stage(4, 'Preprocesamiento',
                  'Crea variable objetivo, codifica y normaliza features',
                  self._etapa_preprocesamiento),

            Stage(5, 'Modelo Supervisado — Random Forest',
                  'Entrena, evalúa y guarda el modelo Random Forest',
                  self._etapa_modelo),

            Stage(6, 'Clustering — K-Means',
                  'Segmenta socios comerciales por perfil de intercambio',
                  self._etapa_clustering),
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # FUNCIONES DE CADA ETAPA
    # ──────────────────────────────────────────────────────────────────────────
    def _etapa_carga(self, ctx: dict) -> dict:
        loader = DataLoader(
            ruta_datos=self.ruta_clean,
            ruta_salida=self.ruta_enriquecido
        )
        loader.ejecutar()
        return ctx

    def _etapa_eda(self, ctx: dict) -> dict:
        eda = EDA(
            ruta_datos=self.ruta_clean,
            ruta_reportes=self.ruta_reportes
        )
        eda.ejecutar()
        return ctx

    def _etapa_visualizaciones(self, ctx: dict) -> dict:
        viz = EDAVisualizaciones(
            ruta_datos=self.ruta_clean,
            ruta_reportes=self.ruta_reportes
        )
        viz.ejecutar()
        return ctx

    def _etapa_preprocesamiento(self, ctx: dict) -> dict:
        prep = Preprocessor(
            ruta_datos=self.ruta_enriquecido,
            ruta_modelos=self.ruta_modelos
        )
        X_train, X_test, y_train, y_test = prep.ejecutar()
        ctx['X_train'] = X_train
        ctx['X_test']  = X_test
        ctx['y_train'] = y_train
        ctx['y_test']  = y_test
        return ctx

    def _etapa_modelo(self, ctx: dict) -> dict:
        trainer = ModelTrainer(
            ruta_modelos=self.ruta_modelos,
            ruta_reportes=self.ruta_reportes
        )
        ctx['modelo_rf'] = trainer.ejecutar(
            ctx['X_train'], ctx['X_test'],
            ctx['y_train'], ctx['y_test']
        )
        return ctx

    def _etapa_clustering(self, ctx: dict) -> dict:
        clustering = Clustering(
            ruta_datos=self.ruta_enriquecido,
            ruta_modelos=self.ruta_modelos,
            ruta_reportes=self.ruta_reportes
        )
        clustering.ejecutar()
        ctx['modelo_clustering'] = clustering.modelo
        return ctx

    # ──────────────────────────────────────────────────────────────────────────
    # EJECUCIÓN DEL PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self, desde: int = 1):
        """
        Ejecuta el pipeline completo o desde una etapa específica.

        Parámetros
        ----------
        desde : int
            Número de etapa desde la cual iniciar (1-6).
        """
        print('\n' + '█' * 60)
        print('  PIPELINE — Análisis Comercio Exterior México')
        print('  Almacenes y Minería de Datos, FC-UNAM')
        print('█' * 60)

        inicio_total = time.time()
        stages_a_ejecutar = [s for s in self.stages
                             if s.numero >= desde]

        print(f'\n  Etapas a ejecutar: '
              f'{[s.numero for s in stages_a_ejecutar]}')

        for stage in stages_a_ejecutar:
            try:
                self.contexto = stage.ejecutar(self.contexto)
            except Exception as e:
                print(f'\n  ✗ Error en etapa {stage.numero}: {e}')
                print(f'  Pipeline detenido.')
                raise

        duracion_total = time.time() - inicio_total
        self._resumen(duracion_total)

        return self.contexto

    # ──────────────────────────────────────────────────────────────────────────
    # DEMOSTRACIÓN por país
    # ──────────────────────────────────────────────────────────────────────────
    def demo_prediccion(self):
        """
        Carga el modelo guardado y hace una predicción nueva.
        """
        import joblib
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        print('\n' + '=' * 60)
        print('  DEMOSTRACIÓN EN VIVO — Reutilización del modelo')
        print('=' * 60)

        # Cargar modelo y transformadores
        modelo   = joblib.load(f'{self.ruta_modelos}random_forest.joblib')
        scaler   = joblib.load(f'{self.ruta_modelos}scaler.joblib')
        le_cont  = joblib.load(f'{self.ruta_modelos}le_continente.joblib')
        le_pais  = joblib.load(f'{self.ruta_modelos}le_pais.joblib')

        print('\n  ✓ Modelo y transformadores cargados desde disco')
        print('  (sin re-entrenar)\n')

        # Ejemplo: predecir balance México-China en enero 2025
        # con tipo de cambio ~17.15 MXN/USD
        nuevos_datos = pd.DataFrame({
            'Exportaciones':  [500_000_000],   # $500M USD
            'Importaciones':  [8_000_000_000], # $8,000M USD
            'TIPO_CAMBIO':    [17.15],
            'CONTINENTE_ENC': [le_cont.transform(['Asia'])[0]],
            'PAIS_ENC':       [le_pais.transform(['China'])[0]],
            'ANIO':           [2025],
            'MES':            [1]
        })

        datos_scaled   = scaler.transform(nuevos_datos)
        prediccion     = modelo.predict(datos_scaled)
        probabilidad   = modelo.predict_proba(datos_scaled)[0]

        print(f'  Escenario: México-China, Enero 2025')
        print(f'  Exportaciones:  $500,000,000 USD')
        print(f'  Importaciones:  $8,000,000,000 USD')
        print(f'  Tipo de cambio: $17.15 MXN/USD')
        print(f'\n  Predicción: {"SUPERÁVIT 🟢" if prediccion[0] == 1 else "DÉFICIT 🔴"}')
        print(f'  Probabilidad déficit:   {probabilidad[0]*100:.1f}%')
        print(f'  Probabilidad superávit: {probabilidad[1]*100:.1f}%')
        print('\n  ✓ Demostración completada — modelo reutilizado sin re-entrenar')

    # ──────────────────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────────────────────────────────
    def _resumen(self, duracion_total: float):
        """Imprime resumen de ejecución del pipeline."""
        print('\n' + '█' * 60)
        print('  PIPELINE COMPLETADO ✓')
        print('█' * 60)
        print(f'\n  {"Etapa":<35} {"Duración":>10}')
        print(f'  {"-"*47}')
        for stage in self.stages:
            if stage.completada:
                estado = '✓'
                dur    = f'{stage.duracion:.1f}s'
            else:
                estado = '—'
                dur    = 'omitida'
            print(f'  {estado} {stage.nombre:<33} {dur:>10}')
        print(f'\n  Tiempo total: {duracion_total:.1f}s')
        print(f'\n  Archivos generados:')
        print(f'    models/   → random_forest.joblib, kmeans.joblib')
        print(f'    reports/  → gráficas EDA, modelo y clustering')
        print(f'    data/     → dataset enriquecido con tipo de cambio')


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline — Análisis Comercio Exterior México')
    parser.add_argument('--desde', type=int, default=1,
                        help='Etapa desde la cual iniciar (1-6)')
    parser.add_argument('--demo', action='store_true',
                        help='Ejecutar demostración en vivo del modelo')
    args = parser.parse_args()

    pipeline = Pipeline()

    if args.demo:
        pipeline.demo_prediccion()
    else:
        pipeline.ejecutar(desde=args.desde)
