"""
model_trainer.py
Entrenamiento y evaluación del modelo supervisado — Random Forest
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM

Uso:
    python src/model_trainer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings

from preprocessor import Preprocessor

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style='whitegrid', palette='muted')


class ModelTrainer:
    """
    Clase que encapsula el entrenamiento y evaluación del modelo
    Random Forest para predecir superávit/déficit comercial.

    Responsabilidades (Rany):
        - Entrenar Random Forest con GridSearchCV
        - Evaluar con métricas completas
        - Comparar contra línea base (DummyClassifier)
        - Guardar modelo serializado con joblib
        - Generar gráficas de evaluación
    """

    def __init__(self, ruta_modelos: str = 'models/',
                 ruta_reportes: str = 'reports/'):
        self.ruta_modelos  = ruta_modelos
        self.ruta_reportes = ruta_reportes
        self.modelo        = None
        self.mejor_modelo  = None

        os.makedirs(self.ruta_modelos,  exist_ok=True)
        os.makedirs(self.ruta_reportes, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LÍNEA BASE
    # ──────────────────────────────────────────────────────────────────────────
    def evaluar_linea_base(self, X_train, X_test,
                           y_train, y_test) -> float:
        """
        Entrena un clasificador trivial (mayoría) como línea base.
        El modelo real debe superarlo obligatoriamente.
        """
        print('\n── Evaluando línea base (clasificador mayoritario)...')

        dummy = DummyClassifier(strategy='most_frequent',
                                random_state=RANDOM_STATE)
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        acc_base = accuracy_score(y_test, y_pred_dummy)
        print(f'  Accuracy línea base: {acc_base:.4f} ({acc_base*100:.1f}%)')
        print(f'  → El modelo Random Forest debe superar {acc_base*100:.1f}%')

        return acc_base

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ENTRENAMIENTO CON GRIDSEARCHCV
    # ──────────────────────────────────────────────────────────────────────────
    def entrenar(self, X_train, y_train) -> RandomForestClassifier:
        """
        Entrena Random Forest con búsqueda de hiperparámetros
        usando validación cruzada de 5 folds.
        """
        print('\n── Entrenando Random Forest con GridSearchCV...')
        print('   (esto puede tardar unos minutos ☕)')

        param_grid = {
            'n_estimators':      [100, 200],
            'max_depth':         [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf':  [1, 2]
        }

        rf = RandomForestClassifier(random_state=RANDOM_STATE,
                                    n_jobs=-1)

        grid_search = GridSearchCV(
            estimator  = rf,
            param_grid = param_grid,
            cv         = 5,
            scoring    = 'f1_weighted',
            n_jobs     = -1,
            verbose    = 1
        )

        grid_search.fit(X_train, y_train)

        self.mejor_modelo = grid_search.best_estimator_

        print(f'\n  Mejores hiperparámetros:')
        for param, valor in grid_search.best_params_.items():
            print(f'    {param}: {valor}')
        print(f'  Mejor F1 (CV): {grid_search.best_score_:.4f}')

        return self.mejor_modelo

    # ──────────────────────────────────────────────────────────────────────────
    # 3. EVALUACIÓN COMPLETA
    # ──────────────────────────────────────────────────────────────────────────
    def evaluar(self, X_test, y_test, acc_base: float):
        """
        Evalúa el modelo con métricas completas y genera gráficas.
        """
        print('\n── Evaluando modelo en conjunto de prueba...')

        y_pred      = self.mejor_modelo.predict(X_test)
        y_pred_prob = self.mejor_modelo.predict_proba(X_test)[:, 1]

        # Métricas
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')
        f1_w      = f1_score(y_test, y_pred, average='weighted')
        f1_m      = f1_score(y_test, y_pred, average='macro')
        auc       = roc_auc_score(y_test, y_pred_prob)

        print(f'\n  {"Métrica":<25} {"Valor":>8}')
        print(f'  {"-"*35}')
        print(f'  {"Accuracy":<25} {acc:>8.4f}')
        print(f'  {"Precision (weighted)":<25} {precision:>8.4f}')
        print(f'  {"Recall (weighted)":<25} {recall:>8.4f}')
        print(f'  {"F1 (weighted)":<25} {f1_w:>8.4f}')
        print(f'  {"F1 (macro)":<25} {f1_m:>8.4f}')
        print(f'  {"AUC-ROC":<25} {auc:>8.4f}')
        print(f'\n  Línea base:  {acc_base:.4f}')
        print(f'  Mejora:      +{(acc - acc_base)*100:.2f} puntos porcentuales')

        if acc > acc_base:
            print('  ✓ El modelo SUPERA la línea base')
        else:
            print('  ✗ El modelo NO supera la línea base — revisar')

        print(f'\n  Reporte completo:\n')
        print(classification_report(y_test, y_pred,
              target_names=['Déficit (0)', 'Superávit (1)']))

        # Guardar métricas
        self._grafica_confusion(y_test, y_pred)
        self._grafica_roc(y_test, y_pred_prob, auc)
        self._grafica_importancia()

        return acc, auc

    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICAS
    # ──────────────────────────────────────────────────────────────────────────
    def _grafica_confusion(self, y_test, y_pred):
        """Matriz de confusión."""
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Déficit (0)', 'Superávit (1)'],
                    yticklabels=['Déficit (0)', 'Superávit (1)'],
                    ax=ax)
        ax.set_title('Matriz de Confusión — Random Forest')
        ax.set_ylabel('Valor Real')
        ax.set_xlabel('Valor Predicho')
        plt.tight_layout()

        ruta = os.path.join(self.ruta_reportes, 'rf_confusion.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Matriz de confusión guardada en {ruta}')

    def _grafica_roc(self, y_test, y_pred_prob, auc):
        """Curva ROC."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='#5B9BD5', linewidth=2,
                label=f'Random Forest (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1,
                label='Clasificador aleatorio (AUC = 0.50)')
        ax.set_title('Curva ROC — Random Forest')
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        ruta = os.path.join(self.ruta_reportes, 'rf_roc.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Curva ROC guardada en {ruta}')

    def _grafica_importancia(self):
        """Importancia de variables."""
        features = ['Exportaciones', 'Importaciones', 'TIPO_CAMBIO', 'CONTINENTE_ENC',
                    'PAIS_ENC', 'ANIO', 'MES']
        importancias = self.mejor_modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(features)),
               importancias[indices],
               color='#70AD47', edgecolor='white')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([features[i] for i in indices], rotation=30)
        ax.set_title('Importancia de Variables — Random Forest')
        ax.set_ylabel('Importancia')
        plt.tight_layout()

        ruta = os.path.join(self.ruta_reportes, 'rf_importancia.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Importancia de variables guardada en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # 4. GUARDAR MODELO
    # ──────────────────────────────────────────────────────────────────────────
    def guardar_modelo(self):
        """Serializa el modelo entrenado con joblib."""
        ruta = os.path.join(self.ruta_modelos, 'random_forest.joblib')
        joblib.dump(self.mejor_modelo, ruta)
        print(f'\n✓ Modelo guardado en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self, X_train, X_test, y_train, y_test):
        """Ejecuta el pipeline completo de entrenamiento y evaluación."""
        acc_base          = self.evaluar_linea_base(X_train, X_test,
                                                    y_train, y_test)
        self.entrenar(X_train, y_train)
        acc, auc          = self.evaluar(X_test, y_test, acc_base)
        self.guardar_modelo()

        print('\n' + '=' * 60)
        print('  Entrenamiento Random Forest completado ✓')
        print(f'  Accuracy final: {acc:.4f} | AUC: {auc:.4f}')
        print('  Siguiente: ejecutar clustering.py')
        print('=' * 60)

        return self.mejor_modelo


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Preprocesar datos
    prep = Preprocessor(
        ruta_datos='data/comercio_exterior_enriquecido.csv',
        ruta_modelos='models/'
    )
    X_train, X_test, y_train, y_test = prep.ejecutar()

    # Entrenar y evaluar
    trainer = ModelTrainer(
        ruta_modelos='models/',
        ruta_reportes='reports/'
    )
    trainer.ejecutar(X_train, X_test, y_train, y_test)
