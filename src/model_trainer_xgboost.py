"""
model_trainer_xgboost.py
Entrenamiento y evaluación del modelo supervisado — XGBoost
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM

Uso:
    python src/model_trainer_xgboost.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
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


class ModelTrainerXGBoost:
    """
    Clase que encapsula el entrenamiento y evaluación del modelo
    XGBoost para predecir superávit/déficit comercial.

    Responsabilidades (Brenda):
        - Entrenar XGBoost con GridSearchCV
        - Evaluar con métricas completas
        - Comparar contra línea base y contra Random Forest
        - Guardar modelo serializado con joblib
        - Generar gráficas de evaluación
    """

    def __init__(self, ruta_modelos: str = 'models/',
                 ruta_reportes: str = 'reports/'):
        self.ruta_modelos  = ruta_modelos
        self.ruta_reportes = ruta_reportes
        self.mejor_modelo  = None

        os.makedirs(self.ruta_modelos,  exist_ok=True)
        os.makedirs(self.ruta_reportes, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LÍNEA BASE
    # ──────────────────────────────────────────────────────────────────────────
    def evaluar_linea_base(self, X_train, X_test,
                           y_train, y_test) -> float:
        """Clasificador mayoritario como línea base."""
        print('\n── Evaluando línea base (clasificador mayoritario)...')

        dummy = DummyClassifier(strategy='most_frequent',
                                random_state=RANDOM_STATE)
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        acc_base = accuracy_score(y_test, y_pred_dummy)

        print(f'  Accuracy línea base: {acc_base:.4f} ({acc_base*100:.1f}%)')
        print(f'  → XGBoost debe superar {acc_base*100:.1f}%')
        return acc_base

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ENTRENAMIENTO CON GRIDSEARCHCV
    # ──────────────────────────────────────────────────────────────────────────
    def entrenar(self, X_train, y_train) -> XGBClassifier:
        """Entrena XGBoost con búsqueda de hiperparámetros."""
        print('\n── Entrenando XGBoost con GridSearchCV...')
        print('   (esto puede tardar unos minutos ☕)')

        param_grid = {
            'n_estimators':  [100, 200],
            'max_depth':     [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample':     [0.8, 1.0]
        }

        xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )

        grid_search = GridSearchCV(
            estimator  = xgb,
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
    def evaluar(self, X_test, y_test, acc_base: float,
                acc_rf: float = 0.9924):
        """Evalúa XGBoost y compara contra línea base y Random Forest."""
        print('\n── Evaluando XGBoost en conjunto de prueba...')

        y_pred      = self.mejor_modelo.predict(X_test)
        y_pred_prob = self.mejor_modelo.predict_proba(X_test)[:, 1]

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

        print(f'\n  ── Comparación de modelos ──')
        print(f'  Línea base:    {acc_base:.4f}')
        print(f'  Random Forest: {acc_rf:.4f}')
        print(f'  XGBoost:       {acc:.4f}')

        if acc > acc_rf:
            print('  🏆 XGBoost SUPERA a Random Forest')
        elif acc == acc_rf:
            print('  🤝 XGBoost EMPATA con Random Forest')
        else:
            print('  🌲 Random Forest supera a XGBoost')

        print(f'\n  Reporte completo:\n')
        print(classification_report(y_test, y_pred,
              target_names=['Déficit (0)', 'Superávit (1)']))

        self._grafica_confusion(y_test, y_pred)
        self._grafica_roc(y_test, y_pred_prob, auc)
        self._grafica_comparacion(acc_base, acc_rf, acc)

        return acc, auc

    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICAS
    # ──────────────────────────────────────────────────────────────────────────
    def _grafica_confusion(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Déficit (0)', 'Superávit (1)'],
                    yticklabels=['Déficit (0)', 'Superávit (1)'], ax=ax)
        ax.set_title('Matriz de Confusión — XGBoost')
        ax.set_ylabel('Valor Real')
        ax.set_xlabel('Valor Predicho')
        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'xgb_confusion.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Matriz de confusión guardada en {ruta}')

    def _grafica_roc(self, y_test, y_pred_prob, auc):
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='#ED7D31', linewidth=2,
                label=f'XGBoost (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1,
                label='Clasificador aleatorio (AUC = 0.50)')
        ax.set_title('Curva ROC — XGBoost')
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'xgb_roc.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Curva ROC guardada en {ruta}')

    def _grafica_comparacion(self, acc_base, acc_rf, acc_xgb):
        """Gráfica comparativa de los tres modelos."""
        modelos  = ['Línea Base', 'Random Forest', 'XGBoost']
        valores  = [acc_base, acc_rf, acc_xgb]
        colores  = ['#D9D9D9', '#5B9BD5', '#ED7D31']

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(modelos, valores, color=colores,
                      edgecolor='white', width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title('Comparación de Accuracy — Modelos')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=acc_base, color='gray', linestyle='--',
                   linewidth=1, alpha=0.5)

        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', fontsize=11,
                    fontweight='bold')
        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'comparacion_modelos.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Comparación de modelos guardada en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # 4. GUARDAR MODELO
    # ──────────────────────────────────────────────────────────────────────────
    def guardar_modelo(self):
        ruta = os.path.join(self.ruta_modelos, 'xgboost.joblib')
        joblib.dump(self.mejor_modelo, ruta)
        print(f'\n✓ Modelo XGBoost guardado en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self, X_train, X_test, y_train, y_test):
        acc_base = self.evaluar_linea_base(X_train, X_test, y_train, y_test)
        self.entrenar(X_train, y_train)
        acc, auc = self.evaluar(X_test, y_test, acc_base)
        self.guardar_modelo()

        print('\n' + '=' * 60)
        print('  Entrenamiento XGBoost completado ✓')
        print(f'  Accuracy final: {acc:.4f} | AUC: {auc:.4f}')
        print('=' * 60)

        return self.mejor_modelo


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    prep = Preprocessor(
        ruta_datos='data/comercio_exterior_enriquecido.csv',
        ruta_modelos='models/'
    )
    X_train, X_test, y_train, y_test = prep.ejecutar()

    trainer = ModelTrainerXGBoost(
        ruta_modelos='models/',
        ruta_reportes='reports/'
    )
    trainer.ejecutar(X_train, X_test, y_train, y_test)
