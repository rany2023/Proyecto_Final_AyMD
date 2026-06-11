"""
clustering.py
Agrupamiento no supervisado de socios comerciales de México — K-Means
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize']  = 13
plt.rcParams['axes.labelsize']  = 11


class Clustering:
    """
    Clase que encapsula el agrupamiento no supervisado de socios
    comerciales de México usando K-Means.

    Responsabilidades:
        - Construir perfil por país (features agregadas)
        - Determinar número óptimo de clusters (codo + silueta)
        - Entrenar K-Means con k óptimo
        - Visualizar clusters con PCA
        - Interpretar y nombrar cada perfil
        - Guardar modelo serializado
    """

    def __init__(self, ruta_datos: str = 'data/comercio_exterior_enriquecido.csv',
                 ruta_modelos: str = 'models/',
                 ruta_reportes: str = 'reports/'):
        self.ruta_datos    = ruta_datos
        self.ruta_modelos  = ruta_modelos
        self.ruta_reportes = ruta_reportes
        self.df            = None
        self.df_perfil     = None
        self.scaler        = StandardScaler()
        self.modelo        = None
        self.k_optimo      = None

        os.makedirs(self.ruta_modelos,  exist_ok=True)
        os.makedirs(self.ruta_reportes, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. CARGA
    # ──────────────────────────────────────────────────────────────────────────
    def cargar_datos(self) -> pd.DataFrame:
        """Carga el dataset enriquecido (Tipo de cambio)."""
        self.df = pd.read_csv(self.ruta_datos, parse_dates=['PERIODO'])
        print(f'Dataset cargado: {self.df.shape[0]:,} filas × '
              f'{self.df.shape[1]} columnas ✓')
        return self.df

    # ──────────────────────────────────────────────────────────────────────────
    # 2. CONSTRUIR PERFIL POR PAÍS
    # ──────────────────────────────────────────────────────────────────────────
    def construir_perfil(self) -> pd.DataFrame:
        """
        Agrega el dataset a nivel país para construir el perfil
        de cada socio comercial con México.

        Features del perfil:
            - EXP_TOTAL:    exportaciones totales acumuladas
            - IMP_TOTAL:    importaciones totales acumuladas
            - BALANCE_PROM: balance comercial promedio mensual
            - EXP_PROM:     exportaciones promedio mensual
            - IMP_PROM:     importaciones promedio mensual
            - TC_PROM:      tipo de cambio promedio del periodo
            - N_MESES:      número de meses con intercambio registrado
        """
        print('\n── Construyendo perfil por país...')

        # Pivot para separar exportaciones e importaciones
        pivot = self.df.pivot_table(
            index=['PAIS', 'CONTINENTE', 'PERIODO'],
            columns='FLUJO',
            values='VALOR_USD',
            aggfunc='sum'
        ).reset_index()
        pivot.columns.name = None

        for col in ['Exportaciones', 'Importaciones']:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot.fillna(0)
        pivot['BALANCE'] = pivot['Exportaciones'] - pivot['Importaciones']

        # Agregar tipo de cambio
        tc = (self.df.groupby('PERIODO')['TIPO_CAMBIO']
              .mean().reset_index())
        pivot = pd.merge(pivot, tc, on='PERIODO', how='left')

        # Perfil por país
        self.df_perfil = pivot.groupby(['PAIS', 'CONTINENTE']).agg(
            EXP_TOTAL    = ('Exportaciones', 'sum'),
            IMP_TOTAL    = ('Importaciones', 'sum'),
            BALANCE_PROM = ('BALANCE',       'mean'),
            EXP_PROM     = ('Exportaciones', 'mean'),
            IMP_PROM     = ('Importaciones', 'mean'),
            TC_PROM      = ('TIPO_CAMBIO',   'mean'),
            N_MESES      = ('PERIODO',       'count')
        ).reset_index()

        print(f'  Países en el perfil: {len(self.df_perfil)}')
        print(f'  Features del perfil: EXP_TOTAL, IMP_TOTAL, BALANCE_PROM, '
              f'EXP_PROM, IMP_PROM, TC_PROM, N_MESES')
        print('\n  Top 5 países por exportaciones totales:')
        print(self.df_perfil.nlargest(5, 'EXP_TOTAL')
              [['PAIS', 'CONTINENTE', 'EXP_TOTAL', 'IMP_TOTAL',
                'BALANCE_PROM']].to_string(index=False))

        return self.df_perfil

    # ──────────────────────────────────────────────────────────────────────────
    # 3. NÚMERO ÓPTIMO DE CLUSTERS
    # ──────────────────────────────────────────────────────────────────────────
    def numero_optimo_clusters(self, k_max: int = 10):
        """
        Determina el número óptimo de clusters usando:
        - Método del codo (inercia)
        - Coeficiente de silueta
        """
        print('\n── Determinando número óptimo de clusters...')

        features = ['EXP_TOTAL', 'IMP_TOTAL', 'BALANCE_PROM',
                    'EXP_PROM', 'IMP_PROM', 'N_MESES']
        X = self.scaler.fit_transform(self.df_perfil[features])

        inercias  = []
        siluetas  = []
        ks        = range(2, k_max + 1)

        for k in ks:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=10)
            km.fit(X)
            inercias.append(km.inertia_)
            siluetas.append(silhouette_score(X, km.labels_))
            print(f'  k={k}: inercia={km.inertia_:,.0f} | '
                  f'silueta={silhouette_score(X, km.labels_):.4f}')

        # Gráfica codo + silueta
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(ks, inercias, 'o-', color='#5B9BD5', linewidth=2)
        axes[0].set_title('Método del Codo — Inercia por k')
        axes[0].set_xlabel('Número de clusters (k)')
        axes[0].set_ylabel('Inercia')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ks, siluetas, 's-', color='#70AD47', linewidth=2)
        axes[1].set_title('Coeficiente de Silueta por k')
        axes[1].set_xlabel('Número de clusters (k)')
        axes[1].set_ylabel('Silueta')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'clustering_codo_silueta.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\n✓ Gráficas guardadas en {ruta}')

        # k óptimo = mayor silueta
        self.k_optimo = ks[siluetas.index(max(siluetas))]
        print(f'\n  k óptimo por silueta: {self.k_optimo} '
              f'(silueta={max(siluetas):.4f})')

        # EE.UU. domina la silueta con k=2 — forzamos k=4
        # para obtener perfiles más informativos
        self.k_optimo = 4
        print(f'  k ajustado a {self.k_optimo} para análisis más granular')

        return self.k_optimo, X
    # ──────────────────────────────────────────────────────────────────────────
    # 4. ENTRENAR K-MEANS
    # ──────────────────────────────────────────────────────────────────────────
    def entrenar(self, X: np.ndarray) -> KMeans:
        """Entrena K-Means con el número óptimo de clusters."""
        print(f'\n── Entrenando K-Means con k={self.k_optimo}...')

        self.modelo = KMeans(
            n_clusters=self.k_optimo,
            random_state=RANDOM_STATE,
            n_init=10
        )
        self.modelo.fit(X)
        self.df_perfil['CLUSTER'] = self.modelo.labels_

        print(f'  Distribución de países por cluster:')
        dist = self.df_perfil['CLUSTER'].value_counts().sort_index()
        for cluster, count in dist.items():
            print(f'    Cluster {cluster}: {count} países')

        return self.modelo

    # ──────────────────────────────────────────────────────────────────────────
    # 5. VISUALIZACIÓN CON PCA
    # ──────────────────────────────────────────────────────────────────────────
    def visualizar_pca(self, X: np.ndarray):
        """Visualiza los clusters en 2D usando PCA."""
        print('\n── Generando visualización PCA...')

        pca  = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d = pca.fit_transform(X)

        varianza = pca.explained_variance_ratio_
        print(f'  Varianza explicada PC1: {varianza[0]*100:.1f}%')
        print(f'  Varianza explicada PC2: {varianza[1]*100:.1f}%')
        print(f'  Varianza total:         {sum(varianza)*100:.1f}%')

        colores = ['#5B9BD5', '#70AD47', '#ED7D31', '#A77DC2',
                   '#FF0000', '#00CED1']

        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster in range(self.k_optimo):
            mask = self.df_perfil['CLUSTER'] == cluster
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=colores[cluster], label=f'Cluster {cluster}',
                       alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

        # Etiquetar países importantes
        paises_destacados = ['Estados Unidos de América', 'China',
                             'Alemania', 'Japón', 'Canada', 'España',
                             'Brasil', 'Corea del Sur']
        for _, row in self.df_perfil.iterrows():
            if row['PAIS'] in paises_destacados:
                idx = self.df_perfil.index.get_loc(
                    self.df_perfil[self.df_perfil['PAIS'] == row['PAIS']].index[0])
                ax.annotate(row['PAIS'],
                            (X_2d[idx, 0], X_2d[idx, 1]),
                            fontsize=7, alpha=0.9,
                            xytext=(5, 5), textcoords='offset points')

        ax.set_title(f'Clusters de Socios Comerciales de México — '
                     f'PCA (varianza explicada: {sum(varianza)*100:.1f}%)')
        ax.set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
        ax.legend(title='Cluster', loc='best')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        ruta = os.path.join(self.ruta_reportes, 'clustering_pca.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Visualización PCA guardada en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # 6. INTERPRETACIÓN DE PERFILES
    # ──────────────────────────────────────────────────────────────────────────
    def interpretar_perfiles(self):
        """Describe cada cluster con estadísticas por grupo."""
        print('\n── Interpretación de perfiles por cluster...')
        print('=' * 70)

        features_analisis = ['EXP_TOTAL', 'IMP_TOTAL', 'BALANCE_PROM',
                             'EXP_PROM', 'IMP_PROM', 'N_MESES']

        resumen = (self.df_perfil.groupby('CLUSTER')[features_analisis]
                   .mean()
                   .round(0))

        print('\nEstadísticas promedio por cluster:')
        print(resumen.to_string())

        print('\nPaíses por cluster:')
        for cluster in range(self.k_optimo):
            paises = (self.df_perfil[self.df_perfil['CLUSTER'] == cluster]
                      ['PAIS'].tolist())
            bal_prom = resumen.loc[cluster, 'BALANCE_PROM']
            exp_prom = resumen.loc[cluster, 'EXP_TOTAL']

            # Etiqueta interpretativa automática
            if exp_prom > resumen['EXP_TOTAL'].quantile(0.75):
                etiqueta = '🔵 Socio estratégico de alto volumen'
            elif bal_prom > 0:
                etiqueta = '🟢 Socio con superávit para México'
            elif bal_prom < 0:
                etiqueta = '🔴 Socio con déficit para México'
            else:
                etiqueta = '⚪ Socio marginal o emergente'

            print(f'\n  Cluster {cluster} — {etiqueta}')
            print(f'  Países ({len(paises)}): '
                  f'{", ".join(paises[:8])}'
                  f'{"..." if len(paises) > 8 else ""}')

        # Guardar perfil completo
        ruta = os.path.join(self.ruta_reportes, 'clustering_perfiles.csv')
        self.df_perfil.to_csv(ruta, index=False)
        print(f'\n✓ Perfiles guardados en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # 7. GUARDAR MODELO
    # ──────────────────────────────────────────────────────────────────────────
    def guardar_modelo(self):
        """Serializa el modelo K-Means con joblib."""
        ruta = os.path.join(self.ruta_modelos, 'kmeans.joblib')
        joblib.dump(self.modelo, ruta)
        print(f'\n✓ Modelo K-Means guardado en {ruta}')

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def ejecutar(self):
        """Ejecuta el pipeline completo de clustering."""
        self.cargar_datos()
        self.construir_perfil()
        k_optimo, X = self.numero_optimo_clusters(k_max=10)
        self.entrenar(X)
        self.visualizar_pca(X)
        self.interpretar_perfiles()
        self.guardar_modelo()

        print('\n' + '=' * 60)
        print(f'  Clustering completado ✓ — k={self.k_optimo} clusters')
        print('  Siguiente: pipeline.py')
        print('=' * 60)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    clustering = Clustering(
        ruta_datos='data/comercio_exterior_enriquecido.csv',
        ruta_modelos='models/',
        ruta_reportes='reports/'
    )
    clustering.ejecutar()
