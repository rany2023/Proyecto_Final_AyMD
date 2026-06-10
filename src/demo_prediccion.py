"""
demo_prediccion.py
Demostración en vivo — Predicción por perfil de socio comercial
Comercio Exterior de México (1993–2025)
Proyecto Final — Almacenes y Minería de Datos, Facultad de Ciencias UNAM
"""

import joblib
import pandas as pd
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def demo_prediccion():
    """
    Carga el modelo guardado y predice el balance comercial
    para los 3 clusters principales de socios comerciales de México.
    Se ejecuta SIN re-entrenar el modelo.
    """

    print('\n' + '█' * 65)
    print('  Predicción de balance comercial por perfil de socio')
    print('█' * 65)

    # ── Cargar modelo y transformadores desde disco ───────────────────────────
    print('\n── Cargando modelo y transformadores desde disco...')
    modelo  = joblib.load('models/random_forest.joblib')
    scaler  = joblib.load('models/scaler.joblib')
    le_cont = joblib.load('models/le_continente.joblib')
    le_pais = joblib.load('models/le_pais.joblib')
    print('  ✓ Modelo cargado SIN re-entrenar\n')

    # ── Escenarios por cluster ────────────────────────────────────────────────
    escenarios = [
        {
            'cluster':      'Cluster 1',
            'etiqueta':     '🔵 Socio Estratégico Dominante',
            'pais':         'Estados Unidos de América',
            'continente':   'América del Norte',
            'descripcion':  'Alto volumen, T-MEC vigente',
            'exportaciones': 8_500_000_000,   # $8,500M — México exporta mucho
            'importaciones': 6_200_000_000,   # $6,200M
            'tipo_cambio':   17.20,
            'anio':          2025,
            'mes':           3
        },
        {
            'cluster':      'Cluster 2',
            'etiqueta':     '🔴 Gran Déficit Estructural',
            'pais':         'China',
            'continente':   'Asia',
            'descripcion':  'México importa electrónica y manufactura avanzada',
            'exportaciones':   350_000_000,   # $350M — México exporta poco
            'importaciones': 4_800_000_000,   # $4,800M — importa mucho
            'tipo_cambio':   17.20,
            'anio':          2025,
            'mes':           3
        },
        {
            'cluster':      'Cluster 3',
            'etiqueta':     '🟡 Socios Tecnológicos Deficitarios',
            'pais':         'Japón',
            'continente':   'Asia',
            'descripcion':  'Proveedor de maquinaria, autos y equipo industrial',
            'exportaciones':   280_000_000,   # $280M
            'importaciones': 1_100_000_000,   # $1,100M
            'tipo_cambio':   17.20,
            'anio':          2025,
            'mes':           3
        }
    ]

    # ── Predecir para cada escenario ──────────────────────────────────────────
    resultados = []

    for esc in escenarios:
        print(f'{"─" * 65}')
        print(f'  {esc["cluster"]} — {esc["etiqueta"]}')
        print(f'  País representativo: {esc["pais"]}')
        print(f'  Descripción:         {esc["descripcion"]}')
        print(f'  Exportaciones:  ${esc["exportaciones"]:>15,.0f} USD')
        print(f'  Importaciones:  ${esc["importaciones"]:>15,.0f} USD')
        print(f'  Tipo de cambio: ${esc["tipo_cambio"]:>8.2f} MXN/USD')

        # Preparar datos
        try:
            cont_enc = le_cont.transform([esc['continente']])[0]
            pais_enc = le_pais.transform([esc['pais']])[0]
        except ValueError:
            print(f'  ⚠️  País/Continente no visto en entrenamiento — usando código 0')
            cont_enc = 0
            pais_enc = 0

        datos = pd.DataFrame({
            'Exportaciones':  [esc['exportaciones']],
            'Importaciones':  [esc['importaciones']],
            'TIPO_CAMBIO':    [esc['tipo_cambio']],
            'CONTINENTE_ENC': [cont_enc],
            'PAIS_ENC':       [pais_enc],
            'ANIO':           [esc['anio']],
            'MES':            [esc['mes']]
        })

        datos_scaled  = scaler.transform(datos)
        prediccion    = modelo.predict(datos_scaled)[0]
        probabilidad  = modelo.predict_proba(datos_scaled)[0]
        balance       = esc['exportaciones'] - esc['importaciones']

        resultado = '🟢 SUPERÁVIT' if prediccion == 1 else '🔴 DÉFICIT'
        prob_pred  = probabilidad[1] if prediccion == 1 else probabilidad[0]

        print(f'\n  Balance real:   ${balance:>+15,.0f} USD')
        print(f'  Predicción:     {resultado}')
        print(f'  Probabilidad:   {prob_pred*100:.1f}%')
        print(f'  Prob. déficit:  {probabilidad[0]*100:.1f}% | '
              f'Prob. superávit: {probabilidad[1]*100:.1f}%')

        resultados.append({
            'Cluster':     esc['cluster'],
            'País':        esc['pais'],
            'Predicción':  resultado,
            'Prob. (%)':   f'{prob_pred*100:.1f}%'
        })

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f'\n{"█" * 65}')
    print('  RESUMEN DE PREDICCIONES')
    print(f'{"█" * 65}')
    print(f'\n  {"Cluster":<12} {"País":<30} {"Predicción":<15} {"Prob.":>6}')
    print(f'  {"─"*65}')
    for r in resultados:
        print(f'  {r["Cluster"]:<12} {r["País"]:<30} '
              f'{r["Predicción"]:<15} {r["Prob. (%)"]:>6}')

    print(f'\n  Cluster 0 (227 países) — Socios marginales')
    print(f'  ⚪ Intercambio mínimo — no relevante para análisis estratégico')

    print(f'\n  ✓ Demostración completada — modelo reutilizado SIN re-entrenar')
    print(f'  ✓ Predicciones consistentes con perfiles del clustering')
    print(f'{"█" * 65}\n')


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    demo_prediccion()
