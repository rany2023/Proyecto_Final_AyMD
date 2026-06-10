"""
demo_prediccion_canada.py
Prueba de reutilización del modelo XGBoost — Predicción México-Canadá 2025
Proyecto Final — Almacenes y Minería de Datos, FC-UNAM

Uso:
    Desde la raíz del proyecto:
        python src/demo_prediccion_canada.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Ruta base: sube un nivel desde src/ hasta la raíz del proyecto
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGAR MODELO Y TRANSFORMADORES (sin re-entrenar)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO — Predicción México-Canadá 2025")
print("  Reutilización del modelo XGBoost guardado")
print("=" * 60)

modelo  = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
le_cont = joblib.load(os.path.join(MODELS_DIR, "le_continente.joblib"))
le_pais = joblib.load(os.path.join(MODELS_DIR, "le_pais.joblib"))

print("\n  ✓ Modelo y transformadores cargados desde disco")

# ──────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUIR EL ESCENARIO: México-Canadá, 2025
# ──────────────────────────────────────────────────────────────────────────────
tipo_cambio   = 18.0
exportaciones = 600_000_000
importaciones = 400_000_000

continente_enc = le_cont.transform(["América del Norte"])[0]
pais_enc       = le_pais.transform(["Canadá"])[0]

# ──────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESAR Y PREDECIR
# ──────────────────────────────────────────────────────────────────────────────
datos = pd.DataFrame({
    "Exportaciones":  [exportaciones],
    "Importaciones":  [importaciones],
    "TIPO_CAMBIO":    [tipo_cambio],
    "CONTINENTE_ENC": [continente_enc],
    "PAIS_ENC":       [pais_enc],
    "ANIO":           [2025],
    "MES":            [1],
})

datos_scaled = scaler.transform(datos)
X_final      = pd.DataFrame(datos_scaled, columns=datos.columns)

prediccion   = modelo.predict(X_final)[0]
probabilidad = modelo.predict_proba(X_final)[0]
resultado    = "SUPERÁVIT 🟢" if prediccion == 1 else "DÉFICIT 🔴"

# ──────────────────────────────────────────────────────────────────────────────
# 4. MOSTRAR RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Escenario: México ↔ Canadá, Enero 2025")
print("─" * 60)
print(f"  Exportaciones:       ${exportaciones:>15,.0f} USD")
print(f"  Importaciones:       ${importaciones:>15,.0f} USD")
print(f"  Tipo de cambio:       {tipo_cambio} MXN/USD")
print(f"  Año / Mes:            2025 / 1")
print("─" * 60)
print(f"\n  ➤  Predicción:              {resultado}")
print(f"     Probabilidad DÉFICIT:    {probabilidad[0] * 100:.1f}%")
print(f"     Probabilidad SUPERÁVIT:  {probabilidad[1] * 100:.1f}%")
print("\n" + "=" * 60)
print("  ✓ Demostración completada")
print("=" * 60 + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. GRÁFICA
# ──────────────────────────────────────────────────────────────────────────────
etiquetas = ["Déficit", "Superávit"]
valores   = [probabilidad[0] * 100, probabilidad[1] * 100]
colores   = ["#E74C3C", "#2ECC71"]

fig, ax = plt.subplots(figsize=(7, 5))
barras = ax.bar(etiquetas, valores, color=colores, width=0.4,
                edgecolor="white", linewidth=1.2)

for barra, val in zip(barras, valores):
    ax.text(barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 1.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=14, fontweight="bold")

ax.axhline(y=50, color="gray", linestyle="--", linewidth=1,
           alpha=0.6, label="Umbral 50%")
ax.set_ylim(0, 115)
ax.set_ylabel("Probabilidad (%)", fontsize=12)
ax.set_title("Predicción XGBoost — México ↔ Canadá, 2025\n"
             f"Resultado: {'SUPERÁVIT' if prediccion == 1 else 'DÉFICIT'}",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
ruta = os.path.join(REPORTS_DIR, "prediccion_canada_2025.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Gráfica guardada en reports/prediccion_canada_2025.png")
