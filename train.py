import joblib
from pathlib import Path

print("=" * 60)
print("TREINANDO MODELOS")
print("=" * 60)

Path("models").mkdir(exist_ok=True)

print("\n[1/2] Treinando modelo de baralho (Baralho_Optimal)...")
import Baralho_Optimal as deck_mod

print("\n[2/2] Calculando character strength (inicio)...")
import inicio as pred_mod

print("\nModelos em models/:")
for f in sorted(Path("models").glob("*.pkl")):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name} ({size_mb:.1f} MB)")
print("=" * 60)
