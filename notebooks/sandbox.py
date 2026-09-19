from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ACDC_DIR = DATA_DIR / "ACDC"

print(ACDC_DIR)
print(list(ACDC_DIR.iterdir())[:5])