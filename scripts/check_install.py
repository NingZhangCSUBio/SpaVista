"""Fast, data-free validation of the SpaVista downstream repository."""

from importlib import import_module
from pathlib import Path
import compileall
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    assert compileall.compile_dir(ROOT, quiet=1), "Python syntax validation failed."
    modules = [
        "spavista.features",
        "spavista.config",
        "spavista.models",
        "tasks.spatial_gene_expression.dataset",
        "tasks.spatial_gene_expression.model",
        "tasks.tumor_detection.run",
        "tasks.protein_abundance.run",
        "tasks.mif_generation.run",
        "tasks.high_resolution.run",
    ]
    for name in modules:
        import_module(name)
        print(f"[OK] {name}")
    print("[OK] SpaVista downstream repository is importable.")


if __name__ == "__main__":
    main()
