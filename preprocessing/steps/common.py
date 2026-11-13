from pathlib import Path
import subprocess
import numpy as np

def execute(command):
    subprocess.run([str(x) for x in command], check=True)

def fresh(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path.name}")
    path.mkdir(parents=True)
    return path

def aligned(a, b):
    return (a.GetSize() == b.GetSize() and
            all(np.allclose(x, y, atol=1e-5) for x, y in [
                (a.GetOrigin(), b.GetOrigin()), (a.GetSpacing(), b.GetSpacing()),
                (a.GetDirection(), b.GetDirection())]))
