"""Run MATLAB ROAST pipeline from Python via MATLAB batch runner."""

import subprocess
from pathlib import Path

MATLAB_EXE = Path(r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe")
MATLAB_SCRIPT_DIR = Path(__file__).parent.parent / "matlab"


def run_roast_leadfield_3d() -> None:
    """Execute the MATLAB ROAST 3D leadfield generator script."""
    if not MATLAB_EXE.exists():
        msg = f"MATLAB executable not found at: {MATLAB_EXE}"
        raise FileNotFoundError(msg)

    matlab_cmd = f"addpath('{MATLAB_SCRIPT_DIR.as_posix()}'); generate_roast_leadfield_3d;"
    cmd = [
        str(MATLAB_EXE),
        "-batch",
        matlab_cmd,
    ]
    print(f"Executing MATLAB ROAST script: {matlab_cmd}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    print(res.stdout)


if __name__ == "__main__":
    run_roast_leadfield_3d()
