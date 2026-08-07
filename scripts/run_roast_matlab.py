"""Run MATLAB ROAST pipeline from Python via MATLAB batch runner."""

import subprocess
from pathlib import Path

MATLAB_EXE = Path(r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe")
MATLAB_SCRIPT_DIR = Path(__file__).parent.parent / "matlab"


_DEFAULT_ZERO_PADDING = 60
_DEFAULT_RETURN_ELECTRODE = "Ex8"
_DEFAULT_OUTPUT_FILE = "data/roast_leadfield_3d.mat"


def run_roast_leadfield_3d(
    electrodes: list[str] | None = None,
    zero_padding: int = _DEFAULT_ZERO_PADDING,
    return_electrode: str = _DEFAULT_RETURN_ELECTRODE,
    output_file: str | Path = _DEFAULT_OUTPUT_FILE,
) -> None:
    """Execute the MATLAB ROAST 3D leadfield generator script."""
    if not MATLAB_EXE.exists():
        msg = f"MATLAB executable not found at: {MATLAB_EXE}"
        raise FileNotFoundError(msg)

    args_list: list[str] = []
    if electrodes:
        elec_str = "{'" + "', '".join(electrodes) + "'}"
        args_list.append(f"'channelLabels', {elec_str}")
    if zero_padding != _DEFAULT_ZERO_PADDING:
        args_list.append(f"'zeroPadding', {zero_padding}")
    if return_electrode != _DEFAULT_RETURN_ELECTRODE:
        args_list.append(f"'returnElectrode', '{return_electrode}'")
    if str(output_file) != _DEFAULT_OUTPUT_FILE:
        args_list.append(f"'outputFile', '{Path(output_file).as_posix()}'")

    args_str = ", ".join(args_list)
    call_str = f"generate_roast_leadfield_3d({args_str});" if args_str else "generate_roast_leadfield_3d;"

    matlab_cmd = f"addpath('{MATLAB_SCRIPT_DIR.as_posix()}'); {call_str}"
    cmd = [
        str(MATLAB_EXE),
        "-batch",
        matlab_cmd,
    ]
    print(f"Executing MATLAB ROAST script: {matlab_cmd}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
        print(res.stdout)
    except subprocess.CalledProcessError as err:
        if err.stdout:
            print("=== MATLAB STDOUT ===")
            print(err.stdout)
        if err.stderr:
            print("=== MATLAB STDERR ===")
            print(err.stderr)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MATLAB ROAST 3D leadfield generation.")
    parser.add_argument("--electrodes", nargs="+", help="Specific scalp electrodes (e.g. TP9 CP5)")
    parser.add_argument(
        "--zero-padding",
        type=int,
        default=_DEFAULT_ZERO_PADDING,
        help="Zero padding voxels (default: 60)",
    )
    parser.add_argument(
        "--return-electrode",
        default=_DEFAULT_RETURN_ELECTRODE,
        help="Return electrode name (default: Ex8)",
    )
    parser.add_argument(
        "--output-file",
        default=_DEFAULT_OUTPUT_FILE,
        help="Output MAT file path",
    )
    args = parser.parse_args()

    run_roast_leadfield_3d(
        electrodes=args.electrodes,
        zero_padding=args.zero_padding,
        return_electrode=args.return_electrode,
        output_file=args.output_file,
    )
