"""Run MATLAB ROAST pipeline from Python via MATLAB batch runner."""

import subprocess
from pathlib import Path

MATLAB_EXE = Path(r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe")
MATLAB_SCRIPT_DIR = Path(__file__).parent.parent / "matlab"


_DEFAULT_ZERO_PADDING = 60
_DEFAULT_RETURN_ELECTRODE = "Ex8"
_DEFAULT_OUTPUT_FILE = "data/roast_field_projection_3d.mat"
_DEFAULT_BATCH_SIZE = 5


def _build_matlab_cmd(
    electrodes: list[str] | None = None,
    zero_padding: int = _DEFAULT_ZERO_PADDING,
    return_electrode: str = _DEFAULT_RETURN_ELECTRODE,
    output_file: str | Path = _DEFAULT_OUTPUT_FILE,
    max_solves: int | None = None,
) -> list[str]:
    """Build MATLAB batch command string and arguments."""
    args_list: list[str] = []
    if electrodes:
        elec_str = "{'" + "', '".join(electrodes) + "'}"
        args_list.append(f"'channelLabels', {elec_str}")
    if max_solves and max_solves > 0:
        args_list.append(f"'maxSolves', {max_solves}")
    if zero_padding != _DEFAULT_ZERO_PADDING:
        args_list.append(f"'zeroPadding', {zero_padding}")
    if return_electrode != _DEFAULT_RETURN_ELECTRODE:
        args_list.append(f"'returnElectrode', '{return_electrode}'")
    if str(output_file) != _DEFAULT_OUTPUT_FILE:
        args_list.append(f"'outputFile', '{Path(output_file).as_posix()}'")

    args_str = ", ".join(args_list)
    call_str = f"generate_roast_field_projection_3d({args_str});" if args_str else "generate_roast_field_projection_3d;"
    matlab_cmd = f"addpath('{MATLAB_SCRIPT_DIR.as_posix()}'); {call_str}"
    return [str(MATLAB_EXE), "-batch", matlab_cmd]


def _run_single_matlab_call(cmd: list[str]) -> None:
    """Execute a single MATLAB batch command with real-time stdout streaming."""
    with subprocess.Popen(  # noqa: S603 -- trusted matlab executable and command
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
        retcode = proc.wait()
        if retcode != 0:
            msg = f"MATLAB exited with non-zero returncode: {retcode}"
            raise subprocess.CalledProcessError(retcode, cmd, msg)


def run_roast_field_projection_3d(
    electrodes: list[str] | None = None,
    zero_padding: int = _DEFAULT_ZERO_PADDING,
    return_electrode: str = _DEFAULT_RETURN_ELECTRODE,
    output_file: str | Path = _DEFAULT_OUTPUT_FILE,
    batch_size: int | None = _DEFAULT_BATCH_SIZE,
) -> None:
    """Execute the MATLAB ROAST 3D field projection generator script.

    Divides execution into process-isolated batches of size ``batch_size`` to eliminate
    memory leaks, Desktop Heap exhaustion, and graphic handle accumulation.
    """
    if not MATLAB_EXE.exists():
        msg = f"MATLAB executable not found at: {MATLAB_EXE}"
        raise FileNotFoundError(msg)

    out_path = Path(output_file)
    checkpoint_path = out_path.parent / f"{out_path.stem}_checkpoint.mat"

    if electrodes:
        cmd = _build_matlab_cmd(
            electrodes=electrodes,
            zero_padding=zero_padding,
            return_electrode=return_electrode,
            output_file=output_file,
        )
        print(f"Executing MATLAB ROAST script: {cmd[-1]}")
        _run_single_matlab_call(cmd)
        return

    batch_num = 1
    while not out_path.exists() or checkpoint_path.exists():
        cmd = _build_matlab_cmd(
            zero_padding=zero_padding,
            return_electrode=return_electrode,
            output_file=output_file,
            max_solves=batch_size,
        )
        print("\n=======================================================")
        print(f" Starting MATLAB ROAST Batch #{batch_num} (batch_size={batch_size})")
        print("=======================================================\n")
        _run_single_matlab_call(cmd)

        if out_path.exists() and not checkpoint_path.exists():
            print(f"\nAll channels completed! Final output generated at: {out_path}")
            break

        print(f"\nBatch #{batch_num} completed. Process memory reclaimed by OS. Launching next batch...\n")
        batch_num += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MATLAB ROAST 3D field projection generation.")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Number of newly solved channels per MATLAB process invocation (default: 5)",
    )
    args = parser.parse_args()

    run_roast_field_projection_3d(
        electrodes=args.electrodes,
        zero_padding=args.zero_padding,
        return_electrode=args.return_electrode,
        output_file=args.output_file,
        batch_size=args.batch_size,
    )
