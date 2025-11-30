import pytest
import subprocess
from pathlib import Path
import numpy as np
import json
import numpy.testing as npt
import os


ROOT_DIR = Path(__file__).parent.parent
OUTPUT_PATH = ROOT_DIR / "test_test_output"
SNAPSHOT_PATH = ROOT_DIR / "tests/snapshots/test_snapshot"
config_path = ROOT_DIR / "configOverrides.json"

RTOL = 1e-5
ATOL = 1e-8


@pytest.fixture(autouse=True, scope="session")
def apply_test_override():
    """
    Ensures that configOverrides.json begins with the test override block
    before any tests run.
    """
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    data = {
        "simulationMode": "test",
        "iterations": 8192,
        "gridWidth": 512,
        "gridHeight": 512,
        "threadsPerBlockX": 32,
        "threadsPerBlockY": 32,
        "downloadFrequency": 1024,
    }

    with open("configOverrides.json", "w") as f:
        json.dump(data, f, indent=4)

    subprocess.run(
        ["./bin/main", "--output", str(OUTPUT_PATH)],
        check=True
    )


def test_validate_cuda_kernel_output_against_snapshot():
    """
    Compare the output of the CUDA test kernel against a stored snapshot.

    This function reads metadata and complex-valued simulation slices from both
    the expected snapshot file and the newly generated output file. It verifies
    that:
      • Header fields (width, height, iteration count, download frequency) match.
      • Each 2D slice of complex64 simulation data matches within numerical
        tolerances (RTOL and ATOL).

    The function raises assertion errors if metadata differs or if any slice
    deviates beyond the allowed tolerance, identifying the iteration at which
    the mismatch occurred.
    """
    with open(SNAPSHOT_PATH, "rb") as snapshot_file:
        with open(OUTPUT_PATH, "rb") as output_file:
            header_line = snapshot_file.readline()
            snapshot_header_data = json.loads(header_line)

            snapshot_width = int(snapshot_header_data["width"])
            snapshot_height = int(snapshot_header_data["height"])
            snapshot_iterations = int(snapshot_header_data["iterations"])
            snapshot_downloadFrequency = int(snapshot_header_data["downloadFrequency"])

            header_line = output_file.readline()
            output_header_data = json.loads(header_line)

            output_width = int(output_header_data["width"])
            output_height = int(output_header_data["height"])
            output_iterations = int(output_header_data["iterations"])
            output_downloadFrequency = int(output_header_data["downloadFrequency"])

            assert output_width == snapshot_width, "widths must be equal"
            assert output_height == snapshot_height, "heights must be equal"
            assert output_iterations == snapshot_iterations, "heights must be equal"
            assert (
                output_downloadFrequency == snapshot_downloadFrequency
            ), "downloadFrequency must be equal"

            width = output_width
            height = output_height
            slice_size = width * height
            current_iter = 0
            max_iter = output_iterations // output_downloadFrequency

            while current_iter < max_iter:
                snapshot_flat_slice = np.fromfile(
                    snapshot_file, dtype=np.complex64, count=slice_size
                )
                output_flat_slice = np.fromfile(
                    output_file, dtype=np.complex64, count=slice_size
                )

                snapshot_array_2d = snapshot_flat_slice.reshape((height, width))
                output_array_2d = output_flat_slice.reshape((height, width))

                npt.assert_allclose(
                    snapshot_array_2d,
                    output_array_2d,
                    rtol=RTOL,
                    atol=ATOL,
                    err_msg=f"Arrays differ at iteration {current_iter}",
                )

                current_iter += 1
