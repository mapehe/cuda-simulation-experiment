import pytest
import subprocess
from pathlib import Path
import numpy as np
import json
import numpy.testing as npt

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_PATH = ROOT_DIR / "gpe_test_output"
SNAPSHOT_PATH = ROOT_DIR / "tests/snapshots/gpe_snapshot"

RTOL = 1e-5
ATOL = 1e-8


def test_environment_paths_exist():
    """
    Verifies that the critical file paths required for testing exist on the disk.
    """
    assert ROOT_DIR.exists(), f"CRITICAL: Root directory not found at {ROOT_DIR}"

    assert SNAPSHOT_PATH.exists(), (
        f"Missing Snapshot File: {SNAPSHOT_PATH}\n"
        "The file 'gpe_snapshot' is missing from 'tests/snapshots/'?"
    )
    assert OUTPUT_PATH.exists(), (
        f"Missing Output Directory: {OUTPUT_PATH}\n"
        "The file 'gpe_test_output' is missing from '.'?"
    )


def test_wavefunction_evolution_fidelity():
    """
    Validates the simulation output against a reference snapshot to ensure numerical fidelity.

    This test performs the following verifications:
    1. Metadata Consistency: Checks that simulation dimensions, iteration
    counts, and download frequencies match the snapshot header.
    2. Data Integrity: Compares the complex wavefunction data slice-by-slice
    against the snapshot using standard tolerance.
    3. Physical Validity: Asserts that the wavefunction remains normalized (L2
    norm ≈ 1.0) at every time step to ensure probability conservation.
    """
    with open(SNAPSHOT_PATH, "rb") as snapshot_file:
        with open(OUTPUT_PATH, "rb") as output_file:
            header_line = snapshot_file.readline()
            snapshot_header_data = json.loads(header_line)

            snapshot_width = int(snapshot_header_data["width"])
            snapshot_height = int(snapshot_header_data["height"])
            snapshot_iterations = int(snapshot_header_data["iterations"])
            snapshot_downloadFrequency = int(snapshot_header_data["downloadFrequency"])
            snapshot_dx = float(snapshot_header_data["parameterData"]["dx"])
            snapshot_dy = float(snapshot_header_data["parameterData"]["dy"])

            header_line = output_file.readline()
            output_header_data = json.loads(header_line)

            output_width = int(output_header_data["width"])
            output_height = int(output_header_data["height"])
            output_iterations = int(output_header_data["iterations"])
            output_downloadFrequency = int(output_header_data["downloadFrequency"])
            output_dx = float(output_header_data["parameterData"]["dx"])
            output_dy = float(output_header_data["parameterData"]["dy"])

            assert output_width == snapshot_width, "widths must be equal"
            assert output_height == snapshot_height, "heights must be equal"
            assert output_iterations == snapshot_iterations, "heights must be equal"
            assert (
                output_downloadFrequency == snapshot_downloadFrequency
            ), "downloadFrequency must be equal"
            assert output_dx == snapshot_dx
            assert output_dy == snapshot_dy

            width = output_width
            height = output_height
            slice_size = width * height
            dx = output_dx
            dy = output_dy
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

                probability_sum = np.sum(np.abs(snapshot_array_2d)**2) * (dx * dy)
                npt.assert_allclose(probability_sum, 1.0, rtol=RTOL, err_msg="Total probability is not 1")

                current_iter += 1
