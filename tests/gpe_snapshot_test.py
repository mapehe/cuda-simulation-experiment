import pytest
import subprocess
from pathlib import Path
import numpy as np
import json
import numpy.testing as npt
import os
from .util import read_gpe_snapshots, load_config
import time

ROOT_DIR = Path(__file__).parent.parent
timestamp_str = "gpe_test_output_%s" % time.strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = ROOT_DIR / timestamp_str
SNAPSHOT_PATH = ROOT_DIR / "tests/snapshots/gpe_snapshot"

RTOL = 1e-2
ATOL = 1e-8


@pytest.fixture(autouse=True, scope="session")
def apply_test_override():
    """
    Ensures that configOverrides.json begins with the test override block
    before any tests run.
    """
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    config = load_config()
    if config is None:
        raise RuntimeError("Failed to load config.json")

    config["grossPitaevskii"] = {
        "iterations": 4096,
        "gridWidth": 512,
        "gridHeight": 512,
        "threadsPerBlockX": 32,
        "threadsPerBlockY": 32,
        "downloadFrequency": 512,
        "L": 1.0,
        "x0": 0.15,
        "y0": 0.15,
        "kx": 0,
        "ky": 0,
        "sigma": 0.1,
        "omega": 0,
        "amp": 1.0,

        "trapStr": 10e4,
        "V_bias": 0,
        "r_0": 0,
        "sigma2": 0,

        "absorbStrength": 0,
        "absorbWidth": 0,

        "dt": 6e-7,
        "g": 10e1
    }

    with open("configOverrides.json", "w") as f:
        json.dump(config, f, indent=4)
    subprocess.run(["./bin/main", "--output", str(OUTPUT_PATH), "--mode", "grossPitaevskii", "--config", "configOverrides.json"], check=True)


def test_wavefunction_evolution_fidelity():
    """
    Validates the simulation output against a reference snapshot to ensure numerical fidelity.

    This test performs the following verifications:
    1. Metadata Consistency: Checks that simulation dimensions, iteration
    counts, and download frequencies match the snapshot header.
    2. Data Integrity: Compares the complex wavefunction data slice-by-slice
    against the snapshot using standard tolerance.
    """
    with open(SNAPSHOT_PATH, "rb") as snapshot_file:
        with open(OUTPUT_PATH, "rb") as output_file:
            [width, height, slice_size, dx, dy, current_iter, max_iter] = (
                read_gpe_snapshots(snapshot_file, output_file)
            )

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


def test_wavefunction_normalization():
    """
    Validates the simulation output against a reference snapshot to ensure numerical fidelity.

    This test performs the following verifications:
    1. Metadata Consistency: Checks that simulation dimensions, iteration
    counts, and download frequencies match the snapshot header.
    2. Physical Validity: Asserts that the wavefunction remains normalized (L2
    norm ≈ 1.0) at every time step to ensure probability conservation.
    """
    with open(OUTPUT_PATH, "rb") as output_file:
        with open(SNAPSHOT_PATH, "rb") as snapshot_file:
            [width, height, slice_size, dx, dy, current_iter, max_iter] = (
                read_gpe_snapshots(snapshot_file, output_file)
            )
            while current_iter < max_iter:
                output_flat_slice = np.fromfile(
                    output_file, dtype=np.complex64, count=slice_size
                )

                output_array_2d = output_flat_slice.reshape((height, width))
                probability_sum = np.sum(np.abs(output_array_2d) ** 2) * (dx * dy)
                npt.assert_allclose(
                    probability_sum,
                    1.0,
                    rtol=RTOL,
                    err_msg="Total probability is not 1 on iteartion %s" % current_iter,
                )

                current_iter += 1
