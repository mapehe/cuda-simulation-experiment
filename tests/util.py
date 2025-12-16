from pathlib import Path
import numpy as np
import json
import numpy.testing as npt
import os


def read_binary_snapshots(snapshot_file, output_file):
    header_line = snapshot_file.readline()
    snapshot_header_data = json.loads(header_line)

    snapshot_width = int(snapshot_header_data["width"])
    snapshot_height = int(snapshot_header_data["height"])
    snapshot_iterations = int(snapshot_header_data["iterations"])
    snapshot_downloadFrequency = int(snapshot_header_data["downloadFrequency"])

    header_line = output_file.readline()
    output_header_data = json.loads(header_line)

    output_width = int(output_header_data["gridWidth"])
    output_height = int(output_header_data["gridHeight"])
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

    return [
        width,
        height,
        slice_size,
        current_iter,
        max_iter,
        snapshot_header_data,
        output_header_data,
    ]


def read_gpe_snapshots(snapshot_file, output_file):
    [
        width,
        height,
        slice_size,
        current_iter,
        max_iter,
        snapshot_header_data,
        output_header_data,
    ] = read_binary_snapshots(snapshot_file, output_file)

    snapshot_dx = float(snapshot_header_data["parameterData"]["dx"])
    snapshot_dy = float(snapshot_header_data["parameterData"]["dy"])
    output_dx = float(output_header_data["parameterData"]["dx"])
    output_dy = float(output_header_data["parameterData"]["dy"])

    assert output_dx == snapshot_dx
    assert output_dy == snapshot_dy

    dx = output_dx
    dy = output_dy

    return [width, height, slice_size, dx, dy, current_iter, max_iter]
