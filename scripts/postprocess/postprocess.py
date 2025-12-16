#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
from matplotlib.colors import hsv_to_rgb
import imageio.v2 as imageio
import sys
import json

filename = sys.argv[1]  # argv[0] is the script name
output_video = "%s.mp4" % filename


def complex_to_rgb(complex_data):
    amp = np.abs(complex_data)
    phase = np.angle(complex_data)

    amp_normalized = amp / np.max(amp)

    phase_normalized = (phase + np.pi) / (2 * np.pi)

    hsv = np.zeros((complex_data.shape[0], complex_data.shape[1], 3))
    hsv[..., 0] = phase_normalized
    hsv[..., 1] = 1.0
    hsv[..., 2] = amp_normalized

    rgb = hsv_to_rgb(hsv)
    return rgb


script_path = Path(__file__).resolve()
root_dir = script_path.parent.parent.parent
os.chdir(root_dir)

with open(filename, "rb") as f:
    header_line = f.readline()
    header_data = json.loads(header_line)

    width = int(header_data["gridWidth"])
    height = int(header_data["gridHeight"])
    iterations = int(header_data["iterations"])
    downloadFrequency = int(header_data["downloadFrequency"])

    slice_size = width * height
    current_iter = 0
    max_iter = iterations // downloadFrequency

    with imageio.get_writer(output_video, fps=30, macro_block_size=None) as writer:
        while current_iter < max_iter:
            flat_slice = np.fromfile(f, dtype=np.complex64, count=slice_size)
            if flat_slice.size != slice_size:
                print(f"Warning: Incomplete data found at iteration {current_iter}")
                break

            array_2d = flat_slice.reshape((height, width))
            rgb_image = complex_to_rgb(array_2d)
            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
            writer.append_data(rgb_uint8)

            current_iter += 1
            if current_iter % 10 == 0:
                print(f"Processed frame {current_iter}/{iterations}", end="\r")

        print("Finished reading file.")
