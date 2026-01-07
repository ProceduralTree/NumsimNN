import subprocess
from pathlib import Path
import shutil
import pyvista as pv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

EXECUTABLE = Path(__file__).parent / "numsim_parallel"
SETTINGS_FILE = Path(__file__).parent / "settings.txt"

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def update_settings(re, u):
    lines = SETTINGS_FILE.read_text().splitlines()

    new_lines = []
    for line in lines:
        if line.startswith("re"):
            new_lines.append(f"re = {re}")
        elif line.startswith("dirichletTopX"):
            new_lines.append(f"dirichletTopX = {u}")
        else:
            new_lines.append(line)

    SETTINGS_FILE.write_text("\n".join(new_lines))


def extract_velocity(vti_path, u, training_in, training_out):
    # Load the VTI file
    mesh = pv.read(vti_path)

    # Get velocity data
    velocity = mesh.point_data["velocity"]

    # Save to numpy file
    velocity_matrix = velocity[:, :2].reshape(21, 21, 2)
    velocity_mat = (velocity_matrix[:, :, 0], velocity_matrix[:, :, 1])
    velocity_input = np.zeros((1, 21, 21))
    velocity_input[0, -1, 1:-1] = u
    training_in.append(velocity_input)
    training_out.append(velocity_mat)

    return training_in, training_out


def run_experiment(re, u, training_in, training_out):
    # Update settings file
    update_settings(re, u)

    subprocess.run(
        [str(EXECUTABLE), str(SETTINGS_FILE)], check=True, cwd=Path(__file__).parent
    )
    # Move output files to the data directory
    training_in, training_out = extract_velocity(
        Path(__file__).parent / "out" / "output_0000.vti", u, training_in, training_out
    )
    return training_in, training_out


training_in = []
training_out = []
for re in range(500, 1510, 10):
    u = re / 1000
    training_in, training_out = run_experiment(re, u, training_in, training_out)

training_in = np.array(training_in)
training_out = np.array(training_out)
in_max = float(np.max(training_in[:, 0, :, :]))
in_min = float(np.min(training_in[:, 0, :, :]))
out_max_u = float(np.max(training_out[:, 0, :, :]))
out_min_u = float(np.min(training_out[:, 0, :, :]))
out_max_v = float(np.max(training_out[:, 1, :, :]))
out_min_v = float(np.min(training_out[:, 1, :, :]))
min_max_vals = {
    "inputs": {"u": {"max": in_max, "min": in_min}},
    "labels": {
        "u": {"max": out_max_u, "min": out_min_u},
        "v": {"max": out_max_v, "min": out_min_v},
    },
}
with open("../data/min_max.yaml", "w") as f:
    yaml.dump(min_max_vals, f)

# Normalize data to [0, 1]
training_in[:, 0, :, :] += np.abs(in_min)
training_in[:, 0, :, :] /= in_max + np.abs(in_min)
training_out[:, 0, :, :] += np.abs(out_min_u)
training_out[:, 0, :, :] /= out_max_u + np.abs(out_min_u)
training_out[:, 1, :, :] += np.abs(out_min_v)
training_out[:, 1, :, :] /= out_max_v + np.abs(out_min_v)

torch.save(
    {
        "training_in": torch.from_numpy(training_in).float(),
        "training_out": torch.from_numpy(training_out).float(),
    },
    "../data/training_data.pt",
)

