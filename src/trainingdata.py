import subprocess
from pathlib import Path
import shutil
import pyvista as pv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
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
        Path(__file__).parent / "out" / "output_0009.vti", u, training_in, training_out
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
training_in[:, 0, :, :] -= in_min
training_in[:, 0, :, :] /= in_max - in_min
training_out[:, 0, :, :] -= out_min_u
training_out[:, 0, :, :] /= out_max_u - out_min_u
training_out[:, 1, :, :] -= out_min_v
training_out[:, 1, :, :] /= out_max_v - out_min_v
data_set = TensorDataset(
    torch.from_numpy(training_in).float(), torch.from_numpy(training_out).float()
)
split_size_training = int(len(data_set) * 0.8)
split_size_val = int(len(data_set) * 0.1)
split_size_test = len(data_set) - split_size_training - split_size_val
training_set, validation_set, test_set = random_split(
    data_set, [split_size_training, split_size_val, split_size_test]
)
torch.save(training_set, "../data/train_data.pt")
torch.save(test_set, "../data/test_data.pt")
torch.save(validation_set, "../data/validation_data.pt")
