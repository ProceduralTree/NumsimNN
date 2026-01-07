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


def extract_side_velocity(vti_path, u, training_in, training_out):
    # Load the VTI file
    mesh = pv.read(vti_path)

    # Get velocity data
    velocity = mesh.point_data["velocity"]

    # Save to numpy file
    velocity_matrix = velocity[:, :2].reshape(21, 21, 2)
    velocity_mat = (velocity_matrix[:, :, 0], velocity_matrix[:, :, 1])
    velocity_input = np.zeros((1, 21, 21))
    velocity_input[0, 1:-1, 0] = u
    training_in.append(velocity_input)
    training_out.append(velocity_mat)

    return training_in, training_out


def run_side_experiment(re, u, training_in, training_out):
    # Update settings file
    update_settings(re, u)

    subprocess.run(
        [str(EXECUTABLE), str(SETTINGS_FILE)], check=True, cwd=Path(__file__).parent
    )
    # Move output files to the data directory
    training_in, training_out = extract_side_velocity(
        Path(__file__).parent / "out" / "output_0009.vti", u, training_in, training_out
    )
    return training_in, training_out


def extract_big_velocity(vti_path, u, training_in, training_out):
    # Load the VTI file
    mesh = pv.read(vti_path)

    # Get velocity data
    velocity = mesh.point_data["velocity"]

    # Save to numpy file
    velocity_matrix = velocity[:, :2].reshape(41, 41, 2)
    velocity_mat = (velocity_matrix[:, :, 0], velocity_matrix[:, :, 1])
    velocity_input = np.zeros((1, 41, 41))
    velocity_input[0, -1, 1:-1] = u
    training_in.append(velocity_input)
    training_out.append(velocity_mat)

    return training_in, training_out


def run_big_experiment(re, u, training_in, training_out):
    # Update settings file
    update_settings(re, u)

    subprocess.run(
        [str(EXECUTABLE), str(SETTINGS_FILE)], check=True, cwd=Path(__file__).parent
    )
    # Move output files to the data directory
    training_in, training_out = extract_big_velocity(
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

# OOD data
high_re_in = []
high_re_out = []
high_re_in, high_re_out = run_experiment(3000, 3, high_re_in, high_re_out)
high_re_in = np.array(high_re_in)
high_re_out = np.array(high_re_out)
high_re_in[:, 0, :, :] -= in_min
high_re_in[:, 0, :, :] /= in_max - in_min
high_re_out[:, 0, :, :] -= out_min_u
high_re_out[:, 0, :, :] /= out_max_u - out_min_u
high_re_out[:, 1, :, :] -= out_min_v
high_re_out[:, 1, :, :] /= out_max_v - out_min_v

high_re_ood_set = TensorDataset(
    torch.from_numpy(np.array(high_re_in)).float(),
    torch.from_numpy(np.array(high_re_out)).float(),
)
torch.save(high_re_ood_set, "../data/high_re_ood_data.pt")

# Big domain OOD data
lines = SETTINGS_FILE.read_text().splitlines()
new_lines = []
for line in lines:
    if line.startswith("physicalSizeX"):
        new_lines.append(f"physicalSizeX = {4.0}")
    elif line.startswith("physicalSizeY"):
        new_lines.append(f"physicalSizeY = {4.0}")
    elif line.startswith("nCellsX"):
        new_lines.append(f"nCellsX = {40}")
    elif line.startswith("nCellsY"):
        new_lines.append(f"nCellsY = {40}")
    else:
        new_lines.append(line)

SETTINGS_FILE.write_text("\n".join(new_lines))
big_domain_in = []
big_domain_out = []
for re in range(500, 1600, 100):
    u = re / 1000
    big_domain_in, big_domain_out = run_big_experiment(
        re, u, big_domain_in, big_domain_out
    )

big_domain_in = np.array(big_domain_in)
big_domain_out = np.array(big_domain_out)
big_domain_in[:, 0, :, :] -= in_min
big_domain_in[:, 0, :, :] /= in_max - in_min
big_domain_out[:, 0, :, :] -= out_min_u
big_domain_out[:, 0, :, :] /= out_max_u - out_min_u
big_domain_out[:, 1, :, :] -= out_min_v
big_domain_out[:, 1, :, :] /= out_max_v - out_min_v

big_domain_ood_set = TensorDataset(
    torch.from_numpy(np.array(big_domain_in)).float(),
    torch.from_numpy(np.array(big_domain_out)).float(),
)

torch.save(big_domain_ood_set, "../data/big_domain_ood_data.pt")

# Side inlet OOD data
lines = SETTINGS_FILE.read_text().splitlines()
new_lines = []
for line in lines:
    if line.startswith("physicalSizeX"):
        new_lines.append(f"physicalSizeX = {2.0}")
    elif line.startswith("physicalSizeY"):
        new_lines.append(f"physicalSizeY = {2.0}")
    elif line.startswith("nCellsX"):
        new_lines.append(f"nCellsX = {20}")
    elif line.startswith("nCellsY"):
        new_lines.append(f"nCellsY = {20}")
    else:
        new_lines.append(line)

SETTINGS_FILE.write_text("\n".join(new_lines))
side_inlet_in = []
side_inlet_out = []
for re in range(500, 1600, 100):
    u = re / 1000
    side_inlet_in, side_inlet_out = run_side_experiment(
        re, u, side_inlet_in, side_inlet_out
    )

side_inlet_in = np.array(side_inlet_in)
side_inlet_out = np.array(side_inlet_out)
side_inlet_in[:, 0, :, :] -= in_min
side_inlet_in[:, 0, :, :] /= in_max - in_min
side_inlet_out[:, 0, :, :] -= out_min_u
side_inlet_out[:, 0, :, :] /= out_max_u - out_min_u
side_inlet_out[:, 1, :, :] -= out_min_v
side_inlet_out[:, 1, :, :] /= out_max_v - out_min_v

side_inlet_ood_set = TensorDataset(
    torch.from_numpy(np.array(side_inlet_in)).float(),
    torch.from_numpy(np.array(side_inlet_out)).float(),
)

torch.save(side_inlet_ood_set, "../data/side_inlet_ood_data.pt")
