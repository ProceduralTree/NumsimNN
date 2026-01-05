import subprocess
from pathlib import Path
import shutil
import pyvista as pv
import numpy as np

EXECUTABLE = Path("numsim_parallel")
SETTINGS_FILE = Path("settings.txt")

DATA_DIR = Path("data")
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

def extract_velocity(vti_path, output_path, u):
    # Load the VTI file
    mesh = pv.read(vti_path)

    # Get velocity data
    velocity = mesh.point_data['velocity']

    # Save to numpy file
    velocity_matrix = velocity[:, :2].reshape(21, 21, 2)
    velocity_mat = (velocity_matrix[:, :, 0], velocity_matrix[:, :, 1])
    np.save(output_path + '_out_velocity.npy', velocity_mat)
    velocity_input = np.zeros((1, 21, 21))
    velocity_input[0, -1, 1:-1] = u
    np.save(output_path + '_in_velocity.npy', velocity_input)

def run_experiment(re, u):
    # Update settings file
    update_settings(re, u)

    subprocess.run(
        ["./numsim_parallel", "./settings.txt"],
        check=True
    )
    # Move output files to the data directory
    extract_velocity("out/output_0000.vti", str(DATA_DIR / f"re_{re}_u_{u}"), u)
    

for re in range(500, 1510, 10):
    u = re / 1000
    run_experiment(re, u)