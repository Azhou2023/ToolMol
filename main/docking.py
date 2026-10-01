import os
from secrets import token_hex
import shutil
import subprocess
import time

import torch

DOCKING_ROOT = os.environ.get("MOLLEO_DOCKING_ROOT", "/home/andrew/docking")
AUTODOCK_BIN = os.environ.get("MOLLEO_AUTODOCK_BIN", "~/AutoDock-GPU/bin/autodock_gpu_128wi")
RECEPTOR_ROOT = os.environ.get("MOLLEO_RECEPTOR_ROOT", "/home/andrew/BAT-cmet-updated")

def calculate_docking(protein_name, smiles, autodock=AUTODOCK_BIN, protein_file=None, num_devices=torch.cuda.device_count(), starting_device=0):
    # Absolute paths only: this runs from several threads at once, and the
    # process-wide cwd cannot be used to scope a single call's files.
    run_dir = os.path.join(DOCKING_ROOT, token_hex(16))
    ligands_folder = os.path.join(run_dir, "inputs")
    outs_folder = os.path.join(run_dir, "outputs")
    os.makedirs(ligands_folder)
    os.makedirs(outs_folder)
    for device in range(starting_device, starting_device + num_devices):
        os.makedirs(f'{ligands_folder}/{device}')

    try:
        device = starting_device
        smiles = [smiles]*64
        print(protein_name)
        if protein_name == "c-met":
            protein_file = f"{RECEPTOR_ROOT}/docking_files/receptor.maps.fld"
        elif protein_name == "brd4":
            protein_file = f"{RECEPTOR_ROOT}/docking_files_brd4/LMCSS-5uf0_5uez_docked.maps.fld"
        elif protein_name == "acaa1":
            protein_file = f"{RECEPTOR_ROOT}/docking_files_acaa1/2iik.maps.fld"
        else:
            raise Exception("Unknown protein")

        for i, smile in enumerate(smiles):
            subprocess.Popen(f'obabel -:"{smile}" -O {ligands_folder}/{device}/ligand{i}HASH{hash(smile)}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
            device += 1
            if device == starting_device + num_devices:
                device = starting_device
        while True:
            total = 0
            for device in range(starting_device, starting_device + num_devices):
                total += len(os.listdir(f'{ligands_folder}/{device}'))
            if total == len(smiles):
                break
        time.sleep(0.1)
        if len(smiles) == 1:
            subprocess.run(f'{autodock} -M {protein_file} -L {ligands_folder}/0/ligand0.pdbqt -N {outs_folder}/ligand0', shell=True)
        else:
            ps = []
            for device in range(starting_device, starting_device + num_devices):
                ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -B {ligands_folder}/{device}/ligand*.pdbqt -N {outs_folder}/ -D {device + 1}', shell=True, stdout=subprocess.DEVNULL))
            stop = False
            while not stop:
                stop = True
                for p in ps:
                    if p.poll() is None:
                        time.sleep(1)
                        stop = False
        affins = [0 for _ in range(len(smiles))]
        for file in os.listdir(outs_folder):
            if file.endswith('.dlg'):
                content = open(f'{outs_folder}/{file}').read()
                if '0.000   0.000   0.000  0.00  0.00' not in content:
                    try:
                        affins[int(file.split('ligand')[1].split('HASH')[0])] = float([line for line in content.split('\n') if 'RANKING' in line][0].split()[3])
                    except:
                        pass

        return min(affins)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
