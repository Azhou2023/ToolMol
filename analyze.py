import argparse
import glob
import os
import re

import numpy as np
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV


def cluster(smiles):
    if not smiles:
        return []
    morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    fps = [morgan.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    matrix = []
    for i in range(len(fps)):
        for j in range(i):
            matrix.append(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    clusters = Butina.ClusterData(data=matrix, nPts=len(fps), distThresh=0.6, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return [smiles[min(index_cluster)] for index_cluster in clusters]


def analyze_seed(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    ligands = {}
    filtered_ligands = []
    for ligand in sorted(data, key=lambda k: data[k][1]):
        affin = float(data[ligand][0])
        if affin >= 0:
            continue
        qed = float(data[ligand][2])
        sa = float(data[ligand][3])
        ligands[ligand] = [affin, qed, sa]
        if qed > 0.5 and sa < 3.0:
            filtered_ligands.append(ligand)

    def by_affin(k):
        return ligands[k][0]

    c = sorted(cluster(sorted(ligands, key=by_affin)), key=by_affin)
    top_10 = np.mean([ligands[k][0] for k in c[:10]]) if c else float('nan')

    c_filtered = sorted(cluster(sorted(filtered_ligands, key=by_affin)), key=by_affin)
    filtered_mean = np.mean([ligands[k][0] for k in c_filtered[:10]]) if c_filtered else float('nan')

    if c:
        score_list = np.array([
            [1 - (-ligands[k][0] / 15), 1 - ligands[k][1], (ligands[k][2] - 1) / 9]
            for k in c
        ])
        nds = NonDominatedSorting().do(score_list, only_non_dominated_front=True)
        hypervolume = HV(ref_point=np.array([1.0, 1.0, 1.0]))(score_list[nds])
    else:
        hypervolume = float('nan')

    return top_10, filtered_mean, hypervolume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='directory holding seed_<n>.yaml files')
    parser.add_argument('--seeds', type=int, default=None, help='use only the first N seeds')
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.results_dir, 'seed_*.yaml')),
        key=lambda p: int(re.search(r'seed_(\d+)\.yaml$', p).group(1)),
    )
    if not paths:
        parser.error(f"no seed_<n>.yaml files found in {args.results_dir}")
    if args.seeds is not None:
        paths = paths[:args.seeds]

    avg_top_10 = []
    fmean = []
    hv = []
    for path in paths:
        top_10, filtered_mean, hypervolume = analyze_seed(path)
        avg_top_10.append(top_10)
        fmean.append(filtered_mean)
        hv.append(hypervolume)

    print("RUN: " + os.path.basename(os.path.normpath(args.results_dir)))
    print("SEEDS: " + str(len(paths)))
    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(avg_top_10)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(avg_top_10)))
    print("MEAN FILTERED MOLECULES: " + str(np.mean(fmean)))
    print("STD FILTERED MOLECULES: " + str(np.std(fmean)))
    print("HYPERVOLUME: " + str(np.mean(hv)))
    print("STD HYPERVOLUME: " + str(np.std(hv)))


if __name__ == "__main__":
    main()
