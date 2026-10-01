from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import os
from pathlib import Path
import random
import sys
import time
import requests
import yaml
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from dotenv import load_dotenv
from .docking import calculate_docking

load_dotenv()

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

REPO_ROOT = Path(__file__).resolve().parents[1]

API_URL = os.environ.get("BOLTZ_GATEWAY_URL") or "https://boltz-api.nrp-nautilus.io/predict_affinity"
REQUEST_TIMEOUT_SEC = int(os.environ.get("BOLTZ_REQUEST_TIMEOUT_SEC", "900"))
REQUEST_MAX_RETRIES = int(os.environ.get("BOLTZ_REQUEST_MAX_RETRIES", "4"))
REQUEST_RETRY_SEC = float(os.environ.get("BOLTZ_REQUEST_RETRY_SEC", "10"))

POSSIBLE_TARGETS = ["c-met", "brd4", "1com", "acaa1", "pka"]


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

def tuple_to_score(input_tuple, oracle, *args):
    for idx, element in enumerate(input_tuple):
        if args[idx][0] in POSSIBLE_TARGETS:
            affin = -(element/13) if oracle == "boltz" else -(element/15)
        elif args[idx][0] == "qed":
            qed = (1 - element)
        elif args[idx][0] == "sa":
            sa = (1 - element)
        else:
            raise Exception("Invalid evaluator")

    bias = {"qed": 1, "sa": 1, "affin": 1}
    weights = {"qed": 1, "sa": 1, "affin": 1}

    qed = qed * bias["qed"] * weights["qed"]
    sa = sa * bias["sa"] * weights["sa"]
    affin = affin * bias["affin"] * weights["affin"]

    return affin + qed + sa

def calculate_boltz_nautilus(protein_name, ligand_smiles, affin_cache, idx):
    print(f"\n[Worker {str(idx)}] Sending job for {ligand_smiles[:60]}...", flush=True)

    if ligand_smiles in affin_cache[protein_name]:
        print(f"[Worker {str(idx)}] Ligand already exists", flush=True)
        return affin_cache[protein_name][ligand_smiles]

    payload = {"protein_name": protein_name, "ligand": ligand_smiles}

    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            before = time.time()
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    url=API_URL,
                    json=payload,
                    headers={"Connection": "close"},
                    timeout=REQUEST_TIMEOUT_SEC,
                )
            after = time.time()
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                print(f"[Worker {str(idx)}] {status} in {after-before:.1f}s", flush=True)
                print(result)
                if status == "success":
                    return result.get("affinity", 0)
                print(f"[Worker {str(idx)}] Fail: {result.get('error', 'unknown')}", flush=True)
            elif response.status_code in (429, 502, 503, 504):
                print(
                    f"[Worker {str(idx)}] transient {response.status_code} "
                    f"(attempt {attempt}/{REQUEST_MAX_RETRIES})",
                    flush=True,
                )
            else:
                print(f"[Worker {str(idx)}] Error {response.status_code}: {response.text[:200]}", flush=True)
                return 0
        except Exception as e:
            print(
                f"[Worker {str(idx)}] Connection failed (attempt {attempt}/{REQUEST_MAX_RETRIES}): {str(e)}",
                flush=True,
            )
        if attempt < REQUEST_MAX_RETRIES:
            print(f"[Worker {str(idx)}] Retrying...")
            time.sleep(REQUEST_RETRY_SEC)
    print(f"[Worker {str(idx)}] Exhausted retries", flush=True)
    return 0


class Oracle:
    def __init__(self, args=None, config=None):
        self.name = None
        self.max_obj = args.max_obj
        self.min_obj = args.min_obj
        self.max_evaluator = None
        self.min_evaluator = None
        self.task_label = None
        self.args = args
        self.max_oracle_calls = args.max_oracle_calls
        self.freq_log = args.freq_log
        self.mol_buffer = {}
        self.last_log = 0
        self.config = config
        self.affin_cache = defaultdict(dict)
        self.affin_oracle = "boltz"
        self.starting_population = []

    @property
    def budget(self):
        return self.max_oracle_calls

    def parallel_oracle(self, inputs):
        n_scores = len(self.max_evaluator) + len(self.min_evaluator)
        results = [None] * len(inputs)
        buffer_length = len(self.mol_buffer)
        num_added = 0

        def _score_one(idx, smi, val):
            if smi is None:
                zeros = [0] * n_scores
                return idx, smi, zeros, zeros

            single_score = []
            affin_scores = []
            for eva_tuple in self.max_evaluator:
                eva_name, eva = eva_tuple[0], eva_tuple[1]
                if eva_name == 'qed':
                    mol = Chem.MolFromSmiles(smi)
                    single_score.append(1 - eva(mol))
                elif eva_name in POSSIBLE_TARGETS:
                    if val is not None or smi in self.affin_cache[eva_name]:
                        affin = self.affin_cache[eva_name][smi]
                    else:
                        if self.affin_oracle == "boltz":
                            affin = calculate_boltz_nautilus(eva_name, smi, self.affin_cache, idx)
                        elif self.affin_oracle == "docking":
                            affin = calculate_docking(eva_name, smi)
                    affin_scores.append(affin)
                    single_score.append(-affin)
                else:
                    single_score.append(1 - eva(smi))
            for eva_tuple in self.min_evaluator:
                eva_name, eva = eva_tuple[0], eva_tuple[1]
                if eva_name == 'sa':
                    mol = Chem.MolFromSmiles(smi)
                    single_score.append((eva(mol) - 1) / 9)
                elif eva_name in POSSIBLE_TARGETS:
                    if val is not None or smi in self.affin_cache[eva_name]:
                        affin = self.affin_cache[eva_name][smi]
                    else:
                        if self.affin_oracle == "boltz":
                            affin = calculate_boltz_nautilus(eva_name, smi, self.affin_cache, idx)
                        elif self.affin_oracle == "docking":
                            affin = calculate_docking(eva_name, smi)
                    affin_scores.append(affin)
                    single_score.append(affin)
                else:
                    single_score.append(1 - eva(smi))
            return idx, smi, single_score, affin_scores

        tasks = []
        for i, x in enumerate(inputs):
            if len(self.mol_buffer) + num_added >= self.max_oracle_calls:
                tasks.append((i, None, None))
            elif x in self.mol_buffer:
                tasks.append((i, x, self.mol_buffer[x][0]))
            else:
                tasks.append((i, x, None))
                num_added += 1

        with ThreadPoolExecutor(max_workers=min(len(inputs), 20)) as pool:
            futures = {pool.submit(_score_one, *task): task for task in tasks}
            for future in as_completed(futures):
                idx, x, single_score, affin_scores = future.result()
                results[idx] = single_score
                affin_idx = 0
                for eva_tuple in self.max_evaluator:
                    if eva_tuple[0] in POSSIBLE_TARGETS:
                        self.affin_cache[eva_tuple[0]][x] = affin_scores[affin_idx]
                        affin_idx += 1
                for eva_tuple in self.min_evaluator:
                    if eva_tuple[0] in POSSIBLE_TARGETS:
                        self.affin_cache[eva_tuple[0]][x] = affin_scores[affin_idx]
                        affin_idx += 1
                if x is not None and x not in self.mol_buffer and any(single_score):
                    self.mol_buffer[x] = [tuple_to_score(single_score, self.affin_oracle, *self.max_evaluator, *self.min_evaluator), buffer_length + num_added + 1]
                    num_added += 1

        return results

    def calculate_similarity(self, smiles):
        morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = morgan.GetFingerprint(mol)
        max_sim = 0
        for starting_ligand in self.starting_population:
            cmet_mol = Chem.MolFromSmiles(starting_ligand)
            cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
            similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
            max_sim = max(max_sim, similarity)
        return max_sim

    def assign_evaluator(self, args):

        self.max_evaluator = []
        self.min_evaluator = []
        for obj in self.max_obj:
            if obj == "qed":
                self.max_evaluator.append(("qed", QED.qed))
            elif obj in POSSIBLE_TARGETS:
                self.max_evaluator.append((obj, None))

        for obj in self.min_obj:
            if obj == "sa":
                self.min_evaluator.append(("sa", sascorer.calculateScore))
            elif obj == "sim":
                self.min_evaluator.append(("sim", self.calculate_similarity))
            elif obj in POSSIBLE_TARGETS:
                self.min_evaluator.append((obj, None))

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'seed_' + suffix + ".yaml")

        self.sort_buffer()
        new_buffer = {}
        main_target = ""
        for target in list(self.affin_cache.keys()):
            if target in self.min_obj:
                main_target = target
                break
        print("Main target: " + main_target)
        for smiles in self.mol_buffer:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
            new_buffer[smiles] = [self.affin_cache[main_target][smiles], self.mol_buffer[smiles][1], qed, sa]
        with open(output_file_path, 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m, canonical=True) for m in mols]
                n_calls = len(self.mol_buffer)

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)

        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                )

        print({
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "n_oracle": n_calls,
        })

    def __len__(self):
        return len(self.mol_buffer)

    def __call__(self, smiles_lst):
        """
        Score
        """
        score_tuples = self.parallel_oracle(smiles_lst)
        score_list = [tuple_to_score(score_tuple, self.affin_oracle, *self.max_evaluator, *self.min_evaluator) for score_tuple in score_tuples]
        print("Current buffer: " + str(self.mol_buffer))
        self.sort_buffer()
        self.log_intermediate()
        self.last_log = len(self.mol_buffer)
        self.save_result(self.task_label)
        return score_list

    def crowding_distance(self, scores, front):
        distances = [0.0] * len(front)
        num_obj = len(scores[0])

        for i in range(num_obj):
            sorted_idx = sorted(range(len(front)), key=lambda x: scores[front[x]][i])
            min_val = scores[front[sorted_idx[0]]][i]
            max_val = scores[front[sorted_idx[-1]]][i]
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            for j in range(1, len(front) - 1):
                prev_val = scores[front[sorted_idx[j - 1]]][i]
                next_val = scores[front[sorted_idx[j + 1]]][i]
                if max_val - min_val == 0:
                    d = 0.0
                else:
                    d = (next_val - prev_val) / (max_val - min_val)
                distances[sorted_idx[j]] += d
        return distances

    def select_pareto_front(self, smiles_lst):
        score_list = self.parallel_oracle(smiles_lst)
        score_array = np.array(score_list)
        n = 3

        nds = NonDominatedSorting().do(score_array, only_non_dominated_front=False)[:n]
        print(nds)
        pareto_front_smiles = []
        num_appended = 0

        for i in range(len(nds)):
            if num_appended >= self.config["population_size"]:
                break
            print(len(nds[i]))
            if len(nds[i]) + num_appended <= self.config["population_size"]:
                pareto_front_smiles.append(list(np.array(smiles_lst)[nds[i]]))
                num_appended += len(nds[i])
            else:
                print("Population size exceeded, employing crowding distance")
                try:
                    cd = self.crowding_distance(score_list, nds[i])
                    front = list(np.array(smiles_lst)[nds[i]])
                    sorted_front = sorted(front, key=lambda x: cd[front.index(x)], reverse=True)
                    pareto_front_smiles.append(sorted_front[:(self.config["population_size"]-num_appended)])
                except Exception as e:
                    print(e)
                    pareto_front_smiles.append(list(np.array(smiles_lst)[nds[i]])[:(self.config["population_size"]-num_appended)])
                num_appended += len(nds[i])
        print("Pareto front length: " + str(len(pareto_front_smiles)))
        print("Non dominated front: " + str(list(np.array(smiles_lst)[nds[0]])))
        return pareto_front_smiles

    @property
    def finish(self):
        print("Length of buffer: " + str(len(self.mol_buffer)))
        print("Max oracle calls: " + str(self.max_oracle_calls))
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None, config=None):
        self.model_name = "Default"
        self.args = args
        self.config = config
        self.oracle = Oracle(args=self.args, config=self.config)
        self.all_smiles = [line.strip().strip('"') for line in open(REPO_ROOT / "data" / "zinc.tab") if not line.startswith("smiles")]

    def sanitize(self, smiles_list):
        new_smiles_list = []
        smiles_set = set()
        for smiles in smiles_list:
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        continue
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_smiles_list.append(smiles)
                except ValueError:
                    print('bad smiles')
        return new_smiles_list

    def sort_buffer(self):
        self.oracle.sort_buffer()

    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)

    def save_result(self, suffix=None):

        print(f"Saving molecules...")

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'seed_' + suffix + '.yaml')

        self.sort_buffer()
        new_buffer = {}
        main_target = ""
        for target in list(self.oracle.affin_cache.keys()):
            if target in self.oracle.min_obj:
                main_target = target
                break

        for smiles in self.oracle.mol_buffer:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
            new_buffer[smiles] = [self.oracle.affin_cache[main_target][smiles], self.oracle.mol_buffer[smiles][1], qed, sa]
        with open(output_file_path, 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    def finish(self):
        return self.oracle.finish

    def _optimize(self, config):
        raise NotImplementedError

    def optimize(self, seed=0):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        self.oracle.task_label = str(seed)
        self.oracle.config = self.config
        self.oracle.affin_oracle = self.args.oracle
        # initial affin values
        if self.seed <= 4 and self.oracle.affin_oracle == "boltz":
            for eva in self.args.min_obj:
                if eva in POSSIBLE_TARGETS and os.path.isfile(f"{REPO_ROOT}/{self.args.oracle}_caches/{eva}_{str(self.seed)}.yaml"):
                    with open(f"{REPO_ROOT}/{self.args.oracle}_caches/{eva}_{str(self.seed)}.yaml", 'r') as file:
                        self.oracle.affin_cache[eva] = yaml.safe_load(file)
            for eva in self.args.max_obj:
                if eva in POSSIBLE_TARGETS and os.path.isfile(f"{REPO_ROOT}/{self.args.oracle}_caches/{eva}_{str(self.seed)}.yaml"):
                    with open(f"{REPO_ROOT}/{self.args.oracle}_caches/{eva}_{str(self.seed)}.yaml", 'r') as file:
                        self.oracle.affin_cache[eva] = yaml.safe_load(file)

        self._optimize(self.config)
        self.save_result(self.oracle.task_label)
        self.reset()
