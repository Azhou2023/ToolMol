from __future__ import print_function

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List
import math

import numpy as np
from rdkit import rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

from main.pareto_optimizer import BaseOptimizer, calculate_boltz_nautilus, POSSIBLE_TARGETS
from main.docking import calculate_docking

from main.molleo_multi_pareto.GPToss import GPToss

MINIMUM = 1e-10

def make_mating_pool(population_smiles: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    all_tuples = list(zip(population_scores, population_smiles))

    population_scores = [math.pow(10, s + MINIMUM) for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)

    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    return mating_tuples

class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None, config=None):
        super().__init__(args)
        self.model_name = "molleo"

        self.mol_lm = GPToss()
        self.args = args
        self.config = config

    def _optimize(self, config):

        self.oracle.assign_evaluator(self.args)

        # Exploration run
        print(f"{str(len(self.all_smiles))} total SMILES")
        population_smiles = list(np.random.choice(self.all_smiles, config["population_size"], replace=True))

        print(f"{str(len(population_smiles))} SMILES selected")

        # select initial population
        print("Before sanitation: " + str(len(population_smiles)))
        population_smiles = self.sanitize(population_smiles)
        print("After sanitation: " + str(len(population_smiles)))
        self.oracle.starting_population = population_smiles.copy()

        population_scores = self.oracle(population_smiles)
        print(population_scores)

        while True:
            # new_population
            mating_tuples = make_mating_pool(population_smiles, population_scores, config["population_size"])

            before = time.time()
            inputs = [
                (idx, mating_tuples, config["mutation_rate"], self.oracle.min_evaluator,
                self.oracle.max_evaluator, self.oracle.affin_cache)
                for idx in range(config["offspring_size"])
            ]

            gen_futures = {}   # future -> idx
            eval_futures = {}  # future -> (idx, smiles)
            targets = list(set([eva[0] for eva in self.oracle.min_evaluator + self.oracle.max_evaluator]) & set(POSSIBLE_TARGETS))
            target = targets[0] if targets else None
            offspring_smiles = []
            with ThreadPoolExecutor(max_workers=min(config["offspring_size"], 10)) as gen_pool, \
                ThreadPoolExecutor(max_workers=min(config["offspring_size"], 10)) as eval_pool:

                # Submit all generation tasks upfront
                for inp in inputs:
                    idx = inp[0]
                    f = gen_pool.submit(self.mol_lm.edit, *inp)
                    gen_futures[f] = idx

                # As each generation completes, immediately dispatch to evaluation.
                # The gen_pool keeps running its remaining workers concurrently.
                for gen_future in as_completed(gen_futures):
                    idx = gen_futures[gen_future]
                    try:
                        smiles = gen_future.result()
                    except Exception as e:
                        print(f"[Gen {idx}] Generation failed: {e}", flush=True)
                        smiles = None

                    if smiles is not None:
                        offspring_smiles.append(smiles)
                        if target is not None:
                            if self.args.oracle == "boltz":
                                eval_f = eval_pool.submit(calculate_boltz_nautilus, target, smiles, self.oracle.affin_cache, idx)
                            elif self.args.oracle == "docking":
                                eval_f = eval_pool.submit(calculate_docking, target, smiles)
                            eval_futures[eval_f] = (idx, smiles)

                # Collect evaluation results as they complete
                for eval_future in as_completed(eval_futures):
                    idx, smiles = eval_futures[eval_future]
                    try:
                        score = eval_future.result()
                    except Exception as e:
                        print(f"[Eval {idx}] Evaluation failed: {e}", flush=True)
                        score = 0
                    self.oracle.affin_cache[target][smiles] = score
            after = time.time()
            print(f"LLM generation took {str(after-before)} seconds")

            # add new_population
            print("Offspring size: " + str(len(offspring_smiles)))
            population_smiles += offspring_smiles
            population_smiles = self.sanitize(population_smiles)
            print("Population size: " + str(len(population_smiles)))

            n_obj = len(self.oracle.max_evaluator) + len(self.oracle.min_evaluator)
            before = time.time()
            if n_obj > 1:
                # Pareto optimal set
                population_smiles = list(self.oracle.select_pareto_front(population_smiles))
                population_smiles = [item for sublist in population_smiles for item in sublist]
            after = time.time()
            print(f"Boltz calculation took {str(after-before)} seconds")
            population_scores = self.oracle(population_smiles)
            population_tuples = list(zip(population_scores, population_smiles))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
            if n_obj == 1:
                population_tuples = population_tuples[:config["population_size"]]
            population_smiles = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            print("Length of buffer: " + str(len(self.oracle.mol_buffer)))
            print("Max oracle calls: " + str(self.oracle.max_oracle_calls))
            if len(self.oracle.mol_buffer) >= self.oracle.max_oracle_calls:
                print("Finished")
                break
