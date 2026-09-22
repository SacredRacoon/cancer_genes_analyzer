import numpy as np
import logging
from typing import Tuple, List
from tqdm import tqdm

logger = logging.getLogger(__name__)

class UnsupervisedModuleSelector:
    def __init__(self, config: dict, W: np.ndarray, module_names: List[str]):
        self.W = W
        self.module_names = module_names
        self.n_modules = W.shape[1]

        ga_cfg = config.get('ga', {})
        self.population_size = ga_cfg.get('population_size', 50)
        self.mutation_rate = ga_cfg.get('mutation_rate', 0.15)
        self.crossover_rate = ga_cfg.get('crossover_rate', 0.8)
        self.generations = ga_cfg.get('generations', 30)
        self.random_state = ga_cfg.get('random_state', 229)
        self.elitism_ratio = ga_cfg.get('elitism_ratio', 0.1)
        self.min_features = ga_cfg.get('min_features', 2)

        self.rng = np.random.default_rng(self.random_state)
        self.history = {"best_fitness": [], "mean_fitness": []}

        logger.info(f"Unsupervised GA ready pop={self.population_size}, gen={self.generations}, modules={self.n_modules}")

    def _fitness(self, chromosome: np.ndarray) -> float:
        selected = np.where(chromosome == 1)[0]
        if len(selected) < self.min_features:
            return 0.0

        W_selected = self.W[:, selected]

        variance_score = np.sum(np.var(W_selected, axis=0))

        if len(selected) > 1:
            corr_matrix = np.corrcoef(W_selected.T)
            mask = ~np.eye(len(selected), dtype=bool)
            redundancy_penalty = np.mean(np.abs(corr_matrix[mask])) * 0.5
        else:
            redundancy_penalty = 0.0

        return variance_score - redundancy_penalty

    def _initialize_population(self) -> np.ndarray:
        population = []
        for _ in range(self.population_size):
            chrom = np.zeros(self.n_modules, dtype=int)
            n = self.rng.integers(self.min_features, self.n_modules + 1)
            idx = self.rng.choice(self.n_modules, size=n, replace=False)
            chrom[idx] = 1
            population.append(chrom)
        return np.array(population)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array([self._fitness(c) for c in population])

    def _selection(self, population, fitness):
        idx = self.rng.choice(self.population_size, size=3, replace=False)
        return population[idx[np.argmax(fitness[idx])]].copy()

    def _crossover(self, p1, p2):
        if self.rng.random() < self.crossover_rate:
            pt = self.rng.integers(1, self.n_modules)
            return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])
        return p1.copy(), p2.copy()

    def _mutation(self, chrom):
        m = chrom.copy()
        for i in range(self.n_modules):
            if self.rng.random() < self.mutation_rate:
                m[i] = 1 - m[i]
        if m.sum() < self.min_features:
            zeros = np.where(m == 0)[0]
            add = self.rng.choice(zeros, size=min(self.min_features - m.sum(), len(zeros)), replace=False)
            m[add] = 1
        return m

    def run(self, verbose: bool = True) -> Tuple[np.ndarray, float, List[str]]:
        logger.info("Starting unsupervised GA optimization")
        population = self._initialize_population()
        iterator = tqdm(range(self.generations), desc="GA progress", disable=not verbose)

        best_overall = -np.inf

        for gen in iterator:
            fitness = self._evaluate_population(population)
            best_idx = np.argmax(fitness)
            current_best = fitness[best_idx]

            self.history['best_fitness'].append(current_best)
            self.history['mean_fitness'].append(fitness.mean())

            if current_best > best_overall:
                best_overall = current_best

            if verbose:
                n_selected = np.where(population[best_idx] == 1)[0].shape[0]
                iterator.set_description(f"Gen {gen+1} | Best: {current_best:.4f} | Modules: {n_selected}")

            n_elite = max(1, int(self.elitism_ratio * self.population_size))
            elite = population[np.argsort(fitness)[-n_elite:]].copy()

            new_pop = list(elite)
            while len(new_pop) < self.population_size:
                c1, c2 = self._crossover(self._selection(population, fitness), self._selection(population, fitness))
                new_pop.append(self._mutation(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutation(c2))

            population = np.array(new_pop[:self.population_size])

        final_fitness = self._evaluate_population(population)
        best_idx = np.argmax(final_fitness)

        self.best_chromosome = population[best_idx]
        self.best_fitness = final_fitness[best_idx]
        self.best_modules = [self.module_names[i] for i in np.where(self.best_chromosome == 1)[0]]

        logger.info(f"GA finished best fitness: {self.best_fitness:.4f}, selected modules {len(self.best_modules)}")
        return self.best_chromosome, self.best_fitness, self.best_modules