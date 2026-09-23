import numpy as np
import logging
from typing import Tuple, List, Optional
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class UnsupervisedModuleSelector:
    def __init__(self, config: dict, W: np.ndarray, module_names: List[str], V_binary: Optional[np.ndarray], H: Optional[np.ndarray] = None, gene_names: Optional[List[str]] = None):
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

        self.use_mutual_exclusivity = ga_cfg.get('use_mutual_exclusivity', True)
        self.use_manifold_separation = ga_cfg.get('use_manifold_separation', True)
        self.weight_variance = ga_cfg.get('weight_variance', 1.0)
        self.weight_redundancy = ga_cfg.get('weight_redundancy', 0.5)
        self.weight_mutual_exclusivity = ga_cfg.get('weight_mutual_exclusivity', 0.3)
        self.weight_manifold = ga_cfg.get('weight_manifold', 0.4)
        self.top_genes_for_me = ga_cfg.get('top_genes_for_mutual_exclusivity', 5)
        self.subsample_for_manifold = ga_cfg.get('subsample_for_manifold', 2000)
        self.n_clusters_manifold = ga_cfg.get('n_clusters_manifold', 4)

        self.rng = np.random.default_rng(self.random_state)
        self.history = {"best_fitness": [], 
                        "mean_fitness": [],
                        "variance_scores": [],
                        "redundancy_scores": [],
                        "me_scores": [],
                        "manifold_scores": []
                        }

        self.V_binary = V_binary
        self.H = H
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)} if gene_names else None

        logger.info(f"Unsupervised GA ready pop={self.population_size}, gen={self.generations}, modules={self.n_modules}, ME={self.use_mutual_exclusivity}, MS={self.use_manifold_separation}")

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

        me_score = 0.0
        if self.use_mutual_exclusivity and self.V_binary is not None and self.H is not None:
            me_score = self._compute_mutual_exclusivity(selected)

        ms_score = 0.0
        if self.use_manifold_separation and W_selected.shape[1] >= 2:
            ms_score = self._compute_manifold_separation(W_selected)

        total = (
            self.weight_variance * variance_score
            - self.weight_redundancy * redundancy_penalty
            + self.weight_mutual_exclusivity * me_score
            + self.weight_manifold * ms_score
        )

        components = {
            'variance': variance_score,
            'redundancy': redundancy_penalty,
            'mutual_exclusivity': me_score,
            'manifold_separation': ms_score,
            'total': total
        }

        return total, components

    def _compute_mutual_exclusivity(self, selected_modules: np.ndarray) -> float:
        if self.V_binary is None or self.H is None or self.gene_to_idx is None:
            return 0.0

        n_patients = self.V_binary.shape[0]
        scores = []

        for module_idx in selected_modules:
            gene_weights = self.H[module_idx, :]
            top_gene_indices_in_H = np.argsort(gene_weights)[-self.top_genes_for_me:][::-1]

            gene_indices_in_V = []
            for h_idx in top_gene_indices_in_H:
                if h_idx < len(self.gene_names):
                    gene_name = self.gene_names[h_idx]
                    if gene_name in self.gene_to_idx:
                        gene_indices_in_V.append(self.gene_to_idx[gene_name])

            if len(gene_indices_in_V) < 2:
                continue

            gene_matrix = self.V_binary[:, gene_indices_in_V]
            valid_mask = ~np.isnan(gene_matrix).any(axis=1)

            if np.sum(valid_mask) < 10:
                continue

            valid_gene_matrix = gene_matrix[valid_mask].astype(float)
            co_matrix = valid_gene_matrix.T @ valid_gene_matrix

            n_valid_patients = np.sum(valid_mask)
            co_matrix_norm = co_matrix / n_valid_patients

            n_genes = len(gene_indices_in_V)
            mask = ~np.eye(n_genes, dtype=bool)

            mean_co_occurrence = float(np.nanmean(co_matrix_norm[mask]))
            exclusivity = 1.0 - mean_co_occurrence

            scores.append(exclusivity)

        if not scores:
            return 0.0
        
        return float(np.mean(scores))

    def _compute_manifold_separation(self, W_selected: np.ndarray) -> float:
        n_patients = W_selected.shape[0]
        if n_patients > self.subsample_for_manifold:
            indices = self.rng.choice(n_patients, size=self.subsample_for_manifold, replace=False)
            W_sub = W_selected[indices]
        else:
            W_sub = W_selected

        n_clusters = min(self.n_clusters_manifold, W_sub.shape[0] // 2)
        if n_clusters < 2:
            return 0.0

        try:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=min(1000, W_sub.shape[0]),
                n_init=3
            )
            labels = kmeans.fit_predict(W_sub)
            sil_score = silhouette_score(W_sub,labels)

            normalized = (sil_score + 1.0) / 2.0
            return float(normalized)
        except Exception as e:
            logger.debug(f"Manifold separation computation failed {e}")
            return 0.0

    def _initialize_population(self) -> np.ndarray:
        population = []
        for _ in range(self.population_size):
            chrom = np.zeros(self.n_modules, dtype=int)
            n = self.rng.integers(self.min_features, self.n_modules + 1)
            idx = self.rng.choice(self.n_modules, size=n, replace=False)
            chrom[idx] = 1
            population.append(chrom)
        return np.array(population)

    def _evaluate_population(self, population: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        fitness_values = []
        all_components = []
        for c in population:
            fitness, components = self._fitness(c)
            fitness_values.append(fitness)
            all_components.append(components)
        return np.array(fitness_values), all_components

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
            fitness, all_components = self._evaluate_population(population)
            best_idx = np.argmax(fitness)
            current_best = fitness[best_idx]

            self.history['best_fitness'].append(current_best)
            self.history['mean_fitness'].append(fitness.mean())

            if all_components and all_components[best_idx]:
                self.history['variance_scores'].append(all_components[best_idx].get('variance', 0))
                self.history['redundancy_scores'].append(all_components[best_idx].get('redundancy',0))
                self.history['me_scores'].append(all_components[best_idx].get('mutual_exclusivity', 0))
                self.history['manifold_scores'].append(all_components[best_idx].get('manifold_separation', 0))
            if current_best > best_overall:
                best_overall = current_best

            if verbose:
                n_selected = np.where(population[best_idx] == 1)[0].shape[0]
                if all_components and all_components[best_idx]:
                    comp = all_components[best_idx]
                    iterator.set_description(
                        f"Gen {gen+1} | Best: {current_best:.3f} | "
                        f"V:{comp.get('variance',0):.2f} R:{comp.get('redundancy',0):.2f} "
                        f"ME:{comp.get('mutual_exclusivity',0):.2f} MS:{comp.get('manifold_separation',0):.2f} | "
                        f"Modules: {n_selected}"
                    )
                else:
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

        final_fitness_values, _ = self._evaluate_population(population)
        best_idx = np.argmax(final_fitness_values)

        self.best_chromosome = population[best_idx]
        self.best_fitness = final_fitness_values[best_idx]
        self.best_modules = [self.module_names[i] for i in np.where(self.best_chromosome == 1)[0]]

        logger.info(f"GA finished best fitness: {self.best_fitness:.4f}, selected modules {len(self.best_modules)}")
        return self.best_chromosome, self.best_fitness, self.best_modules