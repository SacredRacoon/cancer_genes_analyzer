import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import logging
from src.models.factory import ModelFactory
from typing import Tuple, List

logger = logging.getLogger(__name__)

class SignatureGeneticSelector:
    def __init__(self, config: dict, X_signatures: np.ndarray, y: np.ndarray, signature_names: List[str]):
        self.X = X_signatures
        self.y = y
        self.signature_names = signature_names
        self.n_features = X_signatures.shape[1]

        ga_cfg = config.get('ga', {})
        self.population_size = ga_cfg.get('population_size',50)
        self.mutation_rate = ga_cfg.get('mutation_rate',0.15)
        self.crossover_rate = ga_cfg.get('crossover_rate',0.8)
        self.generations = ga_cfg.get('generations', 30)
        self.random_state = ga_cfg.get('random_state', 228)
        self.elitism_ratio = ga_cfg.get('elitism_ratio', 0.1)
        self.min_features = ga_cfg.get('min_features', 2)

        self.rng = np.random.default_rng(self.random_state)
        self.model_factory = ModelFactory()
        self.model_config = config.get('model', {})
        self.cv_params = {
                'cv': 3, 
                'scoring': 'balanced_accuracy' 
            }

        self.history ={
            "best_fitness": [],
            "mean_fitness": [],
        }
        logger.info(f"genetic selector ready, pop {self.population_size}, gen {self.generations}, features {self.n_features}")

    def _create_model(self):
        return self.model_factory.create_model(self.model_config)
        
    def _initialize_population(self) -> np.ndarray:
        population = []
        for _ in range(self.population_size):
            chromosome = np.zeros(self.n_features, dtype=int)
            n_genes = self.rng.integers(self.min_features, min(self.n_features + 1, 10))
            indices = self.rng.choice(self.n_features, size=n_genes, replace=False)
            chromosome[indices] = 1
            population.append(chromosome)
        return np.array(population)

    def _fitness(self, chromosome: np.ndarray) -> float:
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) < self.min_features:
            return 0.0   
        
        X_selected = self.X[:, selected_indices]
        model = self._create_model()

        try:
            scores = cross_val_score(
                model,
                X_selected,
                self.y,
                **self.cv_params
            )
            base_score = scores.mean()

            complexity_penalty = (len(selected_indices) / self.n_features) *0.05
            return base_score - complexity_penalty
        except Exception:
            return 0.0
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array([self._fitness(chromosome) for chromosome in population])

    def _selection(self, population: np.ndarray, fitness_values: np.ndarray):
        indices = self.rng.choice(self.population_size, size=3, replace=False)
        best_index = indices[np.argmax(fitness_values[indices])]
        return population[best_index].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray):
        if self.rng.random() < self.crossover_rate:
            point = self.rng.integers(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutation(self, chromosome: np.ndarray) -> np.ndarray:
        mutated = chromosome.copy()
        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]

        if mutated.sum() < self.min_features:
            zero_indices = np.where(mutated == 0)[0]
            needed = self.min_features - mutated.sum()
            add_indices = self.rng.choice(zero_indices, 
                                        size=min(needed, 
                                        len(zero_indices)),
                                        replace=False)
            mutated[add_indices] = 1
        return mutated

    def run(self,verbose: bool = True) -> Tuple[np.ndarray, float, List[str]]:
        logger.info("Starting ga optimization")
        population = self._initialize_population()
        iterator = tqdm(range(self.generations), desc = "GA progress", disable=not verbose)
        
        best_overall = -np.inf

        for generation in iterator:
            fitness_values = self._evaluate_population(population)
            best_index = np.argmax(fitness_values)
            current_best = fitness_values[best_index]

            self.history['best_fitness'].append(current_best)
            self.history['mean_fitness'].append(fitness_values.mean())
        
            if current_best > best_overall:
                best_overall = current_best

            if verbose:
                selected_sigs = np.where(population[best_index] ==1)[0]
                iterator.set_description(f"Gen {generation+1} Best F1 {current_best:.4f}")

            n_elite = max(1, int(self.elitism_ratio * self.population_size))
            elite_indices = np.argsort(fitness_values)[-n_elite:]
            elite = population[elite_indices].copy()

            new_population = list(elite)
            while len(new_population) < self.population_size:
                p1 = self._selection(population, fitness_values)
                p2 = self._selection(population, fitness_values)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutation(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutation(c2))

            population = np.array(new_population[:self.population_size])

        final_fitness = self._evaluate_population(population)
        best_final_index = np.argmax(final_fitness)

        self.best_chromosome = population[best_final_index]
        self.best_fitness = final_fitness[best_final_index]
        self.best_signatures = [self.signature_names[i] for i in np.where(self.best_chromosome == 1)[0]]

        logger.info(f"GA finished best fitness {self.best_fitness:.4f}, selected signatures {len(self.best_signatures)}")
        return self.best_chromosome, self.best_fitness, self.best_signatures