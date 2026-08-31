import numpy as np
from ..models import Model
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

class Genetic_selector:
    def __init__(self,x,y,feature_name,
                 population_size=100,
                 mutation_rate=0.1,
                 crossover_rate=0.8,
                 generations=50,
                 random_state=228,
                 elitism_ratio=0.1,
                 min_features=3,
                 model_config=None,
                 cv_params=None,
                 early_stopping_rounds=15,       
                 stagnation_threshold=0.0005):    
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.n_features = x.shape[1]
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.elitism_ratio = elitism_ratio
        self.min_features = min_features
        self.early_stopping_rounds = early_stopping_rounds       
        self.stagnation_threshold = stagnation_threshold         

        self.model_config = model_config or {
            'type': 'random_forest',
            'random_state': random_state
        }

        self.models = Model()

        self.cv_params = cv_params or {
            'cv': 5,
            'scoring': 'f1_macro'
        }

        self.history ={
            "best_fitness": [],
            "mean_fitness": [],
            'best_features': []
        }

    def _create_model(self):
        return self.models.create_model(self.model_config)
        
    def _initialize_population(self):
        population = []

        for _ in range(self.population_size):
            chromosome = np.zeros(self.n_features, dtype=int)
            n_genes = self.rng.integers(self.min_features, self.n_features + 1)
            indices = self.rng.choice(self.n_features, size=n_genes, replace=False)
            chromosome[indices] = 1
            population.append(chromosome)

        return np.array(population)

    def _fitness(self, chromosome):
        selected_indices = np.where(chromosome == 1)[0]

        if len(selected_indices) < self.min_features:
            return 0.0
        
        x_selected = self.x[:, selected_indices]

        model = self._create_model()
        scores = cross_val_score(
            model,
            x_selected,
            self.y,
            **self.cv_params
        )
        return scores.mean()
    
    def _evaluate_population(self, population):
        fitness_value = np.array([self._fitness(chromosome) for chromosome in population])
        return fitness_value

    def _selection(self, population, fitness_values):
        def tournament_selection():
            indices = self.rng.choice(self.population_size, size=3, replace=False)
            best_index = indices[np.argmax(fitness_values[indices])]
            return population[best_index].copy()

        parent1 = tournament_selection()
        parent2 = tournament_selection()

        return parent1, parent2

    def _crossover(self, parent1, parent2):
        if self.rng.random() < self.crossover_rate:
            point = self.rng.integers(1, self.n_features - 1)
            
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])

            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutation(self, chromosome):
        mutated = chromosome.copy()

        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]

        if mutated.sum() < self.min_features:
            zero_indices = np.where(mutated == 0)[0]
            needed = self.min_features - mutated.sum()
            add_indices = self.rng.choice(zero_indices, 
                                          size=min(needed, len(zero_indices)), replace=False)
            mutated[add_indices] = 1

        return mutated

    def run(self,verbose=True):
        population = self._initialize_population()
        iterator = tqdm(range(self.generations)) if verbose else range(self.generations)
        
        best_overall = -np.inf
        stagnation_counter = 0

        for generation in iterator:
            fitness_values = self._evaluate_population(population)

            best_index = np.argmax(fitness_values)
            current_best = fitness_values[best_index]
            
            self.history['best_fitness'].append(current_best)
            self.history['mean_fitness'].append(fitness_values.mean())
            self.history['best_features'].append(population[best_index].copy())

            #Проверка стагнации
            if current_best - best_overall > self.stagnation_threshold:
                best_overall = current_best
                stagnation_counter = 0
                stag_text = ""
            else:
                stagnation_counter += 1
                stag_text = f" [STAG {stagnation_counter}/{self.early_stopping_rounds}]"

            if verbose:
                selected_genes = np.where(population[best_index] == 1)[0]
                gene_names = [self.feature_name[i] for i in selected_genes]

                genes_short = ', '.join(gene_names[:4])
                if len(gene_names) > 4:
                    genes_short += f' +{len(gene_names)-4}'

                iterator.set_description(
                    f"Gen {generation+1}/{self.generations} | Best: {current_best:.4f} | Genes[{len(gene_names)}]: {genes_short}{stag_text}"
                )

            if stagnation_counter >= self.early_stopping_rounds:
                if verbose:
                    print(f"\n⚠ Остановка! Нет улучшений {self.early_stopping_rounds} поколений")
                break

            #Элитизм (блатные)
            n_elite = max(1, int(self.elitism_ratio * self.population_size))
            elite_indices = np.argsort(fitness_values)[-n_elite:]
            elite = population[elite_indices].copy()

            new_population = []
            new_population.extend(elite)

            while len(new_population) < self.population_size:
                parent1, parent2 = self._selection(population, fitness_values)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population[:self.population_size])

        final_fitness = self._evaluate_population(population)
        best_final_index = np.argmax(final_fitness)

        self.best_chromosome = population[best_final_index]
        self.best_fitness = final_fitness[best_final_index]
        self.best_features = [self.feature_name[i] for i in np.where(self.best_chromosome == 1)[0]]

        if verbose:
            print(f"Дело сделано\nЛучший фитнесс {self.best_fitness:.4f}\nОтобрано генов {self.best_features}\nГены(букины) {self.best_chromosome}")

        return self.best_chromosome, self.best_fitness, self.best_features