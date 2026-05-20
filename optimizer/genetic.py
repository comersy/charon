"""Genetic algorithm optimizer for Charon mission sequencing.

Finds the visit order that minimizes total delta-v cost across
all target satellites, subject to fuel and deadline constraints.

All units: km, km/s, seconds, kg.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from core.spacecraft import Spacecraft
from mission.target import Target
from mission.sequence import Sequence


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Individual:
    """A single candidate solution: an ordered list of target indices.

    Attributes:
        genes:    Permutation of target indices (visit order).
        fitness:  Fitness score (lower = better). None if not evaluated.
    """
    genes: list[int]
    fitness: Optional[float] = None

    def __repr__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness:.4f})" \
               if self.fitness is not None else f"Individual(genes={self.genes}, fitness=None)"


# ------------------------------------------------------------------
# Fitness
# ------------------------------------------------------------------

def compute_fitness(
    genes: list[int],
    targets: list[Target],
    spacecraft: Spacecraft,
    t_start: datetime,
    depot_altitude: float = 400.0,
    penalty_infeasible: float = 1e6,
) -> float:
    """Evaluate the fitness of a candidate visit order.

    Fitness = total delta-v (km/s) + penalty for infeasibility.
    Lower is better.

    Args:
        genes:               Ordered list of target indices.
        targets:             Full list of Target objects.
        spacecraft:          Servicer spacecraft.
        t_start:             Mission start time.
        depot_altitude:      Initial parking orbit altitude (km).
        penalty_infeasible:  Large penalty added if sequence is infeasible.

    Returns:
        Scalar fitness value (km/s equivalent).
    """
    ordered = [targets[i] for i in genes]
    seq = Sequence(
        targets=ordered,
        spacecraft=spacecraft,
        t_start=t_start,
        depot_altitude=depot_altitude,
    )
    dv = seq.total_dv()
    penalty = 0.0 if seq.is_feasible() else penalty_infeasible
    return dv + penalty


# ------------------------------------------------------------------
# Genetic operators
# ------------------------------------------------------------------

def random_individual(n: int) -> Individual:
    """Generate a random permutation of n target indices."""
    genes = list(range(n))
    random.shuffle(genes)
    return Individual(genes=genes)


def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    """Select the best individual from a random tournament of size k."""
    contestants = random.sample(population, k)
    return min(contestants, key=lambda ind: ind.fitness or float("inf"))


def order_crossover(parent_a: Individual, parent_b: Individual) -> Individual:
    """Order Crossover (OX1): preserves relative order from both parents.

    Picks a random slice from parent_a and fills remaining genes
    in the order they appear in parent_b.
    """
    n = len(parent_a.genes)
    start, end = sorted(random.sample(range(n), 2))

    child_genes = [None] * n
    child_genes[start:end] = parent_a.genes[start:end]

    fill = [g for g in parent_b.genes if g not in child_genes]
    idx = 0
    for i in range(n):
        if child_genes[i] is None:
            child_genes[i] = fill[idx]
            idx += 1

    return Individual(genes=child_genes)


def swap_mutate(individual: Individual, mutation_rate: float = 0.2) -> Individual:
    """Swap mutation: randomly swap two genes with probability mutation_rate."""
    genes = individual.genes[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(genes)), 2)
        genes[i], genes[j] = genes[j], genes[i]
    return Individual(genes=genes)


# ------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result returned by the genetic algorithm.

    Attributes:
        best_sequence:  Ordered list of Target objects (optimal visit order).
        best_dv:        Total delta-v of the best sequence (km/s).
        best_genes:     Raw index permutation of the best solution.
        generations:    Number of generations run.
        history:        Best fitness value per generation.
        feasible:       Whether the best solution is feasible.
    """
    best_sequence: list[Target]
    best_dv: float
    best_genes: list[int]
    generations: int
    history: list[float]
    feasible: bool

    def __repr__(self):
        return (
            f"OptimizationResult("
            f"dv={self.best_dv:.4f} km/s, "
            f"feasible={self.feasible}, "
            f"generations={self.generations})"
        )


class GeneticOptimizer:
    """Genetic algorithm for multi-target on-orbit servicing sequencing.

    Minimizes total delta-v over all possible visit orders.

    Example:
        optimizer = GeneticOptimizer(
            targets=targets,
            spacecraft=spacecraft,
            t_start=T0,
            pop_size=50,
            n_generations=100,
        )
        result = optimizer.run()
        print(result.best_sequence)
    """

    def __init__(
        self,
        targets: list[Target],
        spacecraft: Spacecraft,
        t_start: datetime,
        depot_altitude: float = 400.0,
        pop_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.2,
        elite_frac: float = 0.1,
        tournament_k: int = 3,
        seed: Optional[int] = None,
        on_generation: Optional[Callable[[int, float], None]] = None,
    ):
        """
        Args:
            targets:        List of Target objects to visit.
            spacecraft:     Servicer spacecraft.
            t_start:        Mission start time (UTC).
            depot_altitude: Initial parking orbit altitude (km).
            pop_size:       Number of individuals per generation.
            n_generations:  Number of generations to evolve.
            mutation_rate:  Probability of swap mutation per individual.
            elite_frac:     Fraction of top individuals kept unchanged.
            tournament_k:   Tournament size for parent selection.
            seed:           Random seed for reproducibility.
            on_generation:  Optional callback(generation, best_fitness)
                            called after each generation (useful for
                            progress bars or live dashboard updates).
        """
        if len(targets) < 2:
            raise ValueError("Need at least 2 targets to optimize a sequence.")

        self.targets = targets
        self.spacecraft = spacecraft
        self.t_start = t_start
        self.depot_altitude = depot_altitude
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.n_elite = max(1, int(pop_size * elite_frac))
        self.tournament_k = tournament_k
        self.on_generation = on_generation

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _evaluate(self, individual: Individual) -> Individual:
        """Compute and attach fitness to an individual."""
        individual.fitness = compute_fitness(
            genes=individual.genes,
            targets=self.targets,
            spacecraft=self.spacecraft,
            t_start=self.t_start,
            depot_altitude=self.depot_altitude,
        )
        return individual

    def run(self) -> OptimizationResult:
        """Run the genetic algorithm and return the best solution found.

        Returns:
            OptimizationResult with the optimal visit sequence.
        """
        n = len(self.targets)

        # Initialize population
        population = [self._evaluate(random_individual(n)) for _ in range(self.pop_size)]
        history = []

        for gen in range(self.n_generations):
            population.sort(key=lambda ind: ind.fitness)
            best_fitness = population[0].fitness
            history.append(best_fitness)

            if self.on_generation:
                self.on_generation(gen, best_fitness)

            # Elitism: keep top individuals unchanged
            next_gen = population[: self.n_elite]

            # Fill rest via selection + crossover + mutation
            while len(next_gen) < self.pop_size:
                parent_a = tournament_select(population, self.tournament_k)
                parent_b = tournament_select(population, self.tournament_k)
                child = order_crossover(parent_a, parent_b)
                child = swap_mutate(child, self.mutation_rate)
                child = self._evaluate(child)
                next_gen.append(child)

            population = next_gen

        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        best_targets = [self.targets[i] for i in best.genes]

        best_seq = Sequence(
            targets=best_targets,
            spacecraft=self.spacecraft,
            t_start=self.t_start,
            depot_altitude=self.depot_altitude,
        )

        return OptimizationResult(
            best_sequence=best_targets,
            best_dv=best_seq.total_dv(),
            best_genes=best.genes,
            generations=self.n_generations,
            history=history,
            feasible=best_seq.is_feasible(),
        )