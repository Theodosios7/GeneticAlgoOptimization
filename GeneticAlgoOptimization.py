import random
from deap import base, creator, tools, algorithms

# Define AWS cost metrics
COST_GP3_STORAGE = 0.08
COST_GP3_IOPS_BASE = 3000
COST_GP3_IOPS_PRICE = 0.005
COST_GP3_THROUGHPUT_BASE = 125
COST_GP3_THROUGHPUT_PRICE = 0.040

COST_GP2 = 0.10
COST_IO2_STORAGE = 0.125
COST_IO2_IOPS_BASE = 32000
COST_IO2_IOPS_PRICE1 = 0.065
COST_IO2_IOPS_PRICE2 = 0.046
COST_IO2_IOPS_PRICE3 = 0.032

COST_IO1_STORAGE = 0.125
COST_IO1_IOPS = 0.065

COST_ST1 = 0.045
COST_SC1 = 0.015

# Define problem constants
VOLUME_SIZES = [50, 100, 200, 500]  # Possible volume sizes in GB
POPULATION_SIZE = 10  # The number of individuals in each generation of the genetic algorithm
GENERATIONS = 5
CXPB, MUTPB = 0.7, 0.2  # Crossover and mutation probabilities

# Define the genetic algorithm components
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.choice, VOLUME_SIZES)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Extract the volume size (first gene) from the individual.
# In genetic algorithms, an individual is a list of genes,
# and each gene represents a potential solution to the optimization problem.
def calculate_cost(individual):
    volume_size = individual[0]

    # Determine cost based on volume size and AWS EBS pricing model:
    # - If volume_size <= COST_GP3_THROUGHPUT_BASE, calculate cost using free tier for IOPS and throughput.
    # - If volume_size > COST_GP3_THROUGHPUT_BASE, consider additional charges for provisioned IOPS and throughput.
    #   Calculate iops_cost, throughput_cost, and total cost (storage + provisioned IOPS + provisioned throughput).
    if volume_size <= COST_GP3_THROUGHPUT_BASE:
        cost = volume_size * COST_GP3_STORAGE
    else:
        iops_cost = max(0, (volume_size - COST_GP3_IOPS_BASE)) * COST_GP3_IOPS_PRICE
        throughput_cost = max(0, (volume_size - COST_GP3_THROUGHPUT_BASE)) * COST_GP3_THROUGHPUT_PRICE
        cost = volume_size * COST_GP3_STORAGE + iops_cost + throughput_cost

    # Returns the calculated cost as a tuple
    return cost,

# Register the genetic algorithm components
toolbox.register("evaluate", calculate_cost)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def monte_carlo_simulation(iterations, toolbox):
    best_alpha = None
    best_fitness = float('inf')  # Initialize with a high value for minimization problems

    for _ in range(iterations):
        # Randomly sample alpha
        alpha = random.uniform(0, 1)

        # Register the crossover operator with the current alpha
        toolbox.register("mate", tools.cxBlend, alpha=alpha)

        # Create and evolve the population using the genetic algorithm
        population = toolbox.population(n=POPULATION_SIZE)
        algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS, stats=None, halloffame=None)

        # Get the best individual and its fitness
        best_individual = tools.selBest(population, k=1)[0]
        fitness = calculate_cost(best_individual)[0]

        # Update the best alpha if a better fitness is found
        if fitness < best_fitness:
            best_fitness = fitness
            best_alpha = alpha

    return best_alpha, best_fitness

# Main function: Sets up the genetic algorithm, evolves the population,
# and prints the best solution's volume size and cost.
# If __name__ == "__main__": block ensures the script's main functionality
# is executed when run.
def main():
    # Perform Monte Carlo simulation to find the best alpha
    best_alpha, best_fitness = monte_carlo_simulation(iterations=100, toolbox=toolbox)

    print(f"Best Alpha: {best_alpha}")
    print(f"Corresponding Fitness: {best_fitness}")

    # Register the best alpha for the final run
    toolbox.register("mate", tools.cxBlend, alpha=best_alpha)

    # Run the genetic algorithm with the best alpha
    population = toolbox.population(n=POPULATION_SIZE)
    algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS, stats=None, halloffame=None)

    # Get the best individual after the final run
    best_individual = tools.selBest(population, k=1)[0]
    best_volume_size = best_individual[0]
    best_cost = calculate_cost(best_individual)[0]

    print(f"Best Volume Size: {best_volume_size} GB")
    print(f"Best Cost: ${best_cost:.2f}")

if __name__ == "__main__":
    main()

