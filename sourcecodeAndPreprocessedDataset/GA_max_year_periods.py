import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\hyperimputed_breastfeeding_data_by_country_formatted.csv'

data = pd.read_csv(file_path)

# Extract unique years and countries
years = data['Year'].unique()
countries = data['Countries, territories and areas'].unique()

# Define the year range based on the dataset (start year to end year)
year_range = list(range(int(data['Year'].min()), int(data['Year'].max()) + 1))

# Function to generate the initial population (random year range selections)
def initial_population(countries, year_range, n_population):
    population = []
    for _ in range(n_population):
        start_year = random.choice(year_range)
        end_year = random.choice([year for year in year_range if year >= start_year])
        population.append((start_year, end_year))  # Each individual is a tuple (start_year, end_year)
    return population

# Fitness function: Maximizes the number of years with full data for all countries
def fitness(individual, data):
    start_year, end_year = individual
    fitness_value = 0

    for year in range(start_year, end_year + 1):
        year_has_missing_data = False
        
        for country in countries:
            country_year_data = data[(data['Countries, territories and areas'] == country) & 
                                     (data['Year'] == year)]
            
            if country_year_data.empty or country_year_data[['Under-five mortality rate (per 1000 live births) - Male',
                                                             'Under-five mortality rate (per 1000 live births) - Female',
                                                             'Number of deaths among children under-five - Male',
                                                             'Number of deaths among children under-five - Female']].eq('').values.any():
                year_has_missing_data = True
                if year in [2020, 2021]:
                    print(f"Missing data for {country} in {year}")
                break
        
        if year_has_missing_data:
            fitness_value -= 5  # Strengthen penalty for missing data
        else:
            fitness_value += 1
    
    return fitness_value

# Tournament selection
def tournament_selection(population, fitness_values, tournament_size=3):
    tournament_contestants = random.sample(range(len(population)), tournament_size)
    best_individual_idx = max(tournament_contestants, key=lambda idx: fitness_values[idx])
    return population[best_individual_idx]

# Single-point crossover: Select either the full year range from parent_1 or parent_2
def single_point_crossover(parent_1, parent_2):
    if random.random() < 0.5:
        return parent_1
    else:
        return parent_2

# Mutation function: randomly change the start or end year in the range
def mutation(individual, year_range):
    start_year, end_year = individual
    if random.random() < 0.5:
        start_year = random.choice(year_range)
    else:
        end_year = random.choice([year for year in year_range if year >= start_year])
    
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    # Ensure mutation retains 2020 or 2021 if possible
    if 2021 not in range(start_year, end_year + 1):
        end_year = 2021

    return (start_year, end_year)

# Genetic Algorithm main loop
def run_ga(data, countries, year_range, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(countries, year_range, n_population)
    
    best_cost_so_far = None
    cost_over_time = []  # List to store best fitness per generation
    elite_size = int(0.02 * n_population)  # Elitism: top 2% of the population

    for generation in range(n_generations):
        print(f"\n--- Generation {generation + 1}/{n_generations} ---")
        
        # Calculate fitness
        fitness_values = [fitness(individual, data) for individual in population]
        
        # Elitism: retain the best individuals
        sorted_indices = np.argsort(fitness_values)[::-1]  
        elites = [population[i] for i in sorted_indices[:elite_size]]

        # Select parents using tournament selection
        parents_list = [tournament_selection(population, fitness_values) for _ in range(int(crossover_per * n_population))]

        # Generate offspring
        offspring_list = []
        for i in range(0, len(parents_list), 2):
            parent_1 = parents_list[i]
            parent_2 = parents_list[i+1] if i+1 < len(parents_list) else parents_list[0]  # Handle odd number of parents
            offspring = single_point_crossover(parent_1, parent_2)

            # Apply mutation with a certain probability
            if random.random() < mutation_per:
                offspring = mutation(offspring, year_range)
            
            offspring_list.append(offspring)

        # Combine elites and new offspring into the new population
        population = elites + offspring_list[:n_population - elite_size]

        # Track the best fitness in this generation
        best_fitness_in_gen = max([fitness(individual, data) for individual in population])
        print(f"Best fitness (alignment) in this generation: {best_fitness_in_gen}")

        # Track the best fitness so far
        if best_cost_so_far is None or best_fitness_in_gen > best_cost_so_far:
            best_cost_so_far = best_fitness_in_gen
            print(f"New best fitness found: {best_cost_so_far}")

        cost_over_time.append(best_fitness_in_gen)

    print("Genetic Algorithm completed.")
    
    # Plot the fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_generations + 1), cost_over_time, label="Best Fitness (Alignment)", color='blue', linestyle='-')
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Alignment Score)")
    plt.title("Fitness vs Generations")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

    return population, cost_over_time

# Parameters for GA
n_population = 50
crossover_per = 0.5
mutation_per = 0.3  
n_generations = 100

# Run the GA
best_population, cost_over_time = run_ga(data, countries, year_range, n_population, n_generations, crossover_per, mutation_per)

# Find the best solution (year range)
best_year_range = max(best_population, key=lambda ind: fitness(ind, data))

# Output the best year range found
start_year, end_year = best_year_range
print(f"\nBest year range for all countries: {start_year} to {end_year}")
