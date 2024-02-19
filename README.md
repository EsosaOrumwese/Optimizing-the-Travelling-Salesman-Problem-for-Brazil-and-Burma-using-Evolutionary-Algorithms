# Optimizing the Travelling Salesman Problem for Brazil and Burma using Evolutionary Algorithms with Varied Operators and Parameters
#### by Esosa Orumwese

## Project Overview
This project addresses the Travelling Salesman Problem (TSP) for two distinct datasets: one consisting of 58 cities in Brazil and the other comprising 14 cities in Burma. The TSP aims to find the most cost-effective route for a salesperson to visit all cities exactly once before returning to the starting point. The fitness of each route is based on the cumulative travel cost. Our objective is to implement an evolutionary algorithm to optimize the cost function for the TSP in both datasets.

## Summary of Findings
From this research, we can see that for the TSP problem, having a crossover that explores the landscape of the solution space effectively while also keeping the information of the parents seen
such as the optimal routes seen, helps a great deal in improving the fitness of the algorithm. Also,
for convergence, when the algorithms were paired with mutation operators that took smaller steps
in the neighbourhood of the solutions, it allowed the algorithm to exploit that neighbourhood eventually
leading to finding solutions with better fitness. When paired with these kinds of mutation
and crossover operators, having a large population size was not necessary as a population of size
50 and 100 paired with a selection pressure of 0.05 to 0.1 resulted in better results at a faster
convergence speed.

## Installation
No installation is required. The project consists of two modules, and the code is straightforward. The README will elaborate on the two modules created and the experiment design. You can find the requirements for running this in `requirements.txt`.

## Usage
This experiment was broken down into 3 groups with the last being the main experiment which you can carry out for reproducibility checks.

### 1. Bulk Experiment Design
The `bulk_experiment` function performs extensive experimentation. It involves varying population size, tournament size, crossover, mutation, and replacement operators in 144 experiments, each executed with 10 different random seeds. This function significantly tests combinations of parameters and operators and **may take a considerable amount of time to run**. Note that in this submission, this was commented out and the data which I generated during my run was uploaded on Google drive and is automatically read in as you run the file. The required `gdown` module for downloading files from Google drive that are large will also be automatically be installed.

### 2. Crossovers + Population Size + Tournament Size Experiment Design

#### Single Point Crossover
The `singleCRSV` function performs experiments with varying mutation and replacement operators while keeping population size and tournament size constant using single-point crossover. It conducts multiple trials based on provided seeds and provides dataframes containing fitness results and the best solution for each combination.

#### Edge Crossover
The `edgeCRSV` function, similar to `singleCRSV`, performs trials on mutation and replacement operators, maintaining constant population and tournament sizes but using edge crossover.

### 3. Main Experiment Design
This section will describe additional functions related to the main experiment design and is **the main experiment that you can test out, if you want** because it handles a single parameter/operator combination at a time.

## File Structure

### 1. Bulk Experiment Design
Contained within the Jupyter notebook itself.

### 2. Crossovers + Population Size + Tournament Size Experiment Design
Implemented in the `experiment_design.py` file, housing functions for single-point and edge crossover experiments.

### 3. Main Experiment Design
Contained in the `evoALg_operators.py` file, housing definitions for operators and functions used in the project. It encompasses two evolutionary algorithms, each focusing on different crossover functions and operator implementations.


This section focuses on the main components of the evolutionary algorithm designed to tackle the Traveling Salesman Problem (TSP) and includes population initialization, fitness calculation, parent selection, crossover functions, mutation methods, and survivor selection.

#### Population Initialization

The `init_pop` function generates a specified number of random solutions with associated fitness values based on the cost matrix provided. It creates a population of solutions for further evolutionary processes.

#### Fitness Function

The `fitness` function calculates the fitness for a single solution in the TSP context. It evaluates the cost of traveling a route encompassing all cities exactly once and returning to the starting point.

#### Parent Selection (Tournament Selection)

The `tournament_selection` function performs tournament selection on a population of solutions. It randomly samples a specified number of solutions without replacement and selects the fittest among them as parents. This process is repeated multiple times to select pairs of parents.

#### Crossover Functions

##### Single Point Crossover (`singlePoint_crossover`)

This function implements single-point crossover, a genetic operator that combines genetic material from two parents to generate two offspring solutions. It randomly selects a crossover point and exchanges the genetic material between parents at that point, producing two new solutions.

##### Edge Crossover (`edge_crossover`)

The `generate_edge_table` function prepares an edge table given two parent solutions, which helps in preserving common edges while ensuring diversity in the offspring solutions. The `element_selector` function facilitates the selection of the next element based on the edge table and probabilities, allowing a balance between exploiting common edges and exploring new possibilities.

#### Mutation Methods

Several mutation methods are available to introduce diversity in the population:

- **Swap Mutation (`swap_mutation`)**
- **Insert Mutation (`insert_mutation`)**
- **Scramble Mutation (`scramble_mutation`)**
- **Inversion Mutation (`inversion_mutation`)**

Each mutation method alters the genetic material of a solution to explore different routes in the search space.

#### Survivor Selection

The survivor selection methods control how offspring solutions replace existing solutions in the population:

- **Replace Worst (`replace_worst`)**: Offsprings replace the worst solutions in the parent population.
- **Replace First Worst (`replace_first_worst`)**: Each offspring replaces the first worst solution in the parent population.

#### Evolutionary Algorithms Implementation

Two evolutionary algorithms are implemented, each utilizing different crossover methods. These methods are properly documented and can be tested if need be:

- **Single Point Crossover Algorithm (`evoALg_singlecross`)**
- **Edge Crossover Algorithm (`evoALg_edgecross`)**

These algorithms run a standard evolutionary process for a specified number of iterations, employing different crossover types and survivor selection strategies to optimize solutions for the TSP.