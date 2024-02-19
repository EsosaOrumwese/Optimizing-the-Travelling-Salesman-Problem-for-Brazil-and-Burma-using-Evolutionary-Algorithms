import numpy as np
import pandas as pd
import progressbar
from collections import Counter
import time
import random

# population initialization
def init_pop(cost_matrix, n_sol=50):
      '''Generates a number of random solutions each with associated fitness'''
      # empty list of solutions
      population = []

      # get the number of cities in the matrix
      n_cities = cost_matrix.shape[0]

      c=0 #counter
      while c < n_sol:
            sol = np.random.choice(a=n_cities, size=n_cities, replace=False) # randomly shuffle up the city numbers to create a solution
            population.append(sol)
            c+=1
      
      # calculate fitness for each solution in population
      pop_with_fitness_pairs = [{'solution':solution, 'fitness':fitness(cost_matrix,solution)} for solution in population]
      pop_with_fitness_pairs = np.array(pop_with_fitness_pairs)
      
      return pop_with_fitness_pairs

# fitness function
def fitness(cost_matrix, solution):
      '''Calculates the fitness for a single solution'''
      cost=0 
      elem_1 = cost_matrix[solution[0]][solution[-1]] #gets cost of from last to first city

      # skip the first city
      for val in np.arange(1,len(solution)):
            cost+= cost_matrix[solution[val-1]][solution[val]] # adds cost of current city with the previous one
      
      return cost+elem_1


#------- PARENT SELECTION ----------
def tournament_selection(solution_fitness_pairs, tournament_size=5, n_tourn=2):
      '''
      Performs tournament selection on a population of solutions. Randomly samples `tournament_size`
      of solutions without replacement and gets the fittest as the parent. The tournament is performed `n_tourn` times.
      Returns two parent solutions.
      '''
      # confirm if solution_fitness pairs is an np.ndarray object which allows for indexing of
      # the array
      #if not(isinstance(solution_fitness_pairs, np.ndarray)):
      #      solution_fitness_pairs = np.array(solution_fitness_pairs)

      mating_pool = [{'solution':[],'fitness':10e10},{'solution':[],'fitness':10e10}] #for storing parent solutions
      n_sol = len(solution_fitness_pairs)
      
      c=0 #counter
      while c<n_tourn:
            for i in np.arange(tournament_size):
                  idx = np.random.randint(0,n_sol) # randomly selects which idx to choose
                  candidate = solution_fitness_pairs[idx]
                  if candidate['fitness'] < mating_pool[c]['fitness']:
                        mating_pool[c] = candidate
            c+=1
      
      return np.array(mating_pool)
         
#------ CROSSOVER FUNCTIONS ----------
# create helper function for swapping and replacing tail
def swapTail(child, child_t, parent_t, crossover_point):
      '''
      child: The full child solution
      child_t: Just the tail of the child
      parent_t: The tail of the parent of the child (the one responsible for its head)
      '''
      swappedOut = [] #list to contain the elements that have been swapped out
      child_h = child[:crossover_point] # get child head
      bool = child_t==parent_t # comparing intended child tail with its head's parent tail

      # Now is time to replace
      cp = crossover_point #just so I don't lose it

      for idx, b in enumerate(bool):
            if b: 
                  child[cp] = child_t[idx] #if both vals in tail are similar, just replace
            else:
                  child[cp] = child_t[idx] #if both vals in tails are diff. store the swapped out one
                  swappedOut.append(parent_t[idx])
            cp+=1
      # Now it's time to replace the duplicates in child with values in swappedOut
      dupes = list(set(child_h) & set(child_t)) # this tells us the values that are similar in both
      dupes_idx = np.where(np.isin(child_h, dupes))[0] # gets the indices for the dupes in child_h
      
      # return child if there are no duplicates
      if len(dupes)==0:
            return child
      
      # remove all elements in swappedOut that are alreading in child
      swappedOut = np.array(swappedOut)[np.where(~(np.isin(swappedOut,child)))]

      # replacing the duplicates child with elements in swappedOut
      counter=0
      for idx in dupes_idx:
            child[idx] = swappedOut[counter]
            counter+=1

      return child

## single point crossover
def singlePoint_crossover(mating_pool, cost_matrix, crsv_rate):
      '''Performs single point crossover at any single random point within the length of the solution array'''
      # ensure mating pool contains 2 parents
      try:
            len(mating_pool) == 2
      except:
            raise Exception("Function requires two parents in mating pool")
      
      p_crsv = random.random()

      # checks for probability of crossover
      if p_crsv > crsv_rate:
            return mating_pool

      # extract parent solution
      parent1 = mating_pool[0]['solution']
      parent2 = mating_pool[1]['solution']

      # get length of the parent
      parent_length = len(parent1)

      # initialize child solutions
      child1 = np.zeros(parent_length, dtype=int)
      child2 = np.zeros(parent_length, dtype=int)

      # randomly select a crossover point (excluding the first and the last positions)
      crossover_point = np.random.randint(1,parent_length-1)

      # apply crossover to create children
      ## transfers the head from the split to the children
      child1[:crossover_point] = parent1[:crossover_point]
      child2[:crossover_point] = parent2[:crossover_point]

      # get tails of parent and children
      parent1_t, parent2_t = parent1[crossover_point:], parent2[crossover_point:]
      child1_t, child2_t = parent2_t.copy(), parent1_t.copy()

      # get valid children by eliminating duplicates
      child1 = swapTail(child1, child1_t, parent1_t, crossover_point)
      child2 = swapTail(child2, child2_t, parent2_t, crossover_point)
      # fitness of children
      fit_child1 = fitness(cost_matrix,child1)
      fit_child2 = fitness(cost_matrix,child2)
      
      return [{'solution':child1, 'fitness':fit_child1},
              {'solution':child2, 'fitness':fit_child2}]

## edge crossover
def generate_edge_table(parent1, parent2):
      '''Generates edge table given both parents'''
      # Create an empty dictionary to store the edge table
      edge_table = {}

      # Initialize the edge table for each element
      for element in parent1:
            edge_table[element] = []

      # Populate the edge table with edges from parent 1
      for i in range(len(parent1)):
            # gets the ith element of parent1
            element1 = parent1[i]
            # gets the next element by i. If i is the last element then it picks 
            # the first index as it's neighbour `9 % 9 = 0` but `2 % 9 = 2`
            element2 = parent1[(i + 1) % len(parent1)]  # Circular list

            # assigns each edge to its respective element. For example if 1 is the edge of 2 then 2
            # edge of 1. While the loop checks the edge on the right, this ensures the edge on the left
            # is also included in the edge table.
            edge_table[element1].append(element2)
            edge_table[element2].append(element1)

      # Populate the edge table with edges from parent 2
      for i in range(len(parent2)):
            element1 = parent2[i]
            element2 = parent2[(i + 1) % len(parent2)]  # Circular list

            edge_table[element1].append(element2)
            edge_table[element2].append(element1)

      return edge_table

def element_selector(edge_table ,curr_element=None, p=0.9):
      
      '''
      Given the current element and the edge table it examines the edge table
      and picks the next element based on mostly common edge and other times the shortest list.
      Ties are split at random.

      Parameters:
            curr_element: The current element. References to it are removed from the edge_table. Default is None for the first initialization
            edge_table: Contains the list of edges per element in both parents
            p = the probability that the next element will be chosen solely based on common edge first. If not, it is chosen
                  based on shortest list. This is done to prevent premature convergence. Default value is 0.9
      Returns:
            Previous element,
            Current element,
            and updated dictionary
      '''

      if curr_element is None:
            curr_element = random.choice(list(edge_table.keys()))

      if len(edge_table) == 1:
            selected_key = list(edge_table.keys())[0]
            edge_table = {}
            return curr_element, selected_key, edge_table

      choices_keys = [key for key, values in edge_table.items() if curr_element in values]
      updated_edge_table = {k: v for k, v in edge_table.items() if k != curr_element}
      choices = {k: v for k, v in updated_edge_table.items() if k in choices_keys}

      if not choices:
            updated_edge_table = {k: [v for v in vs if v != curr_element] for k, vs in updated_edge_table.items()}
            selected_key = random.choice(list(updated_edge_table.keys()))
            return curr_element, selected_key, updated_edge_table

      length = []
      common_edge = []
      for key, values in choices.items():
            c = Counter(values)
            if c[curr_element] == 2:
                  common_edge.append(key)
            length.append(len(set(values)))

      if random.random() <= p and common_edge:
            selected_key = random.choice(common_edge)
            updated_edge_table = {k: [v for v in vs if v != curr_element] for k, vs in updated_edge_table.items()}
            return curr_element, selected_key, updated_edge_table

      min_length = min(length)
      shortest_keys = [key for key, values in choices.items() if len(set(values)) == min_length]
      selected_key = random.choice(shortest_keys)

      updated_edge_table = {k: [v for v in vs if v != curr_element] for k, vs in updated_edge_table.items()}

      return curr_element, selected_key, updated_edge_table


def edge_crossover(mating_pool,cost_matrix,crsv_rate,p=0.9):
      '''
      Implements edge-3 crossover which is designed to ensure that common edges are preserved. 
      This operator only produces one offspring.
      Parameters:
            mating pool: contains 2 parent solutions chosen via a selection method (tournament by default)
            cost_matrix: contains the cost matrix
            crsv: crossover rate. If probability is greater, it returns the first parent
            p = the probability that the next element will be chosen solely based on common edge first. If not, it is chosen
                based on shortest list. This is done to prevent premature convergence. Default value is 0.9
      Returns:
            One child      
      '''
      # checks probability of crossover
      p_crsv = random.random()
      if p_crsv > crsv_rate:
            return mating_pool[0]

      # get parent solutions
      parent1 = mating_pool[0]['solution']
      parent2 = mating_pool[1]['solution']

      # initialize child zero array
      #child = []
      child = np.zeros_like(parent1)

      # generate edge table
      edge_table = generate_edge_table(parent1,parent2)

      # initialize selected_key as None
      selected_key = None

      # child construction
      c = 0
      while c < len(parent1):
            curr_element, selected_key, edge_table = element_selector(edge_table,selected_key,p=p)
            #child.append(curr_element)
            child[c] = curr_element
            c+=1

      # fitness of children
      fit_child = fitness(cost_matrix,child)
      
      return {'solution':np.array(child), 'fitness':fit_child}

#------- MUTATION ---------
# swap mutation
def swap_mutation(solution_fitness_pair, cost_matrix, mut_rate):
      '''Two positions in the solution are selected at random and their values are swapped. 
      Function returns mutated solution and fitness pair'''

      # checks for probability of mutation
      p_mut = random.random()
      if p_mut > mut_rate:
            return solution_fitness_pair


      # get solution
      solution = solution_fitness_pair['solution']

      # get two random positions in the solution
      positions = np.random.choice(np.arange(len(solution)), 2, replace=False)

      # swap values
      solution[positions[0]], solution[positions[1]] = solution[positions[1]], solution[positions[0]]

      # calculate fitness of new solution
      fit = fitness(cost_matrix, solution)

      return {'solution':np.array(solution), 'fitness':fit}

# insert mutation
def insert_mutation(solution_fitness_pair, cost_matrix, mut_rate):
      '''
      The positions are chosen at random and the element in the second occuring position is inserted
      directly after the first one. Function returns mutated solution and fitness pair.
      '''
      # checks for probability of mutation
      p_mut = random.random()
      if p_mut > mut_rate:
            return solution_fitness_pair

      # get solution
      solution = solution_fitness_pair['solution']

      # get two random positions in the solution
      n = len(solution)
      positions = np.random.choice(np.arange(1,n-1), 2, replace=False) # avpod choosing the ends
      positions.sort() # sorting the positions
      p1, p2 = positions[0], positions[1]

      # inserting element at p2 to after p1
      mutant = np.insert(solution, p1+1, solution[p2])  # np.insert inserts before index therefore idx+1 is necessary
      
      # delete duplicate solution
      mutant = np.delete(mutant, p2+1)

      # calculate fitness of new solution
      fit = fitness(cost_matrix, mutant)

      return {'solution':np.array(mutant), 'fitness':fit}

# scramble mutation
def scramble_mutation(solution_fitness_pair, cost_matrix, mut_rate):
      '''
      Here, the entire solution or some randomly chosen subset of values within it have their positions 
      scrambled. Function returns mutated solution and fitness pair.
      '''
      # checks for probability of mutation
      p_mut = random.random()
      if p_mut > mut_rate:
            return solution_fitness_pair

      # get solution
      solution = solution_fitness_pair['solution']

      # get two random positions in the solution
      n = len(solution)
      positions = np.random.choice(np.arange(n), 2, replace=False)
      positions.sort()  
      p1, p2 = positions[0], positions[1]

      # account for situations where the first or/and last index are chosen
      if (p1-1) not in np.arange(n):
            head = np.array([],dtype=int)
      else:
            head = solution[:p1]
      if (p2+1) not in np.arange(n):
            tail = np.array([],dtype=int)
      else:
            tail = solution[p2+1:]
      
      # scramble the middle
      middle = solution[p1:p2+1]
      np.random.shuffle(middle)
      
      # concatenate parts together
      mutant = np.concatenate((head,middle,tail))
      
      # calculate fitness of new solution
      fit = fitness(cost_matrix, mutant)

      return {'solution':mutant, 'fitness':fit}

# inversion mutation
def inversion_mutation(solution_fitness_pair, cost_matrix, mut_rate):
      '''
      Ihis mutation works by randomly selecting two positions in the solution and reversing the order in which
      the values appear between those positions. Function returns mutated solution and fitness pair.
      '''
      # checks for probability of mutation
      p_mut = random.random()
      if p_mut > mut_rate:
            return solution_fitness_pair

      # get solution
      solution = solution_fitness_pair['solution']

      # get two random positions in the solution
      n = len(solution)
      positions = np.random.choice(np.arange(n), 2, replace=False)
      positions.sort()  
      p1, p2 = positions[0], positions[1]

      # account for situations where the first or/and last index are chosen
      if (p1-1) not in np.arange(n):
            head = np.array([],dtype=int) #has to be an int empty set so that the concat form will be int and not float
      else:
            head = solution[:p1]
      if (p2+1) not in np.arange(n):
            tail = np.array([],dtype=int)
      else:
            tail = solution[p2+1:]
      
      # inverse the middle section
      middle = solution[p1:p2+1]
      middle = middle[::-1]
      
      # concatenate parts together
      mutant = np.concatenate((head,middle,tail))
      
      # calculate fitness of new solution
      fit = fitness(cost_matrix, mutant)

      return {'solution':mutant, 'fitness':fit}

#------ SURVIVOR SELECTION -----------
def replace_worst(child_arr, parent_population):
      '''
      Replaces the offsprings replace the worst solutions in the parent population.
      Parameters:
            child_arr: array of child solutions. lambda = size of child_arr
            parent_population: parent population. mu = size of parent_population
      Returns:
            Next generation of parent population
      '''
      mu = len(parent_population)
      lambd = len(child_arr) 
      
      # combine arr
      comb_arr = np.concatenate([parent_population,child_arr])

      # extract fitness values from the child_arr
      fitness_values = np.array([ind['fitness'] for ind in comb_arr])

      # find top mu indices based on fitness
      best_indices = np.argsort(fitness_values)[:mu]
      best_indices = np.sort(best_indices) # to preserve the order in which those best solutions appear

      # replace parent population 
      parent_population = comb_arr[best_indices]
      
      
      return parent_population

def replace_first_worst(child_arr, parent_population):
      '''
      Each offspring replaces the first worst solutions in the parent population.
      Parameters:
            child_arr: array of child solutions. lambda = size of child_arr
            parent_population: parent population. mu = size of parent_population
      Returns:
            Next generation of parent population
      '''
      mu = len(parent_population)
      lambd = len(child_arr) 

      c=0
      for i in np.arange(mu):
            if parent_population[i]['fitness'] >= child_arr[c]['fitness']:
                  parent_population[i] = child_arr[c]
                  c+=1
                  if c >= lambd:
                        return parent_population

      return parent_population

# ------ EVOLUTIONARY ALGORITHMS -----------
# using the singlepoint crossover 
def evoALg_singlecross(cost_matrix, pop_size, tourn_size, mut_type, repl_type, crsv_rate=1, mut_rate=1, set_seed=1, budget_allocate=False, n_iter=10000):
      '''
      Runs a standard evolutionary algorithm for a TSP problem for n_iter times and returns the best solution, mean best solution and execution after every iteration,
      and also the best solution fitness pair at the end of the iteration. Crossover type used is Single Crossover and survivor selection used is mu+lambda selection.
      If budget_allocate is True, then 80% of the iteration will be spent exploring with 20% spent on exploitation
      Parameters:
            cost_matrix: The matrix which contains the cost of travel to each of the cities
            pop_size: Desired population size
            crsv_rate: Crossover rate. ranges between 0 and 1. Default is 1.
            mut_rate: Mutation rate. ranges between 0 and 1. Default is 1.
            tourn_size: Tournament size
            mut_type: Mutation type. Values include 'swap', 'insert', 'inversion' and 'scramble'
            repl_type: Choose replacement function. Values include 'replace worst' and 'replace 1st worst'
            set_seed: Input the seed. This is required for reproducibility. Default value is 1
            budget_allocate: If True, then the mut_rate reduces to 0.3 and crsv increases to 1 after 80% of the iteration is done.
            n_iter: Number of iterations. Default value = 10000
      Returns:
            dictionary containing best fitness per iteration, mean fitness per iteration, and execution time per iteration
            execution time of the algorithm,
            best solution-fitness pair after n_iter iterations
      '''
      # set seed
      np.random.seed(set_seed)      
      
      # start time of algorithm
      start_time = time.time()

      ## Progress bar
      #widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar()]
      #progress = progressbar.ProgressBar(widgets=widgets, maxval = n_iter).start()
      

      child_size = 2 

      best_fitness = np.zeros(n_iter)
      #mean_fitness = np.zeros(n_iter)
      #execution_time_iter = np.zeros(n_iter)
      
      ## initialise population with random candidate solutions and evaluate each candidate
      population = init_pop(cost_matrix=cost_matrix, n_sol=pop_size) # returns solution-fitness of required population size
      
      evals = int(0)
      budget = 0.8*n_iter
      skip_lp = 0 # to avoid running changing mut_rate or crsv_rate repeatedly
      while evals < n_iter:
            #start_iter = time.time()

            # change crsv_rate and mut_rate if we've exceeded our budget allocation
            
            if (evals > budget) & (budget_allocate == True) & (skip_lp == 0):
                  mut_rate, crsv_rate = 1, 0.3
                  skip_lp = 1
            
            # generate offspring solutions
            child_arr = []
            count=0
            while count < child_size:
                  # select parents
                  mating_pool = tournament_selection(population,tourn_size)

                  # recombine and mutate pairs of parents
                  # recombination
                  child = singlePoint_crossover(mating_pool,cost_matrix,crsv_rate) # returns 2 children
                  count+=2
                  for sol in child:
                        # mutation
                        if mut_type == 'swap':
                              sol = swap_mutation(sol,cost_matrix,mut_rate)
                        elif mut_type == 'insert':
                              sol = insert_mutation(sol,cost_matrix,mut_rate)
                        elif mut_type == 'inversion':
                              sol = inversion_mutation(sol,cost_matrix,mut_rate)
                        elif mut_type == 'scramble':
                              sol = scramble_mutation(sol,cost_matrix,mut_rate)
                  child_arr = np.append(child_arr,child)

            # select individuals for the next generation
            if repl_type == 'replace 1st worst':
                  population = replace_first_worst(child_arr,population)
            else:
                  population = replace_worst(child_arr,population)

            #end_iter = time.time()

            # append performance measures
            fitness_values = np.array([sol_pair['fitness'] for sol_pair in population])
            best_fitness[evals] = fitness_values.min()
            #mean_fitness[evals] = fitness_values.mean()
            #execution_time_iter[evals] = end_iter-start_iter

            ## progress bar update
            evals +=1
            #progress.update(evals)
      #progress.finish()
      end_time = time.time()
      # extract fitness values from the child_arr
      fitness_values = np.array([ind['fitness'] for ind in population])
      # sort in increasing order of fitness value i.e. from best to worst
      best_index = np.argsort(fitness_values)[0]
      best_solution = population[best_index]
      algorithm_time = end_time-start_time

      return best_fitness, algorithm_time, best_solution

# using the edge crossover
def evoALg_edgecross(cost_matrix, pop_size, tourn_size, mut_type, repl_type, crsv_rate=1, mut_rate=1, set_seed=1, budget_allocate=False, n_iter=10000):
      '''
      Runs a standard evolutionary algorithm for a TSP problem for n_iter times and returns the best solution, mean best solution and execution after every iteration,
      and also the best solution fitness pair at the end of the iteration. Crossover type used is Edge Crossover and survivor selection used is mu+lambda selection.
      If budget_allocate is True, then 80% of the iteration will be spent exploring with 20% spent on exploitation
      Parameters:
            cost_matrix: The matrix which contains the cost of travel to each of the cities
            pop_size: Desired population size
            crsv_rate: Crossover rate. ranges between 0 and 1. Default value is 1
            mut_rate: Mutation rate. ranges between 0 and 1. Default value is 1
            tourn_size: Tournament size
            mut_type: Mutation type. Values include 'swap', 'insert', 'inversion' and 'scramble'
            repl_type: Choose replacement function. Values include 'replace worst' and 'replace 1st worst'
            set_seed: Input the seed. This is required for reproducibility. Default value is 1
            budget_allocate: If True, then the mut_rate reduces to 0.3 and crsv increases to 1 after 80% of the iteration is done.
            n_iter: Number of iterations. Default value = 10000
      Returns:
            dictionary containing best fitness per iteration, mean fitness per iteration, and execution time per iteration
            execution time of the algorithm,
            best solution-fitness pair after n_iter iterations
      '''
      # set seed
      np.random.seed(set_seed)

      # start time of algorithm
      start_time = time.time()

      ## Progress bar
      #widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar()]
      #progress = progressbar.ProgressBar(widgets=widgets, maxval = n_iter).start()
      

      child_size = 2 

      best_fitness = np.zeros(n_iter)
      #mean_fitness = np.zeros(n_iter)
      #execution_time_iter = np.zeros(n_iter)
      
      ## initialise population with random candidate solutions and evaluate each candidate
      population = init_pop(cost_matrix=cost_matrix, n_sol=pop_size) # returns solution-fitness of required population size
      
      evals = int(0)
      budget = 0.8*n_iter
      skip_lp = 0 # to avoid running changing mut_rate or crsv_rate repeatedly
      while evals < n_iter:
            #start_iter = time.time()

            # change crsv_rate and mut_rate if we've exceeded our budget allocation
            
            if (evals > budget) & (budget_allocate == True) & (skip_lp == 0):
                  mut_rate, crsv_rate = 1, 0.3
                  skip_lp = 1
            
            # generate offspring solutions
            child_arr = []
            count=0
            while count < child_size:
                  # select parents
                  mating_pool = tournament_selection(population,tourn_size)

                  # recombine and mutate pairs of parents
                  # recombination
                  child = edge_crossover(mating_pool,cost_matrix,crsv_rate) # returns 1 child
                  count+=1

                  # mutation
                  if mut_type == 'swap':
                        child = swap_mutation(child,cost_matrix,mut_rate)
                  elif mut_type == 'insert':
                        child = insert_mutation(child,cost_matrix,mut_rate)
                  elif mut_type == 'inversion':
                        child = inversion_mutation(child,cost_matrix,mut_rate)
                  elif mut_type == 'scramble':
                        child = scramble_mutation(child,cost_matrix,mut_rate)
                  child_arr = np.append(child_arr,child)

            # select individuals for the next generation
            if repl_type == 'replace 1st worst':
                  population = replace_first_worst(child_arr,population)
            else:
                  population = replace_worst(child_arr,population)

            #end_iter = time.time()

            # append performance measures
            fitness_values = np.array([sol_pair['fitness'] for sol_pair in population])
            best_fitness[evals] = fitness_values.min()
            #mean_fitness[evals] = fitness_values.mean()
            #execution_time_iter[evals] = end_iter-start_iter

            ## progress bar update
            evals +=1
            #progress.update(evals)
      #progress.finish()
      end_time = time.time()
      # extract fitness values from the child_arr
      fitness_values = np.array([ind['fitness'] for ind in population])
      # sort in increasing order of fitness value i.e. from best to worst
      best_index = np.argsort(fitness_values)[0]
      best_solution = population[best_index]
      algorithm_time = end_time-start_time

      return best_fitness, algorithm_time, best_solution