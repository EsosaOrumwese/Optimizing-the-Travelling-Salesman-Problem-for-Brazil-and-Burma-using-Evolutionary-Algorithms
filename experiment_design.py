import numpy as np
import pandas as pd
import progressbar
from evoAlg_operators  import evoALg_edgecross, evoALg_singlecross

# defining experiment design functions for both CRSV functions

def singleCRSV(pop_size,tourn_size, cost_matrix, seed):
      '''
      Performs len(seed) number of trails on the evolutionary algorithm which varies the mutation operator and replacement operator 
      but keeps the pop_size and tourn_size constant all done with the single point crossover function.
      Parameters:
            pop_size: population size
            tourn_size: tournament size
            cost_matrix: cost matrix containing cost for intercity travel
            seed: Array of seeds for each trial.
      Returns:
            A dataframe containing the fitness results per iteration for each combination for each seed and a dataframe which returns the best solution, 
            its fitness and execution time for each combination for each seed.
      '''
      
      N_iter = 10000 # number of iterations for each algorithm

      # init empty dfs with mut_types and replace_types
      # this data is just for us to get the average fitness and plot convergence curves
      df_swap1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'swap', 'repl_func':'1st Weakest'})
      df_swapWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'swap', 'repl_func':'Weakest'})
      df_insert1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'insert', 'repl_func':'1st Weakest'})
      df_insertWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'insert', 'repl_func':'Weakest'})
      df_inversion1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'inversion', 'repl_func':'1st Weakest'})
      df_inversionWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'inversion', 'repl_func':'Weakest'})
      df_scramble1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'scramble', 'repl_func':'1st Weakest'})
      df_scrambleWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'scramble', 'repl_func':'Weakest'})

      # dataframe for storing best solution and execution time
      best_results = pd.DataFrame(columns=['seed','mut_type','repl_type','solution','fitness','exec_time'])

      ## Progress bar
      print("")
      print("Pop: {}, Tourn: {}, CRSV: SinglePt".format(pop_size, tourn_size))
      #widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar()]
      #progress = progressbar.ProgressBar(widgets=widgets, maxval = N_iter*len(seed)*8).start()
      evals = int(0)

      for N in seed:
            # - - - - - - - S W A P   M U T A T I O N - - - - - - - - -
            # SWAP mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'swap', 'replace 1st worst', set_seed=N)
            df_swap1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'swap', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # SWAP mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'swap', 'replace worst', set_seed=N)
            df_swapWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'swap', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)
      

            # - - - - - - - I N S E R T   M U T A T I O N - - - - - - - - -
            # INSERT mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'insert', 'replace 1st worst', set_seed=N)
            df_insert1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df
            
            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'insert', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # INSERT mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'insert', 'replace worst', set_seed=N)
            df_insertWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'insert', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)

            
            # - - - - - - - I N V E R S I O N   M U T A T I O N - - - - - - - - -
            # INVERSION mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'inversion', 'replace 1st worst', set_seed=N)
            df_inversion1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df
            
            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'inversion', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # INVERSION mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'inversion', 'replace worst', set_seed=N)
            df_inversionWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'inversion', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # - - - - - - - S C R A M B L E   M U T A T I O N - - - - - - - - -
            # SCRAMBLE mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'scramble', 'replace 1st worst', set_seed=N)
            df_scramble1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df
            
            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'scramble', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # SCRAMBLE mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_singlecross(cost_matrix, pop_size, tourn_size, 'scramble', 'replace worst', set_seed=N)
            df_scrambleWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'scramble', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)

            print("Seed {} complete!".format(N))

      # add CRSV type to col
      all_dfs = pd.concat([df_swap1Wk,df_swapWeak,df_insert1Wk,df_insertWeak,df_inversion1Wk,df_inversionWeak,df_scramble1Wk,df_scrambleWeak], ignore_index=True)
      all_dfs['crsv_type'] = 'Single Point'
      best_results['crsv_type'] = 'Single Point'

      # end progress bar
      #progress.finish()

      # return dataframes
      return all_dfs, best_results


def edgeCRSV(pop_size,tourn_size, cost_matrix, seed):
      '''
      Performs len(seed) number of trails on the evolutionary algorithm which varies the mutation operator and replacement operator 
      but keeps the pop_size and tourn_size constant all done with the edge crossover function.
      Parameters:
            pop_size: population size
            tourn_size: tournament size
            cost_matrix: cost matrix containing cost for intercity travel
            seed: Array of seeds for each trial.
      Returns:
            A dataframe containing the fitness results per iteration for each combination for each seed and a dataframe which returns the best solution, 
            its fitness and execution time for each combination for each seed.
      '''
      
      N_iter = 10000 # number of iterations for each algorithm

      # init empty dfs with mut_types and replace_types
      # this data is just for us to get the average fitness and plot convergence curves
      df_swap1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'swap', 'repl_func':'1st Weakest'})
      df_swapWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'swap', 'repl_func':'Weakest'})
      df_insert1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'insert', 'repl_func':'1st Weakest'})
      df_insertWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'insert', 'repl_func':'Weakest'})
      df_inversion1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'inversion', 'repl_func':'1st Weakest'})
      df_inversionWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'inversion', 'repl_func':'Weakest'})
      df_scramble1Wk = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'scramble', 'repl_func':'1st Weakest'})
      df_scrambleWeak = pd.DataFrame({'iteration':np.arange(N_iter),'mut_func':'scramble', 'repl_func':'Weakest'})

      # dataframe for storing best solution and execution time
      best_results = pd.DataFrame(columns=['seed','mut_type','repl_type','solution','fitness','exec_time'])

      ## Progress bar
      print("")
      print("Pop: {}, Tourn: {}, CRSV: Edge".format(pop_size, tourn_size))
      #widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar()]
      #progress = progressbar.ProgressBar(widgets=widgets, maxval = N_iter*len(seed)*8).start()
      evals = int(0)

      for N in seed:
            # - - - - - - - S W A P   M U T A T I O N - - - - - - - - -
            # SWAP mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'swap', 'replace 1st worst', set_seed=N)
            df_swap1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'swap', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # SWAP mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'swap', 'replace worst', set_seed=N)
            df_swapWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'swap', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)
      

            # - - - - - - - I N S E R T   M U T A T I O N - - - - - - - - -
            # INSERT mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'insert', 'replace 1st worst', set_seed=N)
            df_insert1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df
            
            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'insert', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # INSERT mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'insert', 'replace worst', set_seed=N)
            df_insertWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'insert', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)

            
            # - - - - - - - I N V E R S I O N   M U T A T I O N - - - - - - - - -
            # INVERSION mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'inversion', 'replace 1st worst', set_seed=N)
            df_inversion1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df
            
            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'inversion', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # INVERSION mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'inversion', 'replace worst', set_seed=N)
            df_inversionWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'inversion', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # - - - - - - - S C R A M B L E   M U T A T I O N - - - - - - - - -
            # SCRAMBLE mutation with 1st weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'scramble', 'replace 1st worst', set_seed=N)
            df_scramble1Wk[N] = pd.Series(best_fit) # adding the fitness/iter to df
            
            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'scramble', 'repl_type':'1st Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)


            # SCRAMBLE mutation with Weakest replacement
            best_fit, exec_time, best_sol = evoALg_edgecross(cost_matrix, pop_size, tourn_size, 'scramble', 'replace worst', set_seed=N)
            df_scrambleWeak[N] = pd.Series(best_fit) # adding the fitness/iter to df

            # concating the best solution and execution time of algorithm to df
            _ = pd.DataFrame({'seed':N, 'mut_type':'scramble', 'repl_type':'Weakest','solution': [best_sol['solution']], 'fitness': best_sol['fitness'], 'exec_time':exec_time})
            best_results = pd.concat([best_results, _], ignore_index=True)
            ## progress bar update
            evals +=N_iter
            #progress.update(evals)

            print("Seed {} complete!".format(N))

      # add CRSV type to col
      all_dfs = pd.concat([df_swap1Wk,df_swapWeak,df_insert1Wk,df_insertWeak,df_inversion1Wk,df_inversionWeak,df_scramble1Wk,df_scrambleWeak], ignore_index=True)
      all_dfs['crsv_type'] = 'Edge'
      best_results['crsv_type'] = 'Edge'

      # end progress bar
      #progress.finish()

      # return dataframes
      return all_dfs, best_results