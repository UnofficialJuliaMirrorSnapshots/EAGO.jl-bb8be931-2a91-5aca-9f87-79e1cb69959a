"""
    solve_nlp!

Solves the branch and bound problem with the input EAGO optimizer object.
"""
function solve_nlp!(x::Optimizer)

  # initializes Flags
  TT = stdout
  UBDSaveFlag = false
  UBDTempSaveFlag = false
  PostSaveFlag = false
  if (~x.warm_start)
    x.current_iteration_count = 1
    x.current_node_count = 0
  end
  NumberOfCuts = 0

  # terminates when max nodes or iteration is reach, or when node stack is empty
  iterationcountinternal = 0
  while (x.termination_check(x))

    iterationcountinternal += 1

    # Fathom nodes with lower bound greater than global upper bound
    x.global_lower_bound = find_lower_bound(x)
    x.history.lower_bound[x.current_iteration_count] = x.global_lower_bound

    # Sets conditional save flags
    UBDSaveFlag = false
    UBDTempSaveFlag = false
    PostSaveFlag = false
    x.cut_iterations = 0

     # Selects node, deletes it from stack, prints based on verbosity
    CurrentKey,CurrentNode = x.node_selection(x);  x.current_node_count -= 1
    (x.verbosity >= 3) && print_node!(CurrentKey,CurrentNode) # Prints node in full verbosity mode

    # Solves preprocessing/LBD/UBD/postprocessing once to get timing right
    x.current_preprocess_info.feasibility = true
    x.current_postprocess_info.feasibility = true
    #=
    if (x.current_iteration_count == 1)
      OldLowerInfo = deepcopy(x.current_lower_info)
      OldUpperInfo = deepcopy(x.current_upper_info)
      OldPreprocessInfo = deepcopy(x.current_preprocess_info)
      OldPostprocessInfo = deepcopy(x.current_postprocess_info)
      tempNode = copy(CurrentNode)
      (x.verbosity >= 4) && println("started initial preprocessing")
      x.preprocess!(x,tempNode)
      (x.verbosity >= 4) && println("finished initial preprocessing")
      x.lower_problem!(x,tempNode)
      (x.verbosity >= 4) && println("finished initial lower problem")
      x.upper_problem!(x,tempNode)
      (x.verbosity >= 4) && println("finished initial upper problem")
      x.postprocess!(x,tempNode)
      (x.verbosity >= 4) && println("finished initial postprocessing")
      x.current_lower_info = OldLowerInfo
      x.current_upper_info = OldUpperInfo
      x.current_preprocess_info = OldPreprocessInfo
      x.current_postprocess_info = OldPostprocessInfo
    end
    x.current_preprocess_info.feasibility = true
    x.current_postprocess_info.feasibility = true
    =#

    # Performs prepocessing and times
    #redirect_stdout()
    PreprocessTime = @elapsed x.preprocess!(x,CurrentNode)
    #PreprocessTime = x.Preprocess(x,CurrentNode)
    x.history.preprocess_time[x.current_iteration_count] = x.history.preprocess_time[x.current_iteration_count-1]+PreprocessTime
    #redirect_stdout(TT)

    x.current_upper_info.feasibility = true

    if (x.current_preprocess_info.feasibility)
      # solves & times lower bounding problem
      LowerProblemTime = @elapsed x.lower_problem!(x,CurrentNode)
      x.history.lower_bound[x.current_iteration_count] = x.global_lower_bound
      x.history.lower_time[x.current_iteration_count] = x.history.lower_time[x.current_iteration_count-1]+LowerProblemTime
      x.history.lower_count += 1
      print_results!(x,true)

      while x.cut_condition(x)
        x.add_cut!(x); x.cut_iterations += 1
        #redirect_stdout()
        LowerProblemTime = @elapsed x.lower_problem!(x,CurrentNode)
        x.history.lower_bound[x.current_iteration_count] = x.global_lower_bound
        x.history.lower_time[x.current_iteration_count] = x.history.lower_time[x.current_iteration_count-1]+LowerProblemTime
        x.history.lower_count += 1
        #redirect_stdout(TT)
        print_results!(x,true)
      end
      x.history.cut_count[x.current_iteration_count] = x.cut_iterations

      # checks for infeasibility stores solution
      if (x.current_lower_info.feasibility)
        if (~x.convergence_check(x))
          # Solves upper bounding problem
          UpperProblemTime = @elapsed x.upper_problem!(x,CurrentNode)
          #UpperProblemTime = 0.0
          #x.upper_problem!(x,CurrentNode)
          x.history.upper_time[x.current_iteration_count] = x.history.upper_time[x.current_iteration_count-1]+UpperProblemTime
          x.history.upper_count += 1
          UBDTempSaveFlag = true
          print_results!(x,false)

          # Stores information if better feasible upper bound is formed
          if (x.current_upper_info.feasibility)
            if (x.current_upper_info.value < x.global_upper_bound)
              x.feasible_solution_found = true
              x.first_solution_node = x.maximum_node_id
              x.solution_value = x.current_upper_info.value
              x.continuous_solution[:] = x.current_upper_info.solution
              x.history.upper_bound[x.current_iteration_count] = x.solution_value
              UBDSaveFlag = true
            end
          end

          # Performs and times post processing
          #redirect_stdout()
          PostprocessTime = @elapsed x.postprocess!(x,CurrentNode)
          x.history.postprocess_time[x.current_iteration_count] = x.history.postprocess_time[x.current_iteration_count-1]+PostprocessTime
          PostSaveFlag = true
          #redirect_stdout(TT)

          # Checks to see if the node
          if (x.current_postprocess_info.feasibility)
            if x.repeat_check(x,CurrentNode)
              single_storage!(x,CurrentNode)
            else
              Y1,Y2 = x.bisection_function(x,CurrentNode)
              x.node_storage!(x,Y1,Y2)
            end
          end
        elseif (~x.exhaustive_search && x.feasible_solution_found)
        end
      else
        x.history.upper_bound[x.current_iteration_count] = x.history.upper_bound[x.current_iteration_count-1]
      end
      fathom!(x)
    else
      x.current_lower_info.value = -Inf
      x.current_lower_info.feasibility = false
      x.current_upper_info.feasibility = false
    end
    (~UBDSaveFlag) && (x.history.upper_bound[x.current_iteration_count] = x.global_upper_bound)
    (~UBDTempSaveFlag) && (x.history.upper_time[x.current_iteration_count] = x.history.upper_time[x.current_iteration_count-1])
    (~PostSaveFlag) && (x.history.postprocess_time[x.current_iteration_count] = x.history.postprocess_time[x.current_iteration_count-1])
    x.history.count[x.current_iteration_count] = x.current_node_count
    print_iteration!(x)
    x.current_iteration_count += 1
  end

  x.objective_value = x.global_upper_bound
  if (x.optimization_sense == MOI.MAX_SENSE)
        x.objective_value *= -1.0
  end
  print_solution!(x)                              # Prints the solution
end
