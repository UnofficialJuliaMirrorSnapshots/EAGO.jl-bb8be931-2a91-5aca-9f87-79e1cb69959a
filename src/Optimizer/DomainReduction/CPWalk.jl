# Performs constraint walking on nonlinear terms
function cpwalk(x::Optimizer, n::NodeBB)

    # runs at midpoint bound
    midx = (n.upper_variable_bounds + n.lower_variable_bounds)/2.0

    # set working node to n, copies pass parameters from EAGO optimizer
    x.working_evaluator_block.evaluator.current_node = n
    x.working_evaluator_block.evaluator.has_reverse = true

    # Run forward-reverse pass and retreive node
    feas = forward_reverse_pass(x.working_evaluator_block.evaluator, midx)
    n.lower_variable_bounds[:] = x.working_evaluator_block.evaluator.current_node.lower_variable_bounds
    n.upper_variable_bounds[:] = x.working_evaluator_block.evaluator.current_node.upper_variable_bounds

    # resets forward reverse scheme for lower bounding problem
    x.working_evaluator_block.evaluator.has_reverse = false

    return feas
end
