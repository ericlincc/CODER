"""
Implementation of CODER basic version.
"""


struct PCCMParams
    L::Float64
    γ::Float64
end


function pccm(
    problem::GMVIProblem,
    exitcriterion::ExitCriterion,
    parameters::PCCMParams;
    x₀=nothing
    )

    # Init of CODER
    L, γ = parameters.L, parameters.γ
    a, A = 0, 0
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    x, x₋₁ = copy(x₀), copy(x₀)
    x_tilde_sum = zeros(problem.d); x_tilde = copy(x₀)
    p = problem.operator_func.func_map(x₀); p₋₁ = copy(p)
    z = zeros(problem.d); z₋₁ = copy(z); q = copy(z)
    m = problem.d  # Assume that each block is simply a coordinate for now
    
    
    # Run init
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag

        x₋₁ .= x; p₋₁ .= p; z₋₁ .= z
        
        # Step 4
        A₋₁ = A
        a₋₁ = a
        a = (1 + γ * A₋₁) / (2 * L)
        A = A₋₁ + a
        
        # F_x₋₁ = problem.operator_func.func_map(x₋₁)
        for j in 1:m
            # Step 6
            p[j] = problem.operator_func.func_map_block(j, x)

            # Step 7
            # q[j] = p[j] + a₋₁ / a * (F_x₋₁[j] - p₋₁[j])
            q[j] = p[j]

            # Step 8
            z[j] = z₋₁[j] + a * q[j]

            # Step 9
            x[j] = problem.g_func.prox_opr_block(j, x₀[j] - z[j], A)
        end

        x_tilde_sum .+= a .* x

        iteration += m
        if iteration % (m * exitcriterion.loggingfreq) == 0
            x_tilde = x_tilde_sum / A

            elapsedtime = time() - starttime
            optmeasure = problem.func_value(x_tilde)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results, x_tilde, x
end
