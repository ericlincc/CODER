"""
Implementation of CODER-VR version.
"""


struct CODERVRParams
    L::Float64
    M::Float64  # Original Lipschitz smoothness
    γ::Float64
    K::Int64  #  VR parameter 
end


function coder_vr(
    problem::GMVIProblem,
    exitcriterion::ExitCriterion,
    parameters::CODERParams;
    x₀=nothing
    )

    # Init of CODER
    L, M, γ, K = parameters.L, parameters.M, parameters.γ, parameters.K
    a₋₁, A₋₁ = 0, 0; a = min(sqrt(K) / (8 * M), K / (8 * L)); A = A
    β = 2 * M / sqrt(K)
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
        μ = problem.operator_func.func_map(x̂)

        # Step 5
        A₋₁ = A
        a₋₁ = a
        a = (1 + γ * A₋₁) / (2 * L)
        A = A₋₁ + a
        
        for k in 1:K
            for j in 1:m
                # Step 8
                p[j] = problem.operator_func.func_map_block(j, x)

                # Step 9
                t = rand(1:problem.operator_func.n)

                # Step 10
                q[j] = p[j] + a₋₁ / a * (F_x₋₁[j] - p₋₁[j])

                # Step 11
                z[j] = z₋₁[j] + a * q[j]

                # Step 12
                x[j] = problem.g_func.prox_opr_block(j, x₀[j] - z[j], A)
            end
        end

        # Step 15
        a₋₁ = a; A₋₁ = A
        a = min((1 + γ / β) * a₋₁, (1 + A₋₁ * γ) * min(sqrt(K) / (8 * M), K / (8 * L)))
        A = A₋₁ + a

        # Step 16
        x̂ = 1 / K * x̂_sum

        # Step 17


        # Step 18

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
