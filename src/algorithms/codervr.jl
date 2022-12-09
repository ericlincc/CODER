"""
Implementation of CODER-VR version.
"""


struct CODERVRParams
    L::Float64
    M::Float64  # Original Lipschitz smoothness
    γ::Float64
    K::Int64  #  VR parameter 
end


function codervr(
    problem::GMVIProblem,
    exitcriterion::ExitCriterion,
    parameters::CODERVRParams;
    x₀=nothing
    )

    # Init of CODER
    L, M, γ, K = parameters.L, parameters.M, parameters.γ, parameters.K
    a, A = 0.0, 0.0
    β = 2 * M / sqrt(K)
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end


    x̂, x̂₋₁ = copy(x₀), copy(x₀); x₀₀, x₀₋₁ = copy(x₀), copy(x₀); x_out = copy(x₀)
    y₀₀, y₀₋₁ = copy(x₀), copy(x₀)
    z₀₀, z₀₋₁, q₀₀ = zeros(problem.d), zeros(problem.d), zeros(problem.d)
    x̂_sum, x̃_sum, x_out_sum = zeros(problem.d), zeros(problem.d), zeros(problem.d)
    m = problem.d  # Assume that each block is simply a coordinate for now
    
    
    # Run init
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag

        x̂₋₁ .= x̂  # x₋₁₀ .= x₀₀; 
        x̂_sum .= 0; x̃_sum .= 0
        
        # Step 4
        μ = problem.operator_func.func_map(x̂₋₁)

        # Step 5
        a₋₁ = a; A₋₁ = A
        if a₋₁ == 0.0
            a = min(sqrt(K) / (8 * M), K / (8 * L))
        else
            a = min((1 + γ / β) * a₋₁, (1 + A₋₁ * γ) * min(sqrt(K) / (8 * M), K / (8 * L)))
        end
        A = A₋₁ + a

        for _ in 1:K
            x₀₋₁ .= x₀₀; y₀₋₁ .= y₀₀; z₀₋₁ .= z₀₀

            y₀₀ .= x₀₋₁
            for j in 1:m
                # Step 8
                if j >= 2
                    y₀₀[j-1] = x₀₀[j-1] 
                end

                # Step 9
                t = rand(1:problem.operator_func.n)

                # Step 10
                # q₀₀[j] = F_y₀₀[t][j] - F_x̂₋₁[t][j] + μ[j] + a₀₋₁ / a * (F_x₀₋₁[t][j] - F_y₀₋₁[t][j]) + β * (x₀₋₁[j] - x̂₋₁[j])
                if j == 1
                    a₀₋₁ = a₋₁
                else
                    a₀₋₁ = a
                end
                F = problem.operator_func.func_map_block_sample
                q₀₀[j] = F(j, t, y₀₀) - F(j, t, x̂₋₁) + μ[j] + a₀₋₁ / a * (F(j, t, x₀₋₁) - F(j, t, y₀₋₁)) + β * (x₀₋₁[j] - x̂₋₁[j])

                # Step 11
                z₀₀[j] = z₀₋₁[j] + a * q₀₀[j]

                # Step 12
                x₀₀[j] = problem.g_func.prox_opr_block(j, x₀[j] - z₀₀[j] / K, A / K)
            end
            x̂_sum .+= β / (β + γ) * x₀₋₁ + γ / (β + γ) * x₀₀
            x̃_sum .+= x₀₀
        end
        

        # Step 16
        x̂ = 1 / K * x̂_sum
        x̃ = 1 / K * x̃_sum
        x_out_sum .+= a * x̃

        iteration += m * K
        if iteration % (m * exitcriterion.loggingfreq) == 0
            x_out = 1 / A * x_out_sum

            elapsedtime = time() - starttime
            optmeasure = problem.func_value(x_out)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end
    results, x_out, x₀₀
end
