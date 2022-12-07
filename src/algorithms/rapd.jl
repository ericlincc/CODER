# the basic goal is to make the coder impl available for both min-max and min settings
# In all codes, we treat the primal and dual variables in the same way for algorithm;
# the difference is hidden in the underlying impl

# so many call will make things complicated

# α: regularizer coeff for L2, β: regularizer coeff for L1


function elastic_net_function(x, α::Float64, β::Float64)
    return 0.5 * α * sum(x .* x) + β * sum(abs.(x))
end

function _prox_func(_x0, p1, p2)
    _value = 0.0
    if _x0 > p1
        _value = p2 * (_x0 - p1)
    elseif _x0 < -p1
        _value = p2 * (_x0 + p1)
    end
    _value
end


function elastic_net_prox(x, α::Float64, β::Float64, weight::Float64)
    p1 = weight * β
    p2 = 1.0 / (1.0 + weight * α)
    return _prox_func(x, p1, p2)
end

function rapd(B, x0, y0, L, epochs, lam1, lam2, exitcriterion)
    # obj value
    # obj(x) = loss(x[1:dim]) + regularizer(x[1:dim])
    clock_start = now()
    clock_cnt() = now() - clock_start
    # Record
    # func_value = zeros(0)
    # time = zeros(0)
    # A = 1.0/(2 * L)
    x = deepcopy(x0)
    y = deepcopy(y0)
    θ = 1
    τ = 1.0/L
    σ = 1.0/L
    n, d = size(B)
    dBx = B * x / n
    s = dBx
    idx_seq = 1:d
    K = epochs * d


    # Eric records
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    #     init_optmeasure = problem.func_value(x₀)
    res =  1 .- (B * x)
    init_optmeasure = sum(max.(res, 0.0))/n + elastic_net_function(x, lam2, lam1)
    logresult!(results, 1, 0.0, init_optmeasure)



    while !exitflag

        y = max.(-1.0, min.(y + σ * (s .- 1/n), 0))
        i = rand(idx_seq)
        pre_xi = x[i]
        # lam2: reg for l2, lam1: reg for l1
        x[i] = elastic_net_prox(x[i] - τ * (B[:, i]' * y), lam2, lam1, τ)
        tmp =  B[:, i] * (x[i] - pre_xi)
        dBx += 1.0/n * tmp
        s = dBx + θ * d * tmp
        
        iteration += 1
        
        if iteration % (exitcriterion.loggingfreq * d) == 0
            res =  1 .- (B * x)
            optmeasure = sum(max.(res, 0.0))/n + elastic_net_function(x, lam2, lam1)
            
            elapsedtime = time() - starttime
            logresult!(results, iteration, elapsedtime, optmeasure)
            @info "epoch: $(div(iteration, d)), loss: $optmeasure"
            
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
            
        end
    end

    return results, x
end