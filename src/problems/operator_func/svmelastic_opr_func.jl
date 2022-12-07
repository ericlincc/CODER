struct SVMElasticOprFunc
    d::Int64  # Dimension of SVM variable  #TODO: Change d and n
    n::Int64  # Number of data samples
    A  # n by d feature matrix
    b  # Length n label vector
    func_value
    func_map::Function
    func_map_block::Function
    func_map_block_sample::Function

    function SVMElasticOprFunc(data::Data)
        d = size(data.features)[2]
        n = length(data.values)
        A = data.features
        b = data.values

        function func_value(x::Vector{Float64})
            @assert length(x) == d + n

            _x = x[1:d]
            res =  1 .- ((b .* A) * _x)
            sum(max.(res, 0.0)) / n
        end

        function func_map(x::Vector{Float64})
            @assert length(x) == d + n

            ret = zeros(d + n)
            _x = x[1:d]
            for t in 1:n
                ret[1:d] .+= x[d + t] * b[t] .* A[t, :]  # TODO: Performance of A[t, :] is bad. Use matrix multiplications
                ret[d+t] = - (b[t] * (A[t, :] ⋅ _x) - 1)   # TODO: Performance of A[t, :] is bad. Use matrix multiplications
            end

            ret / n
        end

        function func_map_block(j, x::Vector{Float64})
            @assert length(x) == d + n
            @assert 1 <= j <= d + n

            if j <= d
                ret = 0.0
                for t in 1:n
                    ret += x[d+t] * b[t] * A[t, j]
                end
                return ret / n
            else
                t = j - d
                return - (b[t] * (A[t, :] ⋅ x[1:d]) - 1) / n
            end
        end

        function func_map_block_sample(j, t, x::Vector{Float64})
            @assert length(x) == d + n
            @assert 1 <= j <= d + n
            @assert 1 <= t <= n

            if j <= d
                return x[d+t] * b[t] * A[t, j]
            elseif j - d == t
                return - (b[t] * (A[t, :] ⋅ x[1:d]) - 1)
            else
                return 0.0
            end
        end
        
        new(d, n, A, b, func_value, func_map, func_map_block, func_map_block_sample)
    end
end
