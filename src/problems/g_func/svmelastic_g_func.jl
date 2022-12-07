"""Implementation of the regularizer function, g(x), in GMVI in the reformulated SVM with elastic net."""


function _prox_func(_x0, p1, p2)
    _value = 0.0
    if _x0 > p1
        _value = p2 * (_x0 - p1)
    elseif _x0 < -p1
        _value = p2 * (_x0 + p1)
    end
    _value
end


struct SVMElasticGFunc
    λ₁::Float64  # LASSO parameter
    λ₂::Float64  # Ridge parameter
    d::Int64  # Dimension of SVM variable  #TODO: Change d and n
    n::Int64  # Number of data samples 
    func_value::Function
    prox_opr_block::Function


    function SVMElasticGFunc(d, n, λ₁, λ₂)

        function func_value(x::Vector{Float64})
            @assert length(x) == d + n

            _ret_1, _ret_2 = 0.0, 0.0
            for i in 1:d
                _ret_1 += abs(x[i])
                _ret_2 += x[i]^2
            end
            ret = λ₁ * _ret_1 + λ₂ / 2 * _ret_2

            _in = true
            for i in (d+1):(n+d)
                _in = _in && (-1.0 <= x[i] <= 0.0)
            end

            if !_in
                return - Inf
            end
            ret
        end

        function prox_opr_block(j, u, τ)
            @assert 1 <= j <= n + d

            if j <= d
                p1 = τ * λ₁
                p2 = 1.0 / (1.0 + τ * λ₂)
                return _prox_func(u, p1, p2)
            else
                return min(0.0, max(-1.0, u))
            end
        end
        
        new(λ₁, λ₂, d, n, func_value, prox_opr_block)
    end
end
