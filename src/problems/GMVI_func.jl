struct GMVIProblem
    d
    func_value
    operator_func
    g_func

    function GMVIProblem(operator_func, g_func)

        function func_value(x::Vector{Float64})
            operator_func.func_value(x) + g_func.func_value(x)
        end
        
        new(operator_func.d + operator_func.n, func_value, operator_func, g_func)
    end
end
