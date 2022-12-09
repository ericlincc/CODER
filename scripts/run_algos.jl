# julia scripts/run_algo.jl <dataset> <algo> <Lipschitz> <gamma> (<oldLipschitz> <K>)

using CSV
using Dates
using LinearAlgebra
using Logging
using SparseArrays
using JLD2


BLAS.set_num_threads(1)


include("../src/problems/utils/data.jl")
include("../src/problems/utils/data_parsers.jl")
include("../src/problems/operator_func/svmelastic_opr_func.jl")
include("../src/problems/g_func/svmelastic_g_func.jl")
include("../src/problems/GMVI_func.jl")

include("../src/algorithms/utils/exitcriterion.jl")
include("../src/algorithms/utils/results.jl")

include("../src/algorithms/coder.jl")
include("../src/algorithms/codervr.jl")
include("../src/algorithms/rapd.jl")
include("../src/algorithms/pccm.jl")
include("../src/algorithms/prcm.jl")



DATASET_INFO = Dict([
    ("a1a", (123, 1605)),
    ("a9a", (123, 32561)),
    ("gisette", (5000, 6000)),
    ("news20", (1355191, 19996)),
    ("rcv1", (47236, 20242)),
])

# Parameters
# outputdir = "./run_results"
outputdir = "./run_results-lasso_ridge"
# outputdir = "./run_results-lasso"
elasticnet_λ₁ = 1e-4
elasticnet_λ₂ = 1e-4

dataset = ARGS[1]
d, n = DATASET_INFO[dataset]


if !haskey(DATASET_INFO, dataset)
    throw(ArgumentError("Invalid dataset name supplied."))
end
filepath = "../data/libsvm/$(dataset)"


# Exit criterion
maxiter = 1e12
maxtime = 1800
targetaccuracy = 1e-7
loggingfreq = 100
exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)


# Output
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS-sss")
# loggingfilename = "$(outputdir)/$(dataset)-$(ARGS[2])-$(join(ARGS[3:end], "_"))-execution_log-$(timestamp).txt"
# io = open(loggingfilename, "w+")
# logger = SimpleLogger(io)
outputfilename = "$(outputdir)/$(dataset)-$(ARGS[2])-$(join(ARGS[3:end], "_"))-output-$(timestamp).jld2"


# Problem instance instantiation
data = libsvm_parser(filepath, n, d)
F = SVMElasticOprFunc(data)
g = SVMElasticGFunc(d, n, elasticnet_λ₁, elasticnet_λ₂)
problem = GMVIProblem(F, g)

@info "timestamp = $(timestamp)"
@info "Completed initialization."

@info "--------------------------------------------------"
@info "Running on $(dataset) dataset."
@info "--------------------------------------------------"
@info "maxiter = $(maxiter)"
@info "maxtime = $(maxtime)"
@info "targetaccuracy = $(targetaccuracy)"
@info "loggingfreq = $(loggingfreq)"
@info "--------------------------------------------------"


if ARGS[2] == "CODER"
    @info "Running CODER..."

    L = parse(Float64, ARGS[3])
    γ = parse(Float64, ARGS[4])
    @info "Setting L=$(L), γ=$(γ)"

    coder_params = CODERParams(L, γ)
    output_coder = coder(problem, exitcriterion, coder_params)
    save_object(outputfilename, output_coder)
    @info "output saved to $(outputfilename)"

elseif ARGS[2] == "CODERVR"
    @info "Running CODERVR..."

    L = parse(Float64, ARGS[3])
    γ = parse(Float64, ARGS[4])
    M = parse(Float64, ARGS[5])
    K = parse(Float64, ARGS[6])
    @info "Setting L=$(L), γ=$(γ), M=$(M), K=$(K)"

    codervr_params = CODERVRParams(L, M, γ, K)
    output_coder = codervr(problem, exitcriterion, codervr_params)
    save_object(outputfilename, output_coder)
    @info "output saved to $(outputfilename)"

elseif ARGS[2] == "RAPD"
    @info "Running RAPD"

    L = parse(Float64, ARGS[3])
    @info "Setting L=$(L)"

    # TODO: Move these inputs into algo
    x0 = zeros(d)
    y0 = zeros(n)
    epochs = 1e12
    B = data.values .* data.features

    output_rapd = rapd(B, x0, y0, L, epochs, elasticnet_λ₁, elasticnet_λ₂, exitcriterion)
    save_object(outputfilename, output_rapd)
    @info "output saved to $(outputfilename)"

elseif ARGS[2] == "PCCM"
    @info "Running PCCM..."

    L = parse(Float64, ARGS[3])
    γ = parse(Float64, ARGS[4])
    @info "Setting L=$(L), γ=$(γ)"

    pccm_params = PCCMParams(L, γ)
    output_pccm = pccm(problem, exitcriterion, pccm_params)
    save_object(outputfilename, output_pccm)
    @info "output saved to $(outputfilename)"

elseif ARGS[2] == "PRCM"
    @info "Running PRCM..."

    L = parse(Float64, ARGS[3])
    γ = parse(Float64, ARGS[4])
    @info "Setting L=$(L), γ=$(γ)"

    prcm_params = PRCMParams(L, γ)
    output_prcm = prcm(problem, exitcriterion, prcm_params)
    save_object(outputfilename, output_prcm)
    @info "output saved to $(outputfilename)"

else
    @info "Wrong algorithm name supplied"
end
