using Logging
import Random
using LinearAlgebra
using JuMP
import Gurobi

# Set seed and define number of timesteps.
Random.seed!(1234)


include("util.jl")
include("model.jl")


# Load grid data.
grid_data = load_grid_data()
nodes = grid_nodes(grid_data)
lines = grid_lines(grid_data)

# Load asset data.
demands = grid_demand(grid_data)
assets = grid_assets(grid_data)
availability = res_availability(assets)


function run_mono(_T::Int; verbose::Bool=true)
    model = create_original_model(nodes, lines, assets, demands, availability; T_max=_T)
    optimize!(model)

    obj = objective_value(model)
    time = solution_summary(model).solve_time

    verbose && @info "orig. model done" obj t = "$(round(time; digits=3)) sec"
    return (time, obj)
end

# ================================================================================
# ================================================================================
# ================================================================================

function run_admm(_T::Int, orig_obj::Float64; n_iter::Int=100, verbose::Bool=true)
    # Settings for ADMM.
    RHO = 1.0
    TAU = 2.0
    MU = 10.0

    # Build all models (master & sub-problems).
    pf_models = Dict(t => create_model_pf(nodes, lines; T_max=1, mode=:admm) for t in 1:_T);
    d_models = Dict(node => create_model_dispatch(node, assets, demands, availability; T_max=_T, mode=:admm) for node in nodes);

    # Initialize x_avg & λ using a first run.
    λ_exp = Dict((l, t) => 0.0 for l in keys(lines), t in 1:_T);
    λ_pf = Dict((n, t) => 0.0 for n in nodes, t in 1:_T);
    λ_d = Dict((n, t) => 0.0 for n in nodes, t in 1:_T);

    x_pf = Dict();
    x_exp = Dict();
    for (t, _m) in pf_models
        optimize!(_m)
        # @assert termination_status(_m) == OPTIMAL
        x_pf[t] = value.(_m[:aux_netpos][:, 1])
        x_exp[t] = value.(_m[:expansion])
    end

    x_d = Dict();
    for (node, _m) in d_models
        optimize!(_m)
        # @assert termination_status(_m) == OPTIMAL
        x_d[node] = value.(_m[:aux_netpos])
    end

    # todo: these should be `DenseAxisArray`s ...
    x_avg = Dict((n, t) => (x_pf[t][n] + x_d[n][t]) / 2.0 for n in nodes, t in 1:_T);
    x_exp_avg = Dict(l => sum(x_exp[t][l] for t in 1:_T) / length(1:_T) for (l, line) in lines);

    old_x_avg = copy(x_avg)
    old_x_exp_avg = copy(x_exp_avg)

    if verbose
        println("======================================== ADMM ======================================================")
        println("iteration   objective    time (t)     time (p)     primal (r)   dual (s)     ρ            rel-Δ obj.")
        println("----------------------------------------------------------------------------------------------------")
    end

    # Iterate.
    start_time = time()
    time_p = 0.0
    current_obj = Inf
    for i in 1:n_iter
        # Update & re-optimize PF (master) problem.
        _tmp = 0.0
        for (t, _m) in pf_models
            _tmp = max(_tmp, (@timed begin
                # todo: updating the objective instead of recreating it can save a lot (!) of time.
                @objective(
                    _m, Min,
                    (t == 1 ? _m[:obj] : 0.0) +
                    sum(λ_pf[(n, t)] * (_m[:aux_netpos][n, 1] - x_avg[(n, t)]) for n in nodes) +
                    RHO / 2.0 * sum((_m[:aux_netpos][n, 1] - x_avg[(n, t)]) ^ 2 for n in nodes) +
                    sum(λ_exp[(l, t)] * (_m[:expansion][l] - x_exp_avg[l]) for (l, line) in lines) +
                    RHO / 2.0 * sum((_m[:expansion][l] - x_exp_avg[l]) ^ 2 for (l, line) in lines)
                )
                optimize!(_m)
                # @assert termination_status(_m) == OPTIMAL
                x_pf[t] = value.(_m[:aux_netpos][:, 1])
                x_exp[t] = value.(_m[:expansion])
            end).time)
        end

        # Update dispatch (sub) problems.
        for (n, _m) in d_models
            _tmp = max(_tmp, (@timed begin
                if haskey(_m, :obj)
                    @objective(
                        _m, Min,
                        _m[:obj] +
                        sum(λ_d[(n, t)] * (_m[:aux_netpos][t] - x_avg[(n, t)]) for t in 1:_T) +
                        RHO / 2.0 * sum((_m[:aux_netpos][t] - x_avg[(n, t)]) ^ 2 for t in 1:_T)
                    )
                    optimize!(_m)
                    x_d[n] = value.(_m[:aux_netpos])
                end
            end).time)
        end

        time_p += _tmp

        # todo: Everything after this also costs (a little) time, so we should measure it (but it's done inefficiently, so I dropped it).

        # Update x_avg & λ.
        for n in nodes, t in 1:_T
            x_avg[(n, t)] = (x_pf[t][n] + x_d[n][t]) / 2.0
            λ_pf[(n, t)] += RHO * (x_pf[t][n] - x_avg[(n, t)])
            λ_d[(n, t)] += RHO * (x_d[n][t] - x_avg[(n, t)])
        end
        for l in keys(lines)
            x_exp_avg[l] = sum(x_exp[t][l] for t in 1:_T) / length(1:_T)
            for t in 1:_T
                λ_exp[(l, t)] += RHO * (x_exp[t][l] - x_exp_avg[l])
            end
        end

        # Update residuals.
        s = sqrt(norm(vcat(values(x_avg) .- values(old_x_avg), values(x_exp_avg) .- values(old_x_exp_avg))))
        r = sqrt(norm(vcat(
            [x_pf[t][n] - x_avg[(n, t)] for n in nodes for t in 1:_T],
            [x_d[n][t] - x_avg[(n, t)] for n in nodes for t in 1:_T],
            [x_exp[t][l] - x_exp_avg[l] for l in keys(lines) for t in 1:_T],
        )))
        old_x_avg = copy(x_avg)
        old_x_exp_avg = copy(x_exp_avg)

        current_obj = value(pf_models[1][:obj]) + sum(value(_m[:obj]) for (_, _m) in d_models if haskey(_m, :obj))
        verbose && print_iteration(i, current_obj, time() - start_time, time_p, r, s, RHO, abs((current_obj - orig_obj) / orig_obj))

        # Update RHO.
        if r > MU * s
            RHO *= TAU
        elseif s > MU * r
            RHO /= TAU
        end
    end

    return (time_p, current_obj)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Execute if called as a script.
    T = isempty(ARGS) ? 24 : parse(Int, ARGS[1])  # default: 24
    @info "Force compilation"

    # Force compilation.
    global_logger(ConsoleLogger(stderr, Logging.Error))
    orig_obj = run_mono(2; verbose=false)[2]
    run_admm(2, orig_obj; n_iter=1, verbose=false)
    global_logger(ConsoleLogger(stderr, Logging.Info))

    @info "Compilation done"
    @info "Start solving models" T

    # Solve the original model, and then run the ADMM algorithm.
    orig_obj = run_mono(T)[2]
    run_admm(T, orig_obj)
end
