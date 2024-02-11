# this does not really work for the problem at hand



# Settings for Benders decomposition.
BENDERS_REL_OPTIMALITY_GAP = 0.01
BENDERS_AGGREGATE_CUTS = false
BENDERS_ADAPTIVE_REGION = false
BENDERS_ADAPTIVE_REGION_MULT = 1e-6

# Build all models (master & sub-problems).
pf_model = create_model_pf(nodes, lines; T_max=8, mode=(BENDERS_AGGREGATE_CUTS ? :aggregate_benders : :benders))
d_models = Dict(node => create_model_dispatch(node, assets, demands, availability; T_max=8, mode=(BENDERS_AGGREGATE_CUTS ? :aggregate_benders : :benders)) for node in nodes)

if BENDERS_ADAPTIVE_REGION
    @variable(pf_model, incumbent_aux_netpos[n=nodes, t=1:8])       # todo: fix time horizon

    @expression(pf_model, obj_region, sum((pf_model[:aux_netpos][n, t] - incumbent_aux_netpos[n, t]) ^ 2 for n in nodes for t in 1:8))    # todo: fix time horizon
    @objective(pf_model, Min, pf_model[:obj] + BENDERS_ADAPTIVE_REGION_MULT * obj_region)
end

# Start iteration.
println("=== STARTING BENDERS                                                                ===")
println("iteration   lb           ub           rel. gap     time (t)     time (p)     time (p16)")
time_start = time()
time_p = 0.0
time_p16 = 0.0
sub_results = Dict(node => (π=[], obj=0., time=0.) for node in nodes)
ub = Inf
for i in 1:50
    x = nothing
    if i % 5 != 0
        _tmp = @timed optimize!(pf_model)
        time_p += _tmp.time
        time_p16 += _tmp.time
        
        _tmp = @timed value.(pf_model[:aux_netpos])
        x = _tmp.value
        time_p += _tmp.time
        time_p16 += _tmp.time
    else
        x = Containers.DenseAxisArray(rand(-1500.0:100.0:1500.0, (452, 8)), pf_model[:aux_netpos].axes...)
    end

    # Update incumbent in adaptive region.
    if BENDERS_ADAPTIVE_REGION
        fix.(pf_model[:incumbent_aux_netpos], x.data; force=true)
    end

    for (node, _m) in d_models
        # Fix the current incumbent net-positions & solve.
        fix.(_m[:aux_netpos], x[node, :]; force=true)
        _t = @timed optimize!(_m)

        # Get results.
        sub_results[node] = (π=reduced_cost.(_m[:aux_netpos]), obj=objective_value(_m), time=_t.time)
    end
    time_p += maximum(res.time for (_, res) in sub_results)
    time_p16 += maximum(res.time for (_, res) in sub_results) * ceil(length(nodes) / 16)

    # Calculate bounds, ...
    lb = i % 5 == 0 ? 0 : (BENDERS_ADAPTIVE_REGION ? value(pf_model[:obj]) : objective_value(pf_model))
    ub = i % 5 == 0 ? Inf : (min(ub, value(pf_model[:obj] + sum(sub_results[n].obj for n in nodes))))
    rel_gap = (ub - lb) / ub

    print_iteration(i, lb, ub, rel_gap, time() - time_start, time_p, time_p16)
    if rel_gap < BENDERS_REL_OPTIMALITY_GAP
        println("Optimality gap reached.")
        break
    end

    # Add cut(s) to PF (master) problem.
    _tmp = @timed begin
        if BENDERS_AGGREGATE_CUTS
            @constraint(pf_model, pf_model[:aux_cost] >= sum(sub_results[n].obj + sub_results[n].π' * (pf_model[:aux_netpos][n, :] .- x[n, :]) for n in nodes))
        else
            @constraint(pf_model, [n=nodes], pf_model[:aux_cost][n] >= sub_results[n].obj + sub_results[n].π' * (pf_model[:aux_netpos][n, :] .- x[n, :]))
        end
    end
    time_p += _tmp.time
    time_p16 += _tmp.time
end

