const GRB_ENV = Gurobi.Env()


function create_original_model(nodes, lines, assets, demands, availability; T_max=168, pf=8.0)
    T = 1:T_max

    # Create a new model.
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_attribute(model, "Method", 2)
    set_attribute(model, "Crossover", 0)

    # Create variable: phase angles.
    @variable(model, θ[n=nodes, t=T])

    # Create variable: line flows & line expansion variables.
    @variable(model, flow[l=keys(lines), t=T])
    @variable(model, 0 <= expansion[l=keys(lines)] <= lines[l].expansion)

    # Create variable: generation of assets.
    @variable(model, (
        (assets[a].type == "storage" ? (-assets[a].capacity / pf) : 0.0) <=
        gen[a=keys(assets), t=T] <=
        (assets[a].capacity * (assets[a].type == "storage" ? (1.0/pf) : (assets[a].type == "res" ? availability[a][t] : 1.0)))
    ))

    # Create variable: storage state.
    storage_assets = [a for (a, asset) in assets if asset.type == "storage"]
    @variable(model, 0.0 <= storage[s=storage_assets, t=T] <= assets[s].capacity)

    # Create constraint: line flow limits.
    @constraint(model, flow_limits_lb[l=keys(lines), t=T], flow[l, t] >= -lines[l].capacity - expansion[l])
    @constraint(model, flow_limits_ub[l=keys(lines), t=T], flow[l, t] <= lines[l].capacity + expansion[l])

    # Create constraint: temporal storage state.
    prev_t(t) = t == 1 ? T[end] : t - 1
    @constraint(model, state[s=storage_assets, t=T], storage[s, t] == storage[s, prev_t(t)] - gen[s, prev_t(t)])

    # Create constraint: nodal balance.
    @constraint(
        model, nodal_balance[n=nodes, t=T], (
            sum(flow[l, t] for (l, line) in lines if line.to == n) - sum(flow[l, t] for (l, line) in lines if line.from == n) +
            sum(gen[a, t] for (a, asset) in assets if asset.node == n) -
            sum(demands[n][t])
        ) == 0.0
    )

    # Create objective: minimize cost.
    corrected_costs = Dict(
        a => assets[a].type == "thermal" ? asset.cost : (assets[a].type == "storage" ? 0.0 : asset.cost / 100.0)
        for (a, asset) in assets
    )
    @objective(model, Min, sum(gen[a, t] * corrected_costs[a] for (a, asset) in assets for t in T) + sum(expansion[l] * line.cost for (l, line) in lines))

    return model
end

function create_model_pf(nodes, lines; T_max=168, mode, max_netpos=25_000)
    T = 1:T_max

    # Create a new model.
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_attribute(model, "Method", 2)
    set_attribute(model, "Crossover", 0)

    # Create variable: phase angles.
    @variable(model, θ[n=nodes, t=T])

    # Create variable: line flows & line expansion variables.
    @variable(model, flow[l=keys(lines), t=T])
    @variable(model, 0 <= expansion[l=keys(lines)] <= lines[l].expansion)

    # Create variable: auxiliary nodal variables.
    @variable(model, -max_netpos <= aux_netpos[n=nodes, t=T] <= max_netpos)
    if mode == :aggregate_benders
        @variable(model, aux_cost >= 0)
    elseif mode == :benders
        @variable(model, aux_cost[n=nodes] >= 0)
    end

    # Create constraint: line flow limits.
    @constraint(model, flow_limits_lb[l=keys(lines), t=T], flow[l, t] >= -lines[l].capacity - expansion[l])
    @constraint(model, flow_limits_ub[l=keys(lines), t=T], flow[l, t] <= lines[l].capacity + expansion[l])

    # Create constraint: nodal balance.
    @constraint(
        model, nodal_balance[n=nodes, t=T], (
            sum(flow[l, t] for (l, line) in lines if line.to == n) - sum(flow[l, t] for (l, line) in lines if line.from == n) +
            aux_netpos[n, t]
        ) == 0.0
    )

    # Create objective: minimize cost.
    @expression(model, obj, sum(expansion[l] * line.cost for (l, line) in lines))
    if mode == :aggregate_benders
        @objective(model, Min, obj + aux_cost)
    elseif mode == :benders
        @objective(model, Min, obj + sum(aux_cost[n] for n in nodes))
    elseif mode == :admm
        @objective(model, Min, obj) # modified later
    end

    return model
end

function create_model_dispatch(n, _assets, demands, availability; T_max=168, pf=8.0, penalty=1e3, mode)
    assets = Dict(a => asset for (a, asset) in _assets if asset.node == n)
    T = 1:T_max

    # Create a new model.
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_attribute(model, "Method", 2)
    set_attribute(model, "Crossover", 0)

    # Create variable: auxiliary nodal variables.
    @variable(model, aux_netpos[t=T])
    if mode == :admm
    else
        @variable(model, z_pos[t=T] >= 0)
        @variable(model, z_neg[t=T] >= 0)
        fix.(aux_netpos, 0.0)
    end

    if isempty(assets)
        if mode == :admm
            @constraint(model, nodal_balance[t=T], aux_netpos[t] == -demands[n][t])
            @objective(model, Min, 0.0)
            return model
        else
            @constraint(model, nodal_balance[t=T], aux_netpos[t] == -demands[n][t] + z_pos[t] - z_neg[t])
            @objective(model, Min, penalty * sum(z_pos[t] + z_neg[t] for t in T))
            return model
        end
    end

    # Create variable: generation of assets.
    @variable(model, (
        (assets[a].type == "storage" ? (-assets[a].capacity / pf) : 0.0) <=
        gen[a=keys(assets), t=T] <=
        (assets[a].capacity * (assets[a].type == "storage" ? (1.0/pf) : (assets[a].type == "res" ? availability[a][t] : 1.0)))
    ))

    storage_assets = [a for (a, asset) in assets if asset.type == "storage"]
    if !isempty(storage_assets)
        # Create variable: storage state.
        @variable(model, 0.0 <= storage[s=storage_assets, t=T] <= assets[s].capacity)

        # Create constraint: temporal storage state.
        prev_t(t) = t == 1 ? T[end] : t - 1
        @constraint(model, state[s=storage_assets, t=T], storage[s, t] == storage[s, prev_t(t)] - gen[s, prev_t(t)])
    end

    # Create constraint: nodal balance.
    if mode == :admm
        @constraint(model, nodal_balance[t=T], aux_netpos[t] == sum(gen[a, t] for (a, asset) in assets) - demands[n][t])
    else
        @constraint(model, nodal_balance[t=T], aux_netpos[t] == sum(gen[a, t] for (a, asset) in assets) - demands[n][t] + z_pos[t] - z_neg[t])
    end
    # Create objective: minimize cost.
    corrected_costs = Dict(
        a => assets[a].type == "thermal" ? asset.cost : (assets[a].type == "storage" ? 0.0 : asset.cost / 100.0)
        for (a, asset) in assets
    )
    if mode == :admm
        @expression(model, obj, sum(gen[a, t] * corrected_costs[a] for (a, asset) in assets for t in T))
    else
        @expression(model, obj, sum(gen[a, t] * corrected_costs[a] for (a, asset) in assets for t in T) + penalty * sum(z_pos[t] + z_neg[t] for t in T))
    end

    @objective(model, Min, model[:obj])
    return model
end
