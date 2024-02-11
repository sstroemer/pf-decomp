import CSV
using DataFrames
import Printf


function load_grid_data()
    df = CSV.read("data/lines.csv", DataFrames.DataFrame, header=[1, 2], types=String)
    df = filter(row -> row["Column3_TSO"] in ["50HERTZ", "TRANSNETBW", "TENNETGMBH", "AMPRION GMBH"], df)
    df = df[!, ["Substation_1_Full_name", "Substation_2_Full_name", "Column6_Voltage_level(kV)", "Column13_Fixed", "Column15_DLRmax(A)", "Column17_Reactance_X(Ω)"]]

    rename!(df, Dict(
        "Substation_1_Full_name" => "node1",
        "Substation_2_Full_name" => "node2",
        "Column6_Voltage_level(kV)" => "U",
        "Column13_Fixed" => "I_fixed",
        "Column15_DLRmax(A)" => "I_dynamic",
        "Column17_Reactance_X(Ω)" => "X"
    ))

    df.I_fixed = replace(df.I_fixed, "-" => missing)
    df.I_fixed = replace(df.I_fixed, " " => missing)
    df.I_dynamic = replace(df.I_dynamic, "-" => missing)
    df.I_dynamic = replace(df.I_dynamic, " " => missing)


    df.I = parse.(Float64, coalesce.(df.I_fixed, df.I_dynamic))
    df.U = parse.(Float64, df.U)
    df.capacity = sqrt(3.0) .* df.U .* df.I ./ 1000.0
    
    df.X = parse.(Float64, df.X)
    df.X = df.X .* (380.0 ./ df.U) .^ 2

    return df[!, ["node1", "node2", "capacity", "X"]]
end

grid_nodes(df) = unique(vcat(df.node1, df.node2))
grid_lines(df) = Dict("line_$(rownumber(row))" => (
    from=row.node1, to=row.node2, capacity=row.capacity, x=row.X,
    expansion=rand([0.5, 1, 1, 1, 2, 2, 4, 8]) * row.capacity, cost=rand(250:250:2000)
) for row in eachrow(df))

function grid_demand(df)
    nodes = grid_nodes(df)
    ts = moving_average(rand(8760), 4) .* 40_000 .+ 40_000
    
    _tmp = rand(length(nodes)) .* rand([0.0, 0.1, 0.1, 0.5, 0.5, 1.0, 2.0, 5.0], length(nodes))
    _tmp /= sum(_tmp)
    dist = Dict(nodes[i] => _tmp[i] for i in eachindex(nodes))

    return Dict(node => ts .* dist[node] for node in nodes)
end

function grid_assets(df; n=1500)
    nodes = grid_nodes(df)
    return Dict("asset_$i" => (
        node=rand(nodes),
        capacity=rand(50:50:(230_000 / n * 2.0)) * (rand() <= 0.01 ? 15.0 : 0.9),
        type=(rand() <= 0.05) ? "storage" : (rand() <= 0.4) ? "res" : "thermal",
        cost=100.0 + rand() * 500.0
    ) for i in 1:n)
end

function res_availability(assets)
    return Dict(
        name => moving_average(rand(8760) .* ((rand(8760) .<= 0.2) * 0.6 .+ 0.4), 8)
        for (name, prop) in assets if prop.type == "res"
    )
end

function moving_average(vector, period)
    n = length(vector)
    smoothed_vector = similar(vector)
    for i in 1:n
        start_index = max(1, i - period + 1)
        end_index = min(n, i)
        smoothed_vector[i] = sum(vector[start_index:end_index]) / (end_index - start_index + 1)
    end
    return smoothed_vector
end

# source: https://jump.dev/JuMP.jl/stable/tutorials/algorithms/benders_decomposition/
function print_iteration(k, args...)
    f(x) = Printf.@sprintf("%12.4e", x)
    println(lpad(k, 9), " ", join(f.(args), " "))
    return
end

function res_to_markdown(res::Vector)
    # Assume all tuples have the same structure, so use the first one for column names
    column_names = propertynames(first(res))
    header_row = "|" * join(String.(column_names), "|") * "|"
    
    # Create the separator row
    separator_row = "|" * join([":---" for _ in column_names], "|") * "|"
    
    # Create data rows
    data_rows = join(["|" * join((round(getfield(t, name); digits=3) for name in column_names), "|") * "|" for t in res], "\n")
    
    # Combine all parts to form the markdown table
    markdown_table = join([header_row, separator_row, data_rows], "\n")
    return markdown_table
end
