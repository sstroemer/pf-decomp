include("src/admm.jl")


@info "Force compilation"
global_logger(ConsoleLogger(stderr, Logging.Error))
orig_obj = run_mono(2; verbose=false)[2]
run_admm(2, orig_obj; n_iter=1, verbose=false)
global_logger(ConsoleLogger(stderr, Logging.Info))
@info "Compilation done"

res = []
for T in [336] # [12, 24, 48, 72, 96, 120, 144, 168]
    @info "Running T = $T"
    res1 = run_mono(T; verbose=false)
    res2 = run_admm(T, res1[2]; n_iter=50, verbose=false)
    push!(res, (T = T, delta_rel = abs((res2[2] - res1[2]) / res1[2]), time_parallel = res2[1], time_original = res1[1]))

    open("results.md", "w") do file
        write(file, res_to_markdown(res))
    end
end
