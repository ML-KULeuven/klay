using LogicCircuits
using BenchmarkTools
using DataFrames: DataFrame, Tables
using CUDA

var_range = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
device = "cuda"

results_folder = "results/sdd_juice_$(device)"
if !isdir(results_folder)
    mkdir(results_folder)
end

for nb_vars in var_range
    for seed in 0:9

        sdd_path = "results_old/sdd/v$(nb_vars)_$(seed).sdd"
        vtree_path = "results_old/sdd/v$(nb_vars)_$(seed).vtree"
        paths = (sdd_path, vtree_path)
        formats = (SddFormat(), VtreeFormat())
        sdd = read(paths, StructLogicCircuit, formats)

        weights = rand(Float32, nb_vars)
        if device == "cuda"
            weights = CuArray(weights)
        end
        data = DataFrame(reshape(weights, 1, :), :auto)
        bit_sdd = same_device(BitCircuit(sdd, data), data)


        timings = Vector{Float32}()
        for i in 1:12
            weights = rand(Float32, nb_vars)
            if device == "cuda"
                weights = CuArray(weights)
            end
            data = DataFrame(reshape(weights, 1, :), :auto)
            t = @elapsed satisfies_flows(bit_sdd, data)
            append!(timings, t)
        end

        avg = mean(timings[3:end])
        println(nb_vars, " ", avg)
        filepath = "results/sdd_juice_$(device)/v$(nb_vars)_$(seed).txt"
        json = "{\"backward\": $(avg)}"
        open(filepath, "w") do f
            write(f, json)
        end
    end
end
