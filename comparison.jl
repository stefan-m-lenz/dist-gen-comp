import Pkg
# Tell Julia to use the environment specified by Manifest.toml and Project.toml
Pkg.activate(".")
# Install all specified packages
Pkg.instantiate()

# Ensure all packages are precompiled before the parallel workers are started
import JLD
import Flux
import Distributions
import Distributed
import GLM
import StatsModels
import DataFrames
import BoltzmannMachines
import SplitApplyCombine
import Statistics


using Distributed
# Use all processes on this machine:
addprocs()
# Or use multiple machines:
# addprocs([("lenz@imbi2",:auto), ("lenz@imbi10",:auto), ("lenz@imbi12",:auto)];
#        topology = :master_worker)

# Load the required packages and modules on all workers
@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere include("./evaluations.jl")

# Define the hyperparameters
seeds = 1:15
modeldefinitions = [
      Dict("model" => "mice");
      map(i -> Dict("model" => "dbm", "epochs" => i), 50:20:2000);
      Dict("model" => "vae", "epochs" => 50:20:2000);
      Dict("model" => "gan", "epochs" => 50:20:2000);
      ]

# Run the experiment
@time result = Evaluations.evaluateoddsratios(modeldefinitions, seeds);

# Write output to tab-separated file
using DelimitedFiles
resultmat = permutedims(hcat(map(collect, result)...))
resultmat = vcat(permutedims(string.(collect(keys(result[1])))), resultmat)
writedlm("result.tsv", resultmat, "\t")

