
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
import CSV
using DataFrames

# read result from comparison.jl
oddsratioresults = CSV.read("results_odds.tsv", DataFrame; delim = '\t', header = 1)

# pick best models with respect to oddsdelta
modeldf = DataFrame([g[findmin(g.oddsdelta)[2],:]
      for g in groupby(oddsratioresults, [:datasetindex, :modeltype, :nsites])])


using Distributed
addprocs()
@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere include("evaluate_odds.jl")
@everywhere include("membershipattack.jl")

# for all models: train model, generate data and perform membership attack
@time attackresults = pmap(i -> MembershipAttack.traingenattack(;
      seed = modeldf[i, :seed],
      modeltype = modeldf[i, :modeltype],
      epochs = modeldf[i, :epochs],
      datasetindex = modeldf[i, :datasetindex],
      nsites = modeldf[i, :nsites]), 1:nrow(modeldf))

resultdf = DataFrame(vcat(map(collect, attackresults)...))
CSV.write("results_membership.tsv", resultdf; delim ='\t')

