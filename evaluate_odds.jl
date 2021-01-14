module EvaluateOdds

include("./gans.jl")
include("./vaes.jl")
include("./micegen.jl")
import JLD
using Statistics
import Random
using Distributed
import BoltzmannMachines
using SplitApplyCombine
const BMs = BoltzmannMachines

using LinearAlgebra

function logoddsratio(x)
   N, dim = size(x)
   OR = LowerTriangular(ones(dim,dim))
   for i = 1:dim
      for j = 1:(i-1)
         a = N - sum(min.(x[:,i] .+ x[:,j], 1))      # ij_00
         diff = x[:,i] .- x[:,j]
         b = sum(diff .== -1)                        # ij_01
         c = sum(diff .== 1)                         # ij_10
         d = sum(x[:,i] .* x[:,j])                   # ij_11
         OR[i,j] = max(a,0.5)*max(d,0.5)/max(b,0.5)/max(c,0.5)
      end
   end
   return LowerTriangular(log.(OR))
end


function oddsdelta(x1, x2)
    mean((logoddsratio(x1) .- logoddsratio(x2)).^2)
end


function my_fitdbm(x::Matrix{Float64}, epochs::Int)
   dbm = BMs.fitdbm(x;
         learningrate = 0.001,
         pretraining = [
               BMs.TrainLayer(nhidden = size(x, 2));
               BMs.TrainLayer(nhidden = 10)],
         epochspretraining = epochs,
         epochs = 20)
end

function getdata(datasetindex)
   JLD.load("data/$(datasetindex)_50.jld")["$(datasetindex)_50"]
end

const nvariables = 50
const ntest = 100
const nsamples = 500

function validationdataset(i)
   getdata(i)[1001:2000, 1:nvariables]
end

const datasetids = 1:30
const validationoddsratios = Dict([(i, logoddsratio(validationdataset(i))) for i in datasetids])
const training_datasets = Dict([(i, getdata(i)[1:nsamples, 1:nvariables]) for i in datasetids])
const test_datasets = Dict([(i, getdata(i)[(nsamples+1):(nsamples + ntest), 1:nvariables])
      for i in datasetids])


# Collect the data sets from the different sites and evaluate the results
function evaluatexgen(fitgenresults)
   xgens = map(r -> r[:xgen], fitgenresults)
   xgens = filter(x -> length(size(x)) == 2, xgens)
   xgen = vcat(xgens...)
   if isempty(xgen)
      delta = Inf
   else
      validationodds = validationoddsratios[fitgenresults[1][:datasetindex]]
      delta = mean((logoddsratio(xgen) .- validationodds).^2)
   end

   oddsdeltatrain = minimum(map(r -> r[:oddsdeltatrain], fitgenresults))

   # The different sites have been trained with the same model, the same seed,
   # and the same number of epochs, so it suffices to take the value of the
   # first array element.
   (seed = fitgenresults[1][:seed],
         datasetindex = fitgenresults[1][:datasetindex],
         modeltype = fitgenresults[1][:modeltype],
         epochs = fitgenresults[1][:epoch],
         nsites = fitgenresults[1][:nsites],
         oddsdeltatrain = oddsdeltatrain,
         oddsdelta = delta)
end


function evaluateoddsratios(modeldefinitions, seeds)
   nsites = [1,2,5,20]

   args = Iterators.product(
      seeds,
      datasetids,
      nsites,
      modeldefinitions)

   pmap(EvaluateOdds.evaluateodds, args; retry_delays = ExponentialBackOff(n = 5))
end


function evaluateodds((seed, datasetindex, nsites, modeldefinition))
   evaluatexgen(map(datasetpart ->
      fitgenerate(seed, datasetindex, nsites, datasetpart, modeldefinition),
         1:nsites))
end

function _initvae()
   VAEs.initvae(nvariables = nvariables, nhidden = nvariables, nz = 10)
end

function _trainvae!(model, x, epochs)
   VAEs.trainvae!(model, x; epochs = epochs, learningrate = 0.002)
end

function _initgan()
   GANs.initgan(nvariables)
end

function _traingan!(model, x, epochs)
   GANs.traingan!(model, x; int_epochs = epochs)
end

function extract_trainingdata(datasetindex, nsites, datasetpart)
   xtrain = training_datasets[datasetindex]
   npersite = floor(Int, nsamples / nsites)
   xpart = xtrain[((datasetpart-1)*npersite+1):(datasetpart*npersite),:]
end

function fitgenerate(
      seed,
      datasetindex,
      nsites,
      datasetpart,
      modeldefinition)

   function evaluation((model_evaluation, xgen, oddsdeltatrain, epoch))
      (seed = seed, datasetindex = datasetindex,
         model_evaluation = model_evaluation,
         nsites = nsites, datasetpart = datasetpart,
         modeltype = modeldefinition["model"],
         epoch = epoch,
         xgen = xgen,
         oddsdeltatrain = oddsdeltatrain)
   end

   Random.seed!(seed)

   xtest = test_datasets[datasetindex]
   xpart = extract_trainingdata(datasetindex, nsites, datasetpart)

   ngensamples = size(xpart, 1)

   if modeldefinition["model"] == "dbm"
      return evaluation(evaldbm(modeldefinition, xpart, xtest))

   elseif modeldefinition["model"] == "mice"
      return evaluation(evalmice(modeldefinition, xpart, xtest))

   elseif modeldefinition["model"] == "gan"
      return evaluation(eval_multepochs(modeldefinition, xpart, xtest;
         initfun = _initgan,
         trainfun! = _traingan!,
         samplefun = model -> GANs.samples(model, ngensamples)))

   elseif modeldefinition["model"] == "vae"
      return evaluation(eval_multepochs(modeldefinition, xpart, xtest;
         initfun = _initvae,
         trainfun! = _trainvae!,
         samplefun = model -> VAEs.samples(model, ngensamples)))

   elseif modeldefinition["model"] == "im"
      return evaluation(evalim(modeldefinition, xpart, xtest))

   else
      error("unknown model")
   end
end


function evaldbm(modeldefinition, xpart, xtest)
   ngensamples = size(xpart, 1)
   epochs = modeldefinition["epochs"]
   dbm = my_fitdbm(xpart, epochs)
   xgen = BMs.samples(dbm, ngensamples)
   model_evaluation = oddsdelta(xgen, xtest)
   oddsdeltatrain = oddsdelta(xpart, xgen)
   model_evaluation, xgen, oddsdeltatrain, epochs
end


function evalmice(modeldefinition, xpart, xtest)
   ngensamples = size(xpart, 1)
   model_evaluation = Inf
   oddsdeltatrain = Inf
   xgen = []
   try
      model = MICE.fitmice(xpart)
      xgen = MICE.genmice(model, ngensamples)
      model_evaluation = oddsdelta(xgen, xtest)
      oddsdeltatrain = oddsdelta(xpart, xgen)
   catch ex
   end
   epoch = 0 # not relevant for mice
   model_evaluation, xgen, oddsdeltatrain, epoch
end


function evalim(modeldefinition, xpart, xtest)
   ngensamples = size(xpart, 1)
   # odds ratio of independent variables is 1
   varmeans = mean(xpart, dims = 1)
   xgen = float.(rand(Float64, size(xpart)) .< varmeans)
   model_evaluation = oddsdelta(xgen, xtest)
   oddsdeltatrain = oddsdelta(xpart, xgen)
   epoch = 0 # not relevant for im
   model_evaluation, xgen, oddsdeltatrain, epoch
end


function eval_multepochs(modeldefinition, xpart, xtest;
      initfun::Function, trainfun!::Function, samplefun::Function)

   epochs = vcat(0, modeldefinition["epochs"])
   bestoddsdelta = Inf
   bestxgen = zeros(1,1)
   bestepoch = 0

   model = initfun()

   for i in 2:length(epochs)
      trainfun!(model, xpart, epochs[i] - epochs[i-1])
      # let sampling not change the random numbers for model training
      rng = copy(Random.default_rng());
      xgen = samplefun(model)
      copy!(Random.default_rng(), rng);
      newoddsdelta = oddsdelta(xgen, xtest)
      if newoddsdelta < bestoddsdelta
         bestoddsdelta = newoddsdelta
         bestxgen = xgen
         bestepoch = epochs[i]
      end
   end
   model_evaluation = bestoddsdelta
   xgen = bestxgen
   oddsdeltatrain = oddsdelta(xpart, xgen)
   model_evaluation, xgen, oddsdeltatrain, bestepoch
end

end
