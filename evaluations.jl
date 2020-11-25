module Evaluations

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
const nvalidation = 1000

function validationdataset(i)
   getdata(i)[1000:2000, 1:nvariables]
end

const datasetids = 1:30
const validationoddsratios = map(i -> logoddsratio(validationdataset(i)), datasetids)
const training_datasets = map(i -> getdata(i)[1:nsamples, 1:nvariables], datasetids)
const test_datasets = map(i -> getdata(i)[(nsamples+1):(nsamples + ntest), 1:nvariables],
      datasetids)


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

   (seed = fitgenresults[1][:seed],
         datasetindex = fitgenresults[1][:datasetindex],
         modeltype = fitgenresults[1][:modeltype],
         nsites = fitgenresults[1][:nsites],
         oddsdeltatrain = oddsdeltatrain,
         oddsdelta = delta)
end


function evaluateoddsratios(modeldefinitions, seeds)
   datasetindex = 1:10
   nsites = [1,2,5,20]

   args = Iterators.product(
      seeds,
      datasetindex,
      nsites,
      modeldefinitions)

   pmap(Evaluations.evaluateodds, args; retry_delays = ExponentialBackOff(n = 5))
end


function evaluateodds((seed, datasetindex, nsites, modeldefinition))
   evaluatexgen(map(datasetpart ->
      fitgenerate(seed, datasetindex, nsites, datasetpart, modeldefinition),
         1:nsites))
end


function fitgenerate(
      seed,
      datasetindex,
      nsites,
      datasetpart,
      modeldefinition)

   function evaluation((model_evaluation, xgen, oddsdeltatrain))
      (seed = seed, datasetindex = datasetindex,
         model_evaluation = model_evaluation,
         nsites = nsites, datasetpart = datasetpart,
         modeltype = modeldefinition["model"],
         xgen = xgen,
         oddsdeltatrain = oddsdeltatrain)
   end

   Random.seed!(seed)

   xtrain = training_datasets[datasetindex]
   xtest = test_datasets[datasetindex]

   npersite = floor(Int, nsamples / nsites)
   xpart = xtrain[((datasetpart-1)*npersite+1):(datasetpart*npersite),:]

   ngensamples = size(xpart, 1)

   if modeldefinition["model"] == "dbm"
      return evaluation(evaldbm(modeldefinition, xpart, xtest))

   elseif modeldefinition["model"] == "mice"
      return evaluation(evalmice(modeldefinition, xpart, xtest))

   elseif modeldefinition["model"] == "gan"
      return evaluation(eval_multepochs(modeldefinition, xpart, xtest;
         initfun = () -> GANs.initgan(nvariables),
         trainfun! = (model, x, epochs) ->
               GANs.traingan!(model, x; int_epochs = epochs),
         samplefun = model -> GANs.samples(model, ngensamples)))

   elseif modeldefinition["model"] == "vae"
      return evaluation(eval_multepochs(modeldefinition, xpart, xtest;
         initfun = () ->
               VAEs.initvae(nvariables = nvariables,
                     nhidden = nvariables, nz = 10),
         trainfun! = (model, x, epochs) ->
               VAEs.trainvae!(model, x; epochs = epochs,
                     learningrate = 0.002),
         samplefun = model -> VAEs.samples(model, ngensamples)))

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
   model_evaluation, xgen, oddsdeltatrain
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
   model_evaluation, xgen, oddsdeltatrain
end


function eval_multepochs(modeldefinition, xpart, xtest;
      initfun::Function, trainfun!::Function, samplefun::Function)

   epochs = vcat(0, modeldefinition["epochs"])
   bestoddsdelta = Inf
   bestxgen = zeros(1,1)

   model = initfun()

   for i in 2:length(epochs)
      trainfun!(model, xpart, epochs[i] - epochs[i-1])
      xgen = samplefun(model)
      newoddsdelta = oddsdelta(xgen, xtest)
      if newoddsdelta < bestoddsdelta
         bestoddsdelta = newoddsdelta
         bestxgen = xgen
      end
   end
   model_evaluation = bestoddsdelta
   xgen = bestxgen
   oddsdeltatrain = oddsdelta(xpart, xgen)
   model_evaluation, xgen, oddsdeltatrain
end

end
