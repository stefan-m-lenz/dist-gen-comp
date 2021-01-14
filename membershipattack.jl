
module MembershipAttack

import ..EvaluateOdds
import Random
using Statistics

function matrixtorowdict(mat)
   matrows = [mat[j,:] for j in 1:size(mat, 1)]
   Dict([(row, count(x -> x == row, matrows)) for row in matrows])
end

function matrixtorowset(mat)
   Set([mat[j,:] for j in 1:size(mat, 1)])
end

const training_dicts = [matrixtorowdict(EvaluateOdds.training_datasets[datasetindex])
      for datasetindex in EvaluateOdds.datasetids]

function get_membership_testdata(datasetindex)
   EvaluateOdds.getdata(datasetindex)[1001:(1000 + EvaluateOdds.nsamples), 1:EvaluateOdds.nvariables]
end

const test_dicts = [matrixtorowdict(get_membership_testdata(datasetindex))
      for datasetindex in EvaluateOdds.datasetids]

function membership_attack(datasetindex, genset::Set, distance)
   trainingdict = training_dicts[datasetindex]
   testdict = test_dicts[datasetindex]
   true_pos = 0
   true_neg = 0
   false_neg = 0
   false_pos = 0

   # nocc: number of occurences of same vector
   for (testsample, nocc) in testdict
      samples_nearby = false
      for gensample in genset
         if sum(abs.(testsample .- gensample)) <= distance
            samples_nearby = true
            break
         end
      end

      if !samples_nearby # attacker assumes: sample is not in training set
         true_neg += nocc
      else # attacker assumes: sample is in the training set
         false_pos += nocc
      end
   end

   for (trainingsample, nocc) in trainingdict
      samples_nearby = false
      for gensample in genset
         if sum(abs.(trainingsample .- gensample)) <= distance
            samples_nearby = true
            break
         end
      end

      if !samples_nearby # attacker assumes: sample is not in training set
         false_neg += nocc
      else # attacker assumes: sample is in the training set
         true_pos += nocc
      end
   end

   (true_pos = true_pos, true_neg = true_neg,
      false_pos = false_pos, false_neg = false_neg,
      distance = distance)
end

const distances = [0, 2, 3, 5, 6, 8, 10]


function train_and_sample(modeltype, trainingdata, epochs)
   ngensamples = size(trainingdata, 1)

   if modeltype == "dbm"
      dbm = EvaluateOdds.my_fitdbm(trainingdata, epochs)
      return EvaluateOdds.BMs.samples(dbm, ngensamples)

   elseif modeltype == "mice"
      ret = []
      try
         model = EvaluateOdds.MICE.fitmice(trainingdata)
         ret = EvaluateOdds.MICE.genmice(model, ngensamples)
      catch ex
      end
      return ret

   elseif modeltype == "im"
      varmeans = mean(trainingdata, dims = 1)
      return float.(rand(Float64, (ngensamples, size(trainingdata,2))) .< varmeans)

   elseif modeltype == "vae"
      vae = EvaluateOdds._initvae()
      EvaluateOdds._trainvae!(vae, trainingdata, epochs)
      return EvaluateOdds.VAEs.samples(vae, ngensamples)

   elseif modeltype == "gan"
      gan = EvaluateOdds._initgan()
      EvaluateOdds._traingan!(gan, trainingdata, epochs)
      return EvaluateOdds.GANs.samples(gan, ngensamples)

   else
      error("Unknown model type $modeltype")
   end
end

function traingenattack(;seed, modeltype, epochs, datasetindex, nsites)
   # train models on parts, sample and combine

   xgens = map(datasetpart -> begin
      Random.seed!(seed)
      xpart = EvaluateOdds.extract_trainingdata(datasetindex, nsites, datasetpart)
      train_and_sample(modeltype, xpart, epochs)
   end, 1:nsites)
   xgens = filter(x -> length(size(x)) == 2, xgens)
   xgen = vcat(xgens...)

   genset = matrixtorowset(xgen)
   [merge((seed = seed, modeltype = modeltype, epochs = epochs, datasetindex = datasetindex, nsites = nsites),
         membership_attack(datasetindex, genset, d)) for d in distances]
end

end # module MembershipAttack

