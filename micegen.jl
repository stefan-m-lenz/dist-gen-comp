module MICE
using GLM, DataFrames
import Random
import StatsModels
using Statistics
import GLM: predict

struct MICEmodel
   varorder
   p1
   regmodels
end

struct PredictSameModel
   value
end

predict(p::PredictSameModel, x) = repeat([p.value], size(x,1))

function formula_n(n, onevaluevariables)
   diversevariables = .!(onevaluevariables[1:(n-1)])
   StatsModels.FormulaTerm(Term(Symbol("x" * string(n))),
         Tuple(map( i -> Term(Symbol("x" * string(i))), (1:(n-1))[diversevariables])))
end

function fitmice(x::Matrix{Float64})
   nsamples, nvariables = size(x)

   varorder = Random.shuffle(1:nvariables)
   df = DataFrame(x[:, varorder])
   onevaluevariables = map(i -> reduce(==, df[:,i]), 1:nvariables)
   colinearvariables = map(i -> any(map(j -> df[:,i] == df[:,j], 1:(i-1))), 1:nvariables)
   p1 = mean(df[:,1])
   regmodels = Array{Any, 1}(undef, nvariables-1)
   for i in 2:nvariables
      if onevaluevariables[i-1] == true
         regmodels[i-1] = PredictSameModel(df[1,i-1])
      else
         formula = formula_n(i, onevaluevariables .| colinearvariables)
         try
            regmodels[i-1] = glm(formula, df, Bernoulli(), LogitLink())
         catch ex
            colinearvariables[i-1] = true
            formula = formula_n(i, onevaluevariables .| colinearvariables)
            regmodels[i-1] = glm(formula, df, Bernoulli(), LogitLink())
         end
      end
   end
   MICEmodel(varorder, p1, regmodels)
end

function genmice(m::MICEmodel, nsamples)
   x1 = float.(rand(nsamples) .< m.p1)
   df = DataFrame(x1 = x1)
   nvariables = length(m.varorder)
   for i in 2:nvariables
      df[:,i] = round.(rand(nsamples) .< predict(m.regmodels[i-1], df[:,1:(i-1)]))
   end
   df = df[:,invperm(m.varorder)]
   convert(Array, df)
end

end