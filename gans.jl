module GANs

using Flux
using Distributions
import Random
import Statistics
using LinearAlgebra

struct GAN
   g # generator
   gg # non-trainable generator copy
   d # discriminator
   dd # non-trainable discriminator copy
   pz # code distribution
end

Flux.@treelike GAN
freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)


logpdf(b::Bernoulli, y::Bool) = y * log(b.p + eps()) + (1 - y) * log(1 - b.p + eps())

GAN(G::Flux.Chain, D::Flux.Chain; pz=randn) = GAN(G, freeze(G), D, freeze(D), pz)

getcode(gan::GAN) = Float64.(gan.pz(size(Flux.params(gan.g).order[1],2)))
getcode(gan::GAN, n::Int) = Float64.(gan.pz(size(Flux.params(gan.g).order[1],2), n))
generate(gan::GAN) = ((gan.g(getcode(gan)).data) .>= 0.5) .* 1.0
generate(gan::GAN, n::Int) = ((gan.g(getcode(gan, n)).data) .>= 0.5) .* 1.0


Dloss(gan::GAN, X, Z) = - Float64(0.5)*(Statistics.mean(log.(gan.d(X) .+ eps(Float64))) + Statistics.mean(log.(1 .- gan.d(gan.gg(Z)) .+ eps(Float64))))
Gloss(gan::GAN, Z) = - Statistics.mean(log.(gan.dd(gan.g(Z)) .+ eps(Float64)))


function initgan(p)
   generator = Chain(Dense(10, p, NNlib.leakyrelu), Dense(p, p, NNlib.σ))
   discriminator = Chain(Dense(p, 25, NNlib.leakyrelu), Dense(25, 1, NNlib.σ))
   gan = GAN(generator, discriminator, pz=randn)
end


function random_batch_index(x::AbstractArray, batch_size=1; dims=1)
   n = size(x,dims)
   Iterators.partition(Random.shuffle(1:n), batch_size)
end


function traingan!(gan::GAN, x; int_epochs::Int64=10,
      batch_size::Int64=10,etagen = 0.00001,etadisc=0.00001)

   Dopt=ADAM(etadisc)
   Gopt=ADAM(etagen)

   for epoch=1:int_epochs
      loss_epoch = [0.0, 0.0, 0]
      for I in random_batch_index(x, batch_size)
         m = length(I)
         # sample data and generate codes
         z = getcode(gan,m)

         # discriminator training
         Dl = Dloss(gan, x[I,:]',z)
         Flux.Tracker.back!(Dl)
         for param in Flux.params(gan.d)
            Flux.Tracker.update!(Dopt, param, Flux.Tracker.grad(param))
         end

         # generator training
         Gl = Gloss(gan, z)
         Flux.Tracker.back!(Gl)
         for param in Flux.params(gan.g)
            Flux.Tracker.update!(Gopt, param, Flux.Tracker.grad(param))
         end
      end
   end
end

function samples(gan, nsamples=1000)
   return convert(Array{Float64,2},Transpose(generate(gan,nsamples)))
end


end # module