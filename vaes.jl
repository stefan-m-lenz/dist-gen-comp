
module VAEs

# """ Init VAE as struct consisting the network for the variable reduction (encoder), the mu and logsigma layer and the the decoder network """
using Flux
using Flux: throttle, params
import Random
using LinearAlgebra

struct VAE
	encoder
	mu
	logsigma
	decoder
end

function initvae(; nvariables, nhidden, nz)
   VAE(Flux.Chain(Flux.Dense(nvariables,nhidden,tanh)),
      Flux.Dense(nhidden,nz),Flux.Dense(nhidden,nz),
      Flux.Chain(Flux.Dense(nz,nhidden,tanh),
      Flux.Dense(nhidden,nvariables,Flux.NNlib.σ)))
end


function normalz(mu,logsigma)
	return mu  + exp(logsigma) * randn()
end

function decoderpen(vae)
	return 0.01 * sum(x->sum(x.^2),Flux.params(vae.decoder))
end


function kullbackleiblerqp(mu,logsigma)
	return 0.5 * sum(exp.(2 .* logsigma) + mu.^2 .- 1 .- 2 .* logsigma)
end

function reconstructionloss(xhat,x)
	ce(x,y) =  y * log(x + eps()) + (1 - y) * log(1 - x + eps())
	# logpdf(b::Bernoulli,y::Bool) = y * log(b.p + eps()) + (1-y) * log(1-b.p + eps())
	# return sum(Distributions.logpdf.(Bernoulli.(xhat),x))
	return sum(ce.(xhat,x))
end


""" The functions to sample from the vae conditional or unconditional of the data """
function getzfordata(vae,data)
	transformeddata = vae.encoder(data)

	mu,logsigma = (vae.mu(transformeddata).data,vae.logsigma(transformeddata).data)
	z = normalz.(mu,logsigma)
	z
end


function getsamplefordata(vae,data,prob=true)
	z = getzfordata(vae,data)
	samplegivenz(vae,z,prob)
end

function samplegivenz(vae,z,prob=prob)
	s = vae.decoder(z).data
	if prob == false
		s = Float64.(s .> rand(length(s)))
	end
	s
end

function getlatentvarsfordata(vae,data,prob=true,varlevel=1)
	z = getzfordata(vae,data)
	getlatentvarsforz(vae,z,prob=prob,varlevel=varlevel)
end

function getlatentvarsforz(vae,z,prob=true,varlevel=1)
	latent = vae.decoder[1:varlevel](z).data
	if prob == false
		latent = splitvars(latent)
	end
	latent
end

function splitvars(x)
	return Float64.(x .>= 0)
end

function samplefromtanh(x)
	ifelse(x < (2 * rand() - 1),0.0,1.0)
end

function samples(vae::VAE, nsamples)
   dz = size(vae.mu.W, 1)
	z = randn(dz, nsamples)
	s = vae.decoder(z).data
	copy(Transpose(float.(s .> 0.5)))
end

function trainvae!(vae, x; epochs = 1,learningrate = 0.001,batchsize = 10)

   g(x) = (h = vae.encoder(x); (vae.mu(h), vae.logsigma(h)))
   loss(x) = ((mu,logsigma) = g(x);
         (reconstructionloss(vae.decoder(normalz.(mu,logsigma)),x) -
               kullbackleiblerqp(mu,logsigma)) / batchsize)
   penloss(x) = -loss(x) + decoderpen(vae)

   n,p =size(x)
   optimizer = ADAM(learningrate)
   vaeparams = Flux.params(vae.encoder, vae.mu, vae.logsigma, vae.decoder)
   for i = 1:epochs
       data = [x[i,:]' for i in Iterators.partition(Random.shuffle(1:n), batchsize)]
       Flux.train!(penloss,vaeparams,zip(data),optimizer)
   end
   vae
end



end # module VAEs