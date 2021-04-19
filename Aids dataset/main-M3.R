rm(list=ls())

# Required sources
require(rstanarm)
require(rstan)
require(bridgesampling)
require(JM)

options(mc.cores = parallel::detectCores())

# Longitudinal and survival datasets
dataL <- JM::aids[,c(1:5,8)]
dataS <- JM::aids.id[,c(1:3,8)]

# Numeric binary variable
dataL$prevOI <- as.numeric(dataL$prevOI)-1 # noAIDS=0 and AIDS=1
dataS$prevOI <- as.numeric(dataS$prevOI)-1

# Data for the Stan model
X <- as.matrix(dataS[,4],ncol=1) # design matrix X                       
n <- nrow(X)                     # total number of observations
y <- sqrt(dataL$CD4)             # longitudinal outcomes
ID <- as.numeric(dataL$patient)  # patient IDs
nid <- length(unique(ID))        # number of patients
status <- dataS$death            # vital status (1 = dead, 0 = alive)
times <- dataS$Time              # times to event
visits <- dataL$obstime          # visit times for repeated observations
N <- length(y)                   # total number of longitudinal outcomes
indobs <- which(status==1)       # observed survival times indicator
nobs <- length(indobs)           # number of observed survival times

####################################################################
#                JOINT MODEL WITH LOGNORMAL HAZARD                 #
####################################################################
fitLN3 <- stan(file = "JM-LN-M3.stan", 
               data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
               warmup = 1000,
               iter = 10000,
               chains = 1,
               seed = 2021,
               cores = getOption("mc.cores",1))

print(fitLN3)

parametersLN3 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "mu", "eta")
# plot(fitLN3, plotfun="trace", pars=parametersLN3, inc_warmup=FALSE)
post.parLN3 <- extract(fitLN3, parametersLN3, permuted=TRUE, inc_warmup=FALSE)
fitLN3.bridge <- bridge_sampler(fitLN3, silent=TRUE)
####################################################################


####################################################################
#                  JOINT MODEL WITH GAMMA HAZARD                   #
####################################################################
fitG3 <- stan(file = "JM-Gamma-M3.stan", 
              data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
              warmup = 1000,
              iter = 10000,
              chains = 1,
              seed = 2021,
              cores = getOption("mc.cores",1))

print(fitG3)

parametersG3 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu")
# plot(fitG3, plotfun="trace", pars=parametersG3, inc_warmup=FALSE)
post.parG3 <- extract(fitG3, parametersG3, permuted=TRUE, inc_warmup=FALSE)
fitG3.bridge <- bridge_sampler(fitG3, silent=TRUE)
####################################################################


####################################################################
#        JOINT MODEL WITH POWER GENERALISED WEIBULL HAZARD         #
####################################################################
fitPGW3 <- stan(file = "JM-PGW-M3.stan", 
                data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
                warmup = 1000,
                iter = 10000,
                chains = 1,
                seed = 2021,
                cores = getOption("mc.cores",1))

print(fitPGW3)

parametersPGW3 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu", "delta")
# plot(fitPGW3, plotfun="trace", pars=parametersPGW3, inc_warmup=FALSE)
post.parPGW3 <- extract(fitPGW3, parametersPGW3, permuted=TRUE, inc_warmup=FALSE)
fitPGW3.bridge <- bridge_sampler(fitPGW3, silent=TRUE)
####################################################################


####################################################################
#            JOINT MODEL WITH GENERALISED GAMMA HAZARD             #
####################################################################
fitGG3 <- stan(file = "JM-GG-M3.stan", 
               data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
               warmup = 1000,
               iter = 10000,
               chains = 1,
               seed = 2021,
               cores = getOption("mc.cores",1))

print(fitGG3)

parametersGG3 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu", "delta")
# plot(fitGG3, plotfun="trace", pars=parametersGG3, inc_warmup=FALSE)
post.parGG3 <- extract(fitGG3, parametersGG3, permuted=TRUE, inc_warmup=FALSE)
fitGG3.bridge <- bridge_sampler(fitGG3, silent=TRUE)
####################################################################


# Compute posterior model probabilities (assuming equal prior model probabilities)
post_prob(fitLN3.bridge, fitG3.bridge, fitPGW3.bridge, fitGG3.bridge)
