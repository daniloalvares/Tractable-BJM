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
fitLN2 <- stan(file = "JM-LN-M2.stan", 
               data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
               warmup = 1000,
               iter = 10000,
               chains = 1,
               seed = 2021,
               cores = getOption("mc.cores",1))

print(fitLN2)

parametersLN2 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "mu", "eta")
# plot(fitLN2, plotfun="trace", pars=parametersLN2, inc_warmup=FALSE)
post.parLN2 <- extract(fitLN2, parametersLN2, permuted=TRUE, inc_warmup=FALSE)
fitLN2.bridge <- bridge_sampler(fitLN2, silent=TRUE)
####################################################################


####################################################################
#                  JOINT MODEL WITH GAMMA HAZARD                   #
####################################################################
fitG2 <- stan(file = "JM-Gamma-M2.stan", 
              data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
              warmup = 1000,
              iter = 10000,
              chains = 1,
              seed = 2021,
              cores = getOption("mc.cores",1))

print(fitG2)

parametersG2 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu")
# plot(fitG2, plotfun="trace", pars=parametersG2, inc_warmup=FALSE)
post.parG2 <- extract(fitG2, parametersG2, permuted=TRUE, inc_warmup=FALSE)
fitG2.bridge <- bridge_sampler(fitG2, silent=TRUE)
####################################################################


####################################################################
#        JOINT MODEL WITH POWER GENERALISED WEIBULL HAZARD         #
####################################################################
fitPGW2 <- stan(file = "JM-PGW-M2.stan", 
                data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
                warmup = 1000,
                iter = 10000,
                chains = 1,
                seed = 2021,
                cores = getOption("mc.cores",1))

print(fitPGW2)

parametersPGW2 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu", "delta")
# plot(fitPGW2, plotfun="trace", pars=parametersPGW2, inc_warmup=FALSE)
post.parPGW2 <- extract(fitPGW2, parametersPGW2, permuted=TRUE, inc_warmup=FALSE)
fitPGW2.bridge <- bridge_sampler(fitPGW2, silent=TRUE)
####################################################################


####################################################################
#            JOINT MODEL WITH GENERALISED GAMMA HAZARD             #
####################################################################
fitGG2 <- stan(file = "JM-GG-M2.stan", 
               data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
               warmup = 1000,
               iter = 10000,
               chains = 1,
               seed = 2021,
               cores = getOption("mc.cores",1))

print(fitGG2)

parametersGG2 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu", "delta")
# plot(fitGG2, plotfun="trace", pars=parametersGG2, inc_warmup=FALSE)
post.parGG2 <- extract(fitGG2, parametersGG2, permuted=TRUE, inc_warmup=FALSE)
fitGG2.bridge <- bridge_sampler(fitGG2, silent=TRUE)
####################################################################


# Compute posterior model probabilities (assuming equal prior model probabilities)
post_prob(fitLN2.bridge, fitG2.bridge, fitPGW2.bridge, fitGG2.bridge)
