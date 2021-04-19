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
fitLN1 <- stan(file = "JM-LN-M1.stan", 
               data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
               warmup = 1000,
               iter = 10000,
               chains = 1,
               seed = 2021,
               cores = getOption("mc.cores",1))

print(fitLN1)

parametersLN1 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "mu", "eta")
# plot(fitLN1, plotfun="trace", pars=parametersLN1, inc_warmup=FALSE)
post.parLN1 <- extract(fitLN1, parametersLN1, permuted=TRUE, inc_warmup=FALSE)
fitLN1.bridge <- bridge_sampler(fitLN1, silent=TRUE)
####################################################################


####################################################################
#                  JOINT MODEL WITH GAMMA HAZARD                   #
####################################################################
fitG1 <- stan(file = "JM-Gamma-M1.stan", 
              data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
              warmup = 1000,
              iter = 10000,
              chains = 1,
              seed = 2021,
              cores = getOption("mc.cores",1))

print(fitG1)

parametersG1 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu")
# plot(fitG1, plotfun="trace", pars=parametersG1, inc_warmup=FALSE)
post.parG1 <- extract(fitG1, parametersG1, permuted=TRUE, inc_warmup=FALSE)
fitG1.bridge <- bridge_sampler(fitG1, silent=TRUE)
####################################################################


####################################################################
#        JOINT MODEL WITH POWER GENERALISED WEIBULL HAZARD         #
####################################################################
fitPGW1 <- stan(file = "JM-PGW-M1.stan", 
                data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
                warmup = 1000,
                iter = 10000,
                chains = 1,
                seed = 2021,
                cores = getOption("mc.cores",1))

print(fitPGW1)

parametersPGW1 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu", "delta")
# plot(fitPGW1, plotfun="trace", pars=parametersPGW1, inc_warmup=FALSE)
post.parPGW1 <- extract(fitPGW1, parametersPGW1, permuted=TRUE, inc_warmup=FALSE)
fitPGW1.bridge <- bridge_sampler(fitPGW1, silent=TRUE)
####################################################################


####################################################################
#            JOINT MODEL WITH GENERALISED GAMMA HAZARD             #
####################################################################
fitGG1 <- stan(file = "JM-GG-M1.stan", 
               data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,nbetas=ncol(X),nlambdas=ncol(X)),
               warmup = 1000,
               iter = 10000,
               chains = 1,
               seed = 2021,
               cores = getOption("mc.cores",1))

print(fitGG1)

parametersGG1 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda", "alpha", "eta", "nu", "delta")
# plot(fitGG1, plotfun="trace", pars=parametersGG1, inc_warmup=FALSE)
post.parGG1 <- extract(fitGG1, parametersGG1, permuted=TRUE, inc_warmup=FALSE)
fitGG1.bridge <- bridge_sampler(fitGG1, silent=TRUE)
####################################################################


# Compute posterior model probabilities (assuming equal prior model probabilities)
post_prob(fitLN1.bridge, fitG1.bridge, fitPGW1.bridge, fitGG1.bridge)
