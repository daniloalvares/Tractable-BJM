rm(list=ls())

# Required sources
require(rstanarm)
require(rstan)
require(bridgesampling)
require(joineR)

options(mc.cores = parallel::detectCores())

# Longitudinal and survival datasets
dataL <- epileptic[, c(1:3,9:11)]
dataS <- UniqueVariables(epileptic, c(4,6,9:11), "id")

# Numeric binary variables (treat and gender)
dataL$treat <- as.numeric(dataL$treat)-1    # CBZ=0 and LTG=1
dataS$treat <- as.numeric(dataS$treat)-1
dataL$gender <- as.numeric(dataL$gender)-1  # F=0 and M=1
dataS$gender <- as.numeric(dataS$gender)-1

# Scaling and centering the age
dataL$age <- as.numeric(scale(dataL$age))
dataS$age <- as.numeric(scale(dataS$age))

# Data for the Stan model
X <- dataS[,c(3,4,2)]                     # design matrix X                       
n <- nrow(X)                              # total number of observations
y <- dataL$dose                           # longitudinal outcomes
ID <- dataL$id                            # patient IDs
status <- dataS$with.status2              # vital status (0 = censored, 1 = inadequate seizure control, 2 = unacceptable adverse effects)
times <- dataS$with.time/365.25           # times to event
visits <- dataL$time/365.25               # visit times for repeated observations
N <- length(y)                            # total number of longitudinal outcomes
indobs <- which(status!=0)                # observed survival times indicator
nobs <- length(indobs)                    # number of observed survival times
status_I <- as.numeric(status[indobs]==1) # inadequate seizure control indicator
status_U <- as.numeric(status[indobs]==2) # unacceptable adverse effects indicator


####################################################################
#             JOINT MODEL WITH LOGNORMAL HAZARD (M1)               #
####################################################################
fitLN1 <- stan(file = "JMCR-LN-M1.stan", 
             data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,status_I=status_I,status_U=status_U,nbetas=ncol(X),nlambdas=ncol(X)),        
             warmup = 1000,                 
             iter = 10000,
             chains = 1,
             seed = 2021,
             cores = getOption("mc.cores",1)) 

print(fitLN1)

parametersLN1 <- c("beta_tilde", "gamma", "beta", "Var_b", "rho", "Var_e", "lambda_I", "alpha_I", "eta_I", "nu_I", "lambda_U", "alpha_U", "eta_U", "nu_U")
# plot(parametersLN1, plotfun="trace", pars=parametersLN1, inc_warmup=FALSE)
post.parLN1 <- extract(fitLN1, parametersLN1, permuted=TRUE, inc_warmup=FALSE)
fitLN1.bridge <- bridge_sampler(fitLN1, silent=TRUE)
####################################################################


####################################################################
#             JOINT MODEL WITH LOGNORMAL HAZARD (M2)               #
####################################################################
fitLN2 <- stan(file = "JMCR-LN-M2.stan", 
             data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,status_I=status_I,status_U=status_U,nbetas=ncol(X),nlambdas=ncol(X)),        
             warmup = 1000,                 
             iter = 10000,
             chains = 1,
             seed = 2021,
             cores = getOption("mc.cores",1))

print(fitLN2)

parametersLN2 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda_I", "alpha_I", "eta_I", "nu_I", "lambda_U", "alpha_U", "eta_U", "nu_U")
# plot(fitLN2, plotfun="trace", pars=parametersLN2, inc_warmup=FALSE)
post.parLN2 <- extract(fitLN2, parametersLN2, permuted=TRUE, inc_warmup=FALSE)
fitLN2.bridge <- bridge_sampler(fitLN2, silent=TRUE)
####################################################################


####################################################################
#             JOINT MODEL WITH LOGNORMAL HAZARD (M3)               #
####################################################################
fitLN3 <- stan(file = "JMCR-LN-M3.stan", 
             data = list(y=y,N=N,n=n,nobs=nobs,X1=X,ID=ID,visits=visits,times=times,indobs=indobs,status_I=status_I,status_U=status_U,nbetas=ncol(X),nlambdas=ncol(X)),        
             warmup = 1000,                 
             iter = 10000,
             chains = 1,
             seed = 2021,
             cores = getOption("mc.cores",1)) 

print(fitLN3)

parametersLN3 <- c("beta_tilde", "beta", "Var_b", "rho", "Var_e", "lambda_I", "alpha_I", "eta_I", "nu_I", "lambda_U", "alpha_U", "eta_U", "nu_U")
# plot(fitLN3, plotfun="trace", pars=parametersLN3, inc_warmup=FALSE)
post.parLN3 <- extract(fitLN3, parametersLN3, permuted=TRUE, inc_warmup=FALSE)
fitLN3.bridge <- bridge_sampler(fitLN3, silent=TRUE)
####################################################################


# Compute Bayes factor
bf(fitLN1.bridge, fitLN2.bridge)
bf(fitLN1.bridge, fitLN3.bridge)
bf(fitLN2.bridge, fitLN3.bridge)

# Compute posterior model probabilities (assuming equal prior model probabilities)
post_prob(fitLN1.bridge, fitLN2.bridge, fitLN3.bridge)
