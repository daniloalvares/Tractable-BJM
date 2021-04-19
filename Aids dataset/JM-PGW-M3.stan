functions{
// ------------------------------------------------------
//      POWER GENERALISED WEIBULL SURVIVAL SUBMODEL                
// ------------------------------------------------------
    // Power Generalised Weibull hazard function
    vector loghazPGW(vector t, real eta, real nu, real delta){
         vector[num_elements(t)] out;  
         for(i in 1:num_elements(t)){
            out[i] = log(nu) - log(delta) - nu*log(eta) + (nu-1)*log(t[i]) + (1/delta - 1)*log1p( pow(t[i]/eta, nu) );
         }
         return out;
    }                                                                                     

    // Power Generalised Weibull cumulative hazard function
    vector cumhazPGW(vector t, real eta, real nu, real delta){
         vector[num_elements(t)] out;  
         for(i in 1:num_elements(t)){
            out[i] = -1 + pow(1 + pow(t[i]/eta, nu), 1/delta);
         }
         return out;
    }
// ------------------------------------------------------ 


// ------------------------------------------------------
//     LINEAR PREDICTOR FOR THE LONGITUDINAL SUBMODEL                
// ------------------------------------------------------ 
    vector linear_predictor(matrix X, vector visits, int[] ID, vector beta_tilde, vector beta, matrix bi){
         int N = num_elements(visits);
         vector[N] out;

         out = beta_tilde[1] + beta_tilde[2]*visits + X[ID,]*beta + bi[ID,1] + rows_dot_product(bi[ID,2],visits);

         return out;
    } 
// ------------------------------------------------------ 
}


data{
  int N;
  int n;
  int nobs;
  int nbetas;
  int nlambdas;
  vector[N] y;
  matrix[n,nbetas] X1;
  int<lower=1,upper=n> ID[N];
  vector[N] visits;
  vector[n] times;
  int<lower=1,upper=n> indobs[nobs];
}


parameters{
  vector[2] beta_tilde;
  vector[nbetas] beta;
  vector[nlambdas] lambda;
  real alpha;
  real<lower=0> eta;
  real<lower=0> nu;
  real<lower=0> delta;
  real<lower=0> Var_b[2];
  real<lower=-1, upper=1> rho;
  real<lower=0> Var_e;  
  matrix[n,2] bi;
}


transformed parameters{
  cov_matrix[2] Sigma;

  Sigma[1,1] = Var_b[1];
  Sigma[2,2] = Var_b[2];
  Sigma[1,2] = sqrt(Var_b[1]*Var_b[2])*rho;
  Sigma[2,1] = Sigma[1,2];
}


model{
// ------------------------------------------------------
//        LOG-LIKELIHOOD FOR LONGITUDINAL SUBMODEL                
// ------------------------------------------------------
{
   vector[N] linpred; 

   // Linear predictor
   linpred = linear_predictor(X1, visits, ID, beta_tilde, beta, bi);

   // Longitudinal Normal log-likelihood
   target += normal_lpdf(y | linpred, sqrt(Var_e));
}  
// ------------------------------------------------------
//          LOG-LIKELIHOOD FOR SURVIVAL SUBMODEL                
// ------------------------------------------------------
{
   vector[n] lmodelt;
   vector[n] lmodelh;
   vector[n] explmodelt;
   vector[n] t1;
   vector[nobs] t2;
   vector[nobs] lhaz;
   vector[n] lsurv;

   lmodelt = rep_vector(0.0,n);
   lmodelh = X1*lambda + alpha*bi[1:n,1];
   explmodelt = exp(lmodelt);    
    
   t1 = rows_dot_product(times,explmodelt);
   t2 = t1[indobs];

   // Log-hazard function
   lhaz = loghazPGW( t2, eta, nu, delta ) + lmodelh[indobs];
   // Log-survival function
   lsurv = -rows_dot_product(cumhazPGW( t1, eta, nu, delta ), exp(lmodelh - lmodelt));
    
   // Survival log-likelihood
   target += sum(lhaz) + sum(lsurv); 
} 
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
   // Longitudinal fixed effects
   target += normal_lpdf(beta_tilde | 0, 100);
   target += normal_lpdf(beta | 0, 100);

   // Survival fixed effects
   target += normal_lpdf(lambda | 0, 100);

   // PGW scale parameter
   target += cauchy_lpdf(eta | 0, 1);

   // PGW shape parameters
   target += cauchy_lpdf(nu | 0, 1);
   target += gamma_lpdf(delta | 0.63, 0.546);

   // Association parameters
   target += normal_lpdf(alpha | 0, 100);
   
   // Random-effects
   for(i in 1:n){ target += multi_normal_lpdf(bi[i,1:2] | rep_vector(0.0,2), Sigma); }

   // Random-effects variance
   target += inv_gamma_lpdf(Var_b | 0.01, 0.01);

   // Random-effects correlation
   target += beta_lpdf((rho+1)/2 | 0.5, 0.5);

   // Residual error variance
   target += inv_gamma_lpdf(Var_e | 0.01, 0.01); 
}