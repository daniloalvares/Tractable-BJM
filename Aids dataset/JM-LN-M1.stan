functions{
// ------------------------------------------------------
//              LOGNORMAL SURVIVAL SUBMODEL                
// ------------------------------------------------------ 
    // Lognormal hazard function
    vector loghazLN(vector t, real mu, real eta){
         vector[num_elements(t)] out;  
         for(i in 1:num_elements(t)){
            out[i] = lognormal_lpdf(t[i] | mu, eta) - lognormal_lccdf(t[i] | mu, eta);
         }
         return out;
    }                                                                                     

    // Lognormal cumulative hazard function
    vector cumhazLN(vector t, real mu, real eta){
         vector[num_elements(t)] out;  
         for(i in 1:num_elements(t)){
            out[i] = -lognormal_lccdf(t[i] | mu, eta);
         }
         return out;
    }
// ------------------------------------------------------ 


// ------------------------------------------------------
//     LINEAR PREDICTOR FOR THE LONGITUDINAL SUBMODEL                
// ------------------------------------------------------ 
    vector linear_predictor(matrix X, vector visits, int[] ID, vector beta_tilde, real gamma, vector beta, matrix bi){
         int N = num_elements(visits);
         vector[N] out;

         out = beta_tilde[1] + beta_tilde[2]*visits + gamma*rows_dot_product(visits,X[ID,1]) + X[ID,]*beta + bi[ID,1] + rows_dot_product(bi[ID,2],visits);

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
  real gamma;
  vector[nbetas] beta;
  vector[nlambdas] lambda;
  vector[2] alpha;
  real mu;
  real<lower=0> eta;
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
   linpred = linear_predictor(X1, visits, ID, beta_tilde, gamma, beta, bi);

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

   lmodelt = alpha[2]*(gamma*X1[1:n,1] + bi[1:n,2]);
   lmodelh = X1*lambda + alpha[1]*bi[1:n,1];
   explmodelt = exp(lmodelt);    
    
   t1 = rows_dot_product(times,explmodelt);
   t2 = t1[indobs];

   // Log-hazard function
   lhaz = loghazLN( t2, mu, eta ) + lmodelh[indobs];
   // Log-survival function
   lsurv = -rows_dot_product(cumhazLN( t1, mu, eta ), exp(lmodelh - lmodelt));
    
   // Survival log-likelihood
   target += sum(lhaz) + sum(lsurv); 
} 
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
   // Longitudinal fixed effects
   target += normal_lpdf(beta_tilde | 0, 100);
   target += normal_lpdf(gamma | 0, 100);
   target += normal_lpdf(beta | 0, 100);

   // Survival fixed effects
   target += normal_lpdf(lambda[1] | 0, 100);

   // Lognormal location parameter
   target += normal_lpdf(mu | 0, 100); 

   // Lognormal scale parameter
   target += cauchy_lpdf(eta | 0, 1);

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