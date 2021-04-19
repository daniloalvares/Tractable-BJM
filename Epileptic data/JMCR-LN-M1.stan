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

         out = beta_tilde[1] + beta_tilde[2]*visits + gamma*rows_dot_product(visits,X[ID,3]) + X[ID,]*beta + bi[ID,1] + rows_dot_product(bi[ID,2],visits);

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
  vector[nobs] status_I;
  vector[nobs] status_U;
  int<lower=1,upper=n> indobs[nobs];
}


parameters{
  vector[2] beta_tilde;
  real gamma;  
  vector[nbetas] beta;
  vector[nlambdas] lambda_I;
  vector[nlambdas] lambda_U;  
  vector[2] alpha_I;
  vector[2] alpha_U;
  real eta_I;
  real eta_U;
  real<lower=0> nu_I;
  real<lower=0> nu_U;
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
   vector[n] lmodelt_I;
   vector[n] lmodelt_U;
   vector[n] lmodelh_I;
   vector[n] lmodelh_U;
   vector[n] explmodelt_I;
   vector[n] explmodelt_U;
   vector[n] t1_I;
   vector[n] t1_U;
   vector[nobs] t2_I;
   vector[nobs] t2_U;
   vector[nobs] lhaz_I;
   vector[nobs] lhaz_U;
   vector[n] lsurv_I;
   vector[n] lsurv_U;

   // Inadequate seizure control
   lmodelt_I = alpha_I[2]*(gamma*X1[1:n,3] + bi[1:n,2]);
   lmodelh_I = X1*lambda_I + alpha_I[1]*bi[1:n,1];
   explmodelt_I = exp(lmodelt_I);    
   
   t1_I = rows_dot_product(times,explmodelt_I);
   t2_I = t1_I[indobs];

   // Log-hazard function
   lhaz_I = loghazLN( t2_I, eta_I, nu_I ) + lmodelh_I[indobs];
   // Log-survival function
   lsurv_I = -rows_dot_product(cumhazLN( t1_I, eta_I, nu_I ), exp(lmodelh_I - lmodelt_I));

   // Unacceptable adverse effects
   lmodelt_U = alpha_U[2]*(gamma*X1[1:n,3] + bi[1:n,2]);
   lmodelh_U = X1*lambda_U + alpha_U[1]*bi[1:n,1];
   explmodelt_U = exp(lmodelt_U);    

   t1_U = rows_dot_product(times,explmodelt_U);
   t2_U = t1_U[indobs];

   // Log-hazard function
   lhaz_U = loghazLN( t2_U, eta_U, nu_U ) + lmodelh_U[indobs];
   // Log-survival function
   lsurv_U = -rows_dot_product(cumhazLN( t1_U, eta_U, nu_U ), exp(lmodelh_U - lmodelt_U));
    
   // Competing risks log-likelihood
   target += dot_product(lhaz_I,status_I) + dot_product(lhaz_U,status_U) + sum(lsurv_I) + sum(lsurv_U); 
} 
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
   // Longitudinal fixed effects
   target += normal_lpdf(beta_tilde | 0, 100);
   target += normal_lpdf(gamma | 0, 100);
   target += normal_lpdf(beta | 0, 100);

   // Survival fixed effects
   target += normal_lpdf(lambda_I | 0, 100);
   target += normal_lpdf(lambda_U | 0, 100);

   // Lognormal location parameter
   target += normal_lpdf(eta_I | 0, 100);
   target += normal_lpdf(eta_U | 0, 100);

   // Lognormal scale parameter
   target += cauchy_lpdf(nu_I | 0, 1);
   target += cauchy_lpdf(nu_U | 0, 1);

   // Association parameters
   target += normal_lpdf(alpha_I | 0, 100);
   target += normal_lpdf(alpha_U | 0, 100);
   
   // Random-effects
   for(i in 1:n){ target += multi_normal_lpdf(bi[i,1:2] | rep_vector(0.0,2), Sigma); }

   // Random-effects variance
   target += inv_gamma_lpdf(Var_b | 0.01, 0.01);

   // Random-effects correlation
   target += beta_lpdf((rho+1)/2 | 0.5, 0.5);
   
   // Residual error variance
   target += inv_gamma_lpdf(Var_e | 0.01, 0.01);   
}