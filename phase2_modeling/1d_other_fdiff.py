import stan
import pandas as pd
import numpy as np
import arviz as az
import xarray

stan_code = """
  data {
    int n_items;
    int K; 
    int<lower=1,upper=K> Y[n_items];
    vector[K-1] v;
    real sigma;
  } 
  parameters {
    real a_other;    
    vector[n_items] d_other;    
    real mu_d;
    real<lower=0> sigma_d;
  }  
  model {
    a_other ~ normal(0,1);    
    d_other ~ normal(mu_d, sigma_d);    
    mu_d ~ normal(0,2);    
    sigma_d ~ cauchy(0,5);
    for (j in 1:n_items){
        real p;
        vector[K] Pmf;
        p = inv_logit(a_other - d_other[j]);
        Pmf[1] = Phi((v[1] - p)/sigma);
        Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
        for (k in 2:(K-1)){ 
            Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
        Y[j] ~ categorical(Pmf);
    }
  }
  generated quantities {
    int<lower=1,upper=K> Y_hat[n_items];
    vector[n_items] log_lik;
    for (j in 1:n_items){
        real p_e;
        vector[K] Pmf_e;
        p_e = inv_logit(a_other - d_other[j]);
        Pmf_e[1] = Phi((v[1] - p_e)/sigma);
        Pmf_e[K] = 1 - Phi((v[K-1] - p_e)/sigma);
        for (k in 2:(K-1)){ 
            Pmf_e[k] = Phi((v[k] - p_e)/sigma) - Phi((v[k-1] - p_e)/sigma);}
        Y_hat[j] = categorical_rng(Pmf_e);
        log_lik[j] = categorical_lpmf(Y[j] | Pmf_e);
    }
  }
  """


df = pd.read_csv('other_data_phase2.csv',index_col=0)
params_df = pd.read_csv('1d_true_phase2.csv', index_col=0)

for target in ["HU","AI"]:

  log_liks = []
  loos = []

  if target =="HU":
    ind = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26, 29, 30, 31]
  else:
    ind = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 32, 33]

  for participant_index in range(34):
    if participant_index in ind:
      dat = np.array(df)[participant_index].tolist()
      true_sigma = np.mean(params_df['sigma'])

      data = {"n_items": 16,
                  "K": 12+1,
                  "Y": dat,
                  "sigma":true_sigma,
                  "v":[0, 0.09090909, 0.18181818, 0.27272727, 0.36363636,
            0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
            0.90909091, 1]}

      posterior = stan.build(stan_code, data=data)
      fit = posterior.sample(num_chains=2, num_warmup=500, num_samples=1200)
      df_tmp = fit.to_frame()
      part_log_liks = [np.log(np.mean(np.exp(df_tmp['log_lik.'+str(i)]))) for i in range(1, 17)]
      log_liks.append(part_log_liks)

      stan_data = az.from_pystan(
        posterior=fit,
        posterior_predictive="Y_hat",
        log_likelihood={"Y": "log_lik"},
        coords={"items":np.arange(16)},
        dims={
            "Y":["items"],
            "d_other": ["items"],
            "Y_hat":["items"],
            "log_lik":["items"]
        },
      )

      stan_data.add_groups(
          observed_data=xarray.DataArray(data["Y"]),
          dims={"Y": [np.arange(16)]},
      )

      loo = az.loo(stan_data)['loo']
      loos.append(loo)

  df_ll = pd.DataFrame(log_liks)
  fname = "1d_other_fdiff_phase2_" + target + ".csv"
  df_ll.to_csv(fname)

  df_loo = pd.DataFrame(loos)
  fname = "1d_other_fdiff_phase2_loo_" + target + ".csv"
  df_loo.to_csv(fname)
