import stan
import pandas as pd
import numpy as np
import arviz as az
import xarray

stan_code = """
  functions {
    int real_to_int(real x){
        int current = 0;
        int done = 0;
        int ans = 0;
        while(done != 1) {
            if(x > current && x < current + 2){
                ans = current + 1;
                done = 1;
            } else {
                current += 1;
            }
        }
        return ans;
    }
  }
  data {
    int n_items;
    int K; 
    int<lower=1,upper=K> Y[n_items];
    real a_true;
    vector[n_items] d_true;
    vector[K-1] v;
  } 
  parameters {
    real gamma;
    real lambda;
    real<lower=0> sigma;
    vector[n_items] d;
    real a;
    real<lower=0> sigma_d;
    real<lower=0> sigma_a;
  }  
  model {
    sigma_d ~ cauchy(0,2);
    sigma_a ~ cauchy(0,2);
    sigma ~ cauchy(0,2);
    gamma ~ std_normal();
    lambda ~ std_normal();
    for (j in 1:n_items) {
      d[j] ~ normal(gamma*d_true[j] + lambda, sigma_d);
    }
    a ~ normal(a_true, sigma_a);
    for (j in 1:n_items){
        real p;
        vector[K] Pmf;
        p = inv_logit(a - d[j]);
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
        p_e = inv_logit(a - d[j]);
        Pmf_e[1] = Phi((v[1] - p_e)/sigma);
        Pmf_e[K] = 1 - Phi((v[K-1] - p_e)/sigma);
        for (k in 2:(K-1)){ 
            Pmf_e[k] = Phi((v[k] - p_e)/sigma) - Phi((v[k-1] - p_e)/sigma);}
        Y_hat[j] = categorical_rng(Pmf_e);
        log_lik[j] = categorical_lpmf(Y[j] | Pmf_e);
    }
  }
  """

#participant_index = 10

df = pd.read_csv('self_data_phase2.csv',index_col=0)
params_df = pd.read_csv('1d_true_phase2.csv', index_col=0)

log_liks = []
loos = []

for participant_index in range(34):
  dat = np.array(df)[participant_index].tolist()
  true_ability = np.mean(params_df['a_true.'+str(participant_index+1)])
  true_difficulties = [np.mean(params_df['d_true.' + str(i)]) for i in range(1,17)]

  data = {"n_items": 16,
              "K": 12+1,
              "Y": dat,
              "a_true":true_ability,
              "d_true":true_difficulties,
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
        "d": ["items"],
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

df = pd.DataFrame(log_liks)
df.to_csv('1d_self_phase2.csv')

df_loo = pd.DataFrame(loos)
df_loo.to_csv('1d_self_loos.csv')
