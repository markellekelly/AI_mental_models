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
    int n_participants;
    int K; 
    int<lower=1,upper=K> Y[n_participants,n_items];
    vector[K-1] v;
  } 
  parameters {  
    vector[n_participants] a_true;

    real<lower=0> sigma_d;

    real<lower=0> mu_d;
    real d_true[n_items];

    real<lower=0> sigma;
  }  
  model {
    for (i in 1:n_participants){
        a_true[i] ~ std_normal();
    }

    sigma_d ~ cauchy(0,5);
    mu_d ~ normal(0,2);
    for (i in 1:n_items){
      d_true[i] ~ normal(mu_d, sigma_d);
    }
  
    sigma ~ cauchy(0,2);
    
    for (i in 1:n_participants){
      for (j in 1:n_items){
            real p;
            vector[K] Pmf;
            p = inv_logit(a_true[i] - d_true[j]);
            Pmf[1] = Phi((v[1] - p)/sigma);
            Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
            for (k in 2:(K-1)){ 
                Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
            Y[i,j] ~ categorical(Pmf);
        }
      }
    }
    generated quantities {
      int<lower=1,upper=K> Y_hat[n_participants,n_items];
      matrix[n_participants,n_items] log_lik;
      for (i in 1:n_participants){
        for (j in 1:n_items){
          vector[K] gamma;
          real p_r;
          vector[K] Pmf_r;
          p_r = inv_logit(a_true[i] - d_true[j]);
          Pmf_r[1] = Phi((v[1] - p_r)/sigma);
          Pmf_r[K] = 1 - Phi((v[K-1] - p_r)/sigma);
          for (k in 2:(K-1)){ 
              Pmf_r[k] = Phi((v[k] - p_r)/sigma) - Phi((v[k-1] - p_r)/sigma);}
          Y_hat[i,j] = categorical_rng(Pmf_r);
          log_lik[i,j] = categorical_lpmf(Y[i,j] | Pmf_r);
        }
      }
}
  """

sim = False
filename = "1d_true_phase2.csv" # set to None to save no file

if sim:
  participants=50
  df = pd.read_csv('md_true_sim_mod.csv')
  n=1900
  mat = []
  for i in range(1,participants+1):
      row = []
      for j in range(1,25):
          col = "Y."+str(i)+"."+str(j)
          row.append(int(df.iloc[n][col]))
      mat.append(row)
  dat = list(np.array(mat))

else:
  df = pd.read_csv('data_phase2.csv',index_col=0)
  dat = np.array(df).tolist()
  participants = len(df)

data = {"n_items": 16,
            "n_participants":participants,
            "K": 12+1,
            "Y": dat,
            "v":[0, 0.09090909, 0.18181818, 0.27272727, 0.36363636,
        0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
        0.90909091, 1]}
posterior = stan.build(stan_code, data=data)
fit = posterior.sample(num_chains=3, num_warmup=800, num_samples=1500)
df = fit.to_frame()
if filename:
  df.to_csv(filename)

stan_data = az.from_pystan(
    posterior=fit,
    posterior_predictive="Y_hat",
    log_likelihood={"Y": "log_lik"},
    coords={"participants": np.arange(participants),
            "items":np.arange(16)},
    dims={
        "Y":["participants","items"],
        "d": ["items"],
        "a_true": ["participants"],
        "Y_hat":["participants","items"],
        "d_true": ["items"],
        "log_lik":["participants","items"]
    },
)

stan_data.add_groups(
    observed_data=xarray.DataArray(data["Y"]),
    dims={"Y": [np.arange(participants),np.arange(16)]},
)

print(az.waic(stan_data))
print(az.loo(stan_data))