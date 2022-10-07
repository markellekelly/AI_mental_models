import stan
import pandas as pd
import numpy as np

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
    int n_topics;
    int K; 
    int<lower=1,upper=K> Y[n_items];
    vector[n_topics] a_true;
    vector[n_items] d_true;
    vector[n_items] topics;
    vector[K-1] v;
  } 
  parameters {
    real noise_a1;
    real noise_a2;
    real noise_a3;
    real noise_a4;
    real gamma;
    real lambda;
    real<lower=0> sigma;
    real noise_d;
    real<lower=0> sigma_d;
    real<lower=0> sigma_a;
  }  
  transformed parameters{
    vector[n_items] d;
    vector[n_topics] a; 
    vector[n_topics] noise_a = [noise_a1, noise_a2, noise_a3, noise_a4]';
    a = a_true + noise_a;
    d = gamma*d_true + lambda + noise_d;

    matrix[n_items,n_topics] loadings;
    for (i in 1:n_items){
        int topic = real_to_int(topics[i]);
        row_vector[n_topics] col;
        col = rep_row_vector(0,n_topics);
        col[topic] = 1;
        loadings[i] = col;
    }
  }
  model {
    sigma_d ~ cauchy(0,2);
    sigma_a ~ cauchy(0,2);
    noise_a1 ~ normal(0, sigma_a);
    noise_a2 ~ normal(0, sigma_a);
    noise_a3 ~ normal(0, sigma_a);
    noise_a4 ~ normal(0, sigma_a);
    sigma ~ cauchy(0,2);
    gamma ~ std_normal();
    lambda ~ std_normal();
    noise_d ~ normal(0, sigma_d);
    for (j in 1:n_items){
        real p;
        vector[K] Pmf;
        p = inv_logit(loadings[j] * a - d[j]);
        Pmf[1] = Phi((v[1] - p)/sigma);
        Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
        for (k in 2:(K-1)){ 
            Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
        Y[j] ~ categorical(Pmf);
    }
  }
  """

participant_index = 10

df = pd.read_csv('self_data_pilot.csv',index_col=0)
dat = np.array(df)[participant_index].tolist()

params_df = pd.read_csv('md_fit_pilot.csv', index_col=0)
art_abs_cols = ['a_true.'+str(i)+'.1' for i in range(1,28)]
games_abs_cols = ['a_true.'+str(i)+'.2' for i in range(1,28)]
cities_abs_cols = ['a_true.'+str(i)+'.3' for i in range(1,28)]
math_abs_cols = ['a_true.'+str(i)+'.4' for i in range(1,28)]
true_abilities = [np.mean(params_df['a_true.'+str(participant_index+1)+'.'+str(j)]) for j in range(1,5)]
true_difficulties = [np.mean(params_df['d_true' + str(i)]) for i in range(1,17)]

data = {"n_items": 16,
            "n_topics":4,
            "K": 12+1,
            "Y": dat,
            "a_true":true_abilities,
            "d_true":true_difficulties,
            "topics":[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "v":[0, 0.09090909, 0.18181818, 0.27272727, 0.36363636,
       0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
       0.90909091, 1]}

posterior = stan.build(stan_code, data=data)
fit = posterior.sample(num_chains=2, num_warmup=1000, num_samples=1000)
df = fit.to_frame()
df.to_csv('md_fit_self_pilot.csv')