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
    int n_participants;
    int K; 
    int<lower=1,upper=K> Y[n_participants,n_items];
    vector[n_items] topics;
    vector[K-1] v;
  } 
  parameters {
    corr_matrix[n_topics] Omega;      
    vector<lower=0>[n_topics] tau;    
    matrix[n_participants, n_topics] a_true;

    real<lower=0> sigma_d;

    real mu_d1;
    real mu_d2;
    real mu_d3;
    real mu_d4;
    real mu_d5;
    real mu_d6;
    real mu_d7;
    real mu_d8;
    real mu_d9;
    real mu_d10;
    real mu_d11;
    real mu_d12;
    real mu_d13;
    real mu_d14;
    real mu_d15;
    real mu_d16;
    
    real d_true1;
    real d_true2;
    real d_true3;
    real d_true4;
    real d_true5;
    real d_true6;
    real d_true7;
    real d_true8;
    real d_true9;
    real d_true10;
    real d_true11;
    real d_true12;
    real d_true13;
    real d_true14;
    real d_true15;
    real d_true16;

    real<lower=0> sigma;
  }  
  transformed parameters{

    matrix[n_items,n_topics] loadings;
    for (i in 1:n_items){
        int topic = real_to_int(topics[i]);
        row_vector[n_topics] col;
        col = rep_row_vector(0,n_topics);
        col[topic] = 1;
        loadings[i] = col;
    }

    vector[16] d = [d_true1,d_true2,d_true3,d_true4,d_true5,d_true6,d_true7,d_true8,d_true9,d_true10,
        d_true11,d_true12,d_true13,d_true14,d_true15,d_true16]';

  }
  model {
    tau ~ cauchy(0, 2.5);
    Omega ~ lkj_corr(2);
    
    row_vector[n_topics] zeros;
    zeros = rep_row_vector(0, n_topics);
    for (i in 1:n_participants){
        a_true[i] ~ multi_normal(zeros,quad_form_diag(Omega, tau));
    }

    sigma_d ~ cauchy(0,5);
    mu_d1 ~ std_normal();
    mu_d2 ~ std_normal();
    mu_d3 ~ std_normal();
    mu_d4 ~ std_normal();
    mu_d5 ~ std_normal();
    mu_d6 ~ std_normal();
    mu_d7 ~ std_normal();
    mu_d8 ~ std_normal();
    mu_d9 ~ std_normal();
    mu_d10 ~ std_normal();
    mu_d11 ~ std_normal();
    mu_d12 ~ std_normal();
    mu_d13 ~ std_normal();
    mu_d14 ~ std_normal();
    mu_d15 ~ std_normal();
    mu_d16 ~ std_normal();
    d_true1 ~ normal(mu_d1, sigma_d);
    d_true2 ~ normal(mu_d2, sigma_d);
    d_true3 ~ normal(mu_d3, sigma_d);
    d_true4 ~ normal(mu_d4, sigma_d);
    d_true5 ~ normal(mu_d5, sigma_d);
    d_true6 ~ normal(mu_d6, sigma_d);
    d_true7 ~ normal(mu_d7, sigma_d);
    d_true8 ~ normal(mu_d8, sigma_d);
    d_true9 ~ normal(mu_d9, sigma_d);
    d_true10 ~ normal(mu_d10, sigma_d);
    d_true11 ~ normal(mu_d11, sigma_d);
    d_true12 ~ normal(mu_d12, sigma_d);
    d_true13 ~ normal(mu_d13, sigma_d);
    d_true14 ~ normal(mu_d14, sigma_d);
    d_true15 ~ normal(mu_d15, sigma_d);
    d_true16 ~ normal(mu_d16, sigma_d);
  
    sigma ~ cauchy(0,2);

    for (i in 1:n_participants){
        for (j in 1:n_items){
            real p;
            vector[K] Pmf;
            p = inv_logit(loadings[j] * a_true[i]' - d[j]);
            Pmf[1] = Phi((v[1] - p)/sigma);
            Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
            for (k in 2:(K-1)){ 
                Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
            Y[i,j] ~ categorical(Pmf);
        }
    }
  }
  """

df = pd.read_csv('data_pilot.csv',index_col=0)
dat = np.array(df).tolist()
participants = len(df)

data = {"n_items": 16,
            "n_topics":4,
            "n_participants":participants,
            "K": 12+1,
            "Y": dat,
            "topics":[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "v":[0, 0.09090909, 0.18181818, 0.27272727, 0.36363636,
       0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
       0.90909091, 1]}

posterior = stan.build(stan_code, data=data)
fit = posterior.sample(num_chains=2, num_warmup=1000, num_samples=1000)
df = fit.to_frame()
df.to_csv('md_fit_pilot.csv')