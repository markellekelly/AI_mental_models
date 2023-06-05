import stan
import pandas as pd
import numpy as np
import arviz as az
import xarray
import sys

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
    int<lower=1> n_items;
    int<lower=1> n_topics;
    vector[n_items] topics;
    vector[n_topics] a_true;
    vector[n_items] d_true;
    int<lower=1> K; 
    array[n_items] int<lower=1,upper=K> Y;
    vector[K-1] v;
} 
parameters { 
    real gamma;
    real lambda;
    real<lower=0> sigma;
    real<lower=0> sigma_d;
    real<lower=0> sigma_a;
    vector[n_items] d;
    vector[n_topics] a;
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
}
model {
    sigma_d ~ cauchy(0,2);
    sigma_a ~ cauchy(0,2);
    for (i in 1:n_topics){
        a[i] ~ normal(a_true[i], sigma_a);
    }
    sigma ~ cauchy(0,2);
    gamma ~ std_normal();
    lambda ~ std_normal();
    d ~ normal(gamma*d_true + lambda, sigma_d);
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
generated quantities {
    int<lower=1,upper=K> Y_hat[n_items];
    vector[n_items] log_lik;
    for (j in 1:n_items){
        real p_e;
        vector[K] Pmf_e;
        p_e = inv_logit(loadings[j] * a - d[j]);
        Pmf_e[1] = Phi((v[1] - p_e)/sigma);
        Pmf_e[K] = 1 - Phi((v[K-1] - p_e)/sigma);
        for (k in 2:(K-1)){ 
            Pmf_e[k] = Phi((v[k] - p_e)/sigma) - Phi((v[k-1] - p_e)/sigma);}
        Y_hat[j] = categorical_rng(Pmf_e);
        log_lik[j] = categorical_lpmf(Y[j] | Pmf_e);
    }
}
"""


def get_data():

    df = pd.read_csv('self_data.csv', index_col=0)
    df.drop(['feedback','human','highacc'], axis=1, inplace=True)

    params_df = pd.read_csv('fits/md_true.csv', index_col=0)

    return df, params_df


def run_model(code, data, num_chains=3, num_warmup=800, num_samples=1500):
    '''build and fit model given data dict'''

    posterior = stan.build(code, data=data)
    
    fit = posterior.sample(num_chains=num_chains,
                            num_warmup=num_warmup, 
                            num_samples=num_samples)

    return fit


def gen_arviz_data(data, fit, items=16):
    '''generate an arviz data structure for computing WAIC and LOO scores'''

    stan_data = az.from_pystan(
        posterior=fit,
        posterior_predictive="Y_hat",
        log_likelihood={"Y": "log_lik"},
        coords={"items":np.arange(items),
                "topics":np.arange(4)},
        dims={
            "Y":["items"],
            "d": ["items"],
            "a": ["topics"],
            "Y_hat":["items"],
            "log_lik":["items"]
        },
    )

    stan_data.add_groups(
        observed_data=xarray.DataArray(data["Y"]),
        dims={"Y": [np.arange(items)]},
    )

    return stan_data


def full_model(data, params_df, participants, k):
    '''
    learn a multidimensional IRT model, trained and evaluated on dat
    return the fit dataframe, waic, and loo scores
    '''
    global stan_code
    log_liks = []; loos = []; waics = []; results = []
    
    data = {
        "n_items": 16,
        "n_topics":4,
        "topics":[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "K": k+1,
        "v": list(np.linspace(0,1,k)) # continuous -> discrete cutoff points
    }

    for participant_index in range(participants):

        data['a_true'] = [np.mean(params_df['a_true.'+str(participant_index+1)+'.'+str(j)]) for j in range(1,5)]
        data['d_true'] = [np.mean(params_df['d_true.' + str(i)]) for i in range(1,17)]
        data["Y"] = np.array(df)[participant_index].tolist()

        fit = run_model(stan_code, data, num_chains=2, num_warmup=500, num_samples=1200)
        stan_data = gen_arviz_data(data, fit)

        out = fit.to_frame()
        results.append([np.mean(out[col]) for col in out.columns])
        log_liks.append([np.log(np.mean(np.exp(out['log_lik.'+str(i)]))) for i in range(1, 17)])
        waics.append(az.waic(stan_data).waic)
        loos.append(az.loo(stan_data).loo)

    results_df = pd.DataFrame(results, columns=out.columns)

    return results_df, pd.DataFrame(log_liks), waics, loos



if __name__ == "__main__":

    fname = "fits/md_self"
    df, params_df = get_data()
    participants = len(df)
    k=12

    res, out, waic, loo = full_model(df, params_df, participants, k)
        
    # save results
    res.to_csv(fname + ".csv")
    out.to_csv(fname + "_LLs.csv")
    with open(fname+"_metrics.csv", "w") as f:
        f.write(str(waic))
        f.write("\n")
        f.write(str(loo))
