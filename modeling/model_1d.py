import stan
import pandas as pd
import numpy as np
import arviz as az
import xarray
import sys

stan_code = """
data {
    int<lower=1> n_items;
    int<lower=1> n_participants;
    int<lower=1> K; 
    array[n_participants,n_items] int<lower=1,upper=K> Y;
    vector[K-1] v;
    int<lower=1> n_skip;
    int<lower=1,upper=n_participants> skip_p;
    array[n_participants, n_skip] int<lower=0,upper=16> skip_i;
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
            for (skip in 1:n_skip) {
                if (j == skip_i[skip_p,skip]) {
                    continue;
                }
            }
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

def process_args(args):
    '''parse command line arguments, determine setting (true performance,
       self-assessment, or other-assessment), agent type (other humans, AI
       agents, or both), and feedback condition (both, no feedback, feedback)'''

    def err():
        print("Usage: python3 model_1d.py setting agent_type feedback_condition loocv")
        print("\t setting: true, self, or other")
        print("\t agent_type: both, h (other humans only), or a (AI agents only)")
        print("\t feedback_cond: both, nofb (no-feedback only), or fb (feedback only)")
        print("\t loocv: 0 (no CV, train/evaluate on all data) or 1 (perform LOO CV)")
        sys.exit(1)

    fname = "fits/1d_" # build a name for results file that describes the settings
    num_args = 4
    if not args or len(args) < num_args:
        err()

    setting = args[0].lower()
    settings = ['true','self','other']
    if setting in settings:
        fname += setting
        setting = settings.index(setting)
    else:
        err()


    agent_type = args[1].lower()
    if agent_type not in ['both','h','a']:
        err()
    if agent_type != 'both':
        fname += "_" + agent_type

    feedback_type = args[2].lower()
    if feedback_type not in ['both','nofb','fb']:
        err()
    if feedback_type != 'both':
        fname += "_" + feedback_type


    try:
        loocv = int(args[3])
    except:
        err()
    if loocv == 1:
        fname += '_CV'
    elif loocv != 0: 
        err()

    return setting, agent_type, feedback_type, loocv, fname


def get_data(setting, agent_type, feedback_type):
    # load relevant data (sub)set
    data_files = ['true_data.csv', 'self_data.csv', 'other_data.csv']
    df = pd.read_csv(data_files[setting], index_col=0)

    if agent_type == 'h':
        df = df[df['human']==True]
    elif agent_type == 'a':
        df = df[df['human']==False]

    if feedback_type == 'nofb':
        df=df[df['feedback']==False]
    elif feedback_type == 'fb':
        df=df[df['feedback']==True]
    
    ind = list(df.index)
    df.drop(['feedback','human','highacc'], axis=1, inplace=True)
    dat = np.array(df).tolist()

    return ind, dat


def run_model(code, data, num_chains=3, num_warmup=800, num_samples=1500):
    '''build and fit model given data dict'''

    posterior = stan.build(code, data=data)
    
    fit = posterior.sample(num_chains=num_chains,
                            num_warmup=num_warmup, 
                            num_samples=num_samples)

    return fit


def gen_arviz_data(data, fit, participants, items=16):
    '''generate an arviz data structure for computing WAIC and LOO scores'''

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
        dims={"Y": [np.arange(participants),np.arange(items)]},
    )

    return stan_data


def full_model(data, participants, k):
    '''
    learn a one-dimensional IRT model, trained and evaluated on dat
    return the fit dataframe, waic, and loo scores
    '''
    global stan_code

    fit = run_model(stan_code, data)
    stan_data = gen_arviz_data(data, fit, participants)

    out = fit.to_frame()
    waic = az.waic(stan_data)
    loo = az.loo(stan_data)

    return out, waic, loo


def cv_model(data, participants, k):
    '''
    learn a one-dimensional IRT model, evaluated on held-out data points
    return log-likelihoods, waic, and loo scores
    '''
    global stan_code

    waics = []; loos = []; log_liks = []

    for i in range(1,participants+1):

        data['skip_p'] = i # hold out relevant problem sets for participant i
        fit = run_model(stan_code, data, num_chains=2, num_warmup=600, num_samples=900)
        stan_data = gen_arviz_data(data, fit, participants)

        # save log-likelihoods (on held-out data), WAIC, and LOO scores
        out = fit.to_frame()
        log_liks.append([np.log(np.mean(np.exp(out['log_lik.'+str(i)+'.'+str(j)]))) for j in skip_i[i-1]])
        waics.append(az.waic(stan_data).waic)
        loos.append(az.loo(stan_data).loo)

    return pd.DataFrame(log_liks), waics, loos


if __name__ == "__main__":

    # read in command line args and get corresponding dataset
    setting, agent_type, feedback_type, loocv, fname = process_args(sys.argv[1:])
    ind, dat = get_data(setting, agent_type, feedback_type)
    participants = len(dat)
    k=12

    data = {
        "n_items": 16,
        "n_participants": participants,
        "K": k+1,
        "Y": dat,
        "v": list(np.linspace(0,1,k)), # continuous -> discrete cutoff points
        "skip_p": 1
    }

    if loocv: # cross-validation
        skip_i = np.array(pd.read_csv('last4.csv', index_col=0, header=0).iloc[ind])
        data["skip_i"] = skip_i.tolist()
        data["n_skip"] = len(skip_i[0])
        
        out, waic, loo = cv_model(data, participants, k)

    else: # full dataset
        data["skip_i"] = np.zeros((participants,1), dtype=int).tolist()
        data["n_skip"] = 1

        out, waic, loo = full_model(data, participants, k)
        
    # save results
    out.to_csv(fname + ".csv")
    with open(fname+"_metrics.csv", "w") as f:
        f.write(str(waic))
        f.write("\n")
        f.write(str(loo))
