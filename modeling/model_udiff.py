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
    int n_items;
    int n_topics;
    int K; 
    array[n_items] int<lower=1,upper=K> Y;
    vector[n_topics] a_true;
    vector[n_items] d_true;
    real sigma;
    vector[n_items] problems;
    vector[n_items] topics;
    vector[K-1] v;
    int<lower=1,upper=16> end_ind;
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
generated quantities {
    array[n_items] int<lower=1,upper=K> Y_hat;
    vector[n_items] log_lik;
    for (j in 1:n_items){
        real p_e;
        vector[K] Pmf_e;
        int problem = real_to_int(problems[j]);
        p_e = inv_logit(loadings[j] * a_true - d_true[problem]);
        Pmf_e[1] = Phi((v[1] - p_e)/sigma);
        Pmf_e[K] = 1 - Phi((v[K-1] - p_e)/sigma);
        for (k in 2:(K-1)){ 
            Pmf_e[k] = Phi((v[k] - p_e)/sigma) - Phi((v[k-1] - p_e)/sigma);}
        Y_hat[j] = categorical_rng(Pmf_e);
        log_lik[j] = categorical_lpmf(Y[j] | Pmf_e);
    }
}
"""

def process_args(args):
    '''parse command line arguments, determine agent type (other humans, AI
       agents, or both) and feedback condition (both, no feedback, feedback)'''

    def err():
        print("Usage: python3 model_udiff.py agent_type feedback_condition loocv")
        print("\t agent_type: both, h (other humans only), or a (AI agents only)")
        print("\t feedback_cond: both, nofb (no-feedback only), or fb (feedback only)")
        print("\t loocv: 0 (no CV, train/evaluate on all data) or 1 (perform LOO CV)")
        sys.exit(1)

    fname = "fits/udiff" # build a name for results file that describes the settings
    num_args = 3
    if not args or len(args) < num_args:
        err()

    agent_type = args[0].lower()
    if agent_type not in ['both','h','a']:
        err()
    if agent_type != 'both':
        fname += "_" + agent_type

    feedback_type = args[1].lower()
    if feedback_type not in ['both','nofb','fb']:
        err()
    if feedback_type != 'both':
        fname += "_" + feedback_type

    try:
        loocv = int(args[2])
    except:
        err()
    if loocv == 1:
        fname += '_CV'
    elif loocv != 0: 
        err()

    return agent_type, feedback_type, loocv, fname


def get_data(agent_type, feedback_type):
    # load relevant data (sub)set
    df = pd.read_csv('new_data/other_data_o.csv',index_col=0)

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

    problems = pd.read_csv('problems.csv',index_col=0).iloc[ind].reset_index(drop=True)
    topics = pd.read_csv('topics.csv',index_col=0).iloc[ind].reset_index(drop=True)
    params_self = pd.read_csv('../fits/md_self.csv', index_col=0).iloc[ind].reset_index(drop=True)
    params_true = pd.read_csv('../fits/md_true.csv', index_col=0)

    return df, problems, topics, params_self, params_true


def gen_arviz_data(data, fit, items=16):
    '''generate an arviz data structure for computing WAIC and LOO scores'''

    stan_data = az.from_pystan(
        posterior=fit,
        posterior_predictive="Y_hat",
        log_likelihood={"Y": "log_lik"},
        coords={"items":np.arange(items)},
        dims={
            "Y":["items"],
            "Y_hat":["items"],
            "log_lik":["items"]
        },
    )

    stan_data.add_groups(
        observed_data=xarray.DataArray(data["Y"]),
        dims={"Y": [np.arange(items)]},
    )

    return stan_data


def full_model(data, df, participants, k, problems, topics, params_s):
    '''
    learn a multidimensional IRT model, trained and evaluated on data
    return the fit dataframe, waic, and loo scores
    '''
    global stan_code
    log_liks = []; waics=[]; loos=[]

    for participant_index in range(participants):

        data['a_true'] = [params_s.iloc[participant_index]['a.'+str(i)] for i in range(1,5)]
        data['d_true'] = [params_s.iloc[participant_index]['d.'+str(i)] for i in range(1,17)]
        data['problems'] = list(problems.iloc[participant_index])
        data['topics'] = list(topics.iloc[participant_index])
        data["Y"] = np.array(df)[participant_index].tolist()

        posterior = stan.build(stan_code, data=data)
        fit = posterior.fixed_param()
        stan_data = gen_arviz_data(data, fit)

        out = fit.to_frame()
        log_liks.append([np.log(np.mean(np.exp(out['log_lik.'+str(i)]))) for i in range(1, 17)])
        waics.append(az.waic(stan_data).waic)
        loos.append(az.loo(stan_data).loo)

    return pd.DataFrame(log_liks), waics, loos


def cv_model(data, df, participants, k, problems, topics, params_s):
    '''
    learn a multidimensional IRT model, evaluated on held-out data points
    return log-likelihoods, waic, and loo scores
    '''
    global stan_code
    log_liks = []
    data['a_true'] = [0,0,0,0]

    for participant_index in range(participants):

        data['a_true'] = [params_s.iloc[participant_index]['a.'+str(i)] for i in range(1,5)]
        data['d_true'] = [params_s.iloc[participant_index]['d.'+str(i)] for i in range(1,17)]
        data['problems'] = list(problems.iloc[participant_index])
        data['topics'] = list(topics.iloc[participant_index])
        data["Y"] = np.array(df)[participant_index].tolist()
        
        part_lls = []
        for end_ind in range(1,16):
            data["end_ind"] = end_ind
            posterior = stan.build(stan_code, data=data)
            fit = posterior.fixed_param()
            out = fit.to_frame()
            part_lls.append(np.log(np.mean(np.exp(out['log_lik.'+str(end_ind+1)]))))

        log_liks.append(part_lls)

    return pd.DataFrame(log_liks)


if __name__ == "__main__":

    # read in command line args and get corresponding dataset
    agent_type, feedback_type, loocv, fname = process_args(sys.argv[1:])
    df, problems, topics, params_s, params_t = get_data(agent_type, feedback_type)
    participants = len(df)
    k=12

    data = {
        "n_items": 16,
        "n_topics":4,
        "end_ind":16,
        "K": k+1,
        "v": list(np.linspace(0,1,k)), # continuous -> discrete cutoff points
        "sigma": np.mean(params_t['sigma'])
    }

    if loocv: # time-based holdout
        out = cv_model(data, df, participants, k, problems, topics, params_s)

    else: # full dataset
        out, waic, loo = full_model(data, df, participants, k, problems, topics, params_s)

        with open(fname+"_metrics.csv", "w") as f:
            f.write(str(waic))
            f.write("\n")
            f.write(str(loo))
    
    # save results
    out.to_csv(fname + ".csv")
