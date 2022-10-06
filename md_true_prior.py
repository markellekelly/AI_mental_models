import stan

stan_code="""
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
    int n_topics;
    int K; 
    vector[n_items] topics;
    vector[K-1] v;
  } 
  generated quantities {
    matrix[n_participants, n_topics] a; //each row is a participant, with a column for each topic
    for (i in 1:n_participants){
      real a1 = normal_rng(0,1);
      real a2 = normal_rng(0,1);
      real a3 = normal_rng(0,1);
      real a4 = normal_rng(0,1);
      row_vector[n_topics] indv = [a1, a2, a3, a4];
      a[i] = indv;
    }
    vector[n_items] d;
    d = rep_vector(0.5, n_items);
    d[3] = 4;
    real sigma = 0.1;

    matrix[n_items,n_topics] loadings;
    for (i in 1:n_items){
        int topic = real_to_int(topics[i]);
        row_vector[n_topics] col;
        col = rep_row_vector(0,n_topics);
        col[topic] = 1;
        loadings[i] = col;
    }

    int<lower=1,upper=K> Y[n_participants,n_items];
    for (i in 1:n_participants){
      for (j in 1:n_items){
          real p_r;
          vector[K] Pmf_r;
          p_r = inv_logit(loadings[j] * a[i]' - d[j]);
          Pmf_r[1] = Phi((v[1] - p_r)/sigma);
          Pmf_r[K] = 1 - Phi((v[K-1] - p_r)/sigma);
          for (k in 2:(K-1)){ 
              Pmf_r[k] = Phi((v[k] - p_r)/sigma) - Phi((v[k-1] - p_r)/sigma);}
          Y[i,j] = categorical_rng(Pmf_r);
      }
    }
  }
"""
participants=100

data = {"n_items": 16,
        "n_participants":participants,
        "n_topics":4,
        "K": 12,
        "topics":[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],
        "v":[0.0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]}


prior = stan.build(stan_code, data=data)
fit = prior.fixed_param()
df = fit.to_frame()
df.to_csv('md_true_samps.csv')