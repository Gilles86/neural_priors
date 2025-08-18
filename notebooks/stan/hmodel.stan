data {
    int<lower=1> S; // Number of subjects
    int<lower=1> N; // Total number of observations
    array[N] int<lower=1, upper=2> c;       // Condition: 1, or 2
    array[N] int<lower=10, upper=40> x;     // Number of dots in each trial
    array[N] real<lower=10, upper=40> y;    // Estimates for each trial
    array[N] int<lower=1, upper=S> subject; // Subject ID for each trial
}
parameters { // z_ = std normalized
    array[16] real z_m_0_15;        // z Global mean function for each x value (10 to 25), in the narrow condition
    array[31] real z_m_0_30;        // z Global mean function for each x value (10 to 40), in the wide condition
    real<lower=0> tau;              // Standard deviation of the subject means around the global mean
    array[S, 16] real z_m_s_15;     // z Subject-specific mean function for each x value (10 to 25), in the narrow condition
    array[S, 31] real z_m_s_30;     // z Subject-specific mean function for each x value (10 to 40), in the wide condition
    array[16] real<lower=0> sigma_0_15;     // Global standard deviation function for each x value, in the narrow condition
    array[31] real<lower=0> sigma_0_30;     // Global standard deviation function for each x value, in the wide condition
    real<lower=0> nu;                       // Standard deviation for the truncated normal distribution
    array[S, 16] real z_log_sigma_s_15; // Subject-specific log of standard deviation for each x value, in the narrow condition
    array[S, 31] real z_log_sigma_s_30; // Subject-specific log of standard deviation for each x value, in the wide condition
}
transformed parameters {
    array[16] real m_0_15;          // Global mean function for each x value (10 to 25), in the narrow condition
    array[31] real m_0_30;          // Global mean function for each x value (10 to 40), in the wide condition
    for (j in 1:16) {
        m_0_15[j] = 9. + j + 10 * z_m_0_15[j];
    }
    for (j in 1:31) {
        m_0_30[j] = 9. + j + 10 * z_m_0_30[j];
    }
    array[S, 16] real m_s_15;   // Subject-specific mean function for each x value, in the narrow condition
    array[S, 31] real m_s_30;   // Subject-specific mean function for each x value, in the wide condition
    array[S, 16] real<lower=0> sigma_s_15;  // Subject-specific standard deviation for each x value, in the narrow condition
    array[S, 31] real<lower=0> sigma_s_30;  // Subject-specific standard deviation for each x value, in the wide condition
    for (s in 1:S) {
        for (j in 1:16) {
            m_s_15[s, j]     = m_0_15[j]     + tau * z_m_s_15[s, j];
            sigma_s_15[s, j] = sigma_0_15[j] * exp( nu * z_log_sigma_s_15[s, j]);
        }
        for (j in 1:31) {
            m_s_30[s, j]     = m_0_30[j]     + tau * z_m_s_30[s, j];
            sigma_s_30[s, j] = sigma_0_30[j] * exp( nu * z_log_sigma_s_30[s, j]);
        }
    }
}
model {
    // Priors
    z_m_0_15 ~ std_normal();
    z_m_0_30 ~ std_normal();
    sigma_0_15 ~ normal(2, 2); // Prior for global standard deviation function
    sigma_0_30 ~ normal(3.5, 3.5); 
    tau ~ normal(2, 5);    // Prior for the standard deviation tau of the subjects' means
    nu ~ normal(1, 2);     // Prior for the standard deviation nu of the subjects' std dev

    // Likelihood
    for (s in 1:S) {
        for (j in 1:16) {
            z_m_s_15[s,j]    ~ std_normal(); 
            z_log_sigma_s_15[s,j] ~ std_normal();
        }
        for (j in 1:31) {
            z_m_s_30[s,j]    ~ std_normal(); 
            z_log_sigma_s_30[s,j] ~ std_normal(); 
        }
    }
    // min_value = 10
    for (i in 1:N) {
        int x_index = x[i] - 10 + 1; // Index adjustment for x values
        int subj_i = subject[i];
        if (c[i] == 1) {
            y[i] ~ normal( m_s_15[ subj_i, x_index ], sigma_s_15[ subj_i, x_index ] );
        }
        else if (c[i] == 2) {
            y[i] ~ normal( m_s_30[ subj_i, x_index ], sigma_s_30[ subj_i, x_index ] );
        }
    }
}
generated quantities {
    // averages
    real<lower=0> avg_sigma_0_15 = mean(sigma_0_15); // Average global standard deviation for each condition
    real<lower=0> avg_sigma_0_30 = mean(sigma_0_30);
    //
    array[16] real<lower=0> var_0_15 = square(sigma_0_15);          // Global variance
    array[31] real<lower=0> var_0_30 = square(sigma_0_30);
    // MSE
    array[16] real<lower=0> sqrt_mse_15;         // Sqrt MSE = Sqrt( bias^2 + variance )
    array[31] real<lower=0> sqrt_mse_30;
    for (j in 1:16) {
        real correct_number = j + 9;
        sqrt_mse_15[j] = sqrt( square(m_0_15[j]-correct_number) + var_0_15[j]  );
    }
    for (j in 1:31) {
        real correct_number = j + 9;
        sqrt_mse_30[j] = sqrt( square(m_0_30[j]-correct_number) + var_0_30[j]  );
    }
    // Variance
    real<lower=0> avg_var_0_15 = mean(var_0_15); // Average global variance
    real<lower=0> avg_var_0_30 = mean(var_0_30);
    real<lower=0> avg_var_0_30_narrowrange = mean(var_0_30[1:16]);
    array[S, 16] real<lower=0> var_s_15 = square(sigma_s_15);  // Subject-specific variance for each x value
    array[S, 31] real<lower=0> var_s_30 = square(sigma_s_30);
    array[S, 16] real          delta_var_s; // = var_s_30 - var_s_15;
    array[S] real<lower=0> avg_var_s_15; // Average subject-specific variance
    array[S] real<lower=0> avg_var_s_15_1124; // Average subject-specific variance (11 to 24)
    array[S] real<lower=0> avg_var_s_15_1223; // Average subject-specific variance (12 to 23)
    array[S] real<lower=0> avg_var_s_15_1322;  // Average subject-specific variance (13 to 22)
    array[S] real<lower=0> avg_var_s_30;
    array[S] real<lower=0> avg_var_s_30_narrowrange;
    array[S] real<lower=0> avg_var_s_30_1124;
    array[S] real<lower=0> avg_var_s_30_1223;
    array[S] real<lower=0> avg_var_s_30_1322;
    array[S] real<lower=0> avg_var_s_30_1139;
    array[S] real<lower=0> avg_var_s_30_1238;
    array[S] real<lower=0> avg_var_s_30_1337;
    for (s in 1:S) {
        // delta_var_s[s] = var_s_30[s, 1:16] - var_s_15[s]; // doesn't work, have to loop
        for (j in 1:16) {
            delta_var_s[s, j] = var_s_30[s, j] - var_s_15[s, j];
        }
        avg_var_s_15[s] = mean(var_s_15[s]);
        avg_var_s_15_1124[s] = mean(var_s_15[s, 2:15]); // 11 to 24
        avg_var_s_15_1223[s] = mean(var_s_15[s, 3:14]); // 12 to 23
        avg_var_s_15_1322[s] = mean(var_s_15[s, 4:13]); // 13 to 22
        avg_var_s_30[s] = mean(var_s_30[s]);
        avg_var_s_30_narrowrange[s] = mean(var_s_30[s, 1:16]);
        avg_var_s_30_1124[s] = mean(var_s_30[s, 2:15]); // 11 to 24
        avg_var_s_30_1223[s] = mean(var_s_30[s, 3:14]); // 12 to 23
        avg_var_s_30_1322[s] = mean(var_s_30[s, 4:13]); // 13 to 22
        avg_var_s_30_1139[s] = mean(var_s_30[s, 2:30]); // 11 to 39
        avg_var_s_30_1238[s] = mean(var_s_30[s, 3:29]); // 12 to 38
        avg_var_s_30_1337[s] = mean(var_s_30[s, 4:28]); // 13 to 37
    }
    array[S] real          delta_avg_var_s; // = avg_var_s_30 - avg_var_s_15;
    array[S] real          delta_avg_var_s_narrowrange; // = avg_var_s_30_narrowrange - avg_var_s_15;
    array[S] real          delta_avg_var_s_1124;
    array[S] real          delta_avg_var_s_1223;
    array[S] real          delta_avg_var_s_1322;
    for (s in 1:S) {
        delta_avg_var_s[s] = avg_var_s_30[s] - avg_var_s_15[s];
        delta_avg_var_s_narrowrange[s] = avg_var_s_30_narrowrange[s] - avg_var_s_15[s];
        delta_avg_var_s_1124[s] = avg_var_s_30_1124[s] - avg_var_s_15_1124[s];
        delta_avg_var_s_1223[s] = avg_var_s_30_1223[s] - avg_var_s_15_1223[s];
        delta_avg_var_s_1322[s] = avg_var_s_30_1322[s] - avg_var_s_15_1322[s];
    }

    // Standard deviation
    // already exists: sigma_0_15, sigma_0_30, sigma_s_15, sigma_s_30, avg_sigma_0_15, avg_sigma_0_30
    array[S, 16] real delta_sigma_s; // = sigma_s_30 - sigma_s_15;
    real<lower=0> avg_sigma_0_30_narrowrange = mean(sigma_0_30[1:16]);
    array[S] real<lower=0> avg_sigma_s_15; // Average subject-specific variance
    array[S] real<lower=0> avg_sigma_s_15_1124;
    array[S] real<lower=0> avg_sigma_s_15_1223;
    array[S] real<lower=0> avg_sigma_s_15_1322;
    array[S] real<lower=0> avg_sigma_s_30;
    array[S] real<lower=0> avg_sigma_s_30_narrowrange;
    array[S] real<lower=0> avg_sigma_s_30_1124;
    array[S] real<lower=0> avg_sigma_s_30_1223;
    array[S] real<lower=0> avg_sigma_s_30_1322;
    array[S] real<lower=0> avg_sigma_s_30_1139;
    array[S] real<lower=0> avg_sigma_s_30_1238;
    array[S] real<lower=0> avg_sigma_s_30_1337;
    for (s in 1:S) {
        for (j in 1:16) {
            delta_sigma_s[s, j] = sigma_s_30[s, j] - sigma_s_15[s, j];
        }
        avg_sigma_s_15[s] = mean(sigma_s_15[s]);
        avg_sigma_s_15_1124[s] = mean(sigma_s_15[s, 2:15]); // 11 to 24
        avg_sigma_s_15_1223[s] = mean(sigma_s_15[s, 3:14]); // 12 to 23
        avg_sigma_s_15_1322[s] = mean(sigma_s_15[s, 4:13]); // 13 to 22
        avg_sigma_s_30[s] = mean(sigma_s_30[s]);
        avg_sigma_s_30_narrowrange[s] = mean(sigma_s_30[s, 1:16]);
        avg_sigma_s_30_1124[s] = mean(sigma_s_30[s, 2:15]); // 11 to 24
        avg_sigma_s_30_1223[s] = mean(sigma_s_30[s, 3:14]); // 12 to 23
        avg_sigma_s_30_1322[s] = mean(sigma_s_30[s, 4:13]); // 13 to 22
        avg_sigma_s_30_1139[s] = mean(sigma_s_30[s, 2:30]); // 11 to 39
        avg_sigma_s_30_1238[s] = mean(sigma_s_30[s, 3:29]); // 12 to 38
        avg_sigma_s_30_1337[s] = mean(sigma_s_30[s, 4:28]); // 13 to 37
    }
    array[S] real          delta_avg_sigma_s; // = avg_sigma_s_30 - avg_sigma_s_15;
    array[S] real          delta_avg_sigma_s_narrowrange; // = avg_sigma_s_30_narrowrange - avg_sigma_s_15;
    array[S] real          delta_avg_sigma_s_1124;
    array[S] real          delta_avg_sigma_s_1223;
    array[S] real          delta_avg_sigma_s_1322;
    for (s in 1:S) {
        delta_avg_sigma_s[s] = avg_sigma_s_30[s] - avg_sigma_s_15[s];
        delta_avg_sigma_s_narrowrange[s] = avg_sigma_s_30_narrowrange[s] - avg_sigma_s_15[s];
        delta_avg_sigma_s_1124[s] = avg_sigma_s_30_1124[s] - avg_sigma_s_15_1124[s];
        delta_avg_sigma_s_1223[s] = avg_sigma_s_30_1223[s] - avg_sigma_s_15_1223[s];
        delta_avg_sigma_s_1322[s] = avg_sigma_s_30_1322[s] - avg_sigma_s_15_1322[s];
    }

    // Inverse sigma
    array[16] real<lower=0> inv_sigma_0_15;  // Inverse of global standard deviation
    array[31] real<lower=0> inv_sigma_0_30;
    for (j in 1:16) {
        inv_sigma_0_15[j] = 1. / sigma_0_15[j];
    }
    for (j in 1:31) {
        inv_sigma_0_30[j] = 1. / sigma_0_30[j];
    }
    real<lower=0> avg_inv_sigma_0_15 = mean(inv_sigma_0_15); // Average inverse global standard deviation
    real<lower=0> avg_inv_sigma_0_30 = mean(inv_sigma_0_30);
    real<lower=0> avg_inv_sigma_0_30_narrowrange = mean(inv_sigma_0_30[1:16]);
    array[S, 16] real<lower=0> inv_sigma_s_15;  // Inverse of subject-specific standard deviation for each x value
    array[S, 31] real<lower=0> inv_sigma_s_30;
    array[S, 16] real delta_inv_sigma_s; // = inv_sigma_s_30 - inv_sigma_s_15;
    for (s in 1:S) {
        for (j in 1:31) {
            inv_sigma_s_30[s, j] = 1. / sigma_s_30[s, j];
        }
        for (j in 1:16) {
            inv_sigma_s_15[s, j] = 1. / sigma_s_15[s, j];
            delta_inv_sigma_s[s, j] = inv_sigma_s_30[s, j] - inv_sigma_s_15[s, j];
        }
    }
    array[S] real<lower=0> avg_inv_sigma_s_15; // Average inverse subject-specific standard deviation
    array[S] real<lower=0> avg_inv_sigma_s_15_1124;
    array[S] real<lower=0> avg_inv_sigma_s_15_1223;
    array[S] real<lower=0> avg_inv_sigma_s_15_1322;
    array[S] real<lower=0> avg_inv_sigma_s_30;
    array[S] real<lower=0> avg_inv_sigma_s_30_narrowrange;
    array[S] real<lower=0> avg_inv_sigma_s_30_1124;
    array[S] real<lower=0> avg_inv_sigma_s_30_1223;
    array[S] real<lower=0> avg_inv_sigma_s_30_1322;
    array[S] real<lower=0> avg_inv_sigma_s_30_1139;
    array[S] real<lower=0> avg_inv_sigma_s_30_1238;
    array[S] real<lower=0> avg_inv_sigma_s_30_1337;
    for (s in 1:S) {
        avg_inv_sigma_s_15[s] = mean(inv_sigma_s_15[s]);
        avg_inv_sigma_s_15_1124[s] = mean(inv_sigma_s_15[s, 2:15]); // 11 to 24
        avg_inv_sigma_s_15_1223[s] = mean(inv_sigma_s_15[s, 3:14]); // 12 to 23
        avg_inv_sigma_s_15_1322[s] = mean(inv_sigma_s_15[s, 4:13]); // 13 to 22
        avg_inv_sigma_s_30[s] = mean(inv_sigma_s_30[s]);
        avg_inv_sigma_s_30_narrowrange[s] = mean(inv_sigma_s_30[s, 1:16]);
        avg_inv_sigma_s_30_1124[s] = mean(inv_sigma_s_30[s, 2:15]); // 11 to 24
        avg_inv_sigma_s_30_1223[s] = mean(inv_sigma_s_30[s, 3:14]); // 12 to 23
        avg_inv_sigma_s_30_1322[s] = mean(inv_sigma_s_30[s, 4:13]); // 13 to 22
        avg_inv_sigma_s_30_1139[s] = mean(inv_sigma_s_30[s, 2:30]); // 11 to 39
        avg_inv_sigma_s_30_1238[s] = mean(inv_sigma_s_30[s, 3:29]); // 12 to 38
        avg_inv_sigma_s_30_1337[s] = mean(inv_sigma_s_30[s, 4:28]); // 13 to 37
    }
    array[S] real          delta_avg_inv_sigma_s; // = avg_inv_sigma_s_30 - avg_inv_sigma_s_15;
    array[S] real          delta_avg_inv_sigma_s_narrowrange; // = avg_inv_sigma_s_30_narrowrange - avg_inv_sigma_s_15;
    array[S] real          delta_avg_inv_sigma_s_1124;
    array[S] real          delta_avg_inv_sigma_s_1223;
    array[S] real          delta_avg_inv_sigma_s_1322;
    for (s in 1:S) {
        delta_avg_inv_sigma_s[s] = avg_inv_sigma_s_30[s] - avg_inv_sigma_s_15[s];
        delta_avg_inv_sigma_s_narrowrange[s] = avg_inv_sigma_s_30_narrowrange[s] - avg_inv_sigma_s_15[s];
        delta_avg_inv_sigma_s_1124[s] = avg_inv_sigma_s_30_1124[s] - avg_inv_sigma_s_15_1124[s];
        delta_avg_inv_sigma_s_1223[s] = avg_inv_sigma_s_30_1223[s] - avg_inv_sigma_s_15_1223[s];
        delta_avg_inv_sigma_s_1322[s] = avg_inv_sigma_s_30_1322[s] - avg_inv_sigma_s_15_1322[s];
    }
    
    // Log of variance
    array[16] real log_var_0_15 = log(var_0_15);  // Log of variance for each x value
    array[31] real log_var_0_30 = log(var_0_30);
    real avg_log_var_0_15 = mean(log_var_0_15); // Average log variance
    real avg_log_var_0_30 = mean(log_var_0_30);
    real avg_log_var_0_30_narrowrange = mean(log_var_0_30[1:16]);
    array[S, 16] real log_var_s_15 = log(var_s_15);  // Log of subject-specific variance for each x value
    array[S, 31] real log_var_s_30 = log(var_s_30);
    array[S, 16] real delta_log_var_s; // = log_var_s_30 - log_var_s_15;
    array[S] real avg_log_var_s_15; // Average log subject-specific variance
    array[S] real avg_log_var_s_15_1124;
    array[S] real avg_log_var_s_15_1223;
    array[S] real avg_log_var_s_15_1322;
    array[S] real avg_log_var_s_30;
    array[S] real avg_log_var_s_30_narrowrange;
    array[S] real avg_log_var_s_30_1124;
    array[S] real avg_log_var_s_30_1223;
    array[S] real avg_log_var_s_30_1322;
    array[S] real avg_log_var_s_30_1139;
    array[S] real avg_log_var_s_30_1238;
    array[S] real avg_log_var_s_30_1337;
    for (s in 1:S) {
        for (j in 1:16) {
            delta_log_var_s[s, j] = log_var_s_30[s, j] - log_var_s_15[s, j];
        }
        avg_log_var_s_15[s] = mean(log_var_s_15[s]);
        avg_log_var_s_15_1124[s] = mean(log_var_s_15[s, 2:15]); // 11 to 24
        avg_log_var_s_15_1223[s] = mean(log_var_s_15[s, 3:14]); // 12 to 23
        avg_log_var_s_15_1322[s] = mean(log_var_s_15[s, 4:13]); // 13 to 22
        avg_log_var_s_30[s] = mean(log_var_s_30[s]);
        avg_log_var_s_30_narrowrange[s] = mean(log_var_s_30[s, 1:16]);
        avg_log_var_s_30_1124[s] = mean(log_var_s_30[s, 2:15]); // 11 to 24
        avg_log_var_s_30_1223[s] = mean(log_var_s_30[s, 3:14]); // 12 to 23
        avg_log_var_s_30_1322[s] = mean(log_var_s_30[s, 4:13]); // 13 to 22
        avg_log_var_s_30_1139[s] = mean(log_var_s_30[s, 2:30]); // 11 to 39
        avg_log_var_s_30_1238[s] = mean(log_var_s_30[s, 3:29]); // 12 to 38
        avg_log_var_s_30_1337[s] = mean(log_var_s_30[s, 4:28]); // 13 to 37
    }
    array[S] real delta_avg_log_var_s; // = avg_log_var_s_30 - avg_log_var_s_15;
    array[S] real delta_avg_log_var_s_narrowrange; // = avg_log_var_s_30_narrowrange - avg_log_var_s_15;
    array[S] real delta_avg_log_var_s_1124;
    array[S] real delta_avg_log_var_s_1223;
    array[S] real delta_avg_log_var_s_1322;
    for (s in 1:S) {
        delta_avg_log_var_s[s] = avg_log_var_s_30[s] - avg_log_var_s_15[s];
        delta_avg_log_var_s_narrowrange[s] = avg_log_var_s_30_narrowrange[s] - avg_log_var_s_15[s];
        delta_avg_log_var_s_1124[s] = avg_log_var_s_30_1124[s] - avg_log_var_s_15_1124[s];
        delta_avg_log_var_s_1223[s] = avg_log_var_s_30_1223[s] - avg_log_var_s_15_1223[s];
        delta_avg_log_var_s_1322[s] = avg_log_var_s_30_1322[s] - avg_log_var_s_15_1322[s];
    }
}
