data {
	int<lower=0> N; // number of measured frequencies
	int<lower=0> Ks; // number of basis functions for series distribution
	int<lower=0> Kp; // number of basis functions for parallel distribution
	matrix[N, Ks] As; // stacked A matrix for series distribution ([[A'] [A'']])
	matrix[N, Kp] Ap; // stacked A matrix for parallel distribution  ([[A'] [A'']])
	vector[N] Z; // stacked impedance vector ([Z' Z'']^T)
	vector[N/2] freq; //measured frequencies
	int<lower=0> N_tilde; // number of frequencies to predict
	matrix[N_tilde,Ks] As_tilde; //stacked series A matrix for prediction
	matrix[N_tilde,Kp] Ap_tilde; //stacked parallel A matrix for prediction
	vector[N_tilde/2] freq_tilde; // frequencies to predict
	matrix[Ks,Ks] L0s; // 0th order differentiation matrix for series distribution
	matrix[Ks,Ks] L1s; // 1st order differentiation matrix for series distribution
	matrix[Ks,Ks] L2s; // 2nd order differentiation matrix for series distribution
	matrix[Kp,Kp] L0p; // 0th order differentiation matrix for parallel distribution
	matrix[Kp,Kp] L1p; // 1st order differentiation matrix for parallel distribution
	matrix[Kp,Kp] L2p; // 2nd order differentiation matrix for parallel distribution
	real<lower=0> sigma_min; // noise level floor
	real<lower=0> ups_alpha; // shape for inverse gamma distribution on ups
	real<lower=0> ups_beta; // rate for inverse gamma distribution on ups
	real<lower=0> induc_scale;
	real<lower=0> x_sum_invscale;
	real<lower=0> xp_scale;
	real<lower=0> so_invscale;
}
transformed data {
	vector [N] Rinf_vec = append_row(rep_vector(1,N/2), rep_vector(0,N/2));
	vector [N] induc_vec = append_row(rep_vector(0,N/2), 2*pi()*freq);
	vector [N_tilde] Rinf_vec_tilde = append_row(rep_vector(1,N_tilde/2), rep_vector(0,N_tilde/2));
	vector [N_tilde] induc_vec_tilde = append_row(rep_vector(0,N_tilde/2), 2*pi()*freq_tilde);
}
parameters {
	real<lower=0> Rinf_raw;
	real<lower=0> induc_raw;
	vector<lower=0>[Ks] xs; // series distribution coefficients
	vector<lower=0>[Kp] xp_raw; // raw parallel distribution coefficients
	real<lower=0> sigma_res_raw;
	real<lower=0> alpha_prop_raw;
	real<lower=0> alpha_re_raw;
	real<lower=0> alpha_im_raw;
	vector<lower=0>[N] sigma_out_raw;
	vector<lower=0>[Ks] ups_s_raw;
	vector<lower=0>[Kp] ups_p_raw;
	real<lower=0> d0s_strength;
	real<lower=0> d1s_strength;
	real<lower=0> d2s_strength;
	real<lower=0> d0p_strength;
	real<lower=0> d1p_strength;
	real<lower=0> d2p_strength;
}
transformed parameters {
	real<lower=0> Rinf = Rinf_raw*100;
	real<lower=0> induc = induc_raw*induc_scale;
	vector<lower=0>[Kp] xp = xp_raw*xp_scale; // scaled parallel distribution coefficients
	vector<lower=0>[Ks] qs = sqrt(d0s_strength*square(L0s*xs) + d1s_strength*square(L1s*xs) + d2s_strength*square(L2s*xs));
	vector<lower=0>[Kp] qp = sqrt(d0p_strength*square(L0p*xp_raw) + d1p_strength*square(L1p*xp_raw) + d2p_strength*square(L2p*xp_raw));
	real<lower=0> x_sum_raw = sum(xs) + sum(xp_raw);
	real<lower=0> x_sum = x_sum_raw*x_sum_invscale;
	real<lower=0> sigma_res = sigma_res_raw*0.05;
	real<lower=0> alpha_prop = alpha_prop_raw*0.05;
	real<lower=0> alpha_re = alpha_re_raw*0.05;
	real<lower=0> alpha_im = alpha_im_raw*0.05;
	vector<lower=0>[N] sigma_out = sigma_out_raw*0.05;
	// calculate admittance from parallel distribution and invert to get impedacne
	vector[N] Y_hat = Ap*xp;
	vector[N/2] Y_hat_re = Y_hat[1:N/2];
	vector[N/2] Y_hat_im = Y_hat[N/2+1:N];
	vector[N] Z_hat_p = append_row(Y_hat_re ./ (square(Y_hat_re) + square(Y_hat_im)), -Y_hat_im ./ (square(Y_hat_re) + square(Y_hat_im)));
	// add parallel distribution impedance to series distribution impedance 
	vector[N] Z_hat = Z_hat_p + As*xs + Rinf*Rinf_vec + induc*induc_vec;
	vector[N] Z_hat_re = append_row(Z_hat[1:N/2],Z_hat[1:N/2]);
	vector[N] Z_hat_im = append_row(Z_hat[N/2+1:N],Z_hat[N/2+1:N]);
	vector<lower=0>[N] sigma_tot = sqrt(square(sigma_min) + square(sigma_res) + square(alpha_prop*Z_hat)
									+ square(alpha_re*Z_hat_re) + square(alpha_im*Z_hat_im) + square(sigma_out)
									);
	vector<lower=0>[Ks] ups_s = ups_s_raw * 0.15;
	vector<lower=0>[Kp] ups_p = ups_p_raw * 0.15;
	vector[Ks-2] dups_s;
	vector[Kp-2] dups_p;
	for (k in 1:Ks-2)
		dups_s[k] = 0.5*(ups_s[k+1] - 0.5*(ups_s[k] + ups_s[k+2]))/ups_s[k+1];
	for (k in 1:Kp-2)
		dups_p[k] = 0.5*(ups_p[k+1] - 0.5*(ups_p[k] + ups_p[k+2]))/ups_p[k+1];
}
model {
	d0s_strength ~ inv_gamma(5,5);
	d1s_strength ~ inv_gamma(5,5);
	d2s_strength ~ inv_gamma(5,5);
	d0p_strength ~ inv_gamma(5,5);
	d1p_strength ~ inv_gamma(5,5);
	d2p_strength ~ inv_gamma(5,5);
	x_sum ~ std_normal();
	ups_s_raw ~ inv_gamma(ups_alpha,ups_beta);
	ups_p_raw ~ inv_gamma(ups_alpha,ups_beta);
	Rinf_raw ~ std_normal();
	induc_raw ~ std_normal();
	qs ~ normal(0,ups_s);
	qp ~ normal(0,ups_p);
	dups_s ~ std_normal();
	dups_p ~ std_normal();
	Z ~ normal(Z_hat,sigma_tot);
	sigma_res_raw ~ std_normal();
	alpha_prop_raw ~ std_normal();
	alpha_re_raw ~ std_normal();
	alpha_im_raw ~ std_normal();
	sigma_out_raw ~ exponential(so_invscale);
}
generated quantities {
	vector[N_tilde] Y_hat_tilde = Ap_tilde*xp;
	vector[N_tilde/2] Y_hat_re_tilde = Y_hat[1:N_tilde/2];
	vector[N_tilde/2] Y_hat_im_tilde = Y_hat[N_tilde/2+1:N_tilde];
	vector[N_tilde] Z_hat_p_tilde = append_row(Y_hat_re ./ (square(Y_hat_re) + square(Y_hat_im)), -Y_hat_im ./ (square(Y_hat_re) + square(Y_hat_im)));
	vector[N_tilde] Z_hat_tilde
		= Z_hat_p_tilde + As_tilde*xs + Rinf*Rinf_vec_tilde + induc*induc_vec_tilde;
}