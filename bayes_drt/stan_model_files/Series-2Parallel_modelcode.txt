data {
	int<lower=0> N; // number of measured frequencies
	int<lower=0> Ks; // number of basis functions for series distribution
	int<lower=0> Kp1; // number of basis functions for 1st parallel distribution
	int<lower=0> Kp2; // number of basis functions for 2nd parallel distribution
	matrix[N, Ks] As; // stacked A matrix for series distribution ([[A'] [A'']])
	matrix[N, Kp1] Ap1; // stacked A matrix for 1st parallel distribution  ([[A'] [A'']])
	matrix[N, Kp2] Ap2; // stacked A matrix for 2nd parallel distribution  ([[A'] [A'']])
	vector[N] Z; // stacked impedance vector ([Z' Z'']^T)
	vector[N/2] freq; //measured frequencies
	int<lower=0> N_tilde; // number of frequencies to predict
	matrix[N_tilde,Ks] As_tilde; //stacked series A matrix for prediction
	matrix[N_tilde,Kp1] Ap1_tilde; //stacked 1st parallel A matrix for prediction
	matrix[N_tilde,Kp2] Ap2_tilde; //stacked 2nd parallel A matrix for prediction
	vector[N_tilde/2] freq_tilde; // frequencies to predict
	matrix[Ks,Ks] L0s; // 0th order differentiation matrix for series distribution
	matrix[Ks,Ks] L1s; // 1st order differentiation matrix for series distribution
	matrix[Ks,Ks] L2s; // 2nd order differentiation matrix for series distribution
	matrix[Kp1,Kp1] L0p1; // 0th order differentiation matrix for 1st parallel distribution
	matrix[Kp1,Kp1] L1p1; // 1st order differentiation matrix for 1st parallel distribution
	matrix[Kp1,Kp1] L2p1; // 2nd order differentiation matrix for 1st parallel distribution
	matrix[Kp2,Kp2] L0p2; // 0th order differentiation matrix for 2nd parallel distribution
	matrix[Kp2,Kp2] L1p2; // 1st order differentiation matrix for 2nd parallel distribution
	matrix[Kp2,Kp2] L2p2; // 2nd order differentiation matrix for 2nd parallel distribution
	real<lower=0> sigma_min; // noise level floor
	real<lower=0> ups_alpha; // shape for inverse gamma distribution on ups
	real<lower=0> ups_beta; // rate for inverse gamma distribution on ups
	real<lower=0> induc_scale;
	real<lower=0> x_sum_invscale;
	real<lower=0> xp1_scale;
	real<lower=0> xp2_scale;
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
	vector [Ks] xs; // series distribution coefficients
	vector<lower=0>[Kp1] xp1_raw; // 1st parallel distribution coefficients
	vector<lower=0>[Kp2] xp2_raw; // 2nd parallel distribution coefficients
	real<lower=0> sigma_res_raw;
	real<lower=0> alpha_prop_raw;
	real<lower=0> alpha_re_raw;
	real<lower=0> alpha_im_raw;
	vector<lower=0>[Ks] ups_s_raw;
	vector<lower=0>[Kp1] ups_p1_raw;
	vector<lower=0>[Kp2] ups_p2_raw;
	real<lower=0> d0s_strength;
	real<lower=0> d1s_strength;
	real<lower=0> d2s_strength;
	real<lower=0> d0p1_strength;
	real<lower=0> d1p1_strength;
	real<lower=0> d2p1_strength;
	real<lower=0> d0p2_strength;
	real<lower=0> d1p2_strength;
	real<lower=0> d2p2_strength;
}
transformed parameters {
	real<lower=0> Rinf = Rinf_raw*100;
	real<lower=0> induc = induc_raw*induc_scale;
	vector<lower=0>[Kp1] xp1 = xp1_raw*xp1_scale;
	vector<lower=0>[Kp2] xp2 = xp2_raw*xp2_scale;
	vector<lower=0>[Ks] qs = sqrt(d0s_strength*square(L0s*xs) + d1s_strength*square(L1s*xs) + d2s_strength*square(L2s*xs));
	vector<lower=0>[Kp1] qp1 = sqrt(d0p1_strength*square(L0p1*xp1_raw) + d1p1_strength*square(L1p1*xp1_raw) + d2p1_strength*square(L2p1*xp1_raw));
	vector<lower=0>[Kp2] qp2 = sqrt(d0p2_strength*square(L0p2*xp2_raw) + d1p2_strength*square(L1p2*xp2_raw) + d2p2_strength*square(L2p2*xp2_raw));
	real<lower=0> x_sum_raw = sum(xs) + sum(xp1_raw) + sum(xp2_raw);
	real<lower=0> x_sum = x_sum_raw*x_sum_invscale;
	real<lower=0> sigma_res = sigma_res_raw*0.05;
	real<lower=0> alpha_prop = alpha_prop_raw*0.05;
	real<lower=0> alpha_re = alpha_re_raw*0.05;
	real<lower=0> alpha_im = alpha_im_raw*0.05;
	// calculate admittance from 1st parallel distribution and invert to get impedacne
	vector[N] Y_hat1 = Ap1*xp1;
	vector[N/2] Y_hat_re1 = Y_hat1[1:N/2];
	vector[N/2] Y_hat_im1 = Y_hat1[N/2+1:N];
	vector[N] Z_hat_p1 = append_row(Y_hat_re1 ./ (square(Y_hat_re1) + square(Y_hat_im1)), -Y_hat_im1 ./ (square(Y_hat_re1) + square(Y_hat_im1)));
	// calculate admittance from 2nd parallel distribution and invert to get impedacne
	vector[N] Y_hat2 = Ap2*xp2;
	vector[N/2] Y_hat_re2 = Y_hat2[1:N/2];
	vector[N/2] Y_hat_im2 = Y_hat2[N/2+1:N];
	vector[N] Z_hat_p2 = append_row(Y_hat_re2 ./ (square(Y_hat_re2) + square(Y_hat_im2)), -Y_hat_im2 ./ (square(Y_hat_re2) + square(Y_hat_im2)));
	// add parallel distribution impedances to series distribution impedance 
	vector[N] Z_hat = Z_hat_p1 + Z_hat_p2 + As*xs + Rinf*Rinf_vec + induc*induc_vec;
	vector[N] Z_hat_re = append_row(Z_hat[1:N/2],Z_hat[1:N/2]);
	vector[N] Z_hat_im = append_row(Z_hat[N/2+1:N],Z_hat[N/2+1:N]);
	vector<lower=0>[N] sigma_tot = sqrt(square(sigma_min) + square(sigma_res) + square(alpha_prop*Z_hat)
									+ square(alpha_re*Z_hat_re) + square(alpha_im*Z_hat_im));
	vector<lower=0>[Ks] ups_s = ups_s_raw * 0.15;
	vector<lower=0>[Kp1] ups_p1 = ups_p1_raw * 0.15;
	vector<lower=0>[Kp2] ups_p2 = ups_p2_raw * 0.15;
	vector[Ks-2] dups_s;
	vector[Kp1-2] dups_p1;
	vector[Kp2-2] dups_p2;
	for (k in 1:Ks-2)
		dups_s[k] = 0.5*(ups_s[k+1] - 0.5*(ups_s[k] + ups_s[k+2]))/ups_s[k+1];
	for (k in 1:Kp1-2)
		dups_p1[k] = 0.5*(ups_p1[k+1] - 0.5*(ups_p1[k] + ups_p1[k+2]))/ups_p1[k+1];
	for (k in 1:Kp2-2)
		dups_p2[k] = 0.5*(ups_p2[k+1] - 0.5*(ups_p2[k] + ups_p2[k+2]))/ups_p2[k+1];
}
model {
	d0s_strength ~ inv_gamma(5,5);
	d1s_strength ~ inv_gamma(5,5);
	d2s_strength ~ inv_gamma(5,5);
	d0p1_strength ~ inv_gamma(5,5);
	d1p1_strength ~ inv_gamma(5,5);
	d2p1_strength ~ inv_gamma(5,5);
	d0p2_strength ~ inv_gamma(5,5);
	d1p2_strength ~ inv_gamma(5,5);
	d2p2_strength ~ inv_gamma(5,5);
	x_sum ~ std_normal();
	ups_s_raw ~ inv_gamma(ups_alpha,ups_beta);
	ups_p1_raw ~ inv_gamma(ups_alpha,ups_beta);
	ups_p2_raw ~ inv_gamma(ups_alpha,ups_beta);
	Rinf_raw ~ std_normal();
	induc_raw ~ std_normal();
	qs ~ normal(0,ups_s);
	qp1 ~ normal(0,ups_p1);
	qp2 ~ normal(0,ups_p2);
	dups_s ~ std_normal();
	dups_p1 ~ std_normal();
	dups_p2 ~ std_normal();
	Z ~ normal(Z_hat,sigma_tot);
	sigma_res_raw ~ std_normal();
	alpha_prop_raw ~ std_normal();
	alpha_re_raw ~ std_normal();
	alpha_im_raw ~ std_normal();
}
generated quantities {
	// calculate admittance from 1st parallel distribution and invert to get impedacne
	vector[N] Y_hat1_tilde = Ap1_tilde*xp1;
	vector[N/2] Y_hat_re1_tilde = Y_hat1_tilde[1:N/2];
	vector[N/2] Y_hat_im1_tilde = Y_hat1_tilde[N/2+1:N];
	vector[N] Z_hat_p1_tilde = append_row(Y_hat_re1_tilde ./ (square(Y_hat_re1_tilde) + square(Y_hat_im1_tilde)), -Y_hat_im1_tilde ./ (square(Y_hat_re1_tilde) + square(Y_hat_im1_tilde)));
	// calculate admittance from 2nd parallel distribution and invert to get impedacne
	vector[N] Y_hat2_tilde = Ap2_tilde*xp2;
	vector[N/2] Y_hat_re2_tilde = Y_hat2_tilde[1:N/2];
	vector[N/2] Y_hat_im2_tilde = Y_hat2_tilde[N/2+1:N];
	vector[N] Z_hat_p2_tilde = append_row(Y_hat_re2_tilde ./ (square(Y_hat_re2_tilde) + square(Y_hat_im2_tilde)), -Y_hat_im2_tilde ./ (square(Y_hat_re2_tilde) + square(Y_hat_im2_tilde)));
	vector[N_tilde] Z_hat_tilde
		= Z_hat_p1_tilde + Z_hat_p1_tilde + As_tilde*xs + Rinf*Rinf_vec_tilde + induc*induc_vec_tilde;
}