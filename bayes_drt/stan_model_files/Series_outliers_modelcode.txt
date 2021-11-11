data {
	int<lower=0> N; // number of measured frequencies
	int<lower=0> K; // number of basis functions
	matrix[2*N, K] A; // stacked A matrix ([[A'] [A'']])
	vector[2*N] Z; // stacked impedance vector ([Z' Z'']^T)
	vector[N] freq; //measured frequencies
	matrix[K,K] L0; // 0th order differentiation matrix
	matrix[K,K] L1; // 1st order differentiation matrix
	matrix[K,K] L2; // 2nd order differentiation matrix
	real<lower=0> sigma_min; // noise level floor
	real<lower=0> ups_alpha; // shape for inverse gamma distribution on ups
	real<lower=0> ups_beta; // rate for inverse gamma distribution on ups
	real<lower=0> induc_scale;
	real<lower=0> sigma_out_lambda;
	real<lower=0> sigma_out_alpha;
	real<lower=0> sigma_out_beta;
}
transformed data {
	vector [2*N] Rinf_vec = append_row(rep_vector(1,N), rep_vector(0,N));
	vector [2*N] induc_vec = append_row(rep_vector(0,N), 2*pi()*freq);
}
parameters {
	real<lower=0> Rinf_raw;
	real<lower=0> induc_raw;
	vector[K] x;
	real<lower=0> sigma_res_raw;
	real<lower=0> alpha_prop_raw;
	real<lower=0> alpha_re_raw;
	real<lower=0> alpha_im_raw;
	vector<lower=0>[N] sigma_out_raw;
	vector<lower=0>[N] sigma_out_scale;
	vector<lower=0>[K] ups_raw;
	real<lower=0> d0_strength;
	real<lower=0> d1_strength;
	real<lower=0> d2_strength;
}
transformed parameters {
	real<lower=0> Rinf = Rinf_raw*100;
	real<lower=0> induc = induc_raw*induc_scale;
	vector<lower=0>[K] q = sqrt(d0_strength*square(L0*x) + d1_strength*square(L1*x) + d2_strength*square(L2*x));
	real<lower=0> sigma_res = sigma_res_raw*0.05;
	real<lower=0> alpha_prop = alpha_prop_raw*0.05;
	real<lower=0> alpha_re = alpha_re_raw*0.05;
	real<lower=0> alpha_im = alpha_im_raw*0.05;
	vector<lower=0>[N] sigma_out = sigma_out_raw .* sigma_out_scale *0.05;
	vector[2*N] Z_hat = A*x + Rinf*Rinf_vec + induc*induc_vec;
	vector[2*N] Z_hat_re = append_row(Z_hat[1:N],Z_hat[1:N]);
	vector[2*N] Z_hat_im = append_row(Z_hat[N+1:2*N],Z_hat[N+1:2*N]);
	vector<lower=0>[2*N] sigma_tot = sqrt(square(sigma_min) + square(sigma_res) + square(alpha_prop*Z_hat)
									+ square(alpha_re*Z_hat_re) + square(alpha_im*Z_hat_im) + square(append_row(sigma_out,sigma_out))
									);
	vector<lower=0>[K] ups = ups_raw * 0.15;
	vector[K-2] dups;
	for (k in 1:K-2)
		dups[k] = 0.5*(ups[k+1] - 0.5*(ups[k] + ups[k+2]))/ups[k+1];
}
model {
	d0_strength ~ inv_gamma(5,5);
	d1_strength ~ inv_gamma(5,5);
	d2_strength ~ inv_gamma(5,5);
	ups_raw ~ inv_gamma(ups_alpha,ups_beta);
	Rinf_raw ~ std_normal();
	induc_raw ~ std_normal();
	q ~ normal(0,ups);
	dups ~ std_normal();
	Z ~ normal(Z_hat,sigma_tot);
	sigma_res_raw ~ std_normal();
	alpha_prop_raw ~ std_normal();
	alpha_re_raw ~ std_normal();
	alpha_im_raw ~ std_normal();
	sigma_out_raw ~ exponential(sigma_out_lambda);
	sigma_out_scale ~ inv_gamma(sigma_out_alpha,sigma_out_beta);
}