data {
	int<lower=0> N; // number of measured frequencies
	int<lower=0> K; // number of basis functions
	matrix[N,N] S; 
	matrix[N,N] S_inv;
	matrix[N, K] SA; // stacked A matrix ([[A'] [A'']])
	vector[N] Y; // stacked admittance vector ([Y' Y'']^T)
	vector[N/2] freq; // measured frequencies
	int<lower=0> N_tilde; // number of frequencies to predict
	matrix[N_tilde,K] A_tilde; //stacked A matrix for prediction
	vector[N_tilde/2] freq_tilde; // frequencies to predict
	matrix[K,K] L0; // 0th order differentiation matrix
	matrix[K,K] L1; // 1st order differentiation matrix
	matrix[K,K] L2; // 2nd order differentiation matrix
	real<lower=0> sigma_min; // noise level floor
	real<lower=0> ups_alpha; // shape for inverse gamma distribution on ups
	real<lower=0> ups_beta; // rate for inverse gamma distribution on ups
}
transformed data {
	//vector [N] Rinf_vec = append_row(rep_vector(1,N/2), rep_vector(0,N/2));
	//vector [N] induc_vec = append_row(rep_vector(0,N/2), 2*pi()*freq);
	//vector [N_tilde] Rinf_vec_tilde = append_row(rep_vector(1,N_tilde/2), rep_vector(0,N_tilde/2));
	//vector [N_tilde] induc_vec_tilde = append_row(rep_vector(0,N_tilde/2), 2*pi()*freq_tilde);
}
parameters {
	//real<lower=0> Rinf_raw;
	//real<lower=0> induc;
	vector<lower=0>[K] x;
	real<lower=0> sigma_res_raw;
	real<lower=0> alpha_prop_raw;
	real<lower=0> alpha_re_raw;
	real<lower=0> alpha_im_raw;
	vector<lower=0>[K] ups_raw;
	real<lower=0> d0_strength;
	real<lower=0> d1_strength;
	real<lower=0> d2_strength;
}
transformed parameters {
	//real<lower=0> Rinf = Rinf_raw*100;
	vector<lower=0>[K] q = sqrt(d0_strength*square(L0*x) + d1_strength*square(L1*x) + d2_strength*square(L2*x));
	real<lower=0> sigma_res = sigma_res_raw*0.05;
	real<lower=0> alpha_prop = alpha_prop_raw*0.05;
	real<lower=0> alpha_re = alpha_re_raw*0.05;
	real<lower=0> alpha_im = alpha_im_raw*0.05;
	vector[N] SY_hat = SA*x;
	vector[N] Y_hat = S_inv*SY_hat;
	vector[N] Y_hat_re = append_row(Y_hat[1:N/2],Y_hat[1:N/2]);
	vector[N] Y_hat_im = append_row(Y_hat[N/2+1:N],Y_hat[N/2+1:N]);
	vector<lower=0>[N] sigma_tot = sqrt(square(sigma_min) + square(sigma_res) + square(alpha_prop*Y_hat)
									+ square(alpha_re*Y_hat_re) + square(alpha_im*Y_hat_im)
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
	//Rinf_raw ~ std_normal();
	//induc ~ std_normal();
	q ~ normal(0,ups);
	dups ~ std_normal();
	Y ~ normal(Y_hat,sigma_tot);
	sigma_res_raw ~ std_normal();
}
//generated quantities {
//	vector[N] Y_hat_tilde = A_tilde*x;
//	vector[N/2] Y_hat_re_tilde = Y_hat_tilde[1:N/2];
//	vector[N/2] Y_hat_im_tilde = Y_hat_tilde[N/2+1:N];
//	vector[N] Z_hat_p_tilde = append_row(Y_hat_re_tilde ./ (square(Y_hat_re_tilde) + square(Y_hat_im_tilde)), -Y_hat_im_tilde ./ (square(Y_hat_re_tilde) + square(Y_hat_im_tilde)));
//	vector[N] Z_hat_tilde = Z_hat_p_tilde + Rinf*Rinf_vec_tilde + induc*induc_vec_tilde;
//}