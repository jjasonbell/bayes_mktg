// Multinomial logit model 
// Suitable for typical scanner panel data 
// with a common choice set for each period
// and brand intercepts that apply to multiple UPCs in the choice set

// Morgan Bale, morganarhodes@gmail.com
// Elea McDonnell Feit, eleafeit@gmail.com

data {
  int<lower=1> N;	// num observations
  int<lower=1> M; // sum over weeks of the number of upcs in each week i.e. sum_t(J[t])
	int<lower=1> K; // num covariates
  int<lower=1> T; // num weeks
	int<lower=1> B; // num brands
  int<lower=2> J[T]; // num upcs in week t
	int<lower=1,upper=B> b[M]; // brand for each UPC
	int<lower=1,upper=T> t[N]; // week for each observation
  int<lower=1, upper=M> xstart[T]; // first row in X matrix for week t
	int<lower=0> Y[N]; // choices
	matrix[M,K] X; // attributes (for each week)
}

parameters {
	vector[K] beta;
	vector[B-1] gamma_raw; 
}

transformed parameters{
  vector[B] gamma; 
  vector[M] alpha; // upc intercepts 
  // sum-to-zero constraint on brand coefficients
  gamma = append_row(-sum(gamma_raw), gamma_raw);
  // brand effects transferred to the intercept
  for (m in 1:M) alpha[m] = gamma[b[m]];
  
}

model {
	// model
	for (i in 1:N) 
		Y[i] ~ categorical_logit(segment(alpha, xstart[t[i]], J[t[i]]) + block(X, xstart[t[i]], 1, J[t[i]], K)*beta);
}


