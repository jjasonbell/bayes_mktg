// Multinomial logit model 
// Suitable for typical scanner panel data with 
// a common choice set for each period

// Morgan Bale, morganarhodes@gmail.com
// Elea McDonnell Feit, eleafeit@gmail.com

data {
  int<lower=1> N;	// num observations
  int<lower=1> M; // sum over weeks of the number of upcs in each week i.e. sum_t(J[t])
	int<lower=1> K; // num covariates
  int<lower=1> T; // num weeks
  int<lower=2> J[T]; // num products in week t
	int<lower=1,upper=T> t[N]; // week for each observation
  int<lower=1, upper=M> xstart[T]; // first row in X matrix for week t
	int<lower=0> Y[N]; // choices
	matrix[M,K] X; // attributes (for each week)
}

parameters {
	vector[K] beta;
}

model {
	// model
	for (i in 1:N) 
		Y[i] ~ categorical_logit(block(X, xstart[t[i]], 1, J[t[i]], K)*beta);
}

