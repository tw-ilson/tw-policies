def score(self, s_t, a_t):
	''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
	    which follows a single linear combination layer of the state input.
	'''
	mu, sd = self.forward(s_t)

	mu_score = (a_t - mu)/(sd**2)
	sd_score = ((a_t - mu)**2) / sd**2 - 1
	# print(mu, sd, a_t)
	mu_grad = np.outer(s_t, mu_score)
	sd_grad = np.outer(s_t, sd_score)
	# print(sd_grad)

	assert not np.isnan(mu_grad).any()
	assert not np.isnan(sd_grad).any()
	self.d_mu_weight += mu_grad
	self.d_sd_weight += sd_grad
	return [mu_grad, sd_grad]
