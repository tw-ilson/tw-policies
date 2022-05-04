def score(self, s_t, a_t):
    ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
	which follows a single linear combination layer of the state input.
	Params
	------
		s_t: state at time t
		a_t: action taken from s_t
    '''
    #compute SoftMax
    sm_probs = self.forward(s_t)
    
    #one-hot encode action
    act_mask = np.zeros(self.action_dim)
    act_mask[a_t] = 1

    #score calculation & gradient update
    score = np.outer(s_t, act_mask) - np.outer(s_t, np.exp(sm_probs))
    self.d_weights += score

    assert not np.isnan(self.d_weights).any()
    return score
