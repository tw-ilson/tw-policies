class REINFORCEAgent(PolicyGradientAgent):

    def __init__(self, 
            policy_fn:Callable,
            gamma: float = 0.99, 
            env: gym.Env = None) -> None:
        super().__init__(gamma, env)
        
        self.pi = policy_fn
        self.G = MonteCarloReturns(self.gamma)


    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        for episode in pbar:
            tau = self.playout()
            self.G.update_baseline(tau)
            for s_t, a_t, r in tau:
                score = self.pi.score(s_t, a_t)
                self.pi.optimize(
                        score,
                        self.G(s_t))

