from learner import PolicyGradientAgent

class ActorCriticAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are introducing a general notion of a baseline in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. This introduces a notion of the value of a state into our policy-based method.
    '''

    def __init__(self, 
            actor_fn:AbstractPolicy, 
            critic_fn:AbstractReturns, 
            env: gym.Env=gym.make('CartPole-v1'),
            batch_size:int=16, 
            gamma: float = 0.98, 
            target_update:int=400,
            buffer_size:int=0,
            ) -> None:
        '''
        Params
        ------
            actor_fn: policy function 'pi'. represents probability denstity distribution (see policy_functions)
            critic_fn: Value-function aka Returns aka Advantage, can be approximation or monte-carlo playouts
            env: OpenAI Gym Environment
            batch_size: the # of steps per policy optimization
            gamma: discount factor
            target_update: target update frequency
            buffer_size: the size of the experienece replay sampling buffer, if 0, most recent steps are always used
        '''


        super().__init__(gamma, env)
        self.batch_size = batch_size

        self.pi = actor_fn
        self.value = critic_fn
        
        self.target_pi

        self.buffered = buffer_size > 0
        if self.buffered:
            self.buffer = ReplayBuffer(buffer_size, self.pi.state_space.shape, self.pi.action_space.shape)

    def train(self, n_iter):
        self.n_iter = n_iter
        avg_batch_rewards = []
        scores = []

        pbar = tqdm(range(1, n_iter+1))
        for episode in pbar:
            pass

