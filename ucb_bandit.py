import numpy as np

class UCB:
    def __init__(self, nActions, c=1):
        self.nActions = nActions # number of arms, actions, price pairs
        self.Q = np.zeros(nActions) # average rewards  (here: profits)
        self.X = np.zeros(nActions) # cumulative rewards (here: profits)
        self.N = np.zeros(nActions) # number of arm pulls
        self.t = 0  
        self.action = None
        self.c = c # scaling factor for the exploration term
        # self.acc_profit = []

    def nextAction(self, expected_profits = None):
        # Possibility to add expected_profits estimated in an alternative way 
        # in the action selection. Q and expected_profits should converge in 
        # the long run, but expected_profits may be more precise quicker.
        # Expected_profits are nontrivial in the case of global budget balance.
        self.t += 1
        # Initial exploration: ensure that each action is selected at least once
        if self.t <= self.nActions:
            self.action = self.t - 1
        # Selection of an price pair maximizing the UCB value
        else:
            ucb_values = self.Q + self.c * np.sqrt((2 * np.log(self.t)) / self.N) 
            if expected_profits is None:
                expected_profits = np.zeros(self.nActions)
            self.action = np.argmax(ucb_values + expected_profits)
        return self.action

    def observeReward(self, profit):
        # Parameters updates
        self.X[self.action] += profit # update the reward vector
        self.N[self.action] += 1 # update the vector of action counts
        # Update the average reward vector
        self.Q[self.action] = self.X[self.action] / self.N[self.action]
        # self.acc_profit.append(profit + (self.acc_profit[-1] if self.acc_profit else 0.0))

    def reset(self):
        self.Q = np.zeros(self.nActions)
        self.X = np.zeros(self.nActions) 
        self.N = np.zeros(self.nActions)
        self.t = 0
        self.action = None
        # self.acc_profit = []

    def results(self):
        return self.acc_profit
    
    
    