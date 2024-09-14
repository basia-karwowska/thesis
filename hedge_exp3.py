import numpy as np

np.random.seed(123)

# Source: andCelli github; look if it is relevant with self.w[indices] indexing


class Hedge:
    def __init__(self, learning_rate, nActions):
        if learning_rate is None:
            learning_rate = np.sqrt(np.log(nActions) / nActions)
        self.learning_rate = learning_rate # epsilon in (0, 1]
        self.nActions = nActions
        self.w = np.full(nActions, 1. / nActions)[:,np.newaxis]
        self.p = np.full(nActions, 1. / nActions)[:,np.newaxis]
        self.action = None

    def nextAction(self, indices = None): # indices gives an option to specify 
        # the subset of arms to choose from (e.g. respecting global budget balance)
        # check to prevent the weight decay
        
        if indices is None: # by default, choice is among all arms
            indices = np.arange(self.nActions)
            self.p = self.w
        else:
            self.p = self.w[indices] / np.sum(self.w[indices]) 
        '''
        if np.isclose(np.sum(self.w[indices]), 0):
            self.p = np.full(self.nActions, 1.0 / self.nActions) # [:,np.newaxis]
        else:
            self.p = self.w
        '''
        '''
        # if we have a subset, we need to modify self p, renormalize and make it smaller vector
        if not np.array_equal(indices, np.arange(self.nActions)):  # choice potentially restricted, probabilities conditional
            # on belonging to allowed actions
        '''
          
        self.action = np.random.choice(indices, p=self.p.reshape(1, -1).squeeze()) 
        
        return self.action

    def observeReward(self, profit, scaling = 1): # loss is the difference between the maximum 
        # theoretical profit (if the price posted to the buyer is 1 and to the
        # seller is 0 and the trade gets accepted by both parties i.e. their
        # valuations are 1 and 0 respectively) and the profit from the actual trade
        loss = 1 / scaling - profit
        self.w = self.w * np.exp(-self.learning_rate * loss) # weights of all
        # the actions are updated in the full feedback setting
        epsilon = 1e-12  # Small regularization constant
        self.w = self.w + epsilon
        self.w = self.w / np.sum(self.w) # normalize weights for numerical stability

class EXP3(Hedge):
    def __init__(self, learning_rate, nActions):
        super().__init__(learning_rate, nActions)

    def nextAction(self, indices = None):
        return super().nextAction(indices)
    
    def observeReward(self, profit, indices = None): # profit - scalar
        if indices is None:
            action_idx = self.action
        else:
            action_idx = np.where(indices == self.action)[0]
        # importance weighting to make the estimate unbiased, one-bit feedback
        p_action = self.p[action_idx] # the actual probability according to which an
        # action was drawn
        profits = np.zeros(self.nActions).reshape(-1,1) 
        profits[self.action] = profit / p_action 
        super().observeReward(profits, scaling = p_action)
        
        
        
        
        

class Hedge2:
    def __init__(self, learning_rate, nActions):
        if learning_rate is None:
            learning_rate = np.sqrt(np.log(nActions) / nActions)
        self.learning_rate = learning_rate # epsilon in (0, 1]
        self.nActions = nActions
        self.w = np.full(nActions, 1. / nActions)[:,np.newaxis]
        self.p = np.full(nActions, 1. / nActions)[:,np.newaxis]
        self.action = None

    def nextAction(self, indices = None): # indices gives an option to specify 
        # the subset of arms to choose from (e.g. respecting global budget balance)
        # check to prevent the weight decay
        
        if indices is None: # by default, choice is among all arms
            indices = np.arange(self.nActions)
        
        if np.isclose(np.sum(self.w[indices]), 0):
            self.p = np.full(self.nActions, 1.0 / self.nActions) # [:,np.newaxis]
        else:
            self.p = self.w
        
        # if we have a subset, we need to modify self p, renormalize and make it smaller vector
        if not np.array_equal(indices, np.arange(self.nActions)):  # choice potentially restricted, probabilities conditional
            # on belonging to allowed actions
            self.p = self.p[indices] / np.sum(self.p[indices])   
        self.action = np.random.choice(indices, p=self.p.reshape(1, -1).squeeze()) 
        
        return self.action

    def observeReward(self, profit, scaling = 1): # loss is the difference between the maximum 
        # theoretical profit (if the price posted to the buyer is 1 and to the
        # seller is 0 and the trade gets accepted by both parties i.e. their
        # valuations are 1 and 0 respectively) and the profit from the actual trade
        loss = 1 / scaling - profit
        self.w = self.w * np.exp(-self.learning_rate * loss) # weights of all
        # the actions are updated in the full feedback setting
        self.w = self.w / np.sum(self.w) # normalize weights for numerical stability

class EXP32(Hedge):
    def __init__(self, learning_rate, nActions):
        super().__init__(learning_rate, nActions)

    def nextAction(self, indices = None):
        return super().nextAction(indices)
    
    def observeReward(self, profit, indices = None): # profit - scalar
        if indices is None:
            action_idx = self.action
        else:
            action_idx = np.where(indices == self.action)[0]
        # importance weighting to make the estimate unbiased, one-bit feedback
        p_action = self.p[action_idx] # the actual probability according to which an
        # action was drawn
        profits = np.zeros(self.nActions).reshape(-1,1) 
        profits[self.action] = profit / p_action 
        super().observeReward(profits, scaling = p_action)
    
        
        
        
        
        
        