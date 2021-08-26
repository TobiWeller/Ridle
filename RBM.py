import logging
import numpy as np
from tqdm import tqdm
from tqdm import trange
import pickle

class RBM():
    """Bernoulli Restricted Boltzmann Machine (RBM)
    Parameters:
    -----------
    n_hidden: int:
        The number of processing nodes (neurons) in the hidden layer. 
    learning_rate: float
        The step length that will be used when updating the weights.
    batch_size: int
        The size of the mini-batch used to calculate each weight update.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    Reference:
        A Practical Guide to Training Restricted Boltzmann Machines 
        URL: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    def batch_iterator(self, iterable, batch_size=1):
        #l = len(iterable)
        l = iterable.shape[0]
        for ndx in range(0, l, batch_size):
            yield iterable[ndx:min(ndx + batch_size, l)]


    def __init__(self, n_hidden=128, learning_rate=0.1, batch_size=10, n_iterations=100):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_hidden = n_hidden
        #self.progressbar = progressbar.ProgressBar()

    def _initialize_weights(self, X):
        n_visible = X.shape[1]
        self.W = np.random.normal(scale=0.1, size=(n_visible, self.n_hidden))
        self.v0 = np.zeros(n_visible)       # Bias visible
        self.h0 = np.zeros(self.n_hidden)   # Bias hidden
        self.grads_first_moment_W = np.zeros((n_visible, self.n_hidden))
        self.grads_second_moment_W = np.zeros((n_visible, self.n_hidden))
        self.grads_first_moment_v = np.zeros(n_visible)
        self.grads_second_moment_v = np.zeros(n_visible)
        self.grads_first_moment_h = np.zeros(self.n_hidden)
        self.grads_second_moment_h = np.zeros(self.n_hidden)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        '''
        self.beta1 = 0.1
        self.beta2 = 0.4
        self.epsilon = 1e-1
        '''
    def fit(self, X, y=None):
        '''Contrastive Divergence training procedure'''

        self._initialize_weights(X)

        self.training_errors = []
        self.training_reconstructions = []
        #for _ in self.progressbar(range(self.n_iterations)):
        t = trange(self.n_iterations, desc='Error: ',leave=True)
        for time in t:
            time += 1
            batch_errors = []
            for batch in self.batch_iterator(X, batch_size=self.batch_size):
                # Positive phase
                positive_hidden = self.sigmoid(batch.dot(self.W) + self.h0)
                hidden_states = self._sample(positive_hidden)
                positive_associations = batch.T.dot(positive_hidden)

                # Negative phase
                negative_visible = self.sigmoid(hidden_states.dot(self.W.T) + self.v0)
                negative_visible = self._sample(negative_visible)
                negative_hidden = self.sigmoid(negative_visible.dot(self.W) + self.h0)
                negative_associations = negative_visible.T.dot(negative_hidden)


                self.grads_first_moment_W = self.beta1 * self.grads_first_moment_W + \
                              (1. - self.beta1) * (positive_associations - negative_associations)
                self.grads_second_moment_W = self.beta2 * self.grads_second_moment_W + \
                              (1. - self.beta2) * (positive_associations - negative_associations)**2

                grads_first_moment_unbiased = self.grads_first_moment_W / (1. - self.beta1**time)
                grads_second_moment_unbiased = self.grads_second_moment_W / (1. - self.beta2**time)
                
                self.W += self.lr * grads_first_moment_unbiased /(np.sqrt(grads_second_moment_unbiased) + self.epsilon)
                  
                
                self.grads_first_moment_h = self.beta1 * self.grads_first_moment_h + \
                              (1. - self.beta1) * (positive_hidden.sum(axis=0) - negative_hidden.sum(axis=0))
                self.grads_second_moment_h = self.beta2 * self.grads_second_moment_h + \
                              (1. - self.beta2) * (positive_hidden.sum(axis=0) - negative_hidden.sum(axis=0))**2
                
                grads_first_moment_unbiased = self.grads_first_moment_h / (1. - self.beta1**time)
                grads_second_moment_unbiased = self.grads_second_moment_h / (1. - self.beta2**time)
                
                self.h0 += self.lr * grads_first_moment_unbiased /(np.sqrt(grads_second_moment_unbiased) + self.epsilon)


                self.grads_first_moment_v = self.beta1 * self.grads_first_moment_v + \
                              (1. - self.beta1) * (batch.sum(axis=0) - negative_visible.sum(axis=0))
                self.grads_second_moment_v = self.beta2 * self.grads_second_moment_v + \
                              (1. - self.beta2) * (batch.sum(axis=0) - negative_visible.sum(axis=0))**2

                grads_first_moment_unbiased = self.grads_first_moment_v / (1. - self.beta1**time)
                grads_second_moment_unbiased = self.grads_second_moment_v / (1. - self.beta2**time)

                self.v0 += self.lr * grads_first_moment_unbiased /(np.sqrt(grads_second_moment_unbiased) + self.epsilon)

                batch_errors.append(np.mean((batch - negative_visible) ** 2))

            self.training_errors.append(np.mean(batch_errors))
            t.set_description('Error: {e}'.format(e = self.training_errors[-1]))
            t.refresh() # to show immediately the update
            # Reconstruct a batch of images from the training set
            idx = np.random.choice(range(X.shape[0]), self.batch_size)
            self.training_reconstructions.append(self.reconstruct(X[idx]))

    # Implemented by me, does the same as reconstruct. But some computations are not needd.
    def predict(self, X):
        # Positive phase
        positive_hidden = self.sigmoid(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        positive_associations = X.T.dot(positive_hidden)

        # Negative phase
        negative_visible = self.sigmoid(hidden_states.dot(self.W.T) + self.v0)
        #negative_visible = self._sample(negative_visible)
        negative_hidden = self.sigmoid(negative_visible.dot(self.W) + self.h0)
        negative_associations = negative_visible.T.dot(negative_hidden)
        
        return negative_visible

    def predict_count(self, X):
        # Positive phase
        positive_hidden = self.sigmoid(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        negative_visible = self.sigmoid(hidden_states.dot(self.W.T) + self.v0)
        
        return negative_visible, hidden_states
        
    def _sample(self, X):
        return X > np.random.random_sample(size=X.shape)

    def reconstruct(self, X):
        positive_hidden = self.sigmoid(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        negative_visible = self.sigmoid(hidden_states.dot(self.W.T) + self.v0)
        return negative_visible
     
    def compress(self, X):
        positive_hidden = self.sigmoid(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        return positive_hidden

    def save(self, name):
        """save class as self.name.txt"""
        with open(name, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        """try load self.name.txt"""
        with open(file, 'rb') as handle:
            tmp_dict  = pickle.load(handle)
        self.__dict__.update(tmp_dict) 


if __name__ == "__main__":
    X = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    rbm = RBM(n_hidden=100, n_iterations=1000, batch_size=25, learning_rate=0.1)
    rbm.fit(X)
    print('Actual')
    print(X)
    print('Predicted')
    print(np.around(rbm.predict(X), decimals=2))