import logging
import numpy as np
from scipy.stats import pearsonr
import time

logger = logging.getLogger(__name__)

class SPSA:

    def __init__(self, train_graphs_a, train_graphs_b, predictor, targets
                , dev_graphs_a=None, dev_graphs_b=None, targets_dev=None
                , init_lr=0.1, A=2, alpha=0.5, gamma=0.05, c=0.01
                , eval_steps=100, n_batch=16, check_every_n_batch=350):

        # predictor must have predict and set_params and get_params
        self.predictor = predictor #.wl_dist_mat_generator
        self.targets = targets
        self.train_graphs_a = train_graphs_a
        self.train_graphs_b = train_graphs_b
        self.dev_graphs_a = dev_graphs_a
        self.dev_graphs_b = dev_graphs_b
        self.targets_dev = targets_dev

        if not dev_graphs_a:
            self.dev_graphs_a = self.train_graphs_a
            self.dev_graphs_b = self.train_graphs_b
            self.targets_dev = self.targets
         
        self.init_lr = init_lr
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.eval_steps = eval_steps
        self.n_batch = n_batch
        self.check_every_n_batch = check_every_n_batch
        
        return None

    def error(self, x, y):
        """Calculates error over two arrays/lists with scalars
        
        Here, the error is defined as 1 - pearsonr, but can be set differently.
        
        Args:
            x (list or array with floats): input 1
            y (list or array with floats): input 2
            
        Returns:
            error
        """

        pr = pearsonr(x, y)[0]
        if pr >= -1 and pr <= 1:
            return 1 - pr
        return 0.0

    def pseudo_grad(self, x, ids, c, rand):
        """estimate gradient

        Args:
            x (numpy array): input parameters
            ids (list): indeces of training examples on which we perform the
                        estimation
            c: constant
            rand: random vector

        Returns:
            estimated gradient
        """

        in_1 = [self.train_graphs_a[i] for i in ids]
        in_2 = [self.train_graphs_b[i] for i in ids]
        
        self.predictor.wl_dist_mat_generator.set_params(x + c * rand)
        a = self.predictor.predict(in_1, in_2)
        error1 = self.error(a, [self.targets[i] for i in ids])
        
        self.predictor.wl_dist_mat_generator.set_params(x - c * rand)
        b = self.predictor.predict(in_1, in_2) 
        error2 = self.error(b, [self.targets[i] for i in ids])
        
        num = error1 - error2
        
        return num / (2 * c * rand), error1, error2, self.predictor.wl_dist_mat_generator.active_params

    def clip(self, grad, x=1):
        """clip values in array"""
        
        clipped = np.clip(grad, -x, x)
        return clipped

    def fit(self):
        """fit the parameters of the underlying predictor"""

        
        
        prstart = pearsonr(self.targets_dev
                , self.predictor.predict(self.dev_graphs_a
                , self.dev_graphs_b))[0]
        
        logger.info("start pearsonr {}".format(prstart))
        
        iters = 1
        eval_steps_done = 0
        grad_norms = []
        errors = []
        best_dev_score = prstart
        best_params = self.predictor.wl_dist_mat_generator.get_params().copy()
        param_shape = best_params.shape
        logger.debug("parameter shape  {}".format(param_shape))

        while True:
            # sample mini batch ids
            i = np.random.randint(0, len(self.train_graphs_a), size=self.n_batch) 
            
            # sample from bernoulli
            rand = np.random.randint(0, 2, size=param_shape)
            rand[rand == 0] = -1
            
            # update c
            c = self.c / iters**self.gamma
            #obtain current params
            x = self.predictor.wl_dist_mat_generator.get_params().copy()
            #compute pseudo grad
            pseudo_grad, error1, error2, active_params = self.pseudo_grad(x, i, c, rand)
            
            active_params /= active_params.sum()
            pseudo_grad *= active_params[:,None]
            #collect some stats
            grad_norms.append(np.linalg.norm(pseudo_grad))
            errors.append(error1)
            errors.append(error2)
            
            #update learning rate
            lr = self.init_lr / (iters + self.A)**self.alpha
            
            #gradient clip
            pseudo_grad = self.clip(pseudo_grad, x=0.01)
            
            #SGD rule
            params = x - lr * pseudo_grad
            
            #update params
            self.predictor.wl_dist_mat_generator.set_params(params)
            iters += 1

            #maybe check results on the development set
            if iters % self.check_every_n_batch == 0:
                logger.info("current params: {}".format(list(params.flatten())))
                logger.info("param keys: {}".format(list(self.predictor.wl_dist_mat_generator.param_keys)))
                
                # some debugging
                logger.debug("mean of grad norms {}; \
                        max of grad values {}".format(np.mean(grad_norms), np.max(pseudo_grad)))
                grad_norms = []
                logger.debug("mean of errors {}; \
                        current learning rate {}; \
                        c={}".format(np.mean(errors), lr, self.c / iters**self.gamma))
                errors = []

                #compute score on dev
                logger.info("conducting evaluation step {}; \
                        processed examples={}; \
                        systime={}".format(eval_steps_done, iters * self.n_batch, time.time()))
                dev_preds = self.predictor.predict(self.dev_graphs_a
                                                    , self.dev_graphs_b)
                pr = pearsonr(self.targets_dev, dev_preds)
                pr = pr[0]
                logger.info("current score {}".format(pr))
                if pr > best_dev_score:
                    print("new high score on dev! Old score={}; \
                            New score={}, \
                            improvement=+{}; \
                            total improvement=+{};\
                            saving params...".format(
                                    best_dev_score, pr, pr - best_dev_score, pr - prstart))
                    best_dev_score = pr
                    best_params = params.copy()
                eval_steps_done += 1
            
            # maybe stop training
            if self.eval_steps == eval_steps_done:
                self.predictor.wl_dist_mat_generator.set_params(best_params)
                return None

        
