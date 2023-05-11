#
# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
# 

import copy
import logging
import pprint
import pyomo.environ as pe
import numpy as np
import hmmlearn.hmm
from munch import Munch
from create_hmm_lp import create_hmm_lp

logger = logging.getLogger("hmmlearn")
logger.setLevel(logging.ERROR)


def state_similarity(s1, s2):
    assert (len(s1) == len(s2)), "ERROR: Cannot compare similarities amongst sequences of hidden states of different lengths: %d vs %d" % (len(s1), len(s2))
    count = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]: 
            count += 1
    return count/len(s1)

def print_differences(s1, s2):
    print("Differences:")
    flag=True
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            flag=False
            print("", i, s1[i], s2[i])
    if flag:
        print("", "None")


class HMMBase(object):

    def __init__(self):
        self.data = Munch()
        self.data.O = []

    def create_ip(self, *, observation_index, emission_probs, data):
        pass

    def run_training_simulations(self, n=None, debug=False, return_observations=False):
        pass

    def generate_observations(self, *, seed=None, debug=False):
        obs = self.run_training_simulations(seed=seed, n=1, return_observations=True)[0]['observations']
        return [ tuple(obs[j][t] for j in obs) for t in range(len(obs[0])) ]

    def generate_observations_and_states(self, *, seed=None, debug=False):
        results = self.run_training_simulations(seed=seed, n=1, return_observations=True)[0]
        obs = results['observations']
        return [ tuple(obs[j][t] for j in obs) for t in range(len(obs[0])) ], results['states']

    def train_HMM(self, debug=False):
        self._estimate_start_probs(debug=debug)
        self._estimate_transition_matrix(debug=debug)
        self._estimate_emissions_matrix(debug=debug)

    def _estimate_start_probs(self, debug=False):
        #
        # Estimate initial hidden state probabilities from data
        #
        start_probs = [0]*self.data.N
        for sim in self.O:
            start_probs[sim['states'][0]] += 1
        for i,v in enumerate(start_probs):
            start_probs[i] = v/len(self.O)

        self.data.start_probs = np.array(start_probs)
        if debug:
            print("")
            print("start_probs")
            pprint.pprint(start_probs)

    def _estimate_transition_matrix(self, debug=False):
        #
        # Estimate hidden state transition probabilities from data
        #
        # Conceptually, we should allow zero transition probabilities.  But
        # the training data might not exhibit all feasible transitions.  Hence,
        # we allow for a low-likelihood transition probability in all cases.
        #
        trans_mat = []
        for i in range(self.data.N):
            trans_mat.append( [1e-4]*self.data.N )

        count = 0
        Tmax = self.data['sim']['Tmax']
        for o in self.O:
            states = o['states']
            for t in range(1,Tmax):
                trans_mat[ states[t-1] ][ states[t] ] += 1
                count += 1
        for i in range(self.data.N):
            rowsum = sum(trans_mat[i])
            for j in range(self.data.N):
                trans_mat[i][j] /= rowsum

        self.data.trans_mat = np.array(trans_mat)

        if debug:
            print("")
            print("trans_mat")
            pprint.pprint(trans_mat)

    def _estimate_emissions_matrix(self, debug=False):
        #
        # Collect the different patterns of observations, and build a custom emissions matrix
        #
        emission_probs = [ ]
        for i in range(self.data.N):
            emission_probs.append( [] )
        omap = {}

        if debug:
            print("A")
            for i in self.data.A:
                print(i,self.data.A[i])

        Tmax = self.data['sim']['Tmax']
        for o in self.O:
            observations = o['observations']
            states = o['states']
            for t in range(Tmax):
                obs =  tuple(observations[j][t] for j in observations)
                if obs not in omap:
                    omap[obs] = len(omap)
                    for i in range(len(emission_probs)):
                        emission_probs[i].append(1e-4)
                emission_probs[ states[t] ] [omap[obs]] += 1

        total_obs = {}
        tmp = []
        for i,v in enumerate(emission_probs):
            total = sum(v)
            tmp.append( [val/total for val in v] )
            total_obs[i] = total
        emission_probs = tmp

        if debug:
            print("Total Obs", total_obs)
            for k,v in omap.items():
                print(v, k, [emission_probs[i][v] for i in range(self.data.N)])

        self.data.Tmax = Tmax
        self.omap = omap
        self.data.emission_probs = emission_probs
        self.data.total_obs = total_obs

    def _encode_observations(self, observations, omap):
        encoded_observations = []
        omap_len = len(omap)
        for obs in observations:
            tmp = [0]*omap_len
            tmp[omap[obs]] = 1
            encoded_observations.append(tmp)
        return encoded_observations

    def _presolve(self, observations):
        #
        # Identify if unexpected observations have been encountered.  If so,
        # augment the emission probabilities and revise the observations.
        #
        unexpected_observations=False
        tmp = []
        for obs in observations:
            if obs in self.omap:
                tmp.append(obs)
            else:
                unexpected_observations=True
                tmp.append(None)

        if unexpected_observations:
            print("WARNING: Correcting emissions probabilities to account for patterns that did not occur in the training data")
            omap = copy.copy(self.omap)
            omap[None] = len(omap)

            emission_probs = copy.copy(self.data.emission_probs)
            for i in range(len(emission_probs)):
                for j in range(len(emission_probs[i])):
                    emission_probs[i][j] = emission_probs[i][j]*self.data.total_obs[i]/(self.data.total_obs[i]+1.0)
                emission_probs[i].append( 1.0/self.data.total_obs[i] )

            return tmp, omap, emission_probs
        else:
            return tmp, self.omap, self.data.emission_probs
        

    def inference_hmmlearn(self, *, seed=None, observations=None, debug=False):
        if debug:
            print("")
            print("HMMLEARN")
            print("")
        observations, omap, emission_probs = self._presolve(observations)
        if debug:
            print("observations:", observations)
            print("emission_probs:")
            for i in range(len(emission_probs)):
                print(i, sum(emission_probs[i]), emission_probs[i])
        encoded_observations = self._encode_observations(observations, omap)

        #
        # Setup HMM
        #
        self.model = hmmlearn.hmm.MultinomialHMM(n_components=self.data.N, n_trials=1)
        self.model.n_features = len(omap)
        self.model.startprob_ = self.data.start_probs
        self.model.emissionprob_ = np.array(emission_probs)
        self.model.transmat_ = self.data.trans_mat
        #
        # Do inference
        #    
        if debug:
            print("sequence:        ", encoded_observations)
        logprob, received = self.model.decode( np.array(encoded_observations) )
        if debug:
            print("predicted states:",received)
            print("logprob",logprob)

        return logprob, received

    def create_lp(self, *, observation_index, emission_probs, data, y_binary=False, cache_indices=False):
        return create_hmm_lp(observation_index, 
                        data.N, data.start_probs, emission_probs, data.trans_mat, 
                        y_binary=y_binary, cache_indices=cache_indices)

    def inference_lp(self, *, seed=None, observations=None, debug=False):
        if debug:
            print("")
            print("LP")
            print("")
        observations, omap, emission_probs = self._presolve(observations)
        observation_index = [omap[obs] for obs in observations]

        M = self.create_lp(observation_index=observation_index, emission_probs=emission_probs, data=self.data)
        opt = pe.SolverFactory('glpk')
        res = opt.solve(M)

        log_likelihood = pe.value(M.o)

        if debug:
            print("sequence:        ", observation_index)
            print("logprob",log_likelihood)
        if log_likelihood < -10**6:
            log_likelihood = -np.inf

        states = [None]*len(observation_index)
        for t,a,b in M.y:
            if pe.value(M.y[t,a,b]) > 0:
                if t+1 < len(observation_index):
                    states[t+1] = b

        if debug:
            self.print_lp_results(M)
            print("predicted states:",states)

        return log_likelihood, states

    def inference_ip(self, *, seed=None, observations=None, debug=False):
        if debug:
            print("")
            print("IP")
            print("")
        observations, omap, emission_probs = self._presolve(observations)
        observation_index = [omap[obs] for obs in observations]

        M = self.create_ip(observation_index=observation_index, emission_probs=emission_probs, data=self.data)
        opt = pe.SolverFactory('glpk')
        res = opt.solve(M)

        if False and debug:
            M.pprint()
            M.display()

        log_likelihood = pe.value(M.o)

        if debug:
            print("sequence:        ", observation_index)
            print("logprob",log_likelihood)
        if log_likelihood < -10**6:
            log_likelihood = -np.inf

        states = [None]*len(observation_index)
        for t,a,b in M.y:
            if pe.value(M.y[t,a,b]) > 0:
                if t+1 < len(observation_index):
                    states[t+1] = b
        if debug:
            self.print_ip_results(M)
            print("predicted states:",states)

        return log_likelihood, states

    def print_lp_results(self, M):
        pass

    def print_ip_results(self, M):
        pass
