#
# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
#

import json
import copy
import logging
import pprint
import pyomo.environ as pe
import numpy as np
import hmmlearn.hmm
from munch import Munch
from .create_hmm_lp import create_hmm_lp

logger = logging.getLogger("hmmlearn")
logger.setLevel(logging.ERROR)


class HMMBase(object):
    def __init__(self):
        self.data = Munch()
        self.data.O = []
        self.data.seed = None

    def create_ip(self, *, observation_index, emission_probs, data):
        pass

    def run_training_simulations(self, n=None, debug=False, return_observations=False, seed=None):
        pass

    def _tuplize_observations(self, observations):
        if type(observations[0]) is list:
            return [
                tuple(observations[j][t] for j in observations)
                for t in range(len(observations[0]))
            ]
        else:
            return observations

    def generate_observations(self, *, seed=None, debug=False):
        if seed is None:
            seed = self.data.seed
        obs = self.run_training_simulations(seed=seed, n=1, return_observations=True)[
            0
        ]["observations"]
        return self._tuplize_observations(results["observations"])

    def generate_observations_and_states(self, *, seed=None, debug=False):
        if seed is None:
            seed = self.data.seed
        results = self.run_training_simulations(
            seed=seed, n=1, return_observations=True
        )[0]
        return self._tuplize_observations(results["observations"]), results["states"]

    def train_HMM(self, debug=False):
        self._estimate_start_probs(debug=debug)
        self._estimate_transition_matrix(debug=debug)
        self._estimate_emissions_matrix(debug=debug)

    def _estimate_start_probs(self, debug=False):
        #
        # Estimate initial hidden state probabilities from data
        #
        start_probs = [0] * self.data.N
        for sim in self.O:
            start_probs[sim["states"][0]] += 1
        total = sum(start_probs)
        for i, v in enumerate(start_probs):
            start_probs[i] = v / total

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
            trans_mat.append([1e-4] * self.data.N)

        count = 0
        Tmax = self.data["sim"]["Tmax"]
        for o in self.O:
            states = o["states"]
            for t in range(1, Tmax):
                trans_mat[states[t - 1]][states[t]] += 1
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
        emission_probs = []
        for i in range(self.data.N):
            emission_probs.append([])
        omap = {}

        Tmax = self.data["sim"]["Tmax"]
        for o in self.O:
            states = o["states"]
            observations = self._tuplize_observations(o["observations"])
            for t in range(Tmax):
                obs = observations[t]
                if obs not in omap:
                    omap[obs] = len(omap)
                    for i in range(len(emission_probs)):
                        emission_probs[i].append(1e-4)
                emission_probs[states[t]][omap[obs]] += 1

        total_obs = {}
        tmp = []
        for i, v in enumerate(emission_probs):
            total = sum(v)
            tmp.append([val / total for val in v])
            total_obs[i] = total
        emission_probs = tmp

        if debug:
            print("Total Obs", total_obs)
            for k, v in omap.items():
                print(v, k, [emission_probs[i][v] for i in range(self.data.N)])

        self.data.Tmax = Tmax
        self.omap = omap
        self.data._emission_probs = emission_probs
        self.data._total_obs = total_obs

    def _encode_observations(self, observations, omap):
        encoded_observations = []
        omap_len = len(omap)
        for obs in observations:
            tmp = [0] * omap_len
            tmp[omap[obs]] = 1
            encoded_observations.append(tmp)
        return encoded_observations

    def _presolve(self, observations):
        #
        # Identify if unexpected observations have been encountered.  If so,
        # augment the emission probabilities and revise the observations.
        #
        num_unexpected_observations = 0
        _observations = []
        for obs in observations:
            if obs in self.omap:
                _observations.append(obs)
            else:
                num_unexpected_observations += 1
                _observations.append(None)

        if num_unexpected_observations > 0:
            print(
                "WARNING: Correcting emissions probabilities to account for patterns that did not occur in the training data"
            )
            omap = copy.copy(self.omap)
            omap[None] = len(omap)

            emission_probs = copy.copy(self.data._emission_probs)
            for i in range(len(emission_probs)):
                foo = [
                    emission_probs[i][j] * self.data._total_obs[i]
                    for j in range(len(emission_probs[i]))
                ]
                total = sum(foo) + num_unexpected_observations
                foo.append(num_unexpected_observations)
                emission_probs[i] = [v / total for v in foo]
                # print("HERE", i,sum(emission_probs[i]))

            return _observations, omap, emission_probs
        else:
            return _observations, self.omap, self.data._emission_probs

    def inference_hmmlearn(self, *, observations=None, debug=False):
        if debug:
            print("")
            print("HMMLEARN")
            print("")
        observations, omap, emission_probs = self._presolve(observations)
        if False and debug:
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
        logprob, received = self.model.decode(np.array(encoded_observations))

        self.results = Munch(M=self.model, log_likelihood=logprob, states=received)
        if debug:
            self.print_hmm_results()
            print("predicted states:", received)
            print("logprob", logprob)

    def create_lp(
        self,
        *,
        observation_index,
        emission_probs,
        data,
        y_binary=False,
        cache_indices=False
    ):
        return create_hmm_lp(
            observation_index,
            data.N,
            data.start_probs,
            emission_probs,
            data.trans_mat,
            y_binary=y_binary,
            cache_indices=cache_indices,
        )

    def inference_lp(self, *, observations=None, debug=False, solver="glpk"):
        if debug:
            print("")
            print("LP")
            print("")
        observations, omap, emission_probs = self._presolve(observations)
        observation_index = [omap[obs] for obs in observations]

        M = self.create_lp(
            observation_index=observation_index,
            emission_probs=emission_probs,
            data=self.data,
        )
        assert M is not None, "No model returned from the create_lp() method"
        opt = pe.SolverFactory(solver)
        res = opt.solve(M)

        log_likelihood = pe.value(M.o)

        if debug:
            print("sequence:        ", observation_index)
            print("logprob", log_likelihood)
        if log_likelihood < -(10**6):
            log_likelihood = -np.inf

        states = [None] * len(observation_index)
        for t, a, b in M.y:
            if pe.value(M.y[t, a, b]) > 0:
                if t + 1 < len(observation_index):
                    states[t + 1] = b

        self.results = Munch(M=M, log_likelihood=log_likelihood, states=states)
        if debug:
            self.print_lp_results()
            print("predicted states:", states)

    def inference_ip(self, *, observations=None, debug=False, solver="glpk"):
        if debug:
            print("")
            print("IP")
            print("")
        observations, omap, emission_probs = self._presolve(observations)
        observation_index = [omap[obs] for obs in observations]

        M = self.create_ip(
            observation_index=observation_index,
            emission_probs=emission_probs,
            data=self.data,
        )
        assert M is not None, "No model returned from the create_ip() method"
        opt = pe.SolverFactory(solver)
        res = opt.solve(M)

        if False and debug:
            M.pprint()
            M.display()

        log_likelihood = pe.value(M.o)

        if debug:
            print("sequence:        ", observation_index)
            print("logprob", log_likelihood)
        if log_likelihood < -(10**6):
            log_likelihood = -np.inf

        states = [None] * len(observation_index)
        for t, a, b in M.y:
            if pe.value(M.y[t, a, b]) > 0:
                if t + 1 < len(observation_index):
                    states[t + 1] = b

        self.results = Munch(M=M, log_likelihood=log_likelihood, states=states)
        if debug:
            self.print_ip_results()
            print("predicted states:", states)

    def get_hmm_results(self, results):
        ans = {}
        ans['n_features'] = results.M.n_features
        ans['start_probs'] = results.M.startprob_.tolist()
        ans['emission_probs'] = results.M.emissionprob_.tolist()
        ans['trans_mat'] = results.M.transmat_.tolist()
        return ans

    def get_lp_results(self, M):
        return {"y: activities": [ [t,a,b, pe.value(M.y[t,a,b])] for t,a,b in M.y if pe.value(M.y[t,a,b]) > 0]}

    def get_ip_results(self, M):
        return {}

    def print_hmm_results(self):
        pprint.pprint( self.get_hmm_results(self.results) )

    def print_lp_results(self):
        pprint.pprint( self.get_lp_results(self.results.M) )

    def print_ip_results(self):
        pprint.pprint( self.get_ip_results(self.results.M) )

    def write_hmm_results(self, filename):
        with open(filename, 'w') as OUTPUT:
            json.dump(self.get_hmm_results(self.results), OUTPUT, sort_keys=True, indent=4, ensure_ascii=False)

    def write_lp_results(self, filename):
        with open(filename, 'w') as OUTPUT:
            json.dump(self.get_lp_results(self.results.M), OUTPUT, sort_keys=True, indent=4, ensure_ascii=False)

    def write_ip_results(self, filename):
        with open(filename, 'w') as OUTPUT:
            json.dump(self.get_ip_results(self.results.M), OUTPUT, sort_keys=True, indent=4, ensure_ascii=False)

