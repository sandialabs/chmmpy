#
# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
#

import math
import pprint
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
        self.model_data = None

    def create_ip(self, *, observations_index, emission_probs, data):
        pass

    def load_observations(self, observations, time=None):
        #
        # WEH - How will this be used?
        #       Should this be defined in the HMMBase class?
        #
        if time is None:
            time = list(range(len(observations)))
        self.O = [dict(time=time, observations=observations)]

        for o in observations:
            assert o in self.omap, "Unexpected observation {}".format(o)

    def load_model(self, *, start_probs, emission_probs, trans_mat, tolerance=None, emission_tolerance=0.0, start_tolerance=0.0, trans_tolerance=0.0):

        if tolerance is not None:
            start_tolerance=tolerance
            emission_tolerance=tolerance
            trans_tolerance=tolerance

        # start_probs
        start_probs_ = [0.0] * self.data.N
        for s, v in start_probs.items():
            start_probs_[self.smap[s]] = v
        if start_tolerance:
            for i in range(self.data.N):
                start_probs_[i] = (start_probs_[i] + start_tolerance) / (
                    1 + self.data.N * start_tolerance
                )
        assert (
            math.fabs(sum(start_probs_) - 1) < 1e-7
        ), "Total start probability is {} but expected 1.0".format(sum(start_probs_))
        self.data.start_probs = np.array(start_probs_)

        # emission probs
        emission_probs_ = []
        for i in range(self.data.N):
            emission_probs_.append([0.0] * len(self.omap))
        for s, v in emission_probs.items():
            for o, p in v.items():
                emission_probs_[self.smap[s]][self.omap[o]] = p
            if emission_tolerance:
                for i in range(len(self.omap)):
                    emission_probs_[self.smap[s]][i] = (
                        emission_probs_[self.smap[s]][i] + emission_tolerance
                    ) / (1 + len(self.omap) * emission_tolerance)
            assert math.fabs(sum(emission_probs_[self.smap[s]]) - 1) < 1e-7
        self.data._emission_probs = emission_probs_

        self.data._total_obs = {i: 1.0 for i in range(self.data.N)}

        # transition matrix
        trans_mat_ = []
        for i in range(self.data.N):
            trans_mat_.append([0.0] * self.data.N)
        for k, v in trans_mat.items():
            s, s_ = k
            trans_mat_[self.smap[s]][self.smap[s_]] = v
        if trans_tolerance:
            for s in self.smap:
                for s_ in self.smap:
                    trans_mat_[self.smap[s]][self.smap[s_]] = (
                        trans_mat_[self.smap[s]][self.smap[s_]] + trans_tolerance
                    ) / (1 + len(self.smap) * trans_tolerance)
        assert math.fabs(sum(trans_mat_[self.smap[s]]) - 1) < 1e-7
        self.data.trans_mat = np.array(trans_mat_)

        self.model_data = Munch(
            start_probs=start_probs,
            emission_probs=emission_probs,
            trans_mat=trans_mat,
            start_tolerance=start_tolerance,
            emission_tolerance=emission_tolerance,
            trans_tolerance=trans_tolerance,
        )

    def run_training_simulations(
        # Always returns a list
        self,
        n=None,
        debug=False,
        return_observations=False,
        seed=None,
    ):
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

    def train_HMM(self, debug=False, start_tolerance=0.0):
        assert len(self.O) > 0, "Expecting simulations in the self.O object"
        if type(self.O[0]["observations"]) is dict:
            Tmax = len(self.O[0]["observations"][0])
        else:
            Tmax = len(self.O[0]["observations"])
        self._estimate_start_probs(debug=debug, start_tolerance=start_tolerance)
        self._estimate_transition_matrix(Tmax=Tmax, debug=debug)
        self._estimate_emissions_matrix(Tmax=Tmax, debug=debug)
        smap = {}
        for i, s in enumerate(self.data.hidden_states):
            smap[s] = i
        self.smap = smap

    def _estimate_start_probs(self, debug=False, start_tolerance=0.0):
        #
        # Estimate initial hidden state probabilities from data
        #
        start_probs = [start_tolerance] * self.data.N
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

    def _estimate_transition_matrix(self, Tmax=None, debug=False):
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

    def _estimate_emissions_matrix(self, Tmax=None, debug=False):
        #
        # Collect the different patterns of observations, and build a custom emissions matrix
        #
        emission_probs = []
        for i in range(self.data.N):
            emission_probs.append([])
        omap = {}

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
        _observations, omap, emission_probs = self._presolve(observations)
        if False and debug:
            print(omap)
            print(self.smap)
            print("startprob_   ", self.data.start_probs)
            print("observations:", _observations)
            print("emission_probs:")
            for i in range(len(emission_probs)):
                print(i, sum(emission_probs[i]), emission_probs[i])
        encoded_observations = self._encode_observations(_observations, omap)

        #
        # Setup HMM
        #
        self.model = hmmlearn.hmm.MultinomialHMM(n_components=self.data.N, n_trials=1, verbose=debug)
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
        # print("HEREx",logprob,received)
        # print("HEREy",self.model._compute_log_likelihood(np.array(encoded_observations)))

        self.results = Munch(
            M=self.model,
            log_likelihood=logprob,
            states=received,
            observations=observations,
            hidden={i: self.data.hidden_states[s] for i, s in enumerate(received)},
        )

        if debug:
            self.print_hmm_results()
            print("predicted states:", received)
            print("logprob", logprob)

    def create_lp(
        self,
        *,
        observations_index,
        emission_probs,
        data,
        y_binary=False,
        cache_indices=True
    ):
        return create_hmm_lp(
            observations_index,
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

        _observations, omap, emission_probs = self._presolve(observations)
        observations_index = [omap[obs] for obs in _observations]
        M = self.create_lp(
            observations_index=observations_index,
            emission_probs=emission_probs,
            data=self.data,
        )
        assert M is not None, "No model returned from the create_lp() method"
        opt = pe.SolverFactory(solver)
        res = opt.solve(M, tee=debug)

        log_likelihood = pe.value(M.o)

        if debug:
            print("sequence:        ", observations_index)
            print("logprob", log_likelihood)
        if log_likelihood < -(10**6):
            log_likelihood = -np.inf

        states = [None] * len(observations_index)
        for t, a, b in M.y:
            if pe.value(M.y[t, a, b]) > 0:
                if t + 1 < len(observations_index):
                    states[t + 1] = b

        self.results = Munch(
            observations=observations,
            start_probs=self.data.start_probs,
            emission_probs=emission_probs,
            trans_mat=self.data.trans_mat,
            M=M,
            log_likelihood=log_likelihood,
            states=states,
            hidden={i: self.data.hidden_states[s] for i, s in enumerate(states)},
        )
        if debug:
            self.print_lp_results()
            print("predicted states:", states)

    def inference_ip(self, *, observations=None, debug=False, solver="glpk"):
        if debug:
            print("")
            print("IP")
            print("")

        _observations, omap, emission_probs = self._presolve(observations)
        observations_index = [omap[obs] for obs in _observations]
        M = self.create_ip(
            observations_index=observations_index,
            emission_probs=emission_probs,
            data=self.data,
        )
        assert M is not None, "No model returned from the create_ip() method"
        opt = pe.SolverFactory(solver)
        res = opt.solve(M, tee=debug)

        if False and debug:
            M.pprint()
            M.display()

        log_likelihood = pe.value(M.o)

        if debug:
            print("sequence:        ", observations_index)
            print("logprob", log_likelihood)
        if log_likelihood < -(10**6):
            log_likelihood = -np.inf

        states = [None] * len(observations_index)
        for t, a, b in M.y:
            if pe.value(M.y[t, a, b]) > 0:
                if t + 1 < len(observations_index):
                    states[t + 1] = b

        self.results = Munch(
            observations=observations,
            start_probs=self.data.start_probs,
            emission_probs=emission_probs,
            trans_mat=self.data.trans_mat,
            M=M,
            log_likelihood=log_likelihood,
            states=states,
            hidden={i: self.data.hidden_states[s] for i, s in enumerate(states)},
        )
        if debug:
            self.print_ip_results()
            print("predicted states:", states)

    def get_hmm_results(self, results):
        ans = {"results": {}, "model": {}}

        ans["model"]["n_features"] = results.M.n_features

        tmp = results.M.startprob_.tolist()
        ans["model"]["start_probs"] = {
            s: tmp[i] for s, i in self.smap.items() if tmp[i] > 0.0
        }

        tmp = results.M.emissionprob_.tolist()
        emission_probs = {}
        for s in self.smap:
            emission_probs[s] = [
                {"key": o, "prob": tmp[self.smap[s]][self.omap[o]]}
                for o in self.omap
                if tmp[self.smap[s]][self.omap[o]] > 0.0
            ]
        ans["model"]["emission_probs"] = emission_probs

        tmp = results.M.transmat_.tolist()
        trans_mat = {}
        for s in self.smap:
            trans_mat[s] = {
                s_: tmp[self.smap[s]][self.smap[s_]]
                for s_ in self.smap
                if tmp[self.smap[s]][self.smap[s_]] > 0.0
            }
        ans["model"]["trans_mat"] = trans_mat

        ans["results"]["observations"] = results.observations
        ans["results"]["log_likelihood"] = results.log_likelihood
        ans["results"]["states"] = results.states.tolist()
        ans["results"]["hidden"] = results.hidden

        return ans

    def get_lp_results(self, results):
        ans = {"results": {}, "model": {}}

        tmp = results.start_probs.tolist()
        ans["model"]["start_probs"] = {
            s: tmp[i] for s, i in self.smap.items() if tmp[i] > 0.0
        }

        tmp = results.emission_probs
        emission_probs = {}
        for s in self.smap:
            emission_probs[s] = [
                {"key": o, "prob": tmp[self.smap[s]][self.omap[o]]}
                for o in self.omap
                if tmp[self.smap[s]][self.omap[o]] > 0.0
            ]
        ans["model"]["emission_probs"] = emission_probs

        tmp = results.trans_mat.tolist()
        trans_mat = {}
        for s in self.smap:
            trans_mat[s] = {
                s_: tmp[self.smap[s]][self.smap[s_]]
                for s_ in self.smap
                if tmp[self.smap[s]][self.smap[s_]] > 0.0
            }
        ans["model"]["trans_mat"] = trans_mat

        ans["results"]["observations"] = {
            i: v for i, v in enumerate(results.observations)
        }
        ans["results"]["y: activities"] = [
            [t, a, b, pe.value(results.M.y[t, a, b])]
            for t, a, b in results.M.y
            if pe.value(results.M.y[t, a, b]) > 0
        ]
        ans["results"]["log_likelihood"] = results.log_likelihood
        ans["results"]["states"] = results.states
        ans["results"]["hidden"] = results.hidden

        invomap = {v: k for k, v in self.omap.items()}
        invsmap = {v: k for k, v in self.smap.items()}
        ans["results"]["invomap"] = invomap
        ans["results"]["invsmap"] = invsmap

        if getattr(self.results.M, "G", None) is not None:
            ans["results"]["objective_coefficient"] = {
                t
                + 1: {
                    "from": invsmap.get(a, "STARTEND"),
                    "to": invsmap.get(b, "STARTEND"),
                    "coef": pe.value(self.results.M.G[t, a, b]),
                }
                for t, a, b in self.results.M.G
                if pe.value(results.M.y[t, a, b]) > 0
            }

        if self.model_data is not None:
            if results.hidden[0] not in self.model_data.start_probs:
                ans["results"]["unexpected start"] = results.hidden[0]

            ans["results"]["unexpected transition"] = {}
            for t in range(len(results.hidden) - 1):
                if (
                    results.hidden[t],
                    results.hidden[t + 1],
                ) not in self.model_data.trans_mat:
                    ans["results"]["unexpected transition"][t] = (
                        results.hidden[t],
                        results.hidden[t + 1],
                    )

            ans["results"]["unexpected emission"] = {}
            for t in range(len(results.hidden)):
                if (
                    results.observations[t]
                    not in self.model_data.emission_probs[results.hidden[t]]
                ):
                    ans["results"]["unexpected emission"][t] = results.observations[t]

        return ans

    def get_ip_results(self, results):
        return {}

    def print_hmm_results(self):
        pprint.pprint(self.get_hmm_results(self.results))

    def print_lp_results(self):
        pprint.pprint(self.get_lp_results(self.results))

    def print_ip_results(self):
        pprint.pprint(self.get_ip_results(self.results))

    def write_hmm_results(self, filename):
        with open(filename, "w") as OUTPUT:
            json.dump(
                self.get_hmm_results(self.results),
                OUTPUT,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )

    def write_lp_results(self, filename):
        with open(filename, "w") as OUTPUT:
            json.dump(
                self.get_lp_results(self.results),
                OUTPUT,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )

    def write_ip_results(self, filename):
        with open(filename, "w") as OUTPUT:
            json.dump(
                self.get_ip_results(self.results),
                OUTPUT,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )

    def write_lp_model(self, filename, debug=False):
        results = getattr(self, "results", None)
        assert results is not None, "No LP model has been generated!"
        if debug:
            results.M.pprint()
            results.M.display()
        results.M.write(filename)

    def write_ip_model(self, filename, debug=False):
        results = getattr(self, "results", None)
        assert results is not None, "No IP model has been generated!"
        if debug:
            results.M.pprint()
            results.M.display()
        results.M.write(filename)
