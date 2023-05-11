#
# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
#
# A simple process detection application using an HMM
#
# We assume the execution of a process with a jobs with fixed lengths.  Jobs may have
# precedence constraints that limit their execution until after a preceding job has completed.
# Additionally, there may be random delays at the beginning of each job.
#
# A simulation is used to generate data from possible executions of the process, and an HMM is
# trained using the simulation data.  Each hidden state in the HMM is associated with a possible 
# state of the process, which corresponds to the set of simultaneous executing jobs.
# 

import yaml
import random
import pprint
from munch import Munch
import pyomo.environ as pe

from cihmm import HMMBase, state_similarity, print_differences


class Model(HMMBase):

    #
    # Load process description
    #
    def load_process(self, process_yaml):
        alldata = self.data
        with open(process_yaml, 'r') as INPUT:
            data = yaml.safe_load(INPUT)
        alldata.J = list(range(data['J']))
        alldata.L = data['L']
        alldata.E = data['E']
        alldata.A = data['A']
        alldata.N = len(alldata.A)
        alldata.sim = data['sim']
        if process_yaml.endswith(".yaml"):
            name = process_yaml[:-4]
        else:
            name = process_yaml[:-3]
        self.name = process_yaml.split('.')[0]

    def create_ip(self, *, observation_index, emission_probs, data):
        M = self.create_lp(observation_index=observation_index, emission_probs=emission_probs, data=data, y_binary=True, cache_indices=True)

        #
        # J: list of job ids
        # L: L[j] is the length of job J
        # E: if (i,j) in E, then job i needs to completed before job j can start
        # A: a mapping from process state to active jobs
        # Tmax: number of timesteps
        #
        J=data.J
        L=data.L
        E=data.E
        Tmax=data.Tmax
        A=data.A

        T = list(range(Tmax))

        #
        # M.z[j,t] is one if job j begins at or before time t
        #   Time step -1 is used to indicate when a job starts before the time window T
        #
        M.z = pe.Var(J, [-1] + T, within=pe.Binary)

        # Y-Z interactions

        M.yz = pe.ConstraintList()
        for t in range(Tmax-1):
            for j in J:
                tau = max(t+1 - L[j], -1)
                # TODO: improve the efficiency of this constraint generation using sparse index sets
                M.yz.add(
                    sum( M.y[t,a,b] for tt,a,b in M.E if t == tt and j in A[b] )  == M.z[j,t+1] - M.z[j,tau]
                    )

        # Z constraints

        def zstep_(m, j, t):
            return m.z[j, t] - m.z[j, t - 1] >= 0
        M.zstep = pe.Constraint(J, T, rule=zstep_)

        def precedence_lb_(m, i, j, t):
            tau = max(t- L[i], -1)
            return m.z[i, tau] - m.z[j, t] >= 0
        M.precedence_lb = pe.Constraint(E, T, rule=precedence_lb_)

        def activity_feasibility_(m, j, t):
            if t + L[j] - 1 >= Tmax:
                return m.z[j, t] == m.z[j, Tmax - 1]
            return pe.Constraint.Skip
        M.activity_feasibility = pe.Constraint(J, T, rule=activity_feasibility_)

        #
        # Force the first job to be in the window
        #
        M.z[0,-1].fix(0)

        return M

    def run_training_simulations(self, seed=None, n=None, debug=False, return_observations=False):
        #
        # prec[j] -> list of predecessors of job j
        #
        prec = {i:[] for i in self.data.J}
        for i,j in self.data.E:
            prec[j].append(i)
        #
        # state[ tuple ] -> state_id
        #
        state = {tuple(val):i for i,val in self.data.A.items()}
        #
        # Generate nruns simulations
        #
        sim = self.data['sim']
        p = sim['p']
        q = sim['q']
        Tmax = sim['Tmax']
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(sim['seed'])

        if debug:
            print("PREC")
            pprint.pprint(prec, compact=True)

        O = []
        nruns = sim['nruns'] if n is None else n
        for n in range(nruns):
            curr = {}
            start_ = {}
            end = {}
            jobs = {t:[] for t in range(Tmax)}
            #
            # Set background noise
            #
            for j in self.data.J:
                curr[j] = random.choices([0,1], [1-q, q], k=Tmax)
            #
            # Randomly select when next job is executed.  Note that this
            # assumes a topological ordering of jobs.
            #
            for j in self.data.J:
                start=0
                for i in prec[j]:
                    start = max(start, end.get(i,0))
                if sim['delays'][j] > 0:
                    start += random.randint(0,sim['delays'][j])
                start_[j] = start
                end[j] = start+self.data.L[j]
                for t in range(start,end[j]):
                    curr[j][t] = 1 if random.uniform(0,1) <= p else 0
                    jobs[t].append(j)
            
            O.append(Munch(observations=curr, states=[state[tuple(jobs[t])] for t in range(Tmax)]))

            if debug:
                print("")
                print("START")
                pprint.pprint(start_, compact=True)
                print("END")
                pprint.pprint(end, compact=True)
    
        if return_observations:
            return O
        self.O = O

    def print_lp_results(self, M):
            print("Y: activities")
            for t,a,b in M.y:
                if pe.value(M.y[t,a,b]) > 0 and b >= 0 and len(self.data.A[b]) > 0:
                    print(t,a,b, self.data.A[b] if b >= 0 else None)

    def print_ip_results(self, M):
            self.print_lp_results(M)
            print("")

            print("Z: job start times")
            tmp = {}
            for i,t in M.z:
                tau = max(t-self.data.L[i],-1)
                if pe.value(M.z[i,t]) - pe.value(M.z[i,tau]) > 0:
                    if t not in tmp:
                        tmp[t] = set()
                    tmp[t].add(i)
            for t in tmp:
                print(t, tmp[t])
            print("")

            for t,a,b in M.FF:
                if pe.value(M.y[t,a,b]) > 0:
                    print("INFEASIBLE", t,a,b)



#
# MAIN
#
model = Model()
model.load_process('process1.yaml')
debug=True
seed=3487098
#seed=1238709723

model.run_training_simulations(n=100, debug=debug)
model.train_HMM(debug=debug)

obs, ground_truth = model.generate_observations_and_states(seed=seed, debug=debug)
print("Observations:")
for i,o in enumerate(obs):
    print(i,model.omap.get(o,None),o)
print("")
print("Ground Truth:", ground_truth)
print("")

ll0, states0 = model.inference_hmmlearn(observations=obs, debug=debug)
print("states", states0)
print("logprob", ll0)
print("")
print("Similarity:", state_similarity(states0, ground_truth))
print_differences(states0, ground_truth)
print("")

ll1, states1 = model.inference_lp(observations=obs, debug=debug)
print("states", states1)
print("logprob", ll1)
print("")
print("Similarity:", state_similarity(states1, ground_truth))
print_differences(states1, ground_truth)
print("")

ll2, states2 = model.inference_ip(observations=obs, debug=debug)
print("states", states2)
print("logprob", ll2)
print("Similarity:", state_similarity(states2, ground_truth))
print_differences(states2, ground_truth)
print("")

