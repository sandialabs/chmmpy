
import sys
import yaml
import random
import pprint
from munch import Munch
import pyomo.environ as pe

from cihmm import HMMBase
from cihmm.util import state_similarity, print_differences, run_all


class TravelHMM_Default(HMMBase):
    """
    The traveler starts and stops at the same city.
    """

    def load_data(self, *, data=None, filename=None):
        if filename is not None:
            with open(filename, 'r') as INPUT:
                data = yaml.safe_load(INPUT)
        assert data is not None

        alldata = self.data
        alldata.N = data['N']+1
        alldata.Costs = {(from_,to_):cost_ for from_,to_,cost_ in data['Costs']}
        alldata.CityNames = data['CityNames']
        alldata.sim = data['sim']
        self.name = data['name']

    def create_ip(self, *, observation_index, emission_probs, data):
        M = self.create_lp(observation_index=observation_index, emission_probs=emission_probs, data=data, y_binary=True, cache_indices=True)

        J=list(range(data.N-1))
        Tmax=data.Tmax
        T = list(range(Tmax))

        #
        # M.start[j] is one iff the tour starts at city j
        #
        M.start = pe.Var(J, within=pe.Binary)
        #
        # M.stop[j] is one iff the tour stops at city j
        #
        M.stop = pe.Var(J, within=pe.Binary)
        #
        # M.z[t] is one if we have seen a city at or before time t
        #
        M.z = pe.Var([-1] + T, within=pe.Binary)
        M.z[-1].fix(0)

        #
        # M.Z[t] is one if we have not seen a city at or after time t
        #
        M.Z = pe.Var(T + [Tmax], within=pe.Binary)
        M.Z[Tmax].fix(0)

        def node_(m, t, j):
            return sum(M.y[t,a,b] for tt,a,b in M.E if tt == t and a == j)
        M.node = pe.Expression(T, list(range(data.N)), rule=node_)

        #
        # CONSTRAINTS
        #

        # Y-Z interactions

        M.yz_start = pe.ConstraintList()
        for t in T:
            for j in J:
                M.yz_start.add( M.node[t,j] <= M.start[j] + 1 - (M.z[t] - M.z[t-1]) )

        M.yz_stop = pe.ConstraintList()
        for t in range(Tmax):
            for j in J:
                M.yz_stop.add( M.node[t,j] <= M.stop[j] + 1 - (M.Z[t] - M.Z[t+1]) )

        M.z_stop = pe.ConstraintList()
        for t in T:
            M.z_stop.add( sum(M.node[t,j] for j in J) == M.z[t] + M.Z[t] - 1)

        # z,Z constraints

        def zstep_(m, t):
            return m.z[t-1] <= m.z[t]
        M.zstep = pe.Constraint(T, rule=zstep_)

        def Zstep_(m, t):
            return m.Z[t] >= m.Z[t+1]
        M.Zstep = pe.Constraint(T, rule=Zstep_)

        def zZ_(m, t):
            return m.z[t] + m.Z[t] >= 1
        M.zZ = pe.Constraint(T, rule=zZ_)

        # start,stop contraints

        M.starting = pe.Constraint(expr=sum(M.start[j] for j in J) == 1)
        M.stopping = pe.Constraint(expr=sum(M.stop[j] for j in J) == 1)

        def same_city_(m, j):
            return m.start[j] == m.stop[j]
        M.same_city = pe.Constraint(J, rule=same_city_)

        return M

    def run_training_simulations(self, seed=None, n=None, debug=False, return_observations=False):
        #
        # Get simulation parameters
        #
        sim = self.data['sim']
        p = sim['p'] # Probability that a city is recognized when observed
        q = sim['q'] # Probability that a picture is sent from a city
        Tmax = sim['Tmax']
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(sim['seed'])
        #
        # Run 'n' simulations
        #
        O = []
        all_cities = list(range(self.data.N-1))
        nruns = sim['nruns'] if n is None else n
        for n in range(nruns):
            #if debug:
            #    sys.stdout.write(".")
            #    sys.stdout.flush()

            budget = random.randint(sim['budget_min'], sim['budget_max'])
            costs = 0
            cities = [self.data.N-1]*Tmax                                 # Unknown city
            observations = [None]*Tmax
            i = random.randint(0,4)
            start = curr = random.randint(0,self.data.N-2)                      # Initial city
            while i<Tmax:
                cities[i] = curr
                if random.uniform(0,1) <= q:
                    if random.uniform(0,1) <= p:
                        observations[i] = curr                              # City is recognized
                    else:
                        observations[i] = random.randint(0,self.data.N-2)   # Guessing the city

                i += 1
                if i == Tmax:
                    break
                if curr != self.data.N:
                    next_city = None
                    random.shuffle(all_cities)
                    for city in all_cities:
                        if city == start:
                            if (curr, city) in self.data.Costs and costs + self.data.Costs[curr,city] <= budget:
                                next_city = city
                                costs += self.data.Costs[curr,city]
                                break
                        else:
                            if (curr, city) in self.data.Costs and costs + self.data.Costs[curr,city] + self.data.Costs[city,start] <= budget:
                                next_city = city
                                costs += self.data.Costs[curr,city]
                                break
                    if next_city is None:
                        assert curr == start, "Expecting to return home!"
                        break
                curr = next_city
            
            print("HERE", cities)
            print("HERE", observations)
            print("HERE", costs)
            print("HERE", budget)
            O.append(Munch(observations=observations, states=cities, costs=costs, budget=budget))

        if return_observations:
            return O
        self.O = O

    def print_lp_results(self, M):
            print("Y: activities")
            for t,a,b in M.y:
                if pe.value(M.y[t,a,b]) > 0:
                    print(t,a,b, pe.value(M.y[t,a,b]))

    def print_ip_results(self, M):
            self.print_lp_results(M)
            print("")

            print("City")
            for t in range(self.data.Tmax):
                for j in range(self.data.N):
                    if pe.value(M.node[t,j]) > 0:
                        print("",t,j, pe.value(M.node[t,j]))
                        break
            print("")

            print("Start city")
            city = None
            for j in M.start:
                print("start",j,pe.value(M.start[j]))
                if pe.value(M.start[j]) > 0:
                    city = j
            print("",city)
            print("")

            print("Stop city")
            city = None
            for j in M.start:
                print("stop",j,pe.value(M.stop[j]))
                if pe.value(M.stop[j]) > 0:
                    city = j
            print("",city)
            print("")

            print("Z: tour start")
            flag=False
            print("z",-1,pe.value(M.z[-1]))
            for t in range(self.data.Tmax):
                print("z",t,pe.value(M.z[t]))
                if pe.value(M.z[t]) - pe.value(M.z[t-1]) > 0:
                    print("",t)
                    flag=True
                    break
            if not flag:
                print("",None)
            print("")

            print("Z: tour stop")
            flag=False
            for t in range(self.data.Tmax):
                print("Z",t,pe.value(M.Z[t]))
                if pe.value(M.Z[t]) - pe.value(M.Z[t+1]) > 0:
                    print("",t)
                    flag=True
                    break
            if not flag:
                print("",None)
            print("")

            for t,a,b in M.FF:
                if pe.value(M.y[t,a,b]) > 0:
                    print("INFEASIBLE", t,a,b)


#
# MAIN
#
model = TravelHMM_Default()
print("-"*70)
print("-"*70)
print("TravelHMM - Default")
print("-"*70)
print("-"*70)
model.load_data(filename='travel1.yaml')
run_all(model, seed=9870983, debug=True)

