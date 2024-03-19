import sys
import yaml
import random
import pprint
from munch import Munch
import pyomo.environ as pe

from chmmpy import HMMBase
from chmmpy.util import state_similarity, print_differences, run_all


class TourHMM_Default(HMMBase):
    """
    The traveler starts and stops at a well-known "home" city.

    Note that this city is duplicated in the HMM logic, so we can distiguish between starting and ending at home.
    """

    def load_data(self, *, data=None, filename=None):
        if filename is not None:
            with open(filename, "r") as INPUT:
                data = yaml.safe_load(INPUT)
        assert data is not None

        alldata = self.data
        alldata.N = data["N"] + 2
        alldata.hidden_states = list(range(alldata.N))
        alldata.Costs = {(from_, to_): cost_ for from_, to_, cost_ in data["Costs"]}
        alldata.CityNames = data["CityNames"]
        alldata.home_city = None
        for i, name in data["CityNames"].items():
            if name == data["Home"]:
                alldata.home_city = i
                break
        assert (
            alldata.home_city is not None
        ), "The name of the home city must match one of the city names"
        alldata.homeagain_city = data["N"]
        for from_, to_ in list(alldata.Costs.keys()):
            if to_ == alldata.home_city:
                alldata.Costs[from_, alldata.homeagain_city] = alldata.Costs[from_, to_]
        alldata.sim = data["sim"]
        alldata.seed = data["sim"].get("seed", None)
        self.name = data["name"]

    def create_ip(self, *, observations_index, emission_probs, data):
        M = self.create_lp(
            observations_index=observations_index,
            emission_probs=emission_probs,
            data=data,
            y_binary=True,
            cache_indices=True,
        )

        J = list(range(data.N - 1))
        Tmax = data.Tmax
        T = list(range(Tmax))

        #
        # M.z[t] is one if we have seen city j for the first time at or before time t
        #
        M.z = pe.Var(J, [-1] + T, within=pe.Binary)
        for j in J:
            M.z[j, -1].fix(0)
        #
        # M.Z[t] is one if we have seen city j for the last time on or after time t
        #
        M.Z = pe.Var(J, T + [Tmax], within=pe.Binary)
        for j in J:
            M.Z[j, Tmax].fix(0)

        #
        # Summarize node activity in the LP
        #
        def node_(m, j, t):
            return sum(M.y[t, a, b] for tt, a, b in M.E if tt == t and a == j)

        M.node = pe.Expression(J, T, rule=node_)

        #
        # CONSTRAINTS
        #

        # Y-Z interactions

        M.z_stop = pe.ConstraintList()
        for t in T:
            for j in J:
                M.z_stop.add(M.node[j, t] == M.z[j, t] + M.Z[j, t] - 1)

        # z,Z constraints

        def zstep_(m, j, t):
            return m.z[j, t - 1] <= m.z[j, t]

        M.zstep = pe.Constraint(J, T, rule=zstep_)

        def Zstep_(m, j, t):
            return m.Z[j, t] >= m.Z[j, t + 1]

        M.Zstep = pe.Constraint(J, T, rule=Zstep_)

        def zZ_(m, j, t):
            return m.z[j, t - 1] + m.Z[j, t] >= 1

        M.zZ = pe.Constraint(J, T, rule=zZ_)

        def start_(m, j, t):
            if j == self.data.home_city:
                return pe.Constraint.Skip
            return M.z[self.data.home_city, t] >= M.z[j, t]

        M.start = pe.Constraint(J, T, rule=start_)

        def stop_(m, j, t):
            if j == self.data.homeagain_city:
                return pe.Constraint.Skip
            return M.z[self.data.homeagain_city, t] <= M.z[j, t]

        M.stop = pe.Constraint(J, T, rule=stop_)

        return M

    def run_training_simulations(
        self, seed=None, n=None, debug=False, return_observations=False
    ):
        #
        # Get simulation parameters
        #
        sim = self.data["sim"]
        p = sim["p"]  # Probability that a city is recognized when observed
        q = sim["q"]  # Probability that a picture is sent from a city
        Tmax = sim["Tmax"]
        random.seed(seed)
        #
        # Run 'n' simulations
        #
        #   City ids [0, ..., N-3] are the original cities
        #   N-2 is the id of the duplicate for the home city
        #   N-1 is the id of an "unknown" city
        #
        O = []
        all_cities = list(range(self.data.N - 2))
        all_cities[self.data.home_city] = self.data.homeagain_city
        nruns = sim["nruns"] if n is None else n
        for n in range(nruns):
            budget = random.randint(sim["budget_min"], sim["budget_max"])
            costs = 0
            cities = [self.data.N - 1] * Tmax  # Unknown city
            observations = [None] * Tmax
            i = random.randint(0, 4)
            curr = self.data.home_city  # Start at home

            random.shuffle(all_cities)
            while all_cities[-1] != self.data.homeagain_city:
                random.shuffle(
                    all_cities
                )  # Order the tour, which will end when we reach home again, or when we run out of money
            next_city_index = 0

            # print("Y", all_cities)
            while i < Tmax:
                # print("y",i,curr)
                cities[i] = curr
                if random.uniform(0, 1) <= q:
                    randcity = random.randint(0, self.data.N - 3)  # Guessing the city
                    if random.uniform(0, 1) <= p:
                        observations[i] = curr  # City is recognized
                    else:
                        observations[i] = randcity
                else:
                    random.uniform(0, 1)
                    randcity = random.randint(0, self.data.N - 3)  # Guessing the city

                # print("X", i,Tmax, curr, self.data.homeagain_city)
                i += 1
                if i == Tmax:
                    break
                if curr == self.data.homeagain_city:
                    break

                if random.uniform(0, 1) <= 0.6:  # Stay in the current city
                    city = curr
                else:
                    city = all_cities[next_city_index]
                    next_city_index += 1
                next_city = None
                while next_city is None:
                    if city == self.data.home_city:
                        if costs + self.data.Costs[curr, city] <= budget:
                            next_city = city
                            costs += self.data.Costs[curr, city]
                        else:
                            assert next_city_index < len(all_cities)
                            city = all_cities[next_city_index]
                            next_city_index += 1
                    elif city == self.data.homeagain_city:
                        next_city = city
                        costs += self.data.Costs[curr, city]
                    elif (curr, city) in self.data.Costs and costs + self.data.Costs[
                        curr, city
                    ] + self.data.Costs[city, self.data.homeagain_city] <= budget:
                        next_city = city
                        costs += self.data.Costs[curr, city]
                    else:
                        assert next_city_index < len(all_cities)
                        city = all_cities[next_city_index]
                        next_city_index += 1
                    # print("XX", next_city)
                if next_city is None:
                    break
                curr = next_city

            # print("HERE", cities)
            # print("HERE", observations)
            # print("HERE", costs)
            # print("HERE", budget)
            O.append(
                Munch(
                    observations=observations, states=cities, costs=costs, budget=budget
                )
            )

        if return_observations:
            return O
        self.O = O

    def get_ip_results(self, results):
        ans = self.get_lp_results(results)
        J = list(range(self.data.N - 1))
        M = results.M

        ans["results"]["City"] = [
            [t, j, pe.value(M.node[j, t])]
            for t in range(self.data.Tmax)
            for j in J
            if pe.value(M.node[j, t]) > 0
        ]

        values = []
        start = []
        flag = False
        for j in J:
            values.append([j, -1, pe.value(M.z[j, -1])])
            for t in range(self.data.Tmax):
                values.append([j, t, pe.value(M.z[j, t])])
                if pe.value(M.z[j, t]) - pe.value(M.z[j, t - 1]) > 0:
                    start.append([j, t])
                    flag = True
                    break
            if not flag:
                start.append([j, None])
        ans["results"]["z: values"] = values
        ans["results"]["z: start"] = start

        values = []
        stop = []
        flag = False
        for j in J:
            for t in range(self.data.Tmax):
                values.append([j, t, pe.value(M.Z[j, t])])
                if pe.value(M.Z[j, t]) - pe.value(M.Z[j, t + 1]) > 0:
                    stop.append([j, t])
                    flag = True
                    break
            if not flag:
                stop.append([j, None])
        ans["results"]["Z: values"] = values
        ans["results"]["Z: start"] = start

        ans["results"]["INFEASIBLE"] = [
            [t, a, b] for t, a, b in M.FF if pe.value(M.y[t, a, b]) > 0
        ]

        return ans


class TourHMM_City3(TourHMM_Default):
    def create_ip(self, *, observations_index, emission_probs, data):
        M = TourHMM_Default.create_ip(
            self,
            observations_index=observations_index,
            emission_probs=emission_probs,
            data=data,
        )

        # Include city 3
        # M.z[3,self.data.Tmax-1].fix(1)

        J = list(range(self.data.N - 1))
        Tmax = data.Tmax
        T = list(range(Tmax))

        def city_3_first_(m, j, t):
            if j == self.data.home_city or j == 3:
                return pe.Constraint.Skip
            return M.z[3, t] >= M.z[j, t]

        M.city_3_first = pe.Constraint(J, T, rule=city_3_first_)

        return M


#
# MAIN
#
debug = False
seed = 98709812

model = TourHMM_Default()
print("-" * 70)
print("-" * 70)
print("TourHMM - Default")
print("-" * 70)
print("-" * 70)
model.load_data(filename="tour1.yaml")
run_all(model, seed=seed, debug=debug, output="results1")

seed = 909831
model = TourHMM_City3()
print("-" * 70)
print("-" * 70)
print("TourHMM - Include City 3")
print("-" * 70)
print("-" * 70)
model.load_data(filename="tour1.yaml")
run_all(model, seed=seed, debug=debug, output="results2")
