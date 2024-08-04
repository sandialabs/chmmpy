import sys
import yaml
import random
from munch import Munch
import pyomo.environ as pe

from chmmpy import HMMBase
from chmmpy.util import run_all


class IrelandTrip_Default(HMMBase):
    """
    The traveler visits one or more tourist destinations in Ireland, where the
    average number of hours spent driving each day is bounded.
    """

    def load_data(self, *, data=None, filename=None):
        if filename is not None:
            with open(filename, "r") as INPUT:
                data = yaml.safe_load(INPUT)
        assert data is not None

        alldata = self.data
        # Add an 'unknown' state
        alldata.N = data["N"] + 1
        alldata.hidden_states = list(range(alldata.N))
        alldata.travel_time = {(from_, to_): hours_ for from_, to_, hours_ in data["TravelTime"]}
        alldata.place_names = {data["PlaceNames"][from_]['name'] for from_ in data["PlaceNames"]}
        alldata.place_abbrv = {data["PlaceNames"][from_]['abbrv'] for from_ in data["PlaceNames"]}
        alldata.max_avg_travel_time = data['max_hours']
        alldata.min_days = data['min_days']
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
        # M.start[j] is one iff the trip starts at city j
        #
        M.start = pe.Var(J, within=pe.Binary)
        #
        # M.stop[j] is one iff the trip stops at city j
        #
        M.stop = pe.Var(J, within=pe.Binary)
        #
        # M.z[t] is one if we have seen a city at or before time t
        #
        M.z = pe.Var([-1] + T, within=pe.Binary)
        #
        # We have not seen a city before the time horizon
        #
        M.z[-1].fix(0)

        #
        # M.Z[t] is one if we have not seen a city at or after time t
        #
        M.Z = pe.Var(T + [Tmax], within=pe.Binary)
        #
        # We have seen a city before the end of the time horizon
        #
        M.Z[Tmax].fix(0)

        #
        # M.node[t,a] is an expression whose value is the log-probability that hidden
        #           state is 'a' at time 't'
        def node_(m, t, j):
            return sum(M.y[t, a, b] for tt, a, b in M.E if tt == t and a == j)
        M.node = pe.Expression(T, list(range(data.N)), rule=node_)

        #
        # CONSTRAINTS
        #

        # Y-Z interactions

        M.yz_start = pe.ConstraintList()
        for t in T:
            for j in J:
                M.yz_start.add(M.node[t, j] <= M.start[j] + 1 - (M.z[t] - M.z[t - 1]))

        M.yz_stop = pe.ConstraintList()
        for t in range(Tmax):
            for j in J:
                M.yz_stop.add(M.node[t, j] <= M.stop[j] + 1 - (M.Z[t] - M.Z[t + 1]))

        M.z_stop = pe.ConstraintList()
        for t in T:
            M.z_stop.add(sum(M.node[t, j] for j in J) == M.z[t] + M.Z[t] - 1)

        # z,Z constraints

        def zstep_(m, t):
            return m.z[t - 1] <= m.z[t]

        M.zstep = pe.Constraint(T, rule=zstep_)

        def Zstep_(m, t):
            return m.Z[t] >= m.Z[t + 1]

        M.Zstep = pe.Constraint(T, rule=Zstep_)

        def zZ_(m, t):
            return m.z[t] + m.Z[t] >= 1

        M.zZ = pe.Constraint(T, rule=zZ_)

        # start,stop contraints

        #
        # One of 
        M.starting = pe.Constraint(expr=sum(M.start[j] for j in J) == 1)
        M.stopping = pe.Constraint(expr=sum(M.stop[j] for j in J) == 1)

        def same_city_(m, j):
            return m.start[j] == m.stop[j]
        M.same_city = pe.Constraint(J, rule=same_city_)

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
        O = []
        all_cities = list(range(self.data.N - 1))
        nruns = sim["nruns"] if n is None else n
        for n in range(nruns):
            if debug:
                sys.stdout.write(".")
                sys.stdout.flush()

            feasible=False
            while not feasible:
                #if debug:
                #    sys.stdout.write("X")
                #    sys.stdout.flush()
                # Default simulation is to visit the "unknown" city every day
                cities = [self.data.N - 1] * Tmax
                observations = [None] * Tmax

                total_hours = 0.0
                ctr=0
                start = curr = 1 # Start/Stop in Dublin
                i = random.randint(0, int(Tmax/4.0))
                done = False
                while i < Tmax:
                    ctr += 1
                    cities[i] = curr
                    if random.uniform(0, 1) <= q:
                        if random.uniform(0, 1) <= p:
                            # City is recognized
                            observations[i] = curr
                        else:
                            # Guessing the city
                            observations[i] = random.randint(0, self.data.N - 2)

                    i += 1
                    if i == Tmax or done:
                        break

                    # Each day, there is a 1/10 chance of continuing the trip
                    if (random.uniform(0, 1) <= 0.1) or (i == Tmax-1):
                        next_city = start
                        done = True
                    else:
                        next_city = random.randint(0, self.data.N - 2)
                    if curr != next_city:
                        total_hours += self.data.travel_time[curr, next_city] if curr < next_city else  self.data.travel_time[next_city, curr]
                        curr = next_city

                #print(total_hours/ctr <= self.data.max_avg_travel_time, ctr >= self.data.min_days, i, ctr)
                feasible = (total_hours/ctr <= self.data.max_avg_travel_time) and (ctr >= self.data.min_days)

            if debug:
                print(cities)
                print(observations)
                print(total_hours)
                print(ctr)
                print(total_hours/ctr)
                print(self.data.max_avg_travel_time)
            O.append(
                Munch(
                    observations=observations, states=cities, total_hours=total_hours, avg_hours=total_hours/ctr, max_avg_hours=self.data.max_avg_travel_time
                )
            )

        if return_observations:
            return O
        self.O = O

    def get_ip_results(self, results):
        ans = self.get_lp_results(results)
        J = list(range(self.data.N - 1))
        M = results.M

        city = []
        for t in range(self.data.Tmax):
            for j in range(self.data.N):
                if pe.value(M.node[t, j]) > 0:
                    city.append([t, j, pe.value(M.node[t, j])])
                    break
        ans["results"]["city"] = city

        city = None
        for j in M.start:
            # print("start",j,pe.value(M.start[j]))
            if pe.value(M.start[j]) > 0:
                city = j
        ans["results"]["start city"] = city

        city = None
        for j in M.start:
            # print("stop",j,pe.value(M.stop[j]))
            if pe.value(M.stop[j]) > 0:
                city = j
        ans["results"]["stop city"] = city

        values = []
        flag = False
        values.append([-1, pe.value(M.z[-1])])
        for t in range(self.data.Tmax):
            values.append([t, pe.value(M.z[t])])
            if pe.value(M.z[t]) - pe.value(M.z[t - 1]) > 0:
                start = t
                flag = True
                break
        if not flag:
            start = None
        ans["results"]["z: values"] = values
        ans["results"]["z: start"] = start

        values = []
        flag = False
        for t in range(self.data.Tmax):
            values.append([t, pe.value(M.Z[t])])
            if pe.value(M.Z[t]) - pe.value(M.Z[t + 1]) > 0:
                stop = t
                flag = True
                break
        if not flag:
            stop = None
        ans["results"]["Z: values"] = values
        ans["results"]["Z: stop"] = stop

        ans["results"]["INFEASIBLE"] = [
            [t, a, b] for t, a, b in M.FF if pe.value(M.y[t, a, b]) > 0
        ]

        return ans


class IrelandTrip_VisitCity3(IrelandTrip_Default):
    def create_ip(self, *, observations_index, emission_probs, data):
        M = IrelandTrip_Default.create_ip(
            self,
            observations_index=observations_index,
            emission_probs=emission_probs,
            data=data,
        )

        Tmax = self.data["sim"]["Tmax"]
        T = list(range(Tmax))
        M.visit_3 = pe.Constraint(expr=sum(M.node[t,3] for t in T) >= 1)

        return M


#
# MAIN
#
debug = False
seed = 987222345614

model = IrelandTrip_Default()
print("-" * 70)
print("-" * 70)
print("IrelandTrip - Default")
print("-" * 70)
print("-" * 70)
model.load_data(filename="ireland.yaml")
run_all(model, seed=seed, debug=debug, output="results1", start_tolerance=1e-7)

model = IrelandTrip_VisitCity3()
print("-" * 70)
print("-" * 70)
print("IrelandTrip - Visit city 3")
print("-" * 70)
print("-" * 70)
model.load_data(filename="ireland.yaml")
run_all(model, seed=seed, debug=debug, output="results2", start_tolerance=1e-7)
