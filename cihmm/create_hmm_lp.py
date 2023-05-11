import math
import pyomo.environ as pe
import numpy as np

#
# Create a linear program that perform inference associated with 
# the Viterbei algorithm.  This LP predicts the most likely hidden 
# states, using a log-likely objective.
#
# TODO - Deal with infeasible solutions.  These have a log-likelihood of -infinity.
#        Right now, we just treat all zero transition probabilities as generating a large
#        negative value.
#
def create_hmm_lp(observations, N, start_probs, emission_probs, trans_mat, y_binary=False, cache_indices=False):
    # N - Number of hidden states
    # start_probs[i] - map from i=1..N to a probability value in 0..1
    # emission_probes[i][k] - probability that output k is generated when in hidden state i
    # trans_mat[i][j] - probability of transitioning from hidden state i to hidden state j
 
    Tmax = len(observations)
    T = list(range(Tmax))

    A = list(range(N))

    F = set()       # (a,b) in transition matrix
    FF = set()      # (t,a,b) such that transition probability == 0
    G = {}          # (t,a,b) such that transition probability > 0
    for t in T:
        if t == 0:
            for i in range(N):
                tmp = start_probs[i]*emission_probs[i][observations[t]]
                if tmp > 0:
                    #print("HERE",(-1,-1,i), tmp, math.log(tmp))
                    G[-1,-1,i] = math.log(tmp)
                else:
                    #print("HERE",(-1,-1,i), tmp, None)
                    FF.add((-1,-1,i))
        else:
            it = np.nditer(trans_mat, flags=['multi_index'])
            for val in it:
                F.add(it.multi_index)
                a,b = it.multi_index
                tmp = val*emission_probs[b][observations[t]]
                #print("----",(t-1,a,b),val,tmp, [observations[t], emission_probs[b][observations[t]]])
                if tmp > 0:
                    #print("HERE",(t-1,a,b), tmp, math.log(tmp))
                    G[t-1,a,b] = math.log(tmp)
                else:
                    #print("HERE",(t-1,a,b), tmp, None, val, emission_probs[b][observations[t]], t, observations[t])
                    FF.add(tuple([t-1]+list(it.multi_index)))

    E = []      # (t,a,b) where
                #           (t-1, -1,  i) when t==0
                #           (t-1,  i, -2) when t==Tmax
                #           (t-1,  a,  b)  when t>0 and t<Tmax
    for t in T:
        if t == 0:
            for i in range(N):
                E.append( (-1, -1, i) )
        else:
            for g in F:
                E.append( tuple([t-1] + list(g)) )
    for i in range(N):
        E.append( (Tmax-1, i, -2) )


    M = pe.ConcreteModel()
    if y_binary:
        M.y = pe.Var(E, bounds=(0,1), within=pe.Boolean)
    else:
        M.y = pe.Var(E, bounds=(0,1))

    # Shortest path constraints

    def flow_(m, t, b):
        if t == 0:
            return m.y[t-1,-1,b] == sum(m.y[t,b,aa] for aa in A  if (b,aa) in F)
        elif t == Tmax-1:
            return sum(m.y[t-1,a,b] for a in A if (a,b) in F) == m.y[t,b,-2]
        else:
            return sum(m.y[t-1,a,b] for a in A if (a,b) in F) == sum(m.y[t,b,aa] for aa in A  if (b,aa) in F)
    M.flow = pe.Constraint(T, A, rule=flow_)

    def flow_start_(m):
        return sum(m.y[-1,-1,b] for b in A) == 1
    M.flow_start = pe.Constraint(rule=flow_start_)

    def flow_end_(m):
        return sum(m.y[Tmax-1,a,-2] for a in A) == 1
    M.flow_end = pe.Constraint(rule=flow_end_)

    M.O = pe.Expression(expr=sum(M.y[t,a,b] for t,a,b in FF))

    M.o = pe.Objective(
            expr=sum(G[t,a,b] * M.y[t,a,b] for t,a,b in G) + -10**6 * M.O, sense=pe.maximize)
                #sum(M.y[t,a,b] for t,a,b in FF),

    if cache_indices:
        # Cache for use in create_ip()
        M.E = E
        M.G = G
        M.F = F
        M.FF = FF

    return M

