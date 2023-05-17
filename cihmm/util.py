def state_similarity(s1, s2):
    assert len(s1) == len(s2), (
        "ERROR: Cannot compare similarities amongst sequences of hidden states of different lengths: %d vs %d"
        % (len(s1), len(s2))
    )
    count = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            count += 1
    return count / len(s1)


def print_differences(s1, s2):
    print("Differences:")
    flag = True
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            flag = False
            print("", i, s1[i], s2[i])
    if flag:
        print("", "None")


def run_all(model, debug=False, seed=None, n=None):
    model.run_training_simulations(n=n, debug=debug)
    model.train_HMM(debug=debug)

    obs, ground_truth = model.generate_observations_and_states(seed=seed, debug=debug)
    print("Observations:")
    for i, o in enumerate(obs):
        print(i, model.omap.get(o, None), o)
    print("")
    print("Ground Truth:", ground_truth)
    print("")

    print("\n\n Viterbei\n")
    ll0, states0 = model.inference_hmmlearn(observations=obs, debug=debug)
    print("predicted states", states0)
    print("logprob", ll0)
    print("")
    print("Similarity:", state_similarity(states0, ground_truth))
    print_differences(states0, ground_truth)
    print("")

    print("\n\n LP\n")
    ll1, states1 = model.inference_lp(observations=obs, debug=debug)
    print("predicted states", states1)
    print("logprob", ll1)
    print("")
    print("Similarity:", state_similarity(states1, ground_truth))
    print_differences(states1, ground_truth)
    print("")

    print("\n\n IP\n")
    ll2, states2 = model.inference_ip(observations=obs, debug=debug)
    print("predicted states", states2)
    print("logprob", ll2)
    print("Similarity:", state_similarity(states2, ground_truth))
    print_differences(states2, ground_truth)
    print("")
