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


def run_all(
    model,
    debug=False,
    training=True,
    seed=None,
    n=None,
    output=None,
    hmmlearn=True,
    lp=True,
    ip=True,
    solver="glpk",
):
    if training:
        if seed is None:
            seed = model.data.seed
        print("Running with seed:", seed)

        model.run_training_simulations(n=n, debug=debug, seed=seed)
        model.train_HMM(debug=debug)

        obs, ground_truth = model.generate_observations_and_states(
            seed=seed, debug=debug
        )
        print("Observations:")
        for i, o in enumerate(obs):
            print(i, model.omap.get(o, None), o)
        print("")
        print("Ground Truth:", ground_truth)
        print("")
    else:
        obs = model.O[0]["observations"]
        ground_truth = None

    if hmmlearn:
        print("\n\n Viterbei\n")
        model.inference_hmmlearn(observations=obs, debug=debug)
        print("predicted states", model.results.states)
        print("logprob", model.results.log_likelihood)
        print("")
        if ground_truth:
            print("Similarity:", state_similarity(model.results.states, ground_truth))
            print_differences(model.results.states, ground_truth)
        if output is not None:
            model.write_hmm_results(output + "_hmm.json")
        print("")

    if lp:
        print("\n\n LP\n")
        model.inference_lp(observations=obs, debug=debug, solver=solver)
        print("predicted states", model.results.states)
        print("logprob", model.results.log_likelihood)
        print("")
        if ground_truth:
            print("Similarity:", state_similarity(model.results.states, ground_truth))
            print_differences(model.results.states, ground_truth)
        if output is not None:
            model.write_lp_results(output + "_lp.json")
        print("")

    if ip:
        print("\n\n IP\n")
        model.inference_ip(observations=obs, debug=debug, solver=solver)
        print("predicted states", model.results.states)
        print("logprob", model.results.log_likelihood)
        if ground_truth:
            print("Similarity:", state_similarity(model.results.states, ground_truth))
            print_differences(model.results.states, ground_truth)
        if output is not None:
            model.write_ip_results(output + "_ip.json")
        print("")
