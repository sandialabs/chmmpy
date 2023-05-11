
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
