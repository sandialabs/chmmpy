//main.cpp

#include "HMM.h"

#include<unordered_set>

//Could also make this a lambda fn, but this is easier
//Checks if the hidden constraints form a tour
bool oracleConstraint(std::vector<int> hid) {
    std::unordered_set<int> visitedStates;
    visitedStates.insert(hid[0]);

    for(int t = 1; t < hid.size(); ++t) {
        if(hid[t] != hid[t-1]) {
            if(visitedStates.count(hid[t]) != 0) {
                return false;
            }
            else {
                visitedStates.insert(hid[t]);
            }
        }
    }

    if(visitedStates.size() != 5) {
        return false;
    }

    return true;
}

int main() {
    bool oracleConstraint(std::vector<int> hid);

    std::vector< std::vector<double> > A{
        {0.80, 0.05, 0.05, 0.05, 0.05},
        {0.05, 0.80, 0.05, 0.05, 0.05},
        {0.05, 0.05, 0.80, 0.05, 0.05},
        {0.05, 0.05, 0.05, 0.80, 0.05},
        {0.05, 0.05, 0.05, 0.05, 0.80}}; //Transition Matrix
    std::vector<double> S =  {0.2, 0.2, 0.2, 0.2, 0.2}; //Start probabilities 
    std::vector< std::vector<double> > E{
        {0.40, 0.05, 0.05, 0.05, 0.05, 0.40},
        {0.05, 0.40, 0.05, 0.05, 0.05, 0.40},
        {0.05, 0.05, 0.40, 0.05, 0.05, 0.40},
        {0.05, 0.05, 0.05, 0.40, 0.05, 0.40},
        {0.05, 0.05, 0.05, 0.05, 0.40, 0.40}}; //Emission Matrix, 6th state is ``unknown''
    
    int T = 20; //Time Horizon
    int counter = 0;
    HMM myHMM(A,S,E,0);

    //Store the observed and hidden variables 
    std::vector<int> obs;   
    std::vector<int> hid;

    //Find a run which satisfies the constraint
    while(true) {
        ++counter;
        //std::cout << counter << "\n";
        myHMM.run(T,obs,hid);

        if(oracleConstraint(hid)) {
            break;
        }

        hid.clear();
        obs.clear();
    }

    std::cout << "Running inference without constraint.\n";
    double logProbNoConstraints;
    std::vector<int> hidGuessNoConstraints = myHMM.aStar(obs,logProbNoConstraints);

    std::cout << "Running inference with constraints.\n";
    double logProbConstraints;
    std::vector<int> hidGuessConstraints = myHMM.aStarOracle(obs, logProbConstraints, oracleConstraint);

    int numDiffNoConstraints = 0;
    int numDiffConstraints = 0;
    for(int t = 0; t < T; ++t) {
        if(hidGuessNoConstraints[t] != hid[t]) {
            ++numDiffNoConstraints;
        }
        if(hidGuessConstraints[t] != hid[t]) {
            ++numDiffConstraints;
        }
    }

    std::cout << "\nLog prob without constraints: " << -logProbNoConstraints << "\n";
    std::cout << "Log prob with constraints: " << logProbConstraints << "\n\n";
    std::cout << "Number of mistakes in inference with no constraints: " << numDiffNoConstraints << "\n";
    std::cout << "Number of mistakes in inference with constraints: " << numDiffConstraints << "\n\n";

    std::cout << "True hidden:\n";
    for(int t = 0; t < T; ++t) {
        std::cout << hid[t];
    }

    std::cout << "\n\nHidden guess with no constraints.\n";
    for(int t = 0; t < T; ++t) {
        std::cout << hidGuessNoConstraints[t];
    }

    std::cout << "\n\nHidden guess with constraints.\n";
    for(int t = 0; t < T; ++t) {
        std::cout << hidGuessConstraints[t];
    }
    std::cout << "\n\n";

    return 0;
}