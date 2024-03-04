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
    int numSolns = 5; //Find top # of solns
    int counter = 0;
    HMM myHMM(A,S,E,1233);//1233 is the seed 

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
    std::vector< std::vector<int> > hidGuessNoConstraints = myHMM.aStarMult(obs,logProbNoConstraints, [](std::vector<int> myHid) -> bool { return true; }, numSolns);

    std::cout << "Running inference with constraints.\n";
    double logProbConstraints;
    std::vector< std::vector<int> > hidGuessConstraints = myHMM.aStarMult(obs, logProbConstraints, oracleConstraint, numSolns);

    std::cout << "\nObserved:\n";
    for(int t = 0; t < T; ++t) {
        std::cout << obs[t];
    }

    std::cout << "\n\nTrue solution:\n";
    for(int t = 0; t < T; ++t) {
        std::cout << hid[t];
    }
    std::cout << "\n\nTop " << numSolns << " solutions with no constraints.\n";
    for(int r = 0; r < numSolns; ++r) {
        for(int t = 0; t < T; ++t) {
            std::cout << hidGuessNoConstraints[r][t];
        }
        std::cout << "\n";
    }

    std::cout << "\n\nTop " << numSolns << " solutions with constraints.\n";
    for(int r = 0; r < numSolns; ++r) {
        for(int t = 0; t < T; ++t) {
            std::cout << hidGuessConstraints[r][t];
        }
        std::cout << "\n";
    }

    return 0;

    return 0;
}