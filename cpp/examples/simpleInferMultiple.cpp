//main.cpp

#include "HMM.h"

bool oracleConstraint(std::vector<int> hid, double numZeros) {
    return (numZeros == count(hid.begin(), hid.end(), 0));
}

int main() {
    //bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector< std::vector<double> > A{{0.899,0.101},{0.099,0.901}}; //Transition Matrix
    std::vector<double> S =  {0.501,0.499}; //Start probabilities 
    std::vector< std::vector<double> > E{{0.699,0.301},{0.299,0.701}}; //Emission Matrix
    
    int T = 25; //Time Horizon
    int numSolns = 5; //Find top # of solutions

    HMM myHMM(A,S,E,1234); //1234 is the seed

    //Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;   
    std::vector<int> hid;
    int numZeros;

    myHMM.run(T,obs,hid);
    numZeros = count(hid.begin(), hid.end(), 0);

    std::cout << "Running inference without constraint.\n";
    double logProbNoConstraints;
    std::vector< std::vector<int> > hidGuessNoConstraints = myHMM.aStarMult(obs,logProbNoConstraints, [](std::vector<int> myHid) -> bool { return true; }, numSolns);

    std::cout << "Running inference with constraints.\n";
    double logProbConstraints;
    std::vector< std::vector<int> > hidGuessConstraints = myHMM.aStarMult(obs, logProbConstraints, [numZeros](std::vector<int> myHid) -> bool { return (numZeros == count(myHid.begin(), myHid.end(), 0)); }, numSolns);

    std::cout << "Observed:\n";
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
}