//main.cpp

#include "HMM.h"


int main() {

    std::vector< std::vector<double> > A{{0.899,0.101},{0.099,0.901}}; //Transition Matrix
    std::vector<double> S =  {0.501,0.499}; //Start probabilities 
    std::vector< std::vector<double> > E{{0.699,0.301},{0.299,0.701}}; //Emission Matrix
    
    int T = 100; //Time Horizon
    int numIt = 1; //Number of runs

    //Initial Guesses
    std::vector< std::vector<double> > AInitial{{0.71,0.29},{0.3,0.7}};
    std::vector<double> SInitial{0.51,0.49};
    std::vector< std::vector<double> > EInitial{{0.81,0.19},{0.2,0.8}};

    HMM toLearn1(AInitial,SInitial,EInitial,0); //0 is the seed of the RNG, can remove and it seeds by time
    HMM toLearn2(AInitial,SInitial,EInitial,0);
    HMM toLearn3(AInitial,SInitial,EInitial,0);
    HMM trueHMM(A,S,E,0);

    //Store the observed and hidden variables as well as the number of zeros
    std::vector< std::vector<int> > obs;   
    std::vector< std::vector<int> > hid;
    std::vector<int> numZeros;
    std::vector< std::function<bool(std::vector<int>)> > constraintOracleVec;

    for(int i = 0; i < numIt; ++i) {
        //std::cout << "Iteration number: " << i << "\n";
        obs.push_back({});
        hid.push_back({});

        trueHMM.run(T,obs[i],hid[i]);
        double numZerosTemp = count(hid[i].begin(), hid[i].end(), 0);
        numZeros.push_back(numZerosTemp);
        constraintOracleVec.push_back([numZerosTemp](std::vector<int> myHid) -> bool { return (numZerosTemp == count(myHid.begin(), myHid.end(), 0));});
    }

    std::cout << "Learning without constraints.\n";
    toLearn1.learn(obs);
    std::cout << "\nLearning with constraints.\n";
    toLearn1.print();
    toLearn2.learnMC(obs,constraintOracleVec,1E-6, 1E6);
    //toLearn3.learn(obs, numZeros);

    std::cout << "Learned parameters without constraints:\n\n";
    toLearn1.print();
    std::cout << "Learned parameters with constraints (MC).\n\n";
    toLearn2.print();
    std::cout << "Learned parameters with constraints.\n\n";
    toLearn3.print();
    std::cout << "True parameter values:\n\n";
    trueHMM.print();

    /*
    for(int r = 0; r < hid.size(); ++r) {
        for(int t = 0; t < T; ++t) {
            std::cout << hid[r][t];
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";*/

    return 0;
}