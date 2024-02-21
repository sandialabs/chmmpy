#include "HMM.h"
#include <vector>

bool oracleConstraint(std::vector<int> hid, double numZeros) {
    return (numZeros == count(hid.begin(), hid.end(), 0));
}

int main() {
   //bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector< std::vector<double> > A{{0.899,0.101},{0.099,0.901}};
    ///std::vector< std::vector<double> > A1{{0.7,0.3},{0.1,0.9}};
    std::vector<double> S =  {0.501,0.499};
    std::vector< std::vector<double> > E{{0.699,0.301},{0.299,0.701}};
    
    int T = 10;
    int numIt = 1;

    //Learning
    std::vector< std::vector<double> > AInitial{{0.61,0.39},{0.4,0.6}};
    std::vector<double> SInitial{0.51,0.49};
    std::vector< std::vector<double> > EInitial{{0.91,0.09},{0.1,0.9}};

    HMM toLearn1(AInitial,SInitial,EInitial);
    HMM toLearn2(AInitial,SInitial,EInitial,4);
    HMM trueHMM(A,S,E,7);

    std::vector< std::vector<int> > obs;
    std::vector< std::vector<int> > hid;
    std::vector<int> numZeros;

    for(int i = 0; i < numIt; ++i) {
        //std::cout << "Iteration number: " << i << "\n";
        obs.push_back({});
        hid.push_back({});

        trueHMM.run(T,obs[i],hid[i]);
        numZeros.push_back(count(hid[i].begin(), hid[i].end(), 0));
        double numZerosTemp = numZeros[i];

    }

    int numZerosTemp = numZeros[0];

    toLearn1.learn(obs[0], numZerosTemp);
    std::cout << "RUN 2" << "\n";
    toLearn2.learnMC(obs[0], [numZerosTemp](std::vector<int> myHid) -> bool { return (numZerosTemp == count(myHid.begin(), myHid.end(), 0));});

    std::cout << "\n" << toLearn1.getA()[0][0] << " " << toLearn1.getA()[0][1] << "\n" << toLearn1.getA()[1][0] << " "  << toLearn1.getA()[1][1] << "\n\n";
    std::cout << toLearn1.getS()[0] << " " << toLearn1.getS()[1] << "\n\n";
    std::cout << toLearn1.getE()[0][0] << " " << toLearn1.getE()[0][1] << "\n" << toLearn1.getE()[1][0] << " "  << toLearn1.getE()[1][1] << "\n\n\n";

    std::cout << toLearn2.getA()[0][0] << " " << toLearn2.getA()[0][1] << "\n" << toLearn2.getA()[1][0] << " "  << toLearn2.getA()[1][1] << "\n\n";
    std::cout << toLearn2.getS()[0] << " " << toLearn2.getS()[1] << "\n\n";
    std::cout << toLearn2.getE()[0][0] << " " << toLearn2.getE()[0][1] << "\n" << toLearn2.getE()[1][0] << " "  << toLearn2.getE()[1][1] << "\n\n\n";



        /*trueHMM.aStar(obs[i],logProb1,numZeros[i]);
        trueHMM.aStarOracle(obs[i],logProb2, [numZerosTemp](std::vector<int> myHid) -> bool { return (numZerosTemp == count(myHid.begin(), myHid.end(), 0));  });
        std::cout << logProb1 << " " << logProb2 << "\n";*/

    /*HMM myHMM(A,S,E);
    std::vector<int> obs;
    std::vector<int> hid;
    myHMM.run(T,obs,hid);

    int numZeros = count(hid.begin(), hid.end(), 0);
    double logProb;
    int numSolns = 10;
    std::vector< std::vector< int > > output = myHMM.aStarMult(obs, logProb, numZeros, numSolns);
    
    int ind1;
    int ind2;

    int maxDiff = 0;
    for(int i = 0; i < numSolns; ++i) {
        for(int j = i+1; j < numSolns; ++j) {
            int diff = 0;
            for(int t = 0; t < T; ++t) {
                if(output[i][t] != output[j][t]) {
                    ++diff;
                }
            }
            if(maxDiff < diff) {
                maxDiff = diff;
                ind1 = i;
                ind2 = j;
            }
        }
    }

    std::cout << "The maximum difference is " << (double(100.*maxDiff))/(double(T)) << "%\n\n";

    for(int i = 0; i < 100; ++i) {
        std::cout << obs[i] << ",";
    }
    std::cout << "\n";
    for(int i = 0; i < 100; ++i) {
        std::cout << output[ind1][i] << ",";
    }
    std::cout << "\n";
    for(int i = 0; i < 100; ++i) {
        std::cout << output[ind2][i] << ",";
    }
    std::cout << "\n\n";
    */

    /*for(int i = 0; i < numIt; ++i) {
        std::cout << "Iteration: " << i << "\n";
        HMM myHMM(A,S,E);
        std::vector<int> obs;
        std::vector<int> hid;
        myHMM.run(T,obs,hid,i+1000);

        int numZeros = count(hid.begin(), hid.end(), 0);

        double logProb1, logProb2;

        auto start2=std::chrono::high_resolution_clock::now();
        std::vector<int> hidGuess2 = myHMM.aStarII(obs,logProb2,numZeros);
        auto finish2=std::chrono::high_resolution_clock::now(); 
        int time = std::chrono::duration_cast<std::chrono::milliseconds>(finish2 - start2).count();

        timeSum += time; 
        timeSumSq += time*time;
        maxTime  = std::max(maxTime,time); 
        std::cout << "A* running time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish2 - start2).count() << " milliseconds \n";
        std::cout << "Number of zeros: " << numZeros << "\n\n";
    }

    std::cout << "Mean run time in milliseconds: " << timeSum/numIt << "\n";
    //std::cout << "Standard deviation: " << std::sqrt(timeSumSq/numIt - timeSum*timeSum/(numIt*numIt)) << "\n";
    std::cout << "Max time: " << maxTime << "\n\n";*/

    return 0;
}