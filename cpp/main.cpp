#include <iostream>
#include <vector>

#include "HMM.h"

int main() {
    std::vector< std::vector<double> > A{{0.899,0.101},{0.099,0.901}};
    std::vector<double> S =  {0.499,0.501};
    std::vector< std::vector<double> > E{{0.699,0.301},{0.299,0.701}};

    HMM myHMM(A,S,E);
    std::vector<int> obs;
    std::vector<int> hid;
    int T = 10000;
    myHMM.run(T,obs,hid);

    int numZeros = count(hid.begin(), hid.end(), 0);

    double logProb1, logProb2;

    auto start1=std::chrono::high_resolution_clock::now();
    std::vector<int> hidGuess1 = myHMM.aStar(obs,logProb1);
    auto finish1=std::chrono::high_resolution_clock::now();

    std::cout << "Viterbi(-ish) running time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish1 - start1).count() << " milliseconds \n"; 

    auto start2=std::chrono::high_resolution_clock::now();
    std::vector<int> hidGuess2 = myHMM.aStar(obs,logProb2,numZeros);
    auto finish2=std::chrono::high_resolution_clock::now(); 

    std::cout << "A* running time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish2 - start2).count() << " milliseconds \n \n";

    int diff1 = 0;
    int diff2 = 0;

    for(int t = 0; t < T; ++t) { 
        if(hid[t] != hidGuess1[t]) {
            ++diff1;
        }
        if(hid[t] != hidGuess2[t]) {
            ++diff2;
        }
    }

    std::cout << "Viterbi(-ish) number of errors: " << diff1 << "\n";
    std::cout << "A* number of errors: " << diff2 << "\n";

    return 0;
}