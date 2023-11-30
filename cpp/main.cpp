#include <iostream>
#include <vector>

#include "HMM.h"

int main() {
    std::vector< std::vector<double> > A{{0.9,0.1},{0.1,0.9}};
    std::vector<double> S =  {0.5,0.5};
    std::vector< std::vector<double> > E{{0.7,0.3},{0.3,0.7}};

    HMM myHMM(A,S,E);
    std::vector<int> obs;
    std::vector<int> hid;
    myHMM.run(15,obs,hid);

    for(auto h: hid) {
        std::cout << h << " ";
    }
    std::cout << std::endl;
    for(auto o: obs) {
        std::cout << o << " ";
    }
    std::cout << "\n";

    std::cout << std::endl;

    return 0;
}