//HMM.h

#ifndef HMM_H
#define HMM_H

#include <vector>
#include <iostream>
#include <random>
#include <queue>
#include <map>
#include <limits>
#include <cstdint>
#include <boost/functional/hash.hpp> //Hash for pairs
#include <time.h> //Used for seed
#include <utility> //Pairs

/*
TODO
----

- Simplify Baum-Welch Style Algorithms with repeated functions
- Fix Monte Carlo method to not break when we have 0's in the transition matrix

*/

//Hash for vectors
//Taken for Stack Overflow (https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/50978188#50978188)
template<typename T>
T xorshift(const T& n,int i);

uint64_t distribute(const uint64_t& n);

template <typename T,typename S>
typename std::enable_if<std::is_unsigned<T>::value,T>::type
constexpr rotl(const T n, const S i);

template <class T>
inline size_t hash_combine(std::size_t& seed, const T& v);

template <typename T> 
struct vectorHash {
    std::size_t operator()(const std::vector<T> &vec) const;
};


//This is the class for dealing with HMMs
//It stores (as private variables) the transition matrix (A), start probabilities (S), and Emission Probabilities (E) as well as the number of hidden states (H) and observed states (O)
class HMM {
    
    private:

        int H; //Number of hidden states
        int O; //Number of observed states
        std::vector< std::vector<double> > A; //Transition matrix, size HxH
        std::vector<double> S; //Start probs, size H
        std::vector< std::vector<double> > E; //Emission probs, size HxO
        std::mt19937 generator; //Needed for running the HMM
        std::uniform_real_distribution<double> dist; //Ditto

        double getRandom();

    public:
        HMM(const std::vector< std::vector<double> > &inputA, const std::vector<double> &inputS, const std::vector< std::vector<double> > &inputE, int seed = time(NULL));
        
        //Get Private Variables
        int getH() const;
        int getO() const;
        std::vector< std::vector<double> > getA() const;
        std::vector<double> getS() const;
        std::vector< std::vector<double> > getE() const;
        double getAEntry(const int h1, const int h2) const;
        double getSEntry(const int h) const;
        double getEEntry(const int h, const int o) const;

        void printS() const;
        void printA() const;
        void printO() const;
        void print() const;

        void run(const int T, std::vector<int> &observedStates, std::vector<int> &hiddenStates);

        //Inference
        std::vector<int> aStar(const std::vector<int> &observations, double &logProb) const;
        std::vector<int> aStar(const std::vector<int> &observations, double &logProb, const int numZeros) const ;
        std::vector< int > aStarOracle(const std::vector<int> &observations, double &logProb, const std::function<bool(std::vector<int>)> &constraintOracle) const;
        std::vector< std::vector< int > > aStarMult(const std::vector<int> &observations, double &logProb, const int numZeros, const int numSolns) const;
        std::vector< std::vector< int > > aStarMult(const std::vector<int> &observations, double &logProb, const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns) const;

        double logProb(const std::vector<int> obs, const std::vector<int> guess) const;

        //Learning Algorithms
        void learn(const std::vector<int> &obs, const int numZeros, const double eps = 10E-6);
        void learn(const std::vector< std::vector<int> > &obs, const std::vector<int> &numZeros, const double eps = 10E-6);
        void learn(const std::vector<int> &obs, const double eps = 10E-6);
        void learn(const std::vector< std::vector<int> > &obs, const double eps = 10E-6);
        void learnMC(const std::vector<int> &obs, const std::function<bool(std::vector<int>)> &constraintOracle, const double eps = 10E-6, const int C = 10E4);

};


#endif