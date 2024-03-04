//HMM.cpp

#include "HMM.h"

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

//--------------------------
//-----Hash for Vectors-----
//--------------------------

//Taken for Stack Overflow (https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/50978188#50978188)
template<typename T>
T xorshift(const T& n,int i){
  return n^(n>>i);
}

// a hash function with another name as to not confuse with std::hash
uint64_t distribute(const uint64_t& n){
  uint64_t p = 0x5555555555555555ull; // pattern of alternating 0 and 1
  uint64_t c = 17316035218449499591ull;// random uneven integer constant; 
  return c*xorshift(p*xorshift(n,32),32);
}

// if c++20 rotl is not available:
template <typename T,typename S>
typename std::enable_if<std::is_unsigned<T>::value,T>::type
constexpr rotl(const T n, const S i){
  const T m = (std::numeric_limits<T>::digits-1);
  const T c = i&m;
  return (n<<c)|(n>>((T(0)-c)&m)); // this is usually recognized by the compiler to mean rotation, also c++20 now gives us rotl directly
}

// call this function with the old seed and the new key to be hashed and combined into the new seed value, respectively the final hash
template <class T>
inline size_t hash_combine(std::size_t& seed, const T& v)
{
    return rotl(seed,std::numeric_limits<size_t>::digits/3) ^ distribute(std::hash<T>{}(v));
}

//Could also do this for containers which aren't vectors
template <typename T> 
std::size_t vectorHash<T>::operator()(const std::vector<T> &vec) const {
    std::size_t seed = vec.size();
    for(auto& i : vec) {
			seed = hash_combine(seed, i);
	}
	return seed;
}


//Return a random number from 0,1
//Needs to be an internal function b/c we are calling random a bunch of different times in the run function, possibly multiple times

double HMM::getRandom() {
    return dist(generator);
}

HMM::HMM(const std::vector< std::vector<double> > &inputA, const std::vector<double> &inputS, const std::vector< std::vector<double> > &inputE, int seed) {
    H = inputA.size();

    //Check if sizes are correct
    if((inputS.size() != H) || (inputE.size() != H)) {
        std::cout << "Error in constructor for HMM, matrices not appropriately sized." << std::endl;
        throw std::exception();
    }

    for(int h = 0; h < H; ++h) {
        if(inputA[h].size() != H) {
            std::cout << "Error in constructor for HMM, A is not a square matrix." << std::endl;
            throw std::exception();
        }
    }

    O = inputE[0].size();

    for(int h = 0; h < H; ++h) {
        if(inputE[h].size() != O) {
            std::cout << "Error in constructor for HMM, E is not a matrix." << std::endl;
            throw std::exception();
        }
    }
    

    //Check if matrices represent probabilities
    double sum = 0;
    for(int h1 = 0; h1 < H; ++h1) {
        sum = 0;
        for(int h2 = 0; h2 < H; ++h2) {
            if(inputA[h1][h2] < 0.) {
                std::cout << "Error in constructor for HMM, A cannot have negative entries." << std::endl;
                throw std::exception();
            }

            sum += inputA[h1][h2];
        }

        if(std::abs(sum-1.) > 10E-6) {
            std::cout << "Error in constructor for HMM, the rows of A must sum to 1." << std::endl;
            throw std::exception();
        }
    }

    sum = 0;
    for(int h = 0; h < H; ++h) {
        if(inputS[h] < 0.) {
            std::cout << "Error in constructor for HMM, S cannot have negative entries." << std::endl;
            throw std::exception();
        }
        sum += inputS[h];
    }
    if(std::abs(sum-1.) > 10E-6) {
        std::cout << "Error in constructor for HMM, the entries of S must sum to 1." << std::endl;
    }

    for(int h = 0; h < H; ++h) {
        sum = 0;
        for(int o = 0; o < O; ++o) {
            if(inputE[h][o] < 0.) {
                std::cout << "Error in constructor for HMM, E cannot have negative entries." << std::endl;
                throw std::exception();
            }

            sum += inputE[h][o];
        }

        if(std::abs(sum-1.) > 10E-6) {
            std::cout << "Error in constructor for HMM, the rows of E must sum to 1." << std::endl;
            throw std::exception();
        }
    }

    A = inputA;
    S = inputS;
    E = inputE;
    
    std::random_device rand_dev;
    std::mt19937 myGenerator(rand_dev());
    generator = myGenerator;
    generator.seed(seed);
    std::uniform_real_distribution<double> myDist(0., 1.);
    dist = myDist;
}


//----------------------------------
//-----Access private variables-----
//----------------------------------

int HMM::getH() const{
    return H;
}

int HMM::getO() const{
    return O;
}

std::vector< std::vector<double> > HMM::getA() const{
    return A;
}

std::vector<double> HMM::getS() const{
    return S;
}

std::vector< std::vector<double> > HMM::getE() const{
    return E;
}

//Range not checked for speed
double HMM::getAEntry(const int h1, const int h2) const{
    return A[h1][h2];
}

double HMM::getSEntry(const int h) const{
    return S[h];
}

double HMM::getEEntry(const int h, const int o) const{
    return E[h][o];
}


//-----------------------
//-----Print the HMM-----
//-----------------------

void HMM::printS() const{
    std::cout << "Start vector:\n";
    for(int i = 0; i < H; ++i) {
        std::cout << S[i] << " ";
    }
    std::cout << "\n\n";
    return;
}

void HMM::printA() const{
    std::cout << "Transmission matrix:\n";
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < H; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    return;
}

void HMM::printO() const{
    std::cout << "Emission matrix: (Columns are hidden states, rows are observed states)\n";
    for(int h = 0;  h < H; ++h) {
        for(int o = 0; o < O; ++o) {
            std::cout << E[h][o] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";
    return;
}

void HMM::print() const{
    printS();
    printA();
    printO();
    return;
}


//---------------------
//-----Run the HMM-----
//---------------------

//This generates the observed states and hidden states running the HMM for T time steps
//Not const b/c of the random stuff
void HMM::run(const int T, std::vector<int> &observedStates, std::vector<int> &hiddenStates) {
    
    observedStates.clear();
    hiddenStates.clear();

    //Initial Hidden State
    double startProb = getRandom();
    double prob = 0;
    for(int h = 0; h < H; ++h) {
        prob += S[h];
        if(startProb < prob) {
            hiddenStates.push_back(h);
            break;
        }
    }

    //Initial Observed State
    double obsProb = getRandom();
    prob = 0;
    for(int o = 0; o < O; ++o) {
        prob += E[hiddenStates[0]][o];
        if(obsProb < prob) {
            observedStates.push_back(o);
            break;
        }
    }

    //All other states
    for(int t = 1; t < T; ++t) {
        startProb = getRandom();
        prob = 0;
        for(int h = 0; h < H; ++h) {
            prob += A[hiddenStates[t-1]][h];
            if(startProb < prob) {
                hiddenStates.push_back(h);
                break;
            }
        }

        obsProb = getRandom();
        prob = 0;
        for(int o = 0; o < O; ++o) {
            prob += E[hiddenStates[t]][o];
            if(obsProb < prob) {
                observedStates.push_back(o);
                break;
            }
        }
    }
    
    
    return;
}

//----------------
//-----A star-----
//----------------

//Does inference with a given set of observations
//logProb is the log of the probability that the given states occur (we use logs as otherwise we could get numerical underflow)
//Uses the A* algorithm for inference 
//Without constraints (such as in this case) this is basically equivalent to running Viterbi with a bit of overhead
std::vector<int> HMM::aStar(const std::vector<int> &observations, double &logProb) const{
    const int T = observations.size();

    //So we don't need to keep recomputing logs
    std::vector< std::vector<double> > logA;
    std::vector<double> logS;
    std::vector< std::vector<double> > logE;

    logA.resize(H);
    logE.resize(H);
    
    for(int h1 = 0; h1 < H; ++h1) {
        logA[h1].resize(H);
        for(int h2 = 0; h2 < H; ++h2) {
            logA[h1][h2] = std::log(A[h1][h2]);
        }
    }

    for(int h = 0; h < H; ++h) {
        logE[h].resize(O);
        for(int o = 0; o < O; ++o) {
            logE[h][o] = std::log(E[h][o]);
        }
    }
    
    std::vector< std::vector<double> > v; //Stands for Viterbi, used as an estimate for how much logprob is left
    v.resize(T);
    for(int t = 0; t < T; ++t) {
        v[t].resize(H);
    }
    
    for(int h = 0; h < H; ++h) {
        v[T-1][h] = 0;
    }

    for(int t = T-2; t >= 0; --t) {
        for(int h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for(int h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t+1][h2] + logA[h1][h2] + logE[h2][observations[t+1]]);
            }
            v[t][h1] = temp;
        }
    }

    std::priority_queue< std::pair<double, std::vector<int> > > openSet;
    std::unordered_map< std::vector<int>, double, vectorHash<int>  > gScore; //log prob so far

    for(int h = 0; h < H; ++h) {
        double tempGScore = std::log(S[h]) + logE[h][observations[0]]; //Avoids extra look-up operation
        gScore[{h}] = tempGScore;
        std::vector<int> tempVec = {h}; //make_pair doesn't like taking in {h}
        openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
    }

    //Actual run run of A* algorithm
    while(!openSet.empty()) {
        auto seq = openSet.top().second;
        openSet.pop();

        int t = seq.size();
        int h1 = seq[t-1];
        double oldGScore = gScore[seq];

        if(t == T) {
            logProb = -oldGScore;
            return seq;
        }

        for(int h2 = 0; h2 < H; ++h2) {
            auto newSeq = seq;
            newSeq.push_back(h2);
            double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
            gScore[newSeq] = tempGScore;
            openSet.push(std::make_pair(tempGScore + v[t][h2], newSeq));
        }
    }

    return {};
}

//---------------------------------
//-----A star with constraints-----
//---------------------------------

//The same as the function above, however here we are allowed to specify the number of times the function is in hidden state 0 with the parameter numZeros
//Could also expand this to be general linear constraints
std::vector<int> HMM::aStar(const std::vector<int> &observations, double &logProb, const int numZeros) const{
    const int T = observations.size();

    //So we don't need to keep recomputing logs
    std::vector< std::vector<double> > logA;
    std::vector<double> logS;
    std::vector< std::vector<double> > logE;

    logA.resize(H);
    logE.resize(H);
    
    for(int h1 = 0; h1 < H; ++h1) {
        logA[h1].resize(H);
        for(int h2 = 0; h2 < H; ++h2) {
            logA[h1][h2] = std::log(A[h1][h2]);
        }
    }

    for(int h = 0; h < H; ++h) {
        logE[h].resize(O);
        for(int o = 0; o < O; ++o) {
            logE[h][o] = std::log(E[h][o]);
        }
    }
    
    std::vector< std::vector<double> > v; //Stands for Viterbi, used as an estimate for how much logprob is left
    v.resize(T);
    for(int t = 0; t < T; ++t) {
        v[t].resize(H);
    }
    
    for(int h = 0; h < H; ++h) {
        v[T-1][h] = 0;
    }

    for(int t = T-2; t >= 0; --t) {
        for(int h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for(int h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t+1][h2] + logA[h1][h2] + logE[h2][observations[t]]);
            }
            v[t][h1] = temp;
        }
    }

    //Dist, current h, time, constraint val
    std::priority_queue< std::tuple<double, int, int, int> > openSet; //Works b/c c++ orders tuples lexigraphically
    std::unordered_map< std::tuple<int, int, int>, double, boost::hash< std::tuple<int,int,int> > > gScore; //pair is h,t, constraintVal
    //TODO make better hash for tuple
    std::unordered_map< std::tuple<int, int, int>, int, boost::hash< std::tuple<int,int,int> > > prev; //Used to recover sequence of hidden states
    for(int h = 0; h < H; ++h) {
        double tempGScore = std::log(S[h]) + logE[h][observations[0]]; //Avoids extra look-up operation
    
        if(h == 0) {
            openSet.push(std::make_tuple(tempGScore + v[0][h],0,1,1));
            gScore[std::make_tuple(0,1,1)] = tempGScore;
        }
        else {
            openSet.push(std::make_tuple(tempGScore + v[0][h],1,1,0));
            gScore[std::make_tuple(1,1,0)] = tempGScore;
        }
    }

    while(!openSet.empty()) { 
        auto tempTuple = openSet.top();
        int h1 = get<1>(tempTuple); //Current state
        int t = get<2>(tempTuple); //Current time
        int fVal = get<3>(tempTuple); //Current fVal

        openSet.pop();
        double oldGScore = gScore.at(std::make_tuple(h1,t,fVal));

        if(t == T) {
            if(fVal == numZeros) { //Make sure we actually satisfy the constraints
                logProb = oldGScore;
                std::vector<int> output;
                output.push_back(h1);

                while(t > 1) { 
                    int h = prev[std::make_tuple(h1,t,fVal)];
                    if(h1 == 0) {
                        --fVal;
                    }
                    --t;
                    output.push_back(h);
                    h1 = h;
                }
                
                std::reverse(output.begin(), output.end());
                return output;
            }
        }

        //Expand in the A* algorithm
        else {
            for(int h2 = 0; h2 < H; ++h2) {
                int newFVal = fVal;
                if(h2 == 0) {
                    ++newFVal;
                }

                if(newFVal <= numZeros) { //Helps reduce the size of the problem - we can't do this if we instead have a general oracle
                    double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                    if(gScore.count(std::make_tuple(h2,t+1,newFVal)) == 0 ) {
                        gScore[std::make_tuple(h2,t+1,newFVal)] = tempGScore;
                        openSet.push(std::make_tuple(tempGScore + v[t][h2], h2,t+1,newFVal));
                        prev[std::make_tuple(h2,t+1,newFVal)] = h1;
                    }
                    else if(tempGScore >  gScore.at(std::make_tuple(h2,t+1,newFVal))) { //Makes sure we don't have empty call to map
                        gScore.at(std::make_tuple(h2,t+1,newFVal)) = tempGScore;
                        openSet.push(std::make_tuple(tempGScore + v[t][h2], h2,t+1,newFVal));
                        prev.at(std::make_tuple(h2,t+1,newFVal)) = h1;
                    }
                }
            }
        }
    }

    return {};
}


//------------------------
//-----A* with oracle-----
//------------------------

//Rather than having some nice function that we can take advantage of the structure of, we just have an oracle which  
//Keeps track of all values not just constraint value. Need for more complicated constraints
//Note: This may produce a different solution from other A* functions. However, they will have the same logProb, and thus occur with the same probability
//Effectively the same as the code above, but we can't restrict the space if we have too many 0's
std::vector< int > HMM::aStarOracle(const std::vector<int> &observations, double &logProb, const std::function<bool(std::vector<int>)> &constraintOracle) const{
    const int T = observations.size();

    //So we don't need to keep recomputing logs
    std::vector< std::vector<double> > logA;
    std::vector<double> logS;
    std::vector< std::vector<double> > logE;

    logA.resize(H);
    logE.resize(H);
    
    for(int h1 = 0; h1 < H; ++h1) {
        logA[h1].resize(H);
        for(int h2 = 0; h2 < H; ++h2) {
            logA[h1][h2] = std::log(A[h1][h2]);
        }
    }

    for(int h = 0; h < H; ++h) {
        logE[h].resize(O);
        for(int o = 0; o < O; ++o) {
            logE[h][o] = std::log(E[h][o]);
        }
    }
    
    std::vector< std::vector<double> > v; //Stands for Viterbi
    v.resize(T);
    for(int t = 0; t < T; ++t) {
        v[t].resize(H);
    }
    
    for(int h = 0; h < H; ++h) {
        v[T-1][h] = 0;
    }

    for(int t = T-2; t >= 0; --t) {
        for(int h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for(int h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t+1][h2] + logA[h1][h2] + logE[h2][observations[t]]);
            }
            v[t][h1] = temp;
        }
    }

    std::vector< int > output;

    //Dist, current h, time, constraint val
    std::priority_queue< std::pair<double, std::vector<int> > > openSet; //Works b/c c++ orders tuples lexigraphically
    std::unordered_map< std::vector<int>, double, boost::hash< std::vector<int> > > gScore; //pair is h,t, constraintVal
    //TODO make better hash for tuple
    //Would gScore be better as a multi-dimensional array? <- probably not, b/c we are hoping it stays sparse
    for(int h = 0; h < H; ++h) {
        double tempGScore = std::log(S[h]) + logE[h][observations[0]]; //Avoids extra look-up operation
    
        if(h == 0) {
            std::vector<int> tempVec = {0}; //Otherwise C++ can't figure out what is happening
            openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
            gScore[{0}] = tempGScore;
        }
        else {
            std::vector<int> tempVec = {1};
            openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
            gScore[{1}] = tempGScore;
        }
    }
    
    while(!openSet.empty()) {
        auto tempPair= openSet.top();
        std::vector<int> currentSequence = get<1>(tempPair);
        int t = currentSequence.size();
        int h1 = currentSequence[t-1];

        openSet.pop();
        double oldGScore = gScore.at(currentSequence);
        if(t == T) {
            if(constraintOracle(currentSequence)) {
                logProb = oldGScore;
                return currentSequence;
            }
        }
        
        else {
            for(int h2 = 0; h2 < H; ++h2) {
                double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                std::vector<int> newSequence = currentSequence;
                newSequence.push_back(h2);

                gScore[newSequence] = tempGScore;
                openSet.push(std::make_pair(tempGScore + v[t][h2],newSequence));
            }
        }
    }
    return {};
}


//---------------------
//-----A* multiple-----
//---------------------

//Returns the top numSolns solutions to the inference problem.
//Uses the same inference technique as A*Oracle, so it is much slower than general A*
std::vector< std::vector< int > > HMM::aStarMult(const std::vector<int> &observations, double &logProb, const int numZeros, const int numSolns) const{
    const int T = observations.size();

    //So we don't need to keep recomputing logs
    std::vector< std::vector<double> > logA;
    std::vector<double> logS;
    std::vector< std::vector<double> > logE;

    logA.resize(H);
    logE.resize(H);
    
    for(int h1 = 0; h1 < H; ++h1) {
        logA[h1].resize(H);
        for(int h2 = 0; h2 < H; ++h2) {
            logA[h1][h2] = std::log(A[h1][h2]);
        }
    }

    for(int h = 0; h < H; ++h) {
        logE[h].resize(O);
        for(int o = 0; o < O; ++o) {
            logE[h][o] = std::log(E[h][o]);
        }
    }
    
    std::vector< std::vector<double> > v; //Stands for Viterbi
    v.resize(T);
    for(int t = 0; t < T; ++t) {
        v[t].resize(H);
    }
    
    for(int h = 0; h < H; ++h) {
        v[T-1][h] = 0;
    }

    for(int t = T-2; t >= 0; --t) {
        for(int h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for(int h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t+1][h2] + logA[h1][h2] + logE[h2][observations[t]]);
            }
            v[t][h1] = temp;
        }
    }

    std::vector< std::vector<int> > output;
    int counter = 0;

    //Dist, current h, time, constraint val
    std::priority_queue< std::pair<double, std::vector<int> > > openSet; //Works b/c c++ orders tuples lexigraphically
    std::unordered_map< std::vector<int>, double, boost::hash< std::vector<int> > > gScore; //pair is h,t, constraintVal
    //TODO make better hash for tuple
    //Would gScore be better as a multi-dimensional array? <- probably not, b/c we are hoping it stays sparse
    for(int h = 0; h < H; ++h) {
        double tempGScore = std::log(S[h]) + logE[h][observations[0]]; //Avoids extra look-up operation
    
        if(h == 0) {
            std::vector<int> tempVec = {0}; //Otherwise C++ can't figure out what is happening
            openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
            gScore[{0}] = tempGScore;
        }
        else {
            std::vector<int> tempVec = {1};
            openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
            gScore[{1}] = tempGScore;
        }
    }
    
    while(!openSet.empty()) {
        auto tempPair= openSet.top();
        std::vector<int> currentSequence = get<1>(tempPair);
        int t = currentSequence.size();
        int h1 = currentSequence[t-1];
        int fVal = 0;
        for(int i = 0; i < t; ++i) {
            if(currentSequence[i] == 0) {
                ++fVal;
            }
        }

        openSet.pop();
        double oldGScore = gScore.at(currentSequence);
        if(t == T) {
            if(fVal == numZeros) {
                logProb = oldGScore;
                output.push_back(currentSequence);
                ++counter;
                if(counter == numSolns) {
                    return output;
                }
            }
        }
        
        else {
            for(int h2 = 0; h2 < H; ++h2) {
                int newFVal = fVal;
                if(h2 == 0) {
                    ++newFVal;
                }

                if(newFVal <= numZeros) {
                    double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                    std::vector<int> newSequence = currentSequence;
                    newSequence.push_back(h2);

                    gScore[newSequence] = tempGScore;
                    openSet.push(std::make_pair(tempGScore + v[t][h2],newSequence));
                }
            }
        }
    }
    return {};
}


//---------------------------------
//-----A* multiple with Oracle-----
//---------------------------------

//Same as above, but we now have an oracle for the constraints
std::vector< std::vector< int > > HMM::aStarMult(const std::vector<int> &observations, double &logProb, const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns) const{
    const int T = observations.size();

    //So we don't need to keep recomputing logs
    std::vector< std::vector<double> > logA;
    std::vector<double> logS;
    std::vector< std::vector<double> > logE;

    logA.resize(H);
    logE.resize(H);
    
    for(int h1 = 0; h1 < H; ++h1) {
        logA[h1].resize(H);
        for(int h2 = 0; h2 < H; ++h2) {
            logA[h1][h2] = std::log(A[h1][h2]);
        }
    }

    for(int h = 0; h < H; ++h) {
        logE[h].resize(O);
        for(int o = 0; o < O; ++o) {
            logE[h][o] = std::log(E[h][o]);
        }
    }
    
    std::vector< std::vector<double> > v; //Stands for Viterbi
    v.resize(T);
    for(int t = 0; t < T; ++t) {
        v[t].resize(H);
    }
    
    for(int h = 0; h < H; ++h) {
        v[T-1][h] = 0;
    }

    for(int t = T-2; t >= 0; --t) {
        for(int h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for(int h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t+1][h2] + logA[h1][h2] + logE[h2][observations[t]]);
            }
            v[t][h1] = temp;
        }
    }

    //Dist, current h, time, constraint val
    std::priority_queue< std::pair<double, std::vector<int> > > openSet; //Works b/c c++ orders tuples lexigraphically
    std::unordered_map< std::vector<int>, double, boost::hash< std::vector<int> > > gScore; //pair is h,t, constraintVal
    //TODO make better hash for tuple
    //Would gScore be better as a multi-dimensional array? <- probably not, b/c we are hoping it stays sparse
    for(int h = 0; h < H; ++h) {
        double tempGScore = std::log(S[h]) + logE[h][observations[0]]; //Avoids extra look-up operation
    
        if(h == 0) {
            std::vector<int> tempVec = {0}; //Otherwise C++ can't figure out what is happening
            openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
            gScore[{0}] = tempGScore;
        }
        else {
            std::vector<int> tempVec = {1};
            openSet.push(std::make_pair(tempGScore + v[0][h],tempVec));
            gScore[{1}] = tempGScore;
        }
    }

    std::vector< std::vector<int> > output;
    int counter = 0;
    
    while(!openSet.empty()) {
        auto tempPair= openSet.top();
        std::vector<int> currentSequence = get<1>(tempPair);
        int t = currentSequence.size();
        int h1 = currentSequence[t-1];

        openSet.pop();
        double oldGScore = gScore.at(currentSequence);
        if(t == T) {
            if(constraintOracle(currentSequence)) {
                output.push_back(currentSequence);
                ++counter;
                if(counter == numSolns) {
                    return output;
                }
            }
        }
        
        else {
            for(int h2 = 0; h2 < H; ++h2) {
                double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                std::vector<int> newSequence = currentSequence;
                newSequence.push_back(h2);

                gScore[newSequence] = tempGScore;
                openSet.push(std::make_pair(tempGScore + v[t][h2],newSequence));
            }
        }
    }
    return {};
}


//---------------------------
//-----Calculate logProb-----
//---------------------------

//Useful in debugging and Monte Carlo learning algorithm
//Should match A* function
double HMM::logProb(const std::vector<int> obs, const std::vector<int> guess) const {
    double output = 0;
    int T = guess.size();
    output += log(S[guess[0]]);
    for(int t = 0; t < T-1; ++t) {
        output += log(A[guess[t]][guess[t+1]]) + log(E[guess[t]][obs[t]]);
    }
    output += log(E[guess[T-1]][obs[T-1]]);
    return output;
}


//-----------------------------------
//-----Learning with constraints-----
//-----------------------------------

//Your HMM should be initalized with your prior guess of the probabilities (referred to as theta in the comments) 
//Only for a single set of observations
//The constraint is the number of zeros in the hidden states, denoted by numZeros
//Epsilon are tolerance
//This would also work if the constraint was a linear combination of the hidden states
void HMM::learn(const std::vector<int> &obs, const int numZeros, const double eps) {   
    int T = obs.size();

    while(true) { 
        //alpha
        std::vector< std::vector< std::vector<double> > > alpha; //alpha[c][h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta, c 0's)
        alpha.resize(numZeros+1);
        for(int c = 0; c <= numZeros; ++c) {
            alpha[c].resize(H);
            for(int h = 0; h < H; ++h) {
                alpha[c][h].resize(T,0.);
                if(((c == 1) && (h == 0)) || ((c == 0) && (h != 0))) {
                    alpha[c][h][0] = S[h]*E[h][obs[0]];
                }
            }
        }

        for(int t = 1; t < T-1; ++t) {
            for(int c = 0; c <= numZeros; ++c) {
                for(int h = 0; h < H; ++h) {
                    for(int h1 = 0; h1 < H; ++h1) {
                        int oldC = c;
                        if(h == 0) {
                            --oldC;
                        }

                        if(oldC >= 0) {
                            alpha[c][h][t] += alpha[oldC][h1][t-1]*A[h1][h];
                        }
                    }
                    alpha[c][h][t] *= E[h][obs[t]];
                }
            }
        }

        //t = T-1
        for(int c = 0; c <= numZeros; ++c) {
            for(int h = 0; h < H; ++h) {   
                if(c == numZeros) {                   
                    for(int h1 = 0; h1 < H; ++h1) {
                        int oldC  = c;
                        if(h == 0) {
                            --oldC;
                        }

                        if(oldC >= 0) {
                            alpha[c][h][T-1] += alpha[oldC][h1][T-2]*A[h1][h];
                        }
                    }
                    alpha[c][h][T-1] *= E[h][obs[T-1]];
                }
            }
        }

        //beta
        std::vector< std::vector< std::vector<double> > > beta; //beta[c][h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta, c 0's )
        beta.resize(numZeros+1);
        for(int c = 0; c <= numZeros; ++c) {
            beta[c].resize(numZeros+1);
            for(int h = 0; h < H; ++h) {
                beta[c][h].resize(T,0.);
                if(c == 0) {
                    beta[c][h][T-1] = 1.;
                }
            }
        }

        for(int t = T-2; t > 0; --t) {
            for(int c = 0; c <= numZeros; ++c) {
                for(int h = 0; h < H; ++h) {
                    for(int h2 = 0; h2 < H; ++h2) {
                        int newC = c;
                        if(h2 == 0) {
                            --newC;
                        }

                        if(newC >= 0) {
                            beta[c][h][t] += beta[newC][h2][t+1]*A[h][h2]*E[h2][obs[t+1]];
                        }
                    }
                }
            }
        }

        //t = 0
        //h[0] = 0
        if(numZeros > 0) { 
            for(int h2 = 0; h2 < H; ++h2) {
                int newC = numZeros -1; 
                if(h2 == 0) {
                    --newC;
                }
                if(newC >= 0) {
                    beta[numZeros-1][0][0] += beta[newC][h2][1]*A[0][h2]*E[h2][obs[1]];
                }
            }
        }

        //h[0] != 0
        for(int h = 1; h < H; ++h) {
            for(int h2 = 0; h2 < H; ++h2) {
                int newC = numZeros; 
                if(h2 == 0) {
                    --newC;
                }

                if(newC >= 0) {
                    beta[numZeros][h][0] += beta[newC][h2][1]*A[h][h2]*E[h2][obs[1]];
                }
            }
        }

        //den = P(O | theta) 
        //Need different denominators because of the scaling
        //This is numerically a VERY weird algorithm
        std::vector<double> den;
        for(int t = 0; t < T; ++t) {
            den.push_back(0.);
            for(int h = 0; h < H; ++h) {
                for(int c = 0; c <= numZeros; ++c) {
                    den[t] += alpha[c][h][t]*beta[numZeros-c][h][t];
                }
            }
        }
        
        //Gamma
        std::vector< std::vector<double> > gamma; //gamma[h][t] = P(H_t = h | Y , theta)
        gamma.resize(H);
        for(int h = 0; h < H; ++h) {
            gamma[h].resize(T);
        }

        for(int h = 0; h < H; ++h) {
            for(int t = 0; t < T; ++t) {
                double num = 0.;
                for(int c = 0; c <= numZeros; ++c) {
                    num += alpha[c][h][t]*beta[numZeros-c][h][t];
                }
                gamma[h][t] = num/den[t];
            }
        }
        
        //xi
        std::vector< std::vector< std::vector<double> > > xi; //xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta) 
        xi.resize(H);
        for(int h1 = 0; h1 < H; ++h1) {
            xi[h1].resize(H);
            for(int h2 = 0; h2 < H; ++h2) {
                xi[h1][h2].resize(T-1);
            }
        }

        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                for(int t = 0; t < T-1; ++t) {
                    double num = 0.;

                    for(int c = 0; c <= numZeros; ++c) {
                        int middleC = 0;
                        if(h2 == 0) {
                            ++middleC;
                        }

                        if(numZeros-middleC-c >= 0) {
                            num += alpha[c][h1][t]*beta[numZeros-middleC-c][h2][t+1];
                        }
                    }
                    num *= A[h1][h2]*E[h2][obs[t+1]];

                    xi[h1][h2][t] = num/den[t];
                }
            }
        }
        
        //New S
        for(int h = 0; h < H; ++h) {
            S[h] = gamma[h][0];
        }
        
        //New E
        for(int h = 0; h < H; ++h) {
            for(int o = 0; o < O; ++o) {
                double num = 0.;
                double newDen = 0.;

                for(int t = 0; t < T; ++t) {
                    if(obs[t] == o) {
                        num += gamma[h][t];
                    }
                    newDen += gamma[h][t];
                }

                E[h][o] = num/newDen;
            }
        }
        
        double tol = 0.;

        //New A
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;

                for(int t = 0; t < T-1; ++t) {
                    num += xi[h1][h2][t];
                    newDen += gamma[h1][t];
                }
                tol = std::max(std::abs(A[h1][h2] - num/newDen), tol); 
                A[h1][h2] = num/newDen;
            }
        }

        std::cout << "Tolerance: " << tol << "\n"; //Can comment this out if too much printing

        if(tol < eps) {
            break;
        }
    }
}


//----------------------------------------------------------------
//-----Constrained Learning with Multiple Set of Observations-----
//----------------------------------------------------------------

//This is the exact same as the algorithm above, but here we allow multiple observations
//Using the same terminology as the Wikipedia page, we use r to deal with constraints
void HMM::learn(const std::vector< std::vector<int> > &obs, const std::vector<int> &numZeros, const double eps) {   
    int T = obs[0].size();
    int R = obs.size();

    while(true) { 
        std::vector< std::vector< std::vector<double> > > totalGamma;
        std::vector< std::vector< std::vector< std::vector<double> > > > totalXi;
        for(int r = 0; r < R; ++r) { 
            //alpha
            std::vector< std::vector< std::vector<double> > > alpha; //alpha[c][h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta, c 0's)
            alpha.resize(numZeros[r]+1);
            for(int c = 0; c <= numZeros[r]; ++c) {
                alpha[c].resize(H);
                for(int h = 0; h < H; ++h) {
                    alpha[c][h].resize(T);

                    if(((c == 1) && (h == 0)) || ((c == 0) && (h != 0))) {
                        alpha[c][h][0] = S[h]*E[h][obs[r][0]];
                    }

                    else{
                        alpha[c][h][0] = 0.;
                    } 
                }
            }

            for(int t = 1; t < T-1; ++t) {
                for(int c = 0; c <= numZeros[r]; ++c) {
                    for(int h = 0; h < H; ++h) {
                        alpha[c][h][t] = 0.;
                        for(int h1 = 0; h1 < H; ++h1) {
                            int oldC = c;
                            if(h1 == 0) {
                                --oldC;
                            }

                            if(oldC >= 0) {
                                alpha[c][h][t] += alpha[oldC][h1][t-1]*A[h1][h];
                            }
                        }
                        alpha[c][h][t] *= E[h][obs[r][t]];
                    }
                }
            }

            //t = T-1
            for(int c = 0; c <= numZeros[r]; ++c) {
                for(int h = 0; h < H; ++h) {   
                    alpha[c][h][T-1] = 0.;
                    if(c == numZeros[r]) {                   
                        for(int h1 = 0; h1 < H; ++h1) {
                            int oldC  = c;
                            if(h1 == 0) {
                                --oldC;
                            }

                            if(oldC >= 0) {
                                alpha[c][h][T-1] += alpha[oldC][h1][T-2]*A[h1][h];
                            }
                        }
                        alpha[c][h][T-1] *= E[h][obs[r][T-1]];
                    }
                }
            }

            //beta
            std::vector< std::vector< std::vector<double> > > beta; //beta[c][h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta, c 0's )
            beta.resize(numZeros[r]+1);
            for(int c = 0; c <= numZeros[r]; ++c) {
                beta[c].resize(numZeros[r]+1);
                for(int h = 0; h < H; ++h) {
                    beta[c][h].resize(T);

                    if(c == 0) {
                        beta[c][h][T-1] = 1;
                    }

                    else {
                        beta[c][h][T-1] = 0;
                    }
                }
            }

            for(int t = T-2; t > 0; --t) {
                for(int c = 0; c <= numZeros[r]; ++c) {
                    for(int h = 0; h < H; ++h) {
                        beta[c][h][t] = 0.;
                        for(int h2 = 0; h2 < H; ++h2) {
                            int newC = c;
                            if(h2 == 0) {
                                --newC;
                            }

                            if(newC >= 0) {
                                beta[c][h][t] += beta[newC][h2][t+1]*A[h][h2]*E[h2][obs[r][t+1]];
                            }
                        }
                    }
                }
            }

            //t = 0
            for(int c = 0; c <= numZeros[r]; ++c) {
                for(int h = 0; h < H; ++h) {
                    beta[c][h][0] = 0.;
                }
            }

            //h[0] = 0
            if(numZeros[r] > 0) { 
                for(int h2 = 0; h2 < H; ++h2) {
                    int newC = numZeros[r] -1; 
                    if(h2 == 0) {
                        --newC;
                    }
                    if(newC >= 0) {
                        beta[numZeros[r]-1][0][0] += beta[newC][h2][1]*A[0][h2]*E[h2][obs[r][1]];
                    }
                }
            }

            //h[0] != 0
            for(int h = 1; h < H; ++h) {
                for(int h2 = 0; h2 < H; ++h2) {
                    int newC = numZeros[r]; 
                    if(h2 == 0) {
                        --newC;
                    }

                    if(newC >= 0) {
                        beta[numZeros[r]][h][0] += beta[newC][h2][1]*A[h][h2]*E[h2][obs[r][1]];
                    }
                }
            }

            //den = P(O | theta) 
            //Need different denominators because of the scaling
            //This is numerically a VERY weird algorithm
            std::vector<double> den;
            for(int t = 0; t < T; ++t) {
                den.push_back(0.);
                for(int h = 0; h < H; ++h) {
                    for(int c = 0; c <= numZeros[r]; ++c) {
                        den[t] += alpha[c][h][t]*beta[numZeros[r]-c][h][t];
                    }
                }
            }
            
            //Gamma
            std::vector< std::vector<double> > gamma; //gamma[h][t] = P(H_t = h | Y , theta)
            gamma.resize(H);
            for(int h = 0; h < H; ++h) {
                gamma[h].resize(T);
            }

            for(int h = 0; h < H; ++h) {
                for(int t = 0; t < T; ++t) {
                    double num = 0.;
                    for(int c = 0; c <= numZeros[r]; ++c) {
                        num += alpha[c][h][t]*beta[numZeros[r]-c][h][t];
                    }
                    gamma[h][t] = num/den[t];
                }
            }

            totalGamma.push_back(gamma);
            
            //xi
            std::vector< std::vector< std::vector<double> > > xi; //xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta) 
            xi.resize(H);
            for(int h1 = 0; h1 < H; ++h1) {
                xi[h1].resize(H);
                for(int h2 = 0; h2 < H; ++h2) {
                    xi[h1][h2].resize(T-1);
                }
            }

            for(int h1 = 0; h1 < H; ++h1) {
                for(int h2 = 0; h2 < H; ++h2) {
                    for(int t = 0; t < T-1; ++t) {
                        double num = 0.;

                        for(int c = 0; c <= numZeros[r]; ++c) {
                            int middleC = 0;
                            if(h2 == 0) {
                                ++middleC;
                            }

                            if(numZeros[r]-middleC-c >= 0) {
                                num += alpha[c][h1][t]*beta[numZeros[r]-middleC-c][h2][t+1];
                            }
                        }
                        num *= A[h1][h2]*E[h2][obs[r][t+1]];

                        xi[h1][h2][t] = num/den[t];
                    }
                }
            }

            totalXi.push_back(xi);
        }
        
        //New S
        for(int h = 0; h < H; ++h) {
            S[h] = 0.;
            for(int r = 0; r < R; ++r) {
                S[h] += totalGamma[r][h][0];
            }
            S[h] /= R;
        }
        
        //New E
        for(int r = 0; r < R; ++r) {
            for(int h = 0; h < H; ++h) {
                for(int o = 0; o < O; ++o) {
                    double num = 0.;
                    double newDen = 0.;

                    for(int t = 0; t < T; ++t) {
                        if(obs[r][t] == o) {
                            num += totalGamma[r][h][t];
                        }
                        newDen += totalGamma[r][h][t];
                    }

                    E[h][o] = num/newDen;
                }
            }
        }
        
        double tol = 0.;

        //New A
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;
                for(int r = 0; r < R; ++r) {                     
                    for(int t = 0; t < T-1; ++t) {
                        num += totalXi[r][h1][h2][t];
                        newDen += totalGamma[r][h1][t];
                    }
                }
                tol = std::max(std::abs(A[h1][h2] - num/newDen), tol); 
                A[h1][h2] = num/newDen;
            }
        }
        std::cout << "Tolerance: " << tol << "\n";
        //tol = 0.;
        if(tol < eps) {
            break;
        }
    }
}


//---------------------------------------
//-----Learning without Constraints------
//---------------------------------------

void HMM::learn(const std::vector<int> &obs, const double eps) {   
    int T = obs.size();

    while(true) { 
        //alpha
        std::vector< std::vector<double> > alpha; //alpha[h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta)
        alpha.resize(H);
        for(int h = 0; h < H; ++h) {
            alpha[h].resize(T,0.);
            alpha[h][0] = S[h]*E[h][obs[0]];
        }
        
        for(int t = 1; t < T; ++t) {
            for(int h = 0; h < H; ++h) {
                for(int h1 = 0; h1 < H; ++h1) {
                    alpha[h][t] += alpha[h1][t-1]*A[h1][h];
                }

                alpha[h][t] *= E[h][obs[t]];
            }
        }

        //beta
        std::vector< std::vector<double> > beta; //beta[h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta)
        beta.resize(H);
        for(int h = 0; h < H; ++h) {
            beta[h].resize(T);
            beta[h][T-1] = 1.;
        }

        
        for(int t = T-2; t >= 0; --t) {
            for(int h = 0; h < H; ++h) {
                for(int h2 = 0; h2 < H; ++h2) {
                    beta[h][t] += beta[h2][t+1]*A[h][h2]*E[h2][obs[t+1]];
                }
            }
        }

        //den = P(O | theta)
        std::vector<double> den(T,0);
        for(int t = 0; t < T; ++t) {
            for(int h = 0; h < H; ++h) {
                den[t] += alpha[h][t]*beta[h][t];
            }
        }
        
        //Gamma
        std::vector< std::vector<double> > gamma; //gamma[h][t] = P(H_t = h | Y , theta)
        gamma.resize(H);
        for(int h = 0; h < H; ++h) {
            gamma[h].resize(T);
        }

        for(int h = 0; h < H; ++h) {
            for(int t = 0; t < T; ++t) {
                gamma[h][t] = alpha[h][t]*beta[h][t]/den[t];
            }
        }
        
        //xi
        std::vector< std::vector< std::vector<double> > > xi; //xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta) 
        xi.resize(H);
        for(int h1 = 0; h1 < H; ++h1) {
            xi[h1].resize(H);
            for(int h2 = 0; h2 < H; ++h2) {
                xi[h1][h2].resize(T-1);
            }
        }

        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                for(int t = 0; t < T-1; ++t) {
                    xi[h1][h2][t] = alpha[h1][t]*beta[h2][t+1]*A[h1][h2]*E[h2][obs[t+1]]/den[t];
                }
            }
        }
        
        //New S
        for(int h = 0; h < H; ++h) {
            S[h] = gamma[h][0];
        }
        
        //New E
        for(int h = 0; h < H; ++h) {
            for(int o = 0; o < O; ++o) {
                double num = 0.;
                double newDen = 0.;

                for(int t = 0; t < T; ++t) {
                    if(obs[t] == o) {
                        num += gamma[h][t];
                    }
                    newDen += gamma[h][t];
                }

                E[h][o] = num/newDen;
            }
        }
        
        double tol = 0.;

        //New A
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;

                for(int t = 0; t < T-1; ++t) {
                    num += xi[h1][h2][t];
                    newDen += gamma[h1][t];
                }
                tol = std::max(std::abs(A[h1][h2] - num/newDen), tol); 
                A[h1][h2] = num/newDen;
            }
        }
        std::cout << "Tolerance: " << tol << "\n";
        //tol = 0.;
        if(tol < eps) {
            break;
        }
    }
}


//Unconstrained learning with multiple observations
//Similar to unconstrained learning with constraints
void HMM::learn(const std::vector< std::vector<int> > &obs, const double eps) {   
    int T = obs[0].size();
    int R = obs.size();
    int numIt = 0;

    while(true) {
        ++numIt;
        std::vector< std::vector< std::vector<double> > > totalGamma;
        std::vector< std::vector< std::vector< std::vector<double> > > > totalXi;

        for(int r = 0; r < R; ++r) { 
            //alpha
            std::vector< std::vector<double> > alpha; //alpha[h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta)
            alpha.resize(H);
            for(int h = 0; h < H; ++h) {
                alpha[h].resize(T,0.);
                alpha[h][0] = S[h]*E[h][obs[r][0]];
            }
            
            for(int t = 1; t < T; ++t) {
                for(int h = 0; h < H; ++h) {
                    for(int h1 = 0; h1 < H; ++h1) {
                        alpha[h][t] += alpha[h1][t-1]*A[h1][h];
                    }

                    alpha[h][t] *= E[h][obs[r][t]];
                }
            }

            //beta
            std::vector< std::vector<double> > beta; //beta[h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta)
            beta.resize(H);
            for(int h = 0; h < H; ++h) {
                beta[h].resize(T);
                beta[h][T-1] = 1.;
            }

            
            for(int t = T-2; t >= 0; --t) {
                for(int h = 0; h < H; ++h) {
                    for(int h2 = 0; h2 < H; ++h2) {
                        beta[h][t] += beta[h2][t+1]*A[h][h2]*E[h2][obs[r][t+1]];
                    }
                }
            }

            //den = P(O | theta) 
            std::vector<double> den(T,0);
            for(int t = 0; t < T; ++t) {
                for(int h = 0; h < H; ++h) {
                    den[t] += alpha[h][t]*beta[h][t];
                }
            }
            
            //Gamma
            std::vector< std::vector<double> > gamma; //gamma[h][t] = P(H_t = h | Y , theta)
            gamma.resize(H);
            for(int h = 0; h < H; ++h) {
                gamma[h].resize(T);
            }

            for(int h = 0; h < H; ++h) {
                for(int t = 0; t < T; ++t) {
                    gamma[h][t] = alpha[h][t]*beta[h][t]/den[t];
                }
            }
            totalGamma.push_back(gamma);
            
            //xi
            std::vector< std::vector< std::vector<double> > > xi; //xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta) 
            xi.resize(H);
            for(int h1 = 0; h1 < H; ++h1) {
                xi[h1].resize(H);
                for(int h2 = 0; h2 < H; ++h2) {
                    xi[h1][h2].resize(T-1);
                }
            }

            for(int h1 = 0; h1 < H; ++h1) {
                for(int h2 = 0; h2 < H; ++h2) {
                    for(int t = 0; t < T-1; ++t) {
                        xi[h1][h2][t] = alpha[h1][t]*beta[h2][t+1]*A[h1][h2]*E[h2][obs[r][t+1]]/den[t];
                    }
                }
            }
            totalXi.push_back(xi);
        }
        
        //New S
        for(int h = 0; h < H; ++h) {
            S[h] = 0.;
            for(int r = 0; r < R; ++r) {
                S[h] += totalGamma[r][h][0];
            }
            S[h] /= R;
        }
        
        //New E
        for(int r = 0; r < R; ++r) {
            for(int h = 0; h < H; ++h) {
                for(int o = 0; o < O; ++o) {
                    double num = 0.;
                    double newDen = 0.;

                    for(int t = 0; t < T; ++t) {
                        if(obs[r][t] == o) {
                            num += totalGamma[r][h][t];
                        }
                        newDen += totalGamma[r][h][t];
                    }

                    E[h][o] = num/newDen;
                }
            }
        }
        
        double tol = 0.;

        //New A
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;
                for(int r = 0; r < R; ++r) {                     
                    for(int t = 0; t < T-1; ++t) {
                        num += totalXi[r][h1][h2][t];
                        newDen += totalGamma[r][h1][t];
                    }
                }
                tol = std::max(std::abs(A[h1][h2] - num/newDen), tol); 
                A[h1][h2] = num/newDen;
            }
        }

        std::cout << "Tolerance: " << tol << "\n";

        if(tol < eps) {
            break;
        }
    }
}


//------------------------------
//-----Monte Carlo Learning-----
//------------------------------

//Will work best/fastest if the sets of hidden states which satisfy the constraints 
//This algorithm is TERRIBLE, I can't even get it to converge in a simple case with T = 10.
//This is currently the only learning algorithm we have for having a constraint oracle rather than ``simple'' constraints
//This also fails to work if we are converging towards values in the transition matrix with 0's (which is NOT uncommon)
void HMM::learnMC(const std::vector<int> &obs, const std::function<bool(std::vector<int>)> &constraintOracle, const double eps, const int C) {
    int T = obs.size();

    while(true) {
        
        std::vector< std::vector<double> > gamma;
        std::vector< std::vector<int> > gammaCounter;
        std::vector< std::vector< std::vector<double> > > xi;
        std::vector< std::vector< std::vector<int> > > xiCounter;

        gamma.resize(H);
        gammaCounter.resize(H);
        xi.resize(H);
        xiCounter.resize(H);
        
        for(int h = 0; h < H; ++h) {
            gamma[h].resize(T, 0.);
            gammaCounter[h].resize(T, 0);
            xi[h].resize(H);
            xiCounter[h].resize(H);

            for(int h1 = 0; h1 < H; ++h1) {
                xi[h][h1].resize(T-1, 0.);
                xiCounter[h][h1].resize(T-1, 0);
            }
        }
        
        //Do Monte Carlo
        int numIt = 1; 
        int checkCont = H*H*C/10; //The check for breaking out of the loop is SLOW, so this makes everything run faster
        double cont = true;
        while(cont) {
            std::vector<int> observed;
            std::vector<int> hidden;

            run(T,observed,hidden);

            if(constraintOracle(hidden)) {
                ++numIt;
                double prob = logProb(observed,hidden);
                prob = std::exp(prob); //Maybe better to stay with logs?    

                ++gammaCounter[hidden[0]][0];
                gamma[hidden[0]][0] += prob;
                for(int t = 1; t < T; ++t) {
                    ++xiCounter[hidden[t-1]][hidden[t]][t-1];
                    xi[hidden[t-1]][hidden[t]][t-1] += prob;
                    ++gammaCounter[hidden[t]][t];
                    gamma[hidden[t]][t] += prob;
                }
            }
            
            if((numIt % checkCont) == 0) {
                std::cout << "Iteration: " << numIt << std::endl;
                ++numIt;
                int minVal = C;
                int temp = 0;

                for(int h1 = 0; h1 < H; ++h1) {
                    for(int h2 = 0; h2 <H; ++h2) {
                        temp = 0;
                        for(int t = 0; t < T-1; ++t) {
                            temp += xiCounter[h1][h2][t];
                            if(xiCounter[h1][h2][t] == 0) { //Avoid dividing by 0
                                minVal = 0;
                            }
                        }
                        minVal = std::min(minVal, temp);
                    }
                }

                for(int h = 0; h < H; ++h) {
                    temp = 0;
                    for(int t = 0; t < T; ++t) {
                        temp += gammaCounter[h][t];
                        if(gammaCounter[h][t] == 0) {
                            minVal = 0;
                        }
                    }
                    minVal = std::min(minVal, temp);
                }

                if(minVal == C) {
                    cont = false;
                }
            }
        }  


        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                for(int t = 0; t < T-1; ++t) {
                    xi[h1][h2][t] /= xiCounter[h1][h2][t];
                }
            }
        }

        //Normalize
        for(int t = 0; t < T-1; ++t) {
            double sum = 0.;
            for(int h1 = 0; h1 < H; ++h1) {
                for(int h2 = 0; h2 < H; ++h2) {
                    sum += xi[h1][h2][t];
                }
            }

            for(int h1 = 0; h1 < H; ++h1) {
                for(int h2 = 0; h2 < H; ++h2) {
                    xi[h1][h2][t] /= sum;
                }
            }
        }

        for(int h = 0; h < H; ++h) {
            for(int t = 0; t < T; ++t) {
                gamma[h][t] /= gammaCounter[h][t];
            }
        }

        //Normalize
        for(int t = 0; t < T; ++t) {
            double sum = 0.;

            for(int h = 0; h < H; ++h) {
                sum += gamma[h][t];
            }

            for(int h = 0; h < H; ++h) {
                gamma[h][t] /= sum;
            }
        }

        //New S
        for(int h = 0; h < H; ++h) {
            S[h] = gamma[h][0];
        }
        
        //New E
        for(int h = 0; h < H; ++h) {
            for(int o = 0; o < O; ++o) {
                double num = 0.;
                double newDen = 0.;

                for(int t = 0; t < T; ++t) {
                    if(obs[t] == o) {
                        num += gamma[h][t];
                    }
                    newDen += gamma[h][t];
                }

                E[h][o] = num/newDen;
            }
        }
        
        double tol = 0.;
        std::vector< std::vector<double> > newA;
        newA.resize(H);
        for(int h = 0; h < H; ++h) {
            newA[h].resize(H);
        }

        //New A
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;

                for(int t = 0; t < T-1; ++t) {
                    num += xi[h1][h2][t];
                    newDen += gamma[h1][t];
                }
                newA[h1][h2] = num/newDen;
            }
        }

        //Normalize A, we need to do this b/c xi and gamma aren't calculated exactly
        for(int h1 = 0; h1 < H; ++h1) {
            double sum = 0; 
            for(int h2 = 0; h2 < H; ++h2) {
                sum += newA[h1][h2];
            }

            for(int h2 = 0; h2 < H; ++h2) {
                newA[h1][h2] /= sum;
                tol = std::max(std::abs(A[h1][h2] - newA[h1][h2]), tol); 
                A[h1][h2] = newA[h1][h2];
            }
        }

        std::cout << "Tolerance: " << tol << "\nA matrix:\n";
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                std::cout << A[h1][h2] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        if(tol < eps) {
            break;
        }

    }
}


