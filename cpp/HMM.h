#pragma once

#include<vector>
#include<iostream>
#include<random>

#include<time.h> //Used for seed

class HMM {
    
    private:

        int H; //Number of hidden states
        int O; //Number of observed states
        std::vector< std::vector<double> > A; //Transition matrix, size HxH
        std::vector<double> S; //Start probs, size H
        std::vector< std::vector<double> > E; //Emission probs, size HxO

    public:

        //---------------------
        //-----Constructor-----
        //---------------------

        HMM(const std::vector< std::vector<double> > &inputA, const std::vector<double> &inputS, const std::vector< std::vector<double> > &inputE) {
            H = inputA.size();

            if((inputA[0].size() != H) || (inputS.size() != H) || (inputE.size() != H)) {
                std::cout << "Error in constructor for HMM, matrices not appropriately sized." << std::endl;
                throw std::exception();
            }

            O = inputE[0].size();

            A = inputA;
            S = inputS;
            E = inputE;
        }

        //----------------------------------
        //-----Access private variables-----
        //----------------------------------

        int getH() {
            return H;
        }

        int getO() {
            return O;
        }

        std::vector< std::vector<double> > getA() {
            return A;
        }

        std::vector<double> getS() {
            return S;
        }

        std::vector< std::vector<double> > getE() {
            return E;
        }

        //Range not checked for speed
        double getAEntry(int h1, int h2) {
            return A[h1][h2];
        }

        double getSEntry(int h) {
            return S[h];
        }

        double getEEntry(int h, int o) {
            return E[h][o];
        }

        //---------------------
        //-----Run the HMM-----
        //---------------------

        void run(int T, std::vector<int> &observedStates, std::vector<int> &hiddenStates, int seed = time(NULL)) {
            observedStates.clear();
            hiddenStates.clear();

            std::random_device rand_dev;
            std::mt19937 generator(rand_dev());
            generator.seed(seed);
            std::uniform_real_distribution<double> dist(0., 1.);

            //Initial Hidden State
            double startProb = dist(generator);
            double prob = 0;
            for(int h = 0; h < H; ++h) {
                prob += S[h];
                if(startProb < prob) {
                    hiddenStates.push_back(h);
                    break;
                }
            }

            //Initial Observed State
            double obsProb = dist(generator);
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
                startProb = dist(generator);
                prob = 0;
                for(int h = 0; h < H; ++h) {
                    prob += A[hiddenStates[t-1]][h];
                    if(startProb < prob) {
                        hiddenStates.push_back(h);
                        break;
                    }
                }
                
                obsProb = dist(generator);
                prob = 0;
                for(int o = 0; o < O; ++o) {
                    prob += E[hiddenStates[t]][o];
                    if(obsProb < prob) {
                        observedStates.push_back(o);
                        break;
                    }
                }
            }
        }
};