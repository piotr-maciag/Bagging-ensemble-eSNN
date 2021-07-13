//
// Created by Piotr Maciąg on 14/06/2021.
//

#ifndef PREDICTION_ESNN_H
#define PREDICTION_ESNN_H

#include "iostream"
#include "algorithm"
#include "vector"
#include "fstream"
#include "sstream"
#include "math.h"
#include "chrono"
#include "random"
#include "iomanip"

using namespace std;

struct neuron
{
    int ID;
    vector<vector<double>> s_weights;
    double outputValue;
    double M;
    double PSP;
    bool fire = true;
}; //output neuron structure

struct inputNeuron
{
    int ID;
    double firingTime;
    int order;
    double mu;
    int rank;
}; //input neuron structure

struct inputAttribute
{
    vector<inputNeuron* > InputNeurons;
    int type = 0; //real 0, ordinal 1, nominal 2
    double I_max, I_min;
};

struct eSNN
{
    vector<neuron *> OutputNeurons;
    vector<inputAttribute *> Attribute;


    int CNO_size = 0;
};

struct Attribute
{
    int type;
    vector<double> values;
};

struct Dataset
{
    vector<Attribute> att;
    vector<double> predictedValues;
    vector<double> realValues;
};

struct Example
{
    vector<double> values;
    vector<int> attributes;
};

extern vector<eSNN *> ensemble;

extern int NIsize;
extern double simTr;
extern double mod;

extern Dataset TrainingDataset;
extern vector<vector<double>> TrainingTargetValues;

extern vector<vector<double>> TestDataset;
extern vector<vector<double>> TrainingTestValues;

void eSNN_Learn(eSNN * eSNN_net, Dataset * sample);
double eSNN_Predict(eSNN *eSNN_net, Example e);
void InitializeInputLayer(eSNN *eSNN_net, Dataset * trainingDataset, Dataset * testDataset);
void ClearStructures(vector<eSNN *> eSNN_Nets, vector<Dataset *> samples, Dataset * trainingDataset, Dataset * testDataset);

#endif //PREDICTION_ESNN_H
