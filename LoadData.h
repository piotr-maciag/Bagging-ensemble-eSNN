//
// Created by Piotr Maciąg on 14/06/2021.
//

#ifndef PREDICTION_LOADDATA_H
#define PREDICTION_LOADDATA_H

#include "iostream"
#include "algorithm"
#include "vector"
#include "fstream"
#include "sstream"
#include "math.h"
#include "chrono"
#include "random"
#include "iomanip"
#include "eSNN.h"
#include <cstdlib>
#include <ctime>

void LoadDataset(string filename, Dataset * trainingDataset);
void LoadTestData(string filename, Dataset * testDataset);
void PrintDataset(Dataset * d);
vector<Dataset *> GenerateSamples(Dataset * trainingDataset, int N);
void SaveResults(string path, Dataset * d);
void SaveRMSE(string path, vector<int> v1, vector<double> v2, vector<double> v3);
void SaveRMSE_NI(string path, vector<int> v1, vector<int> v2, vector<double> v3);
double CalculateMAPE(vector<double> real, vector<double> predicted);
double CalculateIA(vector<double> real, vector<double> predicted);

#endif //PREDICTION_LOADDATA_H
