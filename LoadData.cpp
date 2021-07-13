//
// Created by Piotr Maciąg on 14/06/2021.
//

#include "LoadData.h"

int CountInstances(string fileName) {
    fstream handler;



    handler.open(fileName);

    string line;

    int numInstances = 0;

    while (handler.eof() != true) {

        getline(handler, line);

        if (line != "") {
            numInstances++;
        }
    }

    handler.close();


    return numInstances;
}

void LoadDataset(string filename, Dataset * trainingDataset)
{
    fstream handler;


    int datasetSize = CountInstances(filename); // zlicz l. instancji w pliku

    //std::istringstream s(line);
    //std::string field;
    //while (getline(s, field,','))

    handler.open(filename);

    string line;
    getline(handler, line);
    stringstream linestream(line);
    string dataPortion;

    while(getline(linestream, dataPortion, ',')){

        //getline(linestream, dataPortion, ',');
        double value = stod(dataPortion);
        Attribute at;
        at.type = value;
        trainingDataset->att.push_back(at);
    }

//    getline(linestream, dataPortion, ' ');
//    double value = stod(dataPortion);
//    Attribute at;
//    at.type = value;
//    trainingDataset->att.push_back(at);

    for (int i = 1; i < datasetSize; i++) {
        string line;
        getline(handler, line);
        stringstream linestream(line);
        string dataPortion;

        if (line != "") {

            int i = 0;
            while(getline(linestream, dataPortion, ',')){

                //getline(linestream, dataPortion, ',');
                double value = stod(dataPortion);
                trainingDataset->att[i].values.push_back(value);
                i++;


            }

            getline(linestream, dataPortion, ' ');
            double value = stod(dataPortion);
            //trainingDataset->att[i].values.push_back(value);
            //trainingDataset->realValues.push_back(value);
            trainingDataset->realValues.push_back(trainingDataset->att[0].values.at(trainingDataset->att[0].values.size() - 1));

        }
    }

    trainingDataset->att.erase(trainingDataset->att.begin() + trainingDataset->att.size() - 1);
    handler.close();
}

void LoadTestData(string filename, Dataset * testDataset)
{

}

void PrintDataset(Dataset * d) {

    for(int i = 0; i < d->att.size(); i++)
    {
        cout << d->att[i].type << '\t';
    }

    cout << endl;

    for (int i = 0; i < d->att[0].values.size(); i++) {
        for (int j = 0; j < d->att.size(); j++) {
            cout << d->att[j].values[i] << '\t';
        }
        cout << "RV: " << d->realValues[i] <<  endl;
    }


}

//Clear all structures after each eSNN training and classification
//void ClearStructures() {
//
//    for (int i = 0; i < OutputNeurons.size(); i++) {
//        for(int j = 0; j < OutputNeurons[i].size(); j++)
//            delete OutputNeurons[i][j];
//    }
//
//    OutputNeurons.clear();
//    X.clear();
//    Y.clear();
//
//    for (int k = 0; k < InputNeurons.size(); k++) {
//        for (int j = 0; j < InputNeurons.size(); j++) {
//            delete InputNeurons[k][j];
//        }
//    }
//
//    InputNeurons.clear();
//    Wstream.clear();
//    I_min.clear();
//    I_max.clear();
//    IDS.clear();
//}

void SaveResults(string path, Dataset * d)
{
    fstream results;
    results.open(path, fstream::out);

    for(int i = 0; i < d->realValues.size(); i++)
    {
        results << d->realValues[i] << "," << d->predictedValues[i] << endl;
    }

    results.close();
}

vector<Dataset *> GenerateSamples(Dataset * trainingDataset, int N)
{
    int randomNumber;
    vector<Dataset *> samples;

    for(int i = 0; i < N; i++)
    {
        Dataset * sample = new Dataset;

        for(int j = 0 ; j < trainingDataset->att.size(); j++)
        {
            Attribute at;
            at.type = trainingDataset->att[j].type;
            sample->att.push_back(at);
        }


        for(int j = 0; j < trainingDataset->realValues.size()/1.5; j++)
        {
            randomNumber = (rand() % trainingDataset->realValues.size());
            sample->realValues.push_back(trainingDataset->realValues[randomNumber]);

            for(int k = 0; k < sample->att.size(); k++)
            {
                sample->att[k].values.push_back(trainingDataset->att[k].values[randomNumber]);
            }

        }
        samples.push_back(sample);
    }

    return samples;
}

void SaveRMSE(string path, vector<int> v1, vector<double> v2, vector<double> v3)
{
    fstream results;
    results.open(path, fstream::out);

    results << "N" << "," << "simTr" << "," << "RMSE" << endl;
    for(int j = 0; j < v1.size(); j++) {
        for(int i = 0; i < v2.size(); i++) {
            results << v1[j] << "," << v2[i] << "," << v3[j*v2.size() + i] << endl;
        }
    }

    results.close();
}

void SaveRMSE_NI(string path, vector<int> v1, vector<int> v2, vector<double> v3)
{
    fstream results;
    results.open(path, fstream::out);

    results << "N" << "," << "NIsize" << "," << "RMSE" << endl;

    for(int j = 0; j < v1.size(); j++) {
        for(int i = 0; i < v2.size(); i++) {
            results << v1[j] << "," << v2[i] << "," << v3[j*v2.size() + i] << endl;
        }
    }

    results.close();
}

double CalculateMAPE(vector<double> real, vector<double> predicted)
{
    double MAPE = 0;

    for(int i = 0; i < real.size(); i++)
    {
        MAPE += abs((real[i] - predicted[i])/real[i]);
    }
    MAPE /= real.size();
    return MAPE * 100;
}

double CalculateIA(vector<double> real, vector<double> predicted)
{
    double avgReal = 0;
    for(int i = 0; i < real.size(); i++)
    {
        avgReal += real[i];
    }
    avgReal /= real.size();

    double sumDiffs = 0;

    for(int i = 0; i < real.size(); i++)
    {
        sumDiffs += pow(real[i]-predicted[i], 2);
    }

    double sumDiffsAvg = 0;

    for(int i =0; i < real.size(); i++)
    {
        sumDiffsAvg += pow(abs(predicted[i] - avgReal) + abs(real[i] - avgReal), 2);
    }

    return 1 - (sumDiffs/sumDiffsAvg);
}