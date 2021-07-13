#include <iostream>
#include "eSNN.h"
#include "LoadData.h"

int main() {


     //LoadTestData(testDataset);

   // PrintDataset(testDataset);


   vector<int> ensNum = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60};
   //vector<double> simTrNum = {0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15};
    //vector<double> simTrNum = {0.02};
   //vector<int> NIsizeNum = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80};
   int trials = 5;

   vector<double> RMSE_vect;
    vector<double> MAE_vect;
    vector<double> MAPE_vect;
    vector<double> IA_vect;

   // for(int j = 0; j < ensNum.size(); j++) {
   //     for(int i = 0; i < simTrNum.size(); i++) { //NIsize = 40
        //for(int i = 0; i < NIsizeNum.size(); i++) { //simTR = 0.05

            Dataset *toPredict;
            double RMSEsum = 0;
            double RMSE ;
            double MAEsum = 0;
            double MAE ;
            double MAPEsum = 0;
            double IAsum = 0;
            for(int k = 0; k < trials; k++) {
                Dataset *trainingDataset = new Dataset;
                Dataset *testDataset = new Dataset;
                LoadDataset("../Datasets/DaysAhead/TrainingDataset_4.csv",
                            trainingDataset);
                LoadDataset("../Datasets/DaysAhead/TestDataset_4.csv",
                            testDataset);

                //simTr = simTrNum[i];
                simTr = 0.02;
                NIsize = 55;
                //NIsize = NIsizeNum[i];
                //int N = ensNum[j];
                int N = 50;

                vector<Dataset *> samples = GenerateSamples(trainingDataset, N);

                vector<eSNN *> ensemble;
                for (int i = 0; i < N; i++) {

                    eSNN *eSNN_net = new eSNN;
                    ensemble.push_back(eSNN_net);

                    InitializeInputLayer(eSNN_net, trainingDataset, testDataset);
                    eSNN_Learn(eSNN_net, samples[i]);
                }

                RMSE  = 0;
                MAE = 0;
                toPredict = testDataset;

                for (int i = 0; i < toPredict->att[0].values.size(); i++) {
                    double prediction = 0;
                    double sumPrediction = 0;
                    Example e;
                    for (int k = 0; k < toPredict->att.size(); k++) {
                        e.values.push_back(toPredict->att[k].values[i]);
                        e.attributes.push_back(toPredict->att[k].type);
                    }

                    for (int j = 0; j < N; j++) {
                        prediction = eSNN_Predict(ensemble[j], e);
                        sumPrediction += prediction;
                        samples[j]->predictedValues.push_back(prediction);
                    }

                    toPredict->predictedValues.push_back(sumPrediction / double(N));
                    //cout << toPredict->realValues[i] << " " << toPredict->predictedValues[i] << endl;
                    RMSE += pow((toPredict->realValues[i] - toPredict->predictedValues[i]), 2);
                    MAE += abs((toPredict->realValues[i] - toPredict->predictedValues[i]));

                }

                cout << k << " "  <<  to_string(N) + "_" + to_string(simTr) << " RMSE "
                     << sqrt(RMSE / double(toPredict->predictedValues.size())) << " MAE "
                     << MAE/double(toPredict->predictedValues.size()) << endl;

                RMSEsum += sqrt(RMSE / double(toPredict->predictedValues.size()));
                MAEsum += MAE/double(toPredict->predictedValues.size());
                MAPEsum += CalculateMAPE(toPredict->realValues, toPredict->predictedValues);
                IAsum += CalculateIA(toPredict->realValues, toPredict->predictedValues);
               // RMSE_vect.push_back(sqrt(RMSE / double(toPredict->predictedValues.size())));


               //SaveResults("../Results/Beijing/_Shunyi.csv" + to_string(N) + ".csv", toPredict);
                ClearStructures(ensemble, samples, trainingDataset, testDataset);
            }

            cout << "AVG RMSE: " << RMSEsum / trials << " AVG MAE: " << MAEsum / trials << " AVG MAPe: " <<
            MAPEsum/trials << " AVG IA: "<< IAsum/trials << endl;
            RMSE_vect.push_back(RMSEsum / trials);
            MAE_vect.push_back(MAEsum / trials);
            MAPE_vect.push_back(MAPEsum/trials);
            IA_vect.push_back(IAsum/trials);
     //   }
   //}

    //SaveRMSE_NI("../Results/Prediction/HeatMap/TestingEnsembleRMSE_NISize.csv", ensNum, NIsizeNum, RMSE_vect);
   // SaveRMSE("../Results/Prediction/HeatMap/TestingEnsembleRMSE.csv", ensNum, simTrNum, RMSE_vect);
    //SaveRMSE("../Results/EnsembleComb/TestingEnsembleRMSE.csv", ensNum, simTrNum, RMSE_vect);
    //SaveRMSE("../Results/EnsembleComb/TestingEnsembleMAE.csv", ensNum, simTrNum, MAE_vect);
    //SaveRMSE("../Results/EnsembleComb/TestingEnsembleMAPE.csv", ensNum, simTrNum, MAPE_vect);
    //SaveRMSE("../Results/EnsembleComb/TestingEnsembleIA.csv", ensNum, simTrNum, IA_vect);


    //SaveResults("../Datasets/Warsaw/ResultsTraining.csv", trainingDataset);
}

