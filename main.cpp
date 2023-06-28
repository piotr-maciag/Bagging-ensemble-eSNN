#include <iostream>
#include "eSNN.h"
#include "LoadData.h"

int main() {


    //LoadTestData(testDataset);
    //PrintDataset(testDataset);
//
//   vector<string> pollutants = {"O3", "PM25", "PM10"};
//    vector<string> sites = {"Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng", "Huairou",
//                           "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"};
//    vector<string> hours = {"12", "24", "48", "72"};

//    vector<string> pollutants = {"O3", "PM25"};
//    vector<string> sites = {"Aotizhongxin", "Changping", "Tiantan"};
//    vector<string> hours = {"24"};


    vector<string> pollutants = {"PM10"};
    vector<string> sites = {"Warsaw"};
    vector<string> hours = {"1"};

    vector<int> ensNum = {1, 5, 10, 15, 20, 25, 30, 35, 40, 50};
    //vector<int> ensNum = {70, 100, 150};
    vector<double> simTrNum = {0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 };
    vector<int> NIsizeNum = {10, 20, 30, 40, 50, 60};
    int trials = 1;

    vector<double> RMSE_vect;
    vector<double> MAE_vect;
    vector<double> MAPE_vect;
    vector<double> IA_vect;

    fstream resultsComb;
    resultsComb.open("../Results/Warsaw preprocessed/TestPolOnly/ResultsCombined.csv", fstream::out);
    resultsComb << "Pollutant" << "," << "Site" << "," << "Hour" << ",N" << ",RealValue,PredValue" << endl;

    fstream errorsComb;
    errorsComb.open("../Results/Warsaw preprocessed/TestPolOnly/ErrorsCombined.csv", fstream::out);
    errorsComb << "Pollutant" << "," << "Site" << "," << "Hour" << ",NIsize" << ",simTr,N,RMSE,MAE,MAPE,IA" << endl;



    for (int d1 = 0; d1 < pollutants.size(); d1++) {
        for (int d2 = 0; d2 < sites.size(); d2++) {

            for (int d3 = 0; d3 < hours.size(); d3++) {


                for (int p3 = 0; p3 < NIsizeNum.size(); p3++) { //simTR = 0.05
                    for (int p2 = 0; p2 < simTrNum.size(); p2++) { //NIsize = 40
                        cout << endl;
                        for (int p1 = 0; p1 < ensNum.size(); p1++) {


                            Dataset *toPredict;
                            double RMSEsum = 0;
                            double RMSE;
                            double MAEsum = 0;
                            double MAE;
                            double MAPEsum = 0;
                            double IAsum = 0;

                            simTr = simTrNum[p2];
                            //simTr = 0.02;
                            //NIsize = 60;
                            //Bins = 2;
                            NIsize = NIsizeNum[p3];
                            int N = ensNum[p1];
                            //int N = 50;

                            for (int k = 0; k < trials; k++) {
                                Dataset *trainingDataset = new Dataset;
                                Dataset *testDataset = new Dataset;
//                                LoadDataset("../Datasets/Beijing/" + pollutants[d1] + "_" +
//                                            sites[d2] + "_" + hours[d3] + "_" + "Training.csv",
//                                            trainingDataset);
//                                LoadDataset("../Datasets/Beijing/" + pollutants[d1] + "_" +
//                                            sites[d2] + "_" + hours[d3] + "_" + "Testing.csv",
//                                            testDataset);

                                LoadDataset("../Datasets/Warsaw preprocessed/TrainingDecomposedDatasetPol.csv",
                                            trainingDataset);
                                LoadDataset("../Datasets/Warsaw preprocessed/TestDecomposedDatasetPol.csv",
                                            testDataset);
                                cout << "Loaded" << endl;



//                                cout << endl <<
//                                    pollutants[d1] << "_" <<
//                                    sites[d2] << "_" <<
//                                    hours[d3] << "_" <<
//                                     to_string(NIsize) << "_" <<
//                                     to_string(simTr) << "_" <<
//                                     to_string(N) + "_" <<
//                                     endl;

                                vector<Dataset *> samples;
                                if (N != 1)
                                    samples = GenerateSamples(trainingDataset, N);
                                else {
                                    samples.push_back(trainingDataset);
                                }




                                vector<eSNN *> ensemble;
                                for (int i = 0; i < N; i++) {
                                    //cout << "#" << flush;
                                    eSNN *eSNN_net = new eSNN;
                                    ensemble.push_back(eSNN_net);

                                    InitializeInputLayer(eSNN_net, trainingDataset, testDataset);
                                    eSNN_Learn(eSNN_net, samples[i]);
                                }


                                RMSE = 0;
                                MAE = 0;
                                toPredict = testDataset;

                                for (int i = 0; i < toPredict->att[0].values.size(); i++) {
                                    //cout << i << endl;
                                    //if (i % 1000 == 0) cout << "#" << flush;
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
                                    resultsComb << pollutants[d1] << "," << sites[d2] << "," << hours[d3] << "," <<
                                    ensNum[p1] << "," <<
                                                toPredict->realValues[i] << "," << sumPrediction / double(N) << endl;

                                }

                                //cout << endl;

//                    cout << to_string(N) + "_" + to_string(simTr) << "_"
//                         << to_string(NIsize) <<
//                         //" RMSE " << sqrt(RMSE / double(toPredict->predictedValues.size())) << " MAE "
//                         // << MAE / double(toPredict->predictedValues.size()) <<
//                         endl;

                                RMSEsum += sqrt(RMSE / double(toPredict->predictedValues.size()));
                                MAEsum += MAE / double(toPredict->predictedValues.size());
                                MAPEsum += CalculateMAPE(toPredict->realValues, toPredict->predictedValues);
                                IAsum += CalculateIA(toPredict->realValues, toPredict->predictedValues);
                                // RMSE_vect.push_back(sqrt(RMSE / double(toPredict->predictedValues.size())));


                                //SaveResults("../Results/Beijing/_Shunyi.csv" + to_string(N) + ".csv", toPredict);

                                ClearStructures(ensemble, samples, trainingDataset, testDataset);
                                //cout << "pp" << endl;



                            }

                            cout <<
                                 pollutants[d1] << "_" <<
                                 sites[d2] << "_" <<
                                 hours[d3] << "_" <<
                                 to_string(NIsize) << "_" <<
                                 to_string(simTr) << "_" <<
                                 to_string(N) + "\t";

                            errorsComb <<
                                       pollutants[d1] << "," <<
                                       sites[d2] << "," <<
                                       hours[d3] << "," <<
                                       to_string(NIsize) << "," <<
                                       to_string(simTr) << "," <<
                                       to_string(N) + ",";

                            cout << "AVG RMSE: " << RMSEsum / trials << " AVG MAE: " << MAEsum / trials << " AVG MAPe: "
                                 << MAPEsum / trials << " AVG IA: " << IAsum / trials << endl;

                            errorsComb << RMSEsum / trials << "," << MAEsum / trials << "," << MAPEsum / trials << ","
                            << IAsum / trials << endl;


                            RMSE_vect.push_back(RMSEsum / trials);
                            MAE_vect.push_back(MAEsum / trials);
                            MAPE_vect.push_back(MAPEsum / trials);
                            IA_vect.push_back(IAsum / trials);

                            //   }




                        }

                        //SaveRMSE_NI("../Results/Prediction/HeatMap/TestingEnsembleRMSE_NISize.csv", ensNum, NIsizeNum, RMSE_vect);
                        // SaveRMSE("../Results/Prediction/HeatMap/TestingEnsembleRMSE.csv", ensNum, simTrNum, RMSE_vect);
                        //SaveRMSE("../Results/EnsembleComb/TestingEnsembleRMSE.csv", ensNum, simTrNum, RMSE_vect);
                        //SaveRMSE("../Results/EnsembleComb/TestingEnsembleMAE.csv", ensNum, simTrNum, MAE_vect);
                        //SaveRMSE("../Results/EnsembleComb/TestingEnsembleMAPE.csv", ensNum, simTrNum, MAPE_vect);
                        //SaveRMSE("../Results/EnsembleComb/TestingEnsembleIA.csv", ensNum, simTrNum, IA_vect);


                        //SaveResults("../Datasets/Warsaw preprocessed/ResultsTraining.csv", trainingDataset);
                    }
                }

            }
        }
    }
    resultsComb.close();
    errorsComb.close();
}
