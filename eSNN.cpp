//
// Created by Piotr Maciąg on 14/06/2021.
//


#include "eSNN.h"
int NIsize;
double simTr;
double mod = 0.9;

void InitializeInputLayer(eSNN *eSNN_net, Dataset * trainingDataset, Dataset * testDataset) {
    for (int k = 0; k < trainingDataset->att.size(); k++) {
        if (trainingDataset->att[k].type == 0 || trainingDataset->att[k].type == 1) {
            inputAttribute *InputNeuronsVect = new inputAttribute;
            for (int j = 0; j < NIsize; j++) {
                inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
                InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
            }
            eSNN_net->Attribute.push_back(InputNeuronsVect);
        } else if (trainingDataset->att[k].type == 2) {

            inputAttribute * InputNeuronsVect = new inputAttribute;
            double maxTraining = (*max_element(trainingDataset->att[k].values.begin(), trainingDataset->att[k].values.end()));
            double maxTesting = (*max_element(testDataset->att[k].values.begin(), testDataset->att[k].values.end()));

            double max = (maxTraining > maxTesting) ? maxTraining : maxTesting;

            for (int j = 0; j < int(max); j++) {
                inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
                InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
                InputNeuronsVect->type = 2;
            }
           // cout << InputNeuronsVect->type << endl;
            eSNN_net->Attribute.push_back(InputNeuronsVect);

        }
    }

    for (int i = 0; i < trainingDataset->att.size(); i++) {
        double maxTraining = (*max_element(trainingDataset->att[i].values.begin(), trainingDataset->att[i].values.end()));
        double minTraining = (*min_element(trainingDataset->att[i].values.begin(), trainingDataset->att[i].values.end()));

        double maxTesting = (*max_element(testDataset->att[i].values.begin(), testDataset->att[i].values.end()));
        double minTesting = (*min_element(testDataset->att[i].values.begin(), testDataset->att[i].values.end()));

        double max = (maxTraining > maxTesting) ? maxTraining : maxTesting;
        double min = (minTraining < minTesting) ? minTraining : minTesting;


        //cout << "IMax " << max << " " << "Imin " << min << endl;
        eSNN_net->Attribute[i]->I_min = min, eSNN_net->Attribute[i]->I_max = max;

    }


}

void CalculateOrder(eSNN *eSNN_net, Example exmp) {

    for (int k = 0; k < eSNN_net->Attribute.size(); k++) {

        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            eSNN_net->Attribute[k]->InputNeurons[j]->order = -1;
        }


        if (eSNN_net->Attribute[k]->type == 0 || eSNN_net->Attribute[k]->type == 1) {
            double width = (eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / NIsize;

            for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
                double mu = eSNN_net->Attribute[k]->I_min + (j + 1 - 0.5) * width;
                eSNN_net->Attribute[k]->InputNeurons[j]->mu = mu;
            }

            int j;
            if (exmp.values[k] != eSNN_net->Attribute[k]->I_max) {

                j = floor((exmp.values[k] - eSNN_net->Attribute[k]->I_min) / width) + 1;

            } else {
                j = NIsize;

            }

            int l;
            if (j - 1 < NIsize - j) { l = j - 1; } else { l = NIsize - j; }
            // cout << Windows[k][u] << " " << I_min[k] << " " << width << " "
            // << floor((Windows[k][u] - I_min[k]) / width) << " j:: " << j - 1<< endl;

            //GRFs[k][j - 1]->rank = 0;
            eSNN_net->Attribute[k]->InputNeurons[j - 1]->rank = 0;

            if (exmp.values[k] < eSNN_net->Attribute[k]->InputNeurons[j - 1]->mu) {
                for (int n = 1; n <= l; n++) {
                    //cout << "-j: " << j - n - 1 << endl;
                    //cout << "-j: " << j + n - 1 << endl;
                    eSNN_net->Attribute[k]->InputNeurons[j - n - 1]->rank = 2 * n - 1;
                    eSNN_net->Attribute[k]->InputNeurons[j + n - 1]->rank = 2 * n;
                    //GRFs[k][j - n - 1]->rank = 2 * n - 1;
                    //GRFs[k][j + n - 1]->rank = 2 * n;
                }
                for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
                {
                    // cout << "-j: " << j - l - n - 1 << endl;
                    eSNN_net->Attribute[k]->InputNeurons[j - l - n - 1]->rank = 2 * l - 1 + n;
                    //GRFs[k][j - l - n - 1]->rank = 2 * l - 1 + n;
                }
                for (int n = 1; n <= NIsize - j - l; n++) //n is k in algorithms
                {
                    //cout << "-j: " << j + l + n - 1<< endl;
                    eSNN_net->Attribute[k]->InputNeurons[j + l + n - 1]->rank = 2 * l + n;
                    //GRFs[k][j + l + n - 1]->rank = 2 * l + n;
                }
            } else {
                for (int n = 1; n <= l; n++) {
                    // cout << "+j: " << j - n - 1 << endl;
                    //  cout << "+j: " << j + n - 1 << endl;
                    eSNN_net->Attribute[k]->InputNeurons[j - n - 1]->rank = 2 * n;
                    eSNN_net->Attribute[k]->InputNeurons[j + n - 1]->rank = 2 * n - 1;
                    //GRFs[k][j - n - 1]->rank = 2 * n;
                    //GRFs[k][j + n - 1]->rank = 2 * n - 1;
                }
                for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
                {
                    // cout << "+j: " << j - l - n - 1 << endl;
                    eSNN_net->Attribute[k]->InputNeurons[j - l - n - 1]->rank = 2 * l + n;
                    //GRFs[k][j - l - n - 1]->rank = 2 * l + n;
                }
                for (int n = 1; n <= NIsize - j - l; n++) //n is k in algorithms
                {
                    // cout << "+j: " << j + l + n << endl;
                    eSNN_net->Attribute[k]->InputNeurons[j + l + n - 1]->rank = 2 * l + n - 1;
                    //GRFs[k][j + l + n - 1]->rank = 2 * l + n - 1;
                }

            }

        }


        if (eSNN_net->Attribute[k]->type == 2) {
            //cout << eSNN_net->Attribute[k]->InputNeurons.size() << endl;
            for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
                //cout << j << endl;
                if (j == exmp.values[k])
                    eSNN_net->Attribute[k]->InputNeurons[j]->rank = 0;
                else
                    eSNN_net->Attribute[k]->InputNeurons[j]->rank = -1;
            }

        }

        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            //int rank = GRFs[k][j]->rank + (Wsize - u - 1) * NIsize;
            //cout << " j: " << j << " " << rank << " - ";
            eSNN_net->Attribute[k]->InputNeurons[j]->order = eSNN_net->Attribute[k]->InputNeurons[j]->rank + 1;
           // cout << j << " " << eSNN_net->Attribute[k]->InputNeurons[j]->order << endl;
            //<< InputNeurons[k][j].order[InputNeurons[k][j].order.size() -1];
        }

      //  cout << endl;

    }


}


void InitializeNeuron(eSNN *eSNN_net, neuron *n_c, double x, double idx) { //Initalize new neron n_i

    for (int l = 0; l < eSNN_net->Attribute.size(); l++) {
        vector<double> vec;
        n_c->s_weights.push_back(vec);
        for (int j = 0; j < eSNN_net->Attribute[l]->InputNeurons.size(); j++) {
            n_c->s_weights[l].push_back(0);
        }
    }

    for (int l = 0; l < eSNN_net->Attribute.size(); l++) {
        for (int j = 0; j < eSNN_net->Attribute[l]->InputNeurons.size(); j++) {
            n_c->s_weights[l][j] += pow(mod, eSNN_net->Attribute[l]->InputNeurons[j]->order);
            //cout << n_c->s_weights[l][j] << endl;
        }
    }

    n_c->outputValue = x;
    n_c->M = 1;
    n_c->ID = idx;
}

double
CalculateDistance(const vector<vector<double>> &w1,
                  const vector<vector<double>> &w2) { //calculate distance between two weights vectors
    double diffSq = 0.0;

    for (int k = 0; k < w1.size(); k++) {
        for (int j = 0; j < w1.size(); j++) {
            diffSq += pow(w1[k][j] - w2[k][j], 2);
        }
    }

    return sqrt(diffSq);
}

neuron *FindMostSimilar(eSNN *eSNN_net, neuron *n_c) { //find mos similar neurons in terms of synaptic weights

    double minDist = CalculateDistance(n_c->s_weights, eSNN_net->OutputNeurons[0]->s_weights);
    double minIdx = 0;

    if (eSNN_net->OutputNeurons.size() > 1) {
        for (int i = 1; i < eSNN_net->OutputNeurons.size(); i++) {
            double dist = CalculateDistance(n_c->s_weights, eSNN_net->OutputNeurons[i]->s_weights);
            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
    }
    return eSNN_net->OutputNeurons[minIdx];
}

void UpdateRepository(eSNN *eSNN_net, neuron *n_c, double Dub) { //Update neuron n_s in output repository

    neuron *n_s;

    if (eSNN_net->OutputNeurons.size() > 0) {
        n_s = FindMostSimilar(eSNN_net, n_c);
    }

    if (eSNN_net->OutputNeurons.size() > 0 && CalculateDistance(n_c->s_weights, n_s->s_weights) < simTr * Dub) {
        for (int k = 0; k < n_s->s_weights.size(); k++) {
            for (int j = 0; j < n_s->s_weights[k].size(); j++) {
                n_s->s_weights[k][j] = (n_c->s_weights[k][j] + n_s->s_weights[k][j] * n_s->M) / (n_s->M + 1);
            }
        }

        n_s->outputValue = (n_c->outputValue + n_s->outputValue * n_s->M) / (n_s->M + 1);
        n_s->M += 1;
        delete n_c;
    } else {
        eSNN_net->OutputNeurons.push_back(n_c);
    }
}

double CalculateUpperBound(eSNN *eSNN_net) {
    vector<double> v;
    vector<double> v2;

    for (int k = 0; k < eSNN_net->Attribute.size(); k++) {
        if(eSNN_net->Attribute[k]->type == 0 || eSNN_net->Attribute[k]->type == 1) {
        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            double sum = 0;
            sum += pow(mod, j) - pow(mod, NIsize - j - 1);
            //sum += pow(mod, j) - pow(mod, NIsize - j - 1);
            v.push_back(sum);
        }
        }
        else if(eSNN_net->Attribute[k]->type == 2)
        {
            double sum = 2 * pow(mod, 2);
            v2.push_back(sum);
        }
    }

    double diffSq = 0.0;
    for (int j = 0; j < v.size(); j++) {
        diffSq += pow(v[j], 2);
    }

    double sum = 0.0;
    for (int j = 0; j < v2.size(); j++) {
        sum += v2[j];
    }

    return sqrt(diffSq + sum);
}

double PredictValue(eSNN *eSNN_net) {
    for (int i = 0; i < eSNN_net->OutputNeurons.size(); i++) {
        eSNN_net->OutputNeurons[i]->PSP = 0;
    }


    for (int l = 0; l < eSNN_net->Attribute.size(); l++) {
        for (int j = 0; j < eSNN_net->Attribute[l]->InputNeurons.size(); j++) {
            for (int i = 0; i < eSNN_net->OutputNeurons.size(); i++) {
                if (eSNN_net->Attribute[l]->InputNeurons[j]->order != 0) {
                    eSNN_net->OutputNeurons[i]->PSP += eSNN_net->OutputNeurons[i]->s_weights[l][j] *
                                                       pow(mod, eSNN_net->Attribute[l]->InputNeurons[j]->order);
                } else {
                    eSNN_net->OutputNeurons[i]->PSP += 0;
                }

            }
        }
    }



    double maxPSP = -1;
    double maxVals;
    int countMax = 0;

    for (int i = 0; i < eSNN_net->OutputNeurons.size(); i++) {
        if (maxPSP < eSNN_net->OutputNeurons[i]->PSP) {
            maxVals = eSNN_net->OutputNeurons[i]->outputValue * eSNN_net->OutputNeurons[i]->M;
            countMax = eSNN_net->OutputNeurons[i]->M;
            maxPSP = eSNN_net->OutputNeurons[i]->PSP;
        } else if (maxPSP == eSNN_net->OutputNeurons[i]->PSP) {
            maxVals += eSNN_net->OutputNeurons[i]->outputValue * eSNN_net->OutputNeurons[i]->M;
            countMax += eSNN_net->OutputNeurons[i]->M;
        }
    }



    return (maxVals / (double) countMax);
}

void eSNN_Learn(eSNN *eSNN_net, Dataset * trainingDataset) { //main eSNN procedure

    //InitializeInputLayer(eSNN_net, trainingDataset, testDataset);

    double Dub = CalculateUpperBound(eSNN_net);


    //cout << "Phase 1" << endl;

    for (int i = 0; i < trainingDataset->att[0].values.size(); i++) {

       // if (i % 100 == 0)
         //   cout << "#";

        Example exmp;

        for (int j = 0; j < trainingDataset->att.size(); j++) {
            exmp.attributes.push_back(trainingDataset->att[j].type);
            exmp.values.push_back(trainingDataset->att[j].values[i]);
        }

        CalculateOrder(eSNN_net, exmp);

        neuron *n_c = new neuron;
        InitializeNeuron(eSNN_net, n_c, trainingDataset->realValues[i], i);
        UpdateRepository(eSNN_net, n_c, Dub);

    }
}

double eSNN_Predict(eSNN *eSNN_net, Example e) {

    CalculateOrder(eSNN_net, e);

    return PredictValue(eSNN_net);
}

void ClearStructures(vector<eSNN *> eSNN_Nets, vector<Dataset *> samples, Dataset * trainingDataset, Dataset * testDataset)
{

    for(int i = 0; i < eSNN_Nets.size(); i++)
    {
        for(int j = 0; j < eSNN_Nets[i]->OutputNeurons.size(); j++)
        {
            delete eSNN_Nets[i]->OutputNeurons[j];
        }

        for(int j = 0; j < eSNN_Nets[i]->Attribute.size(); j++) {
            for (int k = 0; k < eSNN_Nets[i]->Attribute[j]->InputNeurons.size(); k++) {
                delete eSNN_Nets[i]->Attribute[j]->InputNeurons[k];
            }

            delete eSNN_Nets[i]->Attribute[j];
        }

        delete eSNN_Nets[i];
    }



    for(int i = 0; i < samples.size(); i++)
    {
        delete samples[i];
    }

    delete trainingDataset;
    delete testDataset;

}
