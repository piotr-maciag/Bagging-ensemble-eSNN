//
// Created by Piotr Maciąg on 14/06/2021.
//


#include "eSNN.h"

int NIsize;
double simTr;
double mod = 0.9;
int Bins;


void InitializeInputLayerDist(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset) {
    for (int k = 0; k < trainingDataset->att.size(); k++) {

        inputAttribute *InputNeuronsVect = new inputAttribute;
        for (int j = 0; j < NIsize; j++) {
            inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
            InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
        }

        eSNN_net->Attribute.push_back(InputNeuronsVect);

        double max = -2000000;
        double min = 2000000;

        for (int i = 0; i < trainingDataset->att[k].values.size(); i++) {
            if (max < trainingDataset->att[k].values[i]) max = trainingDataset->att[k].values[i];
            if (min > trainingDataset->att[k].values[i]) min = trainingDataset->att[k].values[i];
        }

        for (int i = 0; i < testDataset->att[k].values.size(); i++) {
            if (max < testDataset->att[k].values[i]) max = testDataset->att[k].values[i];
            if (min > testDataset->att[k].values[i]) min = testDataset->att[k].values[i];
        }


        eSNN_net->Attribute[k]->I_min = min, eSNN_net->Attribute[k]->I_max = max;

        vector<int> neuronsPerBins;

        for (int i = 0; i < Bins; i++) {
            neuronsPerBins.push_back(1);
        }

        double widthBins = double(eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / double(Bins);

        vector<int> ValuesPerBin;

        for (int i = 0; i < Bins; i++) {
            ValuesPerBin.push_back(0);
        }

        for (int i = 0; i < trainingDataset->att[k].values.size(); i++) {
            for (int j = 1; j <= Bins - 1; j++) {
                if (trainingDataset->att[k].values[i] >= (min + (j - 1) * widthBins) &&
                    trainingDataset->att[k].values[i] < (min + (j) * widthBins)) {
                    ValuesPerBin[j - 1]++;
                }
            }
            if (trainingDataset->att[k].values[i] >= (min + (Bins - 1) * widthBins) &&
                trainingDataset->att[k].values[i] <= max) {
                ValuesPerBin[Bins - 1]++;
            }
        }


        int RestNeurons = NIsize - Bins;
        int r = NIsize - Bins;

        for (int i = 0; i < Bins - 1; i++) {
            neuronsPerBins[i] += floor(
                    (double(ValuesPerBin[i]) / double(trainingDataset->att[k].values.size())) * r);
            RestNeurons -= neuronsPerBins[i] + 1;
        }

        neuronsPerBins[Bins - 1] += ceil(
                (double(ValuesPerBin[Bins - 1]) / double(trainingDataset->att[k].values.size())) * r);

        for (int i = 0; i < Bins; i++) {
            int sum = 0;
            double widthPerBin = widthBins / neuronsPerBins[i];

            if (i > 0) sum += neuronsPerBins[i - 1];
            for (int j = 0; j < neuronsPerBins[i]; j++) {
                double mu = (min + (i) * widthBins) + (j + 1 - 0.5) * widthPerBin;
                eSNN_net->Attribute[k]->InputNeurons[sum + j]->mu = mu;
            }
        }

        eSNN_net->Attribute[k]->width =
                double(eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / double(NIsize);
    }
}

void CalculateOrderDist(eSNN *eSNN_net, Example exmp) {

    int min = (exmp.values.size() < eSNN_net->Attribute.size()) ? exmp.values.size() : eSNN_net->Attribute.size();

    for (int k = 0; k < min; k++) {
        int j;

        if (exmp.values[k] > eSNN_net->Attribute[k]->I_min && exmp.values[k] < eSNN_net->Attribute[k]->I_max &&
            eSNN_net->Attribute[k]->width != 0) {

            // j = floor((exmp.values[k] - eSNN_net->Attribute[k]->I_min) / eSNN_net->Attribute[k]->width) + 1;

            double minDist = abs(exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[0]->mu);
            j = 1;
            for (int i = 1; i < eSNN_net->Attribute[k]->InputNeurons.size(); i++) {
                if (minDist > abs(exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[i]->mu)) {
                    j = i + 1;
                    minDist = abs(exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[i]->mu);
                }
            }
        } else if (exmp.values[k] >= eSNN_net->Attribute[k]->I_max) {
            j = NIsize;
        } else if (exmp.values[k] <= eSNN_net->Attribute[k]->I_min) {
            j = 1;
        }

        eSNN_net->Attribute[k]->InputNeurons[j - 1]->rank = 0;

        int l, p;

        l = j - 1;
        p = j + 1;

        int rank = 0;
        double distL = 0;
        double distP = 0;


        while (l >= 1 || p <= NIsize) {
            if (l >= 1) distL = abs(eSNN_net->Attribute[k]->InputNeurons[l - 1]->mu - exmp.values[k]);
            if (p <= NIsize) distP = abs(eSNN_net->Attribute[k]->InputNeurons[p - 1]->mu - exmp.values[k]);


            if (l < 1 && p <= NIsize) {
                rank++;
                eSNN_net->Attribute[k]->InputNeurons[p - 1]->rank = rank;
                p++;
            } else if (l >= 1 && p > NIsize) {
                rank++;
                eSNN_net->Attribute[k]->InputNeurons[l - 1]->rank = rank;
                l--;
            } else if (l >= 1 && p <= NIsize) {
                if (distL < distP) {
                    rank++;
                    eSNN_net->Attribute[k]->InputNeurons[l - 1]->rank = rank;
                    l--;
                } else {
                    rank++;
                    eSNN_net->Attribute[k]->InputNeurons[p - 1]->rank = rank;
                    p++;
                }
            }
        }

        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            eSNN_net->Attribute[k]->InputNeurons[j]->order = eSNN_net->Attribute[k]->InputNeurons[j]->rank;
        }
    }
}

void InitializeInputLayer(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset) {
    for (int k = 0; k < trainingDataset->att.size(); k++) {
        if (trainingDataset->att[k].type == 0 || trainingDataset->att[k].type == 1) {
            inputAttribute *InputNeuronsVect = new inputAttribute;
            for (int j = 0; j < NIsize; j++) {
                inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
                InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
            }
            eSNN_net->Attribute.push_back(InputNeuronsVect);
        } else if (trainingDataset->att[k].type == 2) {

            inputAttribute *InputNeuronsVect = new inputAttribute;
            double maxTraining = (*max_element(trainingDataset->att[k].values.begin(),
                                               trainingDataset->att[k].values.end()));
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
        double maxTraining = (*max_element(trainingDataset->att[i].values.begin(),
                                           trainingDataset->att[i].values.end()));
        double minTraining = (*min_element(trainingDataset->att[i].values.begin(),
                                           trainingDataset->att[i].values.end()));

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
        if (eSNN_net->Attribute[k]->type == 0 || eSNN_net->Attribute[k]->type == 1) {
            for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
                double sum = 0;
                sum += pow(mod, j) - pow(mod, NIsize - j - 1);
                //sum += pow(mod, j) - pow(mod, NIsize - j - 1);
                v.push_back(sum);
            }
        } else if (eSNN_net->Attribute[k]->type == 2) {
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

void eSNN_Learn(eSNN *eSNN_net, Dataset *trainingDataset) { //main eSNN procedure

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

void
ClearStructures(vector<eSNN *> eSNN_Nets, vector<Dataset *> samples, Dataset *trainingDataset, Dataset *testDataset) {

    for (int i = 0; i < eSNN_Nets.size(); i++) {
        for (int j = 0; j < eSNN_Nets[i]->OutputNeurons.size(); j++) {
            delete eSNN_Nets[i]->OutputNeurons[j];
        }

        for (int j = 0; j < eSNN_Nets[i]->Attribute.size(); j++) {
            for (int k = 0; k < eSNN_Nets[i]->Attribute[j]->InputNeurons.size(); k++) {
                delete eSNN_Nets[i]->Attribute[j]->InputNeurons[k];
            }

            delete eSNN_Nets[i]->Attribute[j];
        }

        delete eSNN_Nets[i];
    }

    if (samples.size() != 1) {
        for (int i = 0; i < samples.size(); i++) {
            delete samples[i];
        }
    }

    delete trainingDataset;
    delete testDataset;

}
