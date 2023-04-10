# Bagging-ensemble-eSNN

# Author: Piotr Maciąg, Warsaw University of Technology,
# Contanct: piotr.maciag@pw.edu.pl

## Description

The software contains the implementation of eSNN based predictor for time series that uses the bagging ensembe technique in order to make preidctions more accurate
than other single eSNN network, SNNTorch package and other tested prediction methods.

The repository contains the following Directories/Files

* Datasets - the input datasets containing files for which the prediction is to be made. The datasets have to be split into training and testing parts. Each file needs to contain a dataframe, 
in which each column corresponds to a single feature time series (such as historical pollution, weather attributes). The rows of the dataframe denote timepoints of observations: in the provided dataset each row contains values of selected attributes for a single hour of measurement.
* Results - the output directory into which the resulting files are saved.
* eSNN.h, eSNN.cpp - the files containing implementation of our bagging method.
* LoadData.h, LoadData.cpp - the files containing functions for datasets loading and preprocessing.
* main.cpp - the main file performing prediction over datasets.

