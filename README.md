# AISTATS2023
Code for paper that explores cross-training of ANN models as a mean of supporting/disproving Effective Market Hypotesis

List of files:
 - Main program:
w_series_generic.m 

To configure, find and modify the following fragments:

%% Load the data, initialize partition pareameters (model name)
%saveDataPrefix = 'nasdaq0704_';
%saveDataPrefix = 'dj0704_';
%saveDataPrefix = 'nikkei0704_';
saveDataPrefix = 'dax0704_';

%% Data name (may be different than model)
%dataFile = 'nasdaq_1_3_05-1_28_22.csv';
%dataFile = 'dj_1_3_05-1_28_22.csv';
dataFile = 'nikkei_1_4_05_1_31_22.csv';
%dataFile = 'dax_1_3_05_1_31_22.csv';

%Input data directory
dataDir = '~/data/STOCKS';



 - Libraries:
   * ANN model objects
   
   * Custom ANN layers
   
   * Training and test sets building


 - Accuracy metrics calculation script:
\
