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

% input dimesion (days)
m_in = 30;
% Try different output dimensions (days)
n_out = 30;

%Make sure that uncommented model name corresponds to ANN make wrapper (to do to make it transparent)
modelName = 'gmdhg';
regNet = makeGMDHNet(i, m_in, n_out, floor(sqrt(2*k_hid1/n_out)), floor(sqrt(2*k_hid2/n_out)), 0, 0, 0.01, 0.01, 1, mb_size, X, Y, 1);


 - Libraries:
   * ANN models
BaseNet.m
CNNInputNet.m
LinRegInputNet.m
MLPInputNet.m
RNNInputNet.m
RNNSeqInputNet.m

AnnNet.m
CnnCascadeNet.m
CnnNet.m
GmdhNet.m
KgNet.m
LinRegNet.m
LstmNet.m
LstmSeqNet.m
RbfNet.m
ReluNet.m
SigNet.m
TanhNet.m
TransNet.m

makeANNNet.m
makeConv2DCascadeRegNet.m
makeConv2DRegNet.m
makeGMDHNet.m
makeGRUSeqNet.m
makeKGNet.m
makeLSTMNet.m
makeLSTMSeqNet.m
makeLinReg.m
makeRBFNet.m
makeReLUNet.m
makeSigNet.m
makeTG2Net.m
makeTGNet.m
makeTanhNet.m
makeTransNet.m
   
   * Custom ANN layers
GaussianRBFLayer.m
gmdhLayerGrowN.m
gmdhLayerN.m
gmdhRegression.m
transformerLayer.m
vRegression.m
   
   * Training and test sets building
w_series_generic_test_seq_tensors.m
w_series_generic_test_seq_vtensors.m
w_series_generic_test_tensors.m
w_series_generic_test_vtensors.m
w_series_generic_train_seq_tensors.m
w_series_generic_train_seq_vtensors.m
w_series_generic_train_tensors.m
w_series_generic_train_vtensors.m
w_series_spots.m
w_seriesv_test_tensors.m
w_seriesv_train_tensors.m

   * Auxiliary: scaling, error, graph routines
w_series_generic_calc_mape.m
w_series_generic_calc_rmse.m
w_series_generic_calcv_mape.m
w_series_generic_calcv_rmse.m
w_series_generic_err_graph.m
w_series_generic_errv_graph.m
w_series_generic_minmax_rescale.m
w_series_generic_minmax_scale.m

 - Accuracy metrics calculation script:
pval.R

 - Input data files (stock indexes
nasdaq_1_3_05-1_28_22.csv
dax_1_3_05_1_31_22.csv
dj_1_3_05-1_28_22.csv
nikkei_1_4_05_1_31_22.csv

  - Detailed output files (errors per session)
mape.1.<model>.<training index>.<test index>.*.txt
