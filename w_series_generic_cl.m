%% Clear everything 
clearvars -global;
clear all; close all; clc;

%% Mem cleanup
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end

%% Load the data, initialize partition pareameters
%saveDataPrefix = 'wse_';
%saveDataPrefix = 'nasdaq0704_';
saveDataPrefix = 'dj0704_';
%saveDataPrefix = 'nikkei0704_';
%saveDataPrefix = 'dax0704_';

%saveDataPrefix = '7203toyota_';
%saveDataPrefix = 'nvidia_';
%saveDataPrefix = 'tsla4030_';

%saveDataPrefix = 'AirPassengers1_114_30_';
%saveDataPrefix = 'sun_1_';
%saveDataPrefix = 'SN_y_tot_V2.0_spots_4030_';

save_regNet_fileT = '~/data/ws_';

%dataFile = 'wse_data.csv';
%dataFile = 'nasdaq_1_3_05-1_28_22.csv';
dataFile = 'dj_1_3_05-1_28_22.csv';
%dataFile = 'nikkei_1_4_05_1_31_22.csv';
%dataFile = 'dax_1_3_05_1_31_22.csv';

%dataFile = '7203toyota_1_4_05_1_31_22';
%dataFile = 'nvidia_1_3_05_1_28_22';
%dataFile = 'tsla_6_30_10_1_28_22.csv';

%dataFile = 'AirPassengers1.csv';
%dataFile = 'sun_1.csv';
%dataFile = 'SN_y_tot_V2.0_spots.csv';

dataDir = '~/data/STOCKS';
dataFullName = strcat(dataDir,'/',dataFile);

Me = readmatrix(dataFullName);
[l_whole_ex, ~] = size(Me);

min(Me)
max(Me)
mean(Me)
std(Me)


% input dimesion (days)
m_in = 30;
% Try different output dimensions (days)
n_out = 30;

% Or no future
M = Me;
% Leave space for last full label
l_whole = l_whole_ex - n_out;

% Break the whole dataset in training sessions,
% Set training session length (with m_in datapoints of length m_in), 
l_sess = 3*m_in + n_out;

% Only for 1 whole session (otherwise, comment out)
%l_sess = l_whole;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess); %34; - nikkei cross-training


% Normalization flag
%norm_fl = 1;

ini_rate = 0.01; 
max_epoch = 1000;
    

%% regNet parameters
% Fit ann into minimal loss function (SSE)
%mult = 1;
%k_hid1 = floor(mult * (m_in + 1));
%k_hid2 = floor(mult * (2*m_in + 1));

mb_size = 32;

regNets = cell([n_sess, 1]);

       

%% Train or pre-load regNets
for i = 1:n_sess

    %save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(norm_fl), '_', int2str(n_sess), '.mat');
    %% cross-training
    %%save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(norm_fl), '_35', '.mat');
    %if isfile(save_regNet_file)
    %    fprintf('Loading net %d from %s\n', i, save_regNet_file);
    %    load(save_regNet_file, 'regNet');
    %else

        %norm_fl = 1;
        %regNet = SigNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = SigVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = TanhNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = TanhVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = RbfNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = RbfVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = TransNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = TransVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = LstmNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = LstmVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %%regNet = LstmSeqNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = GmdhNet(m_in, n_out, 0, 0, 0.005, 0.005, 1, 1, mb_size, ini_rate, max_epoch);
        %regNet = GmdhVNet(m_in, n_out, 0, 0, 0.005, 0.005, 1, 1, mb_size, ini_rate, max_epoch);

        norm_fl = 0;
        %regNet = LinRegNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = AnnNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = AnnVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = ReluNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = ReluVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = KgNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = KgVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = CnnNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        regNet = CnnVNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = CnnCascadeNet(m_in, n_out, mb_size, ini_rate, max_epoch);
        %regNet = CnnCascadeVNet(m_in, n_out, mb_size, ini_rate, max_epoch);

        modelName = regNet.name;

        [regNet, X, Y, B, k_ob] = regNet.TrainTensors(M, m_in, n_out, l_sess, n_sess, norm_fl);

        regNet = regNet.Train(i, X, Y);
        %save(save_regNet_file, 'regNet');
    %end

    regNets{i} = regNet;

end
clear('regNet');

%% Test parameters 
% the test input period - same as training period, to cover whole data
l_test = l_sess;

% Test from particular training session
sess_off = 0;
% additional offset after training sessions (usually for the future forecast)
offset = 0;

% Left display margin
l_marg = 1;

%% Test parameters for one last session

% Left display margin
%l_marg = 4100;

% Future session
%M = zeros([l_whole_ex+n_out, 1]);
%M(1:l_whole_ex) = Me;
%[l_whole, ~] = size(M);

% Last current session
%l_whole = l_whole_ex;

% Fit last testing session at the end of data
%offset = l_whole - n_sess*l_sess - m_in - n_out;

% Test from particular training session
%sess_off = n_sess-1;


%% For whole-through test, comment out secion above
% Number of training sessions with following full-size test sessions 
t_sess = floor((l_whole - l_test - m_in) / l_sess);

[X2, Y2, Yh2, Bt, k_tob] = regNets{1}.TestTensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl);

%% test
[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, t_sess, sess_off, k_tob);


%% re-scale in observation bounds
if(norm_fl)
    [Y2, Yh2] = regNets{1}.ReScale(Y2, Yh2, Bt, t_sess, sess_off, k_tob);
end


%% Calculate errors
[S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = regNets{1}.Calc_mape(Y2, Yh2, n_out); 

fprintf('%s, trainN %s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f+-%f MaxAPErr %f+-%f\n', modelName, saveDataPrefix, dataFile, norm_fl, m_in, n_out, n_sess, t_sess, S2, S2Std, mean(ma_err), std(ma_err));


[S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = regNets{1}.Calc_rmse(Y2, Yh2, n_out); 

fprintf('%s, trainN %s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f+-%f MaxRSErr %f+-%f\n', modelName, saveDataPrefix, dataFile, norm_fl, m_in, n_out, n_sess, t_sess, S2Q, S2StdQ, mean(ma_errQ), std(ma_errQ));

%%
% Write per-session errors to a file
fd = fopen( strcat('wsg_cl_mape_', saveDataPrefix, dataFile, '.', regNets{1}.name, '.txt'),'w' );

fprintf(fd, "Sess MeanPE MaxPE MeanRSE MaxRSE\n");

for i = 1:t_sess-sess_off
    fprintf(fd, "%d %f %f %f %f\n", i, S2s(i), ma_err(i), S2sQ(i), ma_errQ(i));
end
fclose(fd);
%% Error and Series Plot
%w_series2_err_graph(Y2, Yh2);
regNets{1}.Err_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);

