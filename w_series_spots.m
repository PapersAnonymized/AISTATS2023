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
%saveDataPrefix = 'dj0704_';
%saveDataPrefix = 'nikkei0704_';
%saveDataPrefix = 'dax0704_';

%saveDataPrefix = '7203toyota_';
%saveDataPrefix = 'nvidia_';
%saveDataPrefix = 'tsla4030_';

%saveDataPrefix = 'AirPassengers1_114_30_';
%saveDataPrefix = 'sun_1_';
saveDataPrefix = 'SN_y_tot_V2.0_spots_4030_';

save_regNet_fileT = '~/data/ws_';

%dataFile = 'wse_data.csv';
%dataFile = 'nasdaq_1_3_05-1_28_22.csv';
%dataFile = 'dj_1_3_05-1_28_22.csv';
%dataFile = 'nikkei_1_4_05_1_31_22.csv';
%dataFile = 'dax_1_3_05_1_31_22.csv';

%dataFile = '7203toyota_1_4_05_1_31_22';
%dataFile = 'nvidia_1_3_05_1_28_22';
%dataFile = 'tsla_6_30_10_1_28_22.csv';

%dataFile = 'AirPassengers1.csv';
%dataFile = 'sun_1.csv';
dataFile = 'SN_y_tot_V2.0_spots.csv';

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
l_sess = l_whole;

% Only for 1 whole session (otherwise, comment out)
%l_sess = l_whole;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess);


% Normalization flag
norm_fl = 0; %0;
    
% Reformat sequence into input training tensor X of m_in observations dimension, 
% label Y (predicted values) of n_out dimension, on training seesion of
% length l_sess (n_sess of them), and min-max boundaries B for each observation, 
% such that number of observations in a session, including training label sequences
% do not touch test period
[X, Xc, Xr, Ys, Y, B, XI, C, k_ob] = w_series_generic_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl);
[Xl, Yl, Bl, k_lob] = w_series_generic_train_seq_tensors(M, l_sess, n_sess, norm_fl);

% Fit ann into minimal loss function (SSE)
mult = 1;
%k_hid = floor(mult * m_ine);
k_hid1 = floor(mult * (m_in + 1));
k_hid2 = floor(mult * (2*m_in + 1));

regNets = cell(n_sess);

%modelName = 'reg';
%modelName = 'ann';
%modelName = 'relu';
%modelName = 'sig';
%modelName = 'tanh';
%modelName = 'lstm_seq';
%%modelName = 'gru_seq';
%modelName = 'lstm';
%modelName = 'gmdhg';
%modelName = 'gmdhg1';
modelName = 'kgate';
%modelName = 'rbfg';
%%modelName = 'cnn';
%modelName = 'cnnc';

%% regNet parameters

mb_size = 2^floor(log2(k_ob)); %32
       

%% Train or pre-load regNets
for i = 1:n_sess

    save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(norm_fl), '_', int2str(n_sess), '.mat');
    if isfile(save_regNet_file)
        fprintf('Loading net %d from %s\n', i, save_regNet_file);
        load(save_regNet_file, 'regNet');
    else
        clear('regNet');
    end


    if exist('regNet') == 0

        %regNet = makeLinReg(i, m_in, n_out, n_sess, Xr, Y);
        %regNet = makeANNNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeReLUNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeSigNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeTanhNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeLSTMSeqNet(i, k_hid1, k_hid2, mb_size, X, Ys);
        %%regNet = makeGRUSeqNet(i, k_hid1, k_hid2, mb_size, X, Ys);
        %regNet = makeLSTMNet(i, k_hid1, k_hid2, mb_size, Xl, Yl);
        %regNet = makeGMDHNet(i, m_in, n_out, floor(sqrt(2*k_hid1/n_out)), floor(sqrt(2*k_hid2/n_out)), 0, 0, 0.01, 0.01, 1, mb_size, X, Y, 1);
        %%regNet = makeGMDHNet(i, m_in, n_out, floor(sqrt(2*k_hid1/n_out))+1, floor(sqrt(2*k_hid2/n_out))+1, 0, 0, 0.01, 0.001, 1, mb_size, X, Y, 1);
        %regNet = makeRBFNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        regNet = makeKGNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %%regNet = makeConv2DRegNet(i, m_in, 1, n_out, k_hid1, k_hid2, mb_size, Xc, Y);
        %regNet = makeConv2DCascadeRegNet(i, m_in, 1, n_out, k_hid1, k_hid2, mb_size, Xc, Y);

        save(save_regNet_file, 'regNet');
    end

    regNets{i} = regNet;

    %clear('cgraph');
    clear('regNet');

end

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

if ~strcmp(modelName,'lstm')
    [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_tensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, 0);
else
    [X2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_seq_tensors(M, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, k_lob, 0);
end


%% test
if (~strcmp(modelName,'lstm_seq')) && (~strcmp(modelName,'gru_seq')) && (~strcmp(modelName,'lstm')) &&...
       (~strcmp(modelName,'cnn')) && (~strcmp(modelName,'cnnc')) && (~strcmp(modelName,'reg'))

    for i = 1:t_sess-sess_off
        predictedScores = predict(regNets{i}, X2(:, :, i)');
        Y2(:, :, i) = predictedScores';
    end

elseif (strcmp(modelName, 'reg'))

    % test fully fit output sequences
    for i = 1:t_sess
        Xr2i = Xr2(:, :, i);
        W1i = regNets{i}(:, : ,i);
        Y2(:, :, i) = W1i * Xr2i;
    end

elseif (strcmp(modelName, 'cnn')) || (strcmp(modelName, 'cnnc'))
    
    for i = 1:t_sess-sess_off
        predictedScores = predict(regNets{i}, Xc2(:, :, :, :, i));
        Y2(:, :, i) = predictedScores';
    end

elseif (strcmp(modelName, 'lstm_seq')) || (strcmp(modelName, 'gru_seq'))

    % LSTM Seq test
    for i = 1:t_sess-sess_off
        
        % Now feeding test data
        for j = 1:k_tob
            lstmNet = resetState(regNets{i});
            % Trick - go through all input sequence to get the first next
            % prediction outside the seqence, discarding intermediate
            % predictions
            %or k = 1:m_in
            %    [lstmNet, Y2(1, j, i)] = predictAndUpdateState(lstmNet, X2(k, j, i));
            %end
            [lstmNet, Y2s(:, j, i)] = predictAndUpdateState(lstmNet, X2(:, j, i)');
            Y2(1, j, i) = Y2s(end, j, i);

            % Continue predicting further output points based on previous
            % predicvtion
            for l = 2:n_out
                %Y2s(1:m_in-l+1, j, i) = X2(l:end, j, i);
                [lstmNet, Y2s(:, j, i)] = predictAndUpdateState(lstmNet, Y2s(:, j, i)');
                Y2(l, j, i) = Y2s(end, j, i);
            end

        end
    end

else

    for i = 1:t_sess-sess_off
        
        % Now feeding test data
        for j = 1:k_tob
            lstmNet = resetState(regNets{i});
            % Trick - go through all input sequence to get the first next
            % prediction outside the seqence, discarding intermediate
            % predictions
            for k = 1:k_lob+1
                [lstmNet, Y2(1, j, i)] = predictAndUpdateState(lstmNet, X2(k, j, i));
            end
            %[lstmNet, Y2s(:, j, i)] = predictAndUpdateState(lstmNet, X2(:, j, i)');
            %Y2(1, j, i) = Y2s(end, j, i);

            % Continue predicting further output points based on previous
            % predicvtion
            for l = 2:n_out
                [lstmNet, Y2(l, j, i)] = predictAndUpdateState(lstmNet, Y2(l-1, j, i)');
            end

        end
    end
end


%% re-scale in observation bounds
if(norm_fl)
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            Y2(:, j, i) = w_series_generic_minmax_rescale(Y2(:, j, i), Bt(1,j,i), Bt(2,j,i));
            Yh2(:, j, i) = w_series_generic_minmax_rescale(Yh2(:, j, i), Bt(1,j,i), Bt(2,j,i));
        end
    end
end


%% Calculate errors
[S2, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = w_series_generic_calc_mape(Y2, Yh2, n_out); 

fprintf('%s, trainN %s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f\n', modelName, saveDataPrefix, dataFile, norm_fl, m_in, n_out, n_sess, t_sess, S2);


[S2, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = w_series_generic_calc_rmse(Y2, Yh2, n_out); 

fprintf('%s, trainN %s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f\n', modelName, saveDataPrefix, dataFile, norm_fl, m_in, n_out, n_sess, t_sess, S2);

%% Error and Series Plot
%w_series2_err_graph(Y2, Yh2);
w_series_generic_err_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);

