classdef GmdhNet < BaseNet & MLPInputNet

    properties
        accTarget1 
        accTarget2 
        accRel1 
        accRel2 
        lMax
        freeze
        max_neuro1
        max_neuro2 
    end

    methods
        function net = GmdhNet(m_in, n_out, accTarget1, accTarget2, accRel1, accRel2, lMax, freeze, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, ini_rate, max_epoch);
            net = net@MLPInputNet();

            net.name = "gmdh";

            net.accTarget1 = accTarget1;
            net.accTarget2 = accTarget2;
            net.accRel1 = accRel1;
            net.accRel2 = accRel2;
            net.lMax = lMax;
            net.freeze = freeze;
        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@MLPInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            net.max_neuro1 = floor(net.k_hid1/net.n_out);
            net.max_neuro2 = floor(net.k_hid2/net.n_out);
        end



        function net = Train(net, i, X, Y)

            fprintf('Training %s Reg net %d\n', net.name, i); 

                                    sOptions = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', net.mb_size, ...
'InitialLearnRate',net.ini_rate*2, ...
'MaxEpochs',net.max_epoch/2);

sOptions2 = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', net.mb_size, ...
'InitialLearnRate',net.ini_rate*2, ...
'MaxEpochs',net.max_epoch/2);

 sOptionsFinal = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', net.mb_size, ...
'InitialLearnRate',net.ini_rate, ...
'MaxEpochs',net.max_epoch/2);  

    % Start growing GMDH net
    prevLayerName = 'inputFeature';
    oLayers = [
        featureInputLayer(net.m_in, 'Name', prevLayerName)
    ];
    cgraph = layerGraph(oLayers);


    fprintf('Training 1st layer %s net %d\n', net.name, i);

    % Start from first GMDH layer
    ll = 1;
    % Target accuracy
    %accTarget = 0.4;
    % Stale accuracy chage threshol
    dAccMin = 0.001;
    % Maximal polinomial length (number of chained gmdh layer) 
    %lMax = 3;

    min_neuro1 = 0;
    min_neuro2 = 0;

    % Number of velocity outputs
    %n_outv = n_oute - n_out;


    [cgraph, regNet, ll, k_hid1_real, curGMDHLayerName, curGMDHRegressionName] =... 
        gmdhLayerGrowN(cgraph, prevLayerName, sOptions, X, Y, net.m_in, i, net.n_out, ll,...
        net.accTarget1, net.accRel1, net.max_neuro1, min_neuro1, dAccMin, net.lMax, net.freeze);
    prevLayerName = curGMDHLayerName;

    if(k_hid1_real > 1)
        % Sum all polynomial candidates into last fully connected layuer and use
        % standard Regression instead of GMDH
        fullyConnectdMidLayerName = 'fullyConnectedLayerMid';
        fullyConnectdMidLayer = fullyConnectedLayer(k_hid1_real, 'Name', fullyConnectdMidLayerName);
        cgraph = addLayers(cgraph, fullyConnectdMidLayer);
        cgraph = connectLayers(cgraph, curGMDHLayerName, fullyConnectdMidLayerName);

        prevLayerName = fullyConnectdMidLayerName;


        fprintf('Training 2nd layer %s net %d\n', net.name, i);

        % Build second GMDH layer
        ll = ll + 1;
        % Target accuracy
        %accTarget = 0.3; %0.015;% 0.05; 0.08;
        % Stale accuracy chage threshol
        dAccMin = 0.001;
        % Maximal polinomial length (number of chained gmdh layer) 
        %lMax = 3;

        [cgraph, regNet, ll, k_hid2_real, curGMDHLayerName, curGMDHRegressionName] =... 
            gmdhLayerGrowN(cgraph, prevLayerName, sOptions2, X, Y, k_hid1_real, i, net.n_out, ll,...
            net.accTarget2, net.accRel2, net.max_neuro2, min_neuro2, dAccMin, net.lMax, net.freeze);
        prevLayerName = curGMDHLayerName;

        if(k_hid2_real > 1)
            fullyConnectdLastLayerName = 'fullyConnectedLayerLast';
            fullyConnectdLastLayer = fullyConnectedLayer(k_hid2_real, 'Name', fullyConnectdLastLayerName);
            cgraph = addLayers(cgraph, fullyConnectdLastLayer);
            cgraph = connectLayers(cgraph, curGMDHLayerName, fullyConnectdLastLayerName);

            fullyConnectedOutLayerName = 'fcOut';
            fullyConnectedOutLayer = fullyConnectedLayer(net.n_out,'Name',fullyConnectedOutLayerName);
            cgraph = addLayers(cgraph, fullyConnectedOutLayer);
            cgraph = connectLayers(cgraph, fullyConnectdLastLayerName, fullyConnectedOutLayerName);

            prevLayerName = fullyConnectedOutLayerName;
        end
    end

    regressionLayerName = "regOut";
    regressionLayerLast = regressionLayer('Name', regressionLayerName);
    %regressionLayerLast = vRegression(regressionLayerName, n_outv);

    cgraph = replaceLayer(cgraph, curGMDHRegressionName, regressionLayerLast);
    cgraph = connectLayers(cgraph, prevLayerName, regressionLayerName);

    
            net.lGraph = cgraph;

            tNet = trainNetwork(X(:,:,i)', Y(:,:,i)', cgraph, sOptionsFinal);

            %net = Train@MLPInputNet(net, X, Y);

            %tNet = trainNetwork(X', Y', net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph; 
        end

        
    end
end