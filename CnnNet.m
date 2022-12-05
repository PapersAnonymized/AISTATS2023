classdef CnnNet < BaseNet & CNNInputNet

    properties

    end

    methods
        function net = CnnNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, ini_rate, max_epoch);
            net = net@CNNInputNet();

            net.name = "cnn";
            
        end



        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@CNNInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            c_in = 1;
    f_h = 5; 
    f_n = 16; 
    f_s = 1;
    
    layers = [
        imageInputLayer([net.m_in c_in 1],'Normalization','none','Name','Input')
        convolution2dLayer([f_h c_in], f_n, 'Stride', [f_s 1],'Name','Conv1')
        %maxPooling2dLayer([p_s 1], 'Stride', [p_s 1],'Name','Pool1')
        flattenLayer('Name','Flat')
        fullyConnectedLayer(net.k_hid1,'Name','Full1')
        reluLayer('Name','Relu1')
        fullyConnectedLayer(net.k_hid2,'Name','Full2')
        reluLayer('Name','Relu2')
        fullyConnectedLayer(net.n_out,'Name','FullC')
        regressionLayer
    ];

            net.lGraph = layerGraph(layers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end



        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@CNNInputNet(net, i, X, Y);

        end


    end

end