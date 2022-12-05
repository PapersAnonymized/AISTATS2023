classdef LstmNet < BaseNet & RNNInputNet

    properties

    end

    methods
        function net = LstmNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, ini_rate, max_epoch);
            net = net@RNNInputNet();

            net.name = "lstm";

        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@RNNInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            sLayers = [
                sequenceInputLayer(1)
                lstmLayer(net.k_hid1)%, 'OutputMode','last')
                lstmLayer(net.k_hid2)%, 'OutputMode','last')
                fullyConnectedLayer(1)
                regressionLayer
            ];

            net.lGraph = layerGraph(sLayers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@RNNInputNet(net, i, X, Y);
 
        end

    end

end