classdef LstmSeqNet < BaseNet & RNNSeqInputNet

    properties

    end

    methods
        function net = LstmSeqNet(m_in, n_out, k_hid1, k_hid2, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, k_hid1, k_hid2, ini_rate, max_epoch);
            net = net@RNNSeqInputNet();

            net.name = "lstm_seq";

            sLayers = [
                sequenceInputLayer(1)
                lstmLayer(k_hid1)%, 'OutputMode','last')
                lstmLayer(k_hid2)%, 'OutputMode','last')
                fullyConnectedLayer(1)
                regressionLayer
            ];

            net.lGraph = layerGraph(sLayers);
        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@RNNSeqInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@RNNSeqInputNet(net, i, X, Y);
 
        end

    end

end