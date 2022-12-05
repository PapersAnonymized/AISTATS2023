classdef ReluNet < BaseNet & MLPInputNet

    properties

    end

    methods
        function net = ReluNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, ini_rate, max_epoch);
            net = net@MLPInputNet();

            net.name = "relu";



        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@MLPInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            layers = [
                featureInputLayer(net.m_in)
                fullyConnectedLayer(net.k_hid1)
                reluLayer
                fullyConnectedLayer(net.k_hid2)
                reluLayer
                fullyConnectedLayer(net.n_out)
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

            net = Train@MLPInputNet(net, i, X, Y);
        end

        
    end
end