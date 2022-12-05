classdef ReluVNet < BaseNetV & MLPInputNetV

    properties

    end

    methods
        function net = ReluVNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNetV(m_in, n_out, ini_rate, max_epoch);
            net = net@MLPInputNetV();

            net.name = "reluV";

        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@MLPInputNetV(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            layers = [
                featureInputLayer(net.m_ine)
                fullyConnectedLayer(net.k_hid1)
                reluLayer
                fullyConnectedLayer(net.k_hid2)
                reluLayer
                fullyConnectedLayer(net.n_oute)
                vRegression('vRegression', net.n_outv)
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

            net = Train@MLPInputNetV(net, i, X, Y);
        end

        
    end
end