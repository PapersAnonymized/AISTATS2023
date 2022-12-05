classdef TransVNet < BaseNetV & MLPInputNetV

    properties

    end

    methods
        function net = TransVNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNetV(m_in, n_out, ini_rate, max_epoch);
            net = net@MLPInputNetV();

            net.name = "transV";

        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@MLPInputNetV(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            oLayers = [
                featureInputLayer(net.m_ine,'Name','inputFeature')
                additionLayer(2,'Name','fcAgate')
                fullyConnectedLayer(net.k_hid1,'Name','fcHidden')
                additionLayer(2,'Name','fcAgate2')
                fullyConnectedLayer(net.k_hid2,'Name','fcHidden2')
                fullyConnectedLayer(net.n_oute,'Name','fcOut')
                vRegression('vRegression', net.n_outv)
            ];
            cgraph = layerGraph(oLayers);

            tLayers = [
                transformerLayer(net.m_ine,'trans')
                fullyConnectedLayer(net.m_ine,'Name','fcTrans')
            ];

            cgraph = addLayers(cgraph, tLayers);
    
            cgraph = connectLayers(cgraph, 'inputFeature', 'trans');
            cgraph = connectLayers(cgraph,'fcTrans','fcAgate/in2');


            t2Layers = [
                transformerLayer(net.k_hid1,'trans2')
                fullyConnectedLayer(net.k_hid1,'Name','fcTrans2')
            ];

            cgraph = addLayers(cgraph, t2Layers);
    
            cgraph = connectLayers(cgraph, 'fcHidden', 'trans2');
            cgraph = connectLayers(cgraph,'fcTrans2','fcAgate2/in2');
    
            net.lGraph = cgraph;


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