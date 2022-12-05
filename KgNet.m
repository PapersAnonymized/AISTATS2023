classdef KgNet < BaseNet & MLPInputNet

    properties

    end

    methods
        function net = KgNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, ini_rate, max_epoch);
            net = net@MLPInputNet();

            net.name = "kgate";

            

        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@MLPInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

            oLayers = [
        featureInputLayer(net.m_in,'Name','inputFeature')
        additionLayer(2,'Name','fcAgate')
        multiplicationLayer(2,'Name','fcMgate')
        fullyConnectedLayer(net.k_hid1,'Name','fcHidden')
        additionLayer(2,'Name','fcAgate2')
        multiplicationLayer(2,'Name','fcMgate2') 
        fullyConnectedLayer(net.k_hid2,'Name','fcHidden2')
        fullyConnectedLayer(net.n_out,'Name','fcOut')
        regressionLayer('Name','regOut')
    ];
    cgraph = layerGraph(oLayers);


    % Allow Linear Transform and Sigmoid activation
    aLayers = [
        fullyConnectedLayer(net.m_in,'Name','fcSig')
        sigmoidLayer('Name','sigAllow')
    ];
    aLayers2 = [
        fullyConnectedLayer(net.k_hid1,'Name','fcSig2')
        sigmoidLayer('Name','sigAllow2')
    ];
    cgraph = addLayers(cgraph, aLayers);
    cgraph = addLayers(cgraph, aLayers2);


    % Update Linear Transform and Tanh activation
    dLayers = [
        fullyConnectedLayer(net.m_in,'Name','fcTanh')
        tanhLayer('Name','tanhUpdate')
    ];
    dLayers2 = [
        fullyConnectedLayer(net.k_hid1,'Name','fcTanh2')
        tanhLayer('Name','tanhUpdate2')
    ];
    cgraph = addLayers(cgraph, dLayers);
    cgraph = addLayers(cgraph, dLayers2);


    % Update Cacade Linear Transform and Hadfamard join with Tanh
    nLayers = [
        fullyConnectedLayer(net.m_in,'Name','fcNorm')
        multiplicationLayer(2,'Name','normMgate')
    ];
    nLayers2 = [
        fullyConnectedLayer(net.k_hid1,'Name','fcNorm2')
        multiplicationLayer(2,'Name','normMgate2')
    ];
    cgraph = addLayers(cgraph, nLayers);
    cgraph = addLayers(cgraph, nLayers2);


    % Cascade-conneect Allow path to main trunk via Haddamard product
    cgraph = connectLayers(cgraph, 'inputFeature', 'fcSig');
    cgraph = connectLayers(cgraph,'sigAllow','fcMgate/in2');

    cgraph = connectLayers(cgraph, 'inputFeature', 'fcSig2');
    cgraph = connectLayers(cgraph,'sigAllow2','fcMgate2/in2');


    % Cascade-connect Update path to main trunk via Addition
    cgraph = connectLayers(cgraph, 'inputFeature', 'fcTanh');
    cgraph = connectLayers(cgraph, 'inputFeature', 'fcTanh2');

    cgraph = connectLayers(cgraph, 'inputFeature', 'fcNorm');
    cgraph = connectLayers(cgraph,'tanhUpdate','normMgate/in2');
    cgraph = connectLayers(cgraph,'normMgate','fcAgate/in2');

    cgraph = connectLayers(cgraph, 'inputFeature', 'fcNorm2');
    cgraph = connectLayers(cgraph,'tanhUpdate2','normMgate2/in2');
    cgraph = connectLayers(cgraph,'normMgate2','fcAgate2/in2');

    
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

            net = Train@MLPInputNet(net, i, X, Y);
        end

        
    end
end