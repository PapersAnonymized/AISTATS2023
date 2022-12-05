classdef CnnCascadeVNet < BaseNetV & CNNInputNetV

    properties

    end

    methods
        function net = CnnCascadeVNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNetV(m_in, n_out, ini_rate, max_epoch);
            net = net@CNNInputNetV();

            net.name = "cnncV";

        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@CNNInputNetV(net, M, m_in, n_out, l_sess, n_sess, norm_fl);


            c_in = 1;
    f_h = 3;
    f_n = 16;
    f_s = 1;
    
    sLayers = [
        imageInputLayer([net.m_ine c_in 1],'Normalization','none','Name','Input')
        convolution2dLayer([f_h c_in], f_n, 'Stride', [f_s 1],'Name','Conv1') 
        flattenLayer('Name','Flat')
    ];
    sgraph = layerGraph(sLayers);

    fb_h = 5;
    fb_n = 16;
    fb_s = 1;

    sbLayers = [
        convolution2dLayer([fb_h c_in], fb_n, 'Stride', [fb_s 1],'Name','Conv1b') 
        flattenLayer('Name','Flatb')
    ];
    sgraph = addLayers(sgraph, sbLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1b');

    fc_h = 7;
    fc_n = 16;
    fc_s = 1;

    scLayers = [
        convolution2dLayer([fc_h c_in], fc_n, 'Stride', [fc_s 1],'Name','Conv1c') 
        %averagePooling2dLayer([pc_s 1], 'Stride', [pc_s 1],'Name','Pool1c')
        flattenLayer('Name','Flatc')
    ];
    sgraph = addLayers(sgraph, scLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1c');

    fd_h = 11;
    fd_n = 16;
    fd_s = 1;

    sdLayers = [
        convolution2dLayer([fd_h c_in], fd_n, 'Stride', [fd_s 1],'Name','Conv1d') 
        flattenLayer('Name','Flatd')
    ];
    sgraph = addLayers(sgraph, sdLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1d');

    fe_h = 13;
    fe_n = 16;
    fe_s = 1;

    seLayers = [
        convolution2dLayer([fe_h c_in], fe_n, 'Stride', [fe_s 1],'Name','Conv1e') 
        flattenLayer('Name','Flate')
    ];
    sgraph = addLayers(sgraph, seLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1e');


    s2Layers = [
        concatenationLayer(1, 5, 'Name', 'Concat')
        fullyConnectedLayer(net.k_hid1,'Name','Full1')
        %batchNormalizationLayer
        reluLayer('Name','Relu1')
        %dropoutLayer%(0.25)
        fullyConnectedLayer(net.k_hid2,'Name','Full2')
        %batchNormalizationLayer
        reluLayer('Name','Relu2')
        %dropoutLayer%(0.25)
        fullyConnectedLayer(net.n_oute,'Name','FullC')
        vRegression('vRegression', net.n_outv)
    ];
    sgraph = addLayers(sgraph, s2Layers);

    sgraph = connectLayers(sgraph, 'Flat', 'Concat/in1');
    sgraph = connectLayers(sgraph, 'Flatb', 'Concat/in2');
    sgraph = connectLayers(sgraph, 'Flatc', 'Concat/in3'); 
    sgraph = connectLayers(sgraph, 'Flatd', 'Concat/in4');
    sgraph = connectLayers(sgraph, 'Flate', 'Concat/in5'); 


            net.lGraph = sgraph;

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@CNNInputNetV(net, i, X, Y);
 
        end

    end

end