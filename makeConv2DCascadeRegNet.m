function [regNet, k_hid1, k_hid2] = makeConv2DCascadeRegNet(i, m_in, c_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    f_h = 3;
    f_n = 16;
    f_s = 1;
    p_s = 3;
    
    sLayers = [
        imageInputLayer([m_in c_in 1],'Normalization','none','Name','Input')
        convolution2dLayer([f_h c_in], f_n, 'Stride', [f_s 1],'Name','Conv1') 
        flattenLayer('Name','Flat')
    ];
    sgraph = layerGraph(sLayers);

    fb_h = 5;
    fb_n = 16;
    fb_s = 1;
    pb_s = 3;
    sbLayers = [
        convolution2dLayer([fb_h c_in], fb_n, 'Stride', [fb_s 1],'Name','Conv1b') 
        flattenLayer('Name','Flatb')
    ];
    sgraph = addLayers(sgraph, sbLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1b');

    fc_h = 7;
    fc_n = 16;
    fc_s = 1;
    pc_s = 3;
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
    pd_s = 3;
    sdLayers = [
        convolution2dLayer([fd_h c_in], fd_n, 'Stride', [fd_s 1],'Name','Conv1d') 
        flattenLayer('Name','Flatd')
    ];
    sgraph = addLayers(sgraph, sdLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1d');

    fe_h = 13;
    fe_n = 16;
    fe_s = 1;
    pe_s = 3;
    seLayers = [
        convolution2dLayer([fe_h c_in], fe_n, 'Stride', [fe_s 1],'Name','Conv1e') 
        flattenLayer('Name','Flate')
    ];
    sgraph = addLayers(sgraph, seLayers);
    sgraph = connectLayers(sgraph, 'Input', 'Conv1e');


    %ff_h = 17;
    %ff_n = 4;
    %ff_s = 1;
    %pf_s = 3;
    %sfLayers = [
    %    convolution2dLayer([ff_h c_in], ff_n, 'Stride', [ff_s 1],'Name','Conv1f') 
    %    flattenLayer('Name','Flatf')
    %];
    %sgraph = addLayers(sgraph, sfLayers);
    %sgraph = connectLayers(sgraph, 'Input', 'Conv1f');

    %fg_h = 797;%.4781 %887;.4746  %997;.4781 %397;.3898 %293;%.5116 %199;.4750 %179;.4731 %157;.4726 %149;.4705 %97 %.462 %499; %.344
    %fg_n = 4;
    %fg_s = 1;
    %pg_s = 3;
    %sgLayers = [
    %    convolution2dLayer([fg_h c_in], fg_n, 'Stride', [fg_s 1],'Name','Conv1g') 
        %averagePooling2dLayer([pg_s 1], 'Stride', [pg_s 1],'Name','Pool1g')
        %layerNormalizationLayer('OffsetInitializer','ones', 'Name','PreFlatgfN')
    %    flattenLayer('Name','Flatg')
    %];
    %sgraph = addLayers(sgraph, sgLayers);
    %sgraph = connectLayers(sgraph, 'Input', 'Conv1g');


    %fz_h = 397; %.5218
    %fz_n = 4;
    %fz_s = 1;
    %pz_s = 3;
    %szLayers = [
        %convolution2dLayer([fg_h c_in], fg_n, 'Stride', [fg_s 1],'Name','Conv1g') 
        %averagePooling2dLayer([pg_s 1], 'Stride', [pg_s 1],'Name','Pool1g')
        %layerNormalizationLayer('OffsetInitializer','ones', 'Name','PreFlatzN') %.4970
    %    flattenLayer('Name','Flatz')
    %];
    %sgraph = addLayers(sgraph, szLayers);
    %sgraph = connectLayers(sgraph, 'Input', 'Flatz');



    %k_hid1 = floor((m_in - f_h + 1) * f_n/f_s/p_s);% + floor((m_in - fb_h + 1) * fb_n/fb_s/pb_s);
    %k_hid2 = 2*k_hid1 + 1;


    s2Layers = [
        concatenationLayer(1, 5, 'Name', 'Concat')
        fullyConnectedLayer(k_hid1,'Name','Full1')
        %batchNormalizationLayer
        reluLayer('Name','Relu1')
        %dropoutLayer%(0.25)
        fullyConnectedLayer(k_hid2,'Name','Full2')
        %batchNormalizationLayer
        reluLayer('Name','Relu2')
        %dropoutLayer%(0.25)
        fullyConnectedLayer(n_out,'Name','FullC')
        regressionLayer
    ];
    sgraph = addLayers(sgraph, s2Layers);

    sgraph = connectLayers(sgraph, 'Flat', 'Concat/in1');
    sgraph = connectLayers(sgraph, 'Flatb', 'Concat/in2');
    sgraph = connectLayers(sgraph, 'Flatc', 'Concat/in3'); 
    sgraph = connectLayers(sgraph, 'Flatd', 'Concat/in4');
    sgraph = connectLayers(sgraph, 'Flate', 'Concat/in5'); 
    %sgraph = connectLayers(sgraph, 'Flatf', 'Concat/in6');
    %sgraph = connectLayers(sgraph, 'Flatg', 'Concat/in7');
    %sgraph = connectLayers(sgraph, 'Flatz', 'Concat/in7');

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000); %,...%);'L2Regularization', 0.001,...
        %'Verbose',true, ...
        %'Plots','training-progress');

    fprintf('Training Conv net %d\n', i);   

    regNet = trainNetwork(X(:, :, :, :, i), Y(:, :, i)', sgraph, sOptions);
    
end