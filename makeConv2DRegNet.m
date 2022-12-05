function [regNet, k_hid1, k_hid2] = makeConv2DRegNet(i, m_in, c_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    f_h = 5; 
    f_n = 16; 
    f_s = 1;
    p_s = 3;
    
    %k_hid1 = floor((m_in - f_h + 1) * f_n/f_s/p_s);

    %f2_h = 50;
    %f2_n = 4;
    %f2_s = 1;
    %p2_s = 1;

    %k_hid1 = floor((m_in - f2_h + 1) * f2_n/f2_s/p2_s);

    %k_hid2 = 2*k_hid1 + 1;
    
    sLayers = [
        imageInputLayer([m_in c_in 1],'Normalization','none','Name','Input')
        convolution2dLayer([f_h c_in], f_n, 'Stride', [f_s 1],'Name','Conv1')
        %maxPooling2dLayer([p_s 1], 'Stride', [p_s 1],'Name','Pool1')
        flattenLayer('Name','Flat')
        fullyConnectedLayer(k_hid1,'Name','Full1')
        reluLayer('Name','Relu1')
        fullyConnectedLayer(k_hid2,'Name','Full2')
        reluLayer('Name','Relu2')
        fullyConnectedLayer(n_out,'Name','FullC')
        regressionLayer
    ];

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000); %,...%);'L2Regularization', 0.001,...
        %'Verbose',true, ...
        %'Plots','training-progress');

    fprintf('Training Conv net %d\n', i);   

    regNet = trainNetwork(X(:, :, :, :, i), Y(:, :, i)', sLayers, sOptions);
    
end