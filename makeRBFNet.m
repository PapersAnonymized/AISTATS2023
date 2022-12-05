function regNet = makeRBFNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    sLayers = [
        featureInputLayer(m_in)
        GaussianRBFLayer('RBF1', m_in, k_hid1)
        GaussianRBFLayer('RBF2', k_hid1, k_hid2)
        fullyConnectedLayer(n_out)
        regressionLayer
    ];

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training RBF net %d\n', i);   

    regNet = trainNetwork(X(:, :, i)', Y(:, :, i)', sLayers, sOptions);
    
end