function regNet = makeTransNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    oLayers = [
        featureInputLayer(m_in,'Name','inputFeature')
        additionLayer(2,'Name','fcAgate')
        fullyConnectedLayer(k_hid1,'Name','fcHidden')
        additionLayer(2,'Name','fcAgate2')
        fullyConnectedLayer(k_hid2,'Name','fcHidden2')
        fullyConnectedLayer(n_out,'Name','fcOut')
        regressionLayer('Name','regOut')
    ];
    cgraph = layerGraph(oLayers);

    tLayers = [
        transformerLayer(m_in,'trans')
        fullyConnectedLayer(m_in,'Name','fcTrans')
    ];

    cgraph = addLayers(cgraph, tLayers);
    
    cgraph = connectLayers(cgraph, 'inputFeature', 'trans');
    cgraph = connectLayers(cgraph,'fcTrans','fcAgate/in2');


    t2Layers = [
        transformerLayer(k_hid1,'trans2')
        fullyConnectedLayer(k_hid1,'Name','fcTrans2')
    ];

    cgraph = addLayers(cgraph, t2Layers);
    
    cgraph = connectLayers(cgraph, 'fcHidden', 'trans2');
    cgraph = connectLayers(cgraph,'fcTrans2','fcAgate2/in2');

    %figure
    %plot(cgraph);

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',2000);

    fprintf('Training Trans net %d\n', i);   

    regNet = trainNetwork(X(:, :, i)', Y(:, :, i)', cgraph, sOptions);
    
end