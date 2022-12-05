function regNet = makeTG2Net(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    oLayers = [
        featureInputLayer(m_in,'Name','inputFeature')
        fullyConnectedLayer(k_hid1,'Name','fcHidden')
        fullyConnectedLayer(k_hid1,'Name','fcV')
        multiplicationLayer(2,'Name','fcMgate') 
        fullyConnectedLayer(k_hid2,'Name','fcHidden2')
        fullyConnectedLayer(k_hid2,'Name','fcV2')
        multiplicationLayer(2,'Name','fcMgate2')
        fullyConnectedLayer(n_out,'Name','fcOut')
        regressionLayer('Name','regOut')
    ];
    cgraph = layerGraph(oLayers);


    % Q Linear Transform
    qLayers = [
        fullyConnectedLayer(k_hid1,'Name','fcQ')
    ];
    kLayers = [
        fullyConnectedLayer(k_hid1,'Name','fcK')
    ];
    qkLayers = [
        % dim, input
        concatenationLayer(1, 2,'Name','concatQK')
        dotProdLayer('dotProdQK', k_hid1)
    ];    

    cgraph = addLayers(cgraph, qLayers);
    cgraph = addLayers(cgraph, kLayers);
    cgraph = addLayers(cgraph, qkLayers);
    
    cgraph = connectLayers(cgraph, 'fcHidden', 'fcQ');
    cgraph = connectLayers(cgraph, 'fcHidden', 'fcK');

    cgraph = connectLayers(cgraph, 'fcQ', 'concatQK/in1');
    cgraph = connectLayers(cgraph, 'fcK', 'concatQK/in2');

    cgraph = connectLayers(cgraph,'dotProdQK','fcMgate/in2');


    % Q2 Linear Transform
    q2Layers = [
        fullyConnectedLayer(k_hid2,'Name','fcQ2')
    ];
    k2Layers = [
        fullyConnectedLayer(k_hid2,'Name','fcK2')
    ];
    qk2Layers = [
        % dim, input
        concatenationLayer(1, 2,'Name','concatQK2')
        dotProdLayer('dotProdQK2', k_hid2)
    ];    

    cgraph = addLayers(cgraph, q2Layers);
    cgraph = addLayers(cgraph, k2Layers);
    cgraph = addLayers(cgraph, qk2Layers);
    
    cgraph = connectLayers(cgraph, 'fcHidden2', 'fcQ2');
    cgraph = connectLayers(cgraph, 'fcHidden2', 'fcK2');

    cgraph = connectLayers(cgraph, 'fcQ2', 'concatQK2/in1');
    cgraph = connectLayers(cgraph, 'fcK2', 'concatQK2/in2');

    cgraph = connectLayers(cgraph,'dotProdQK2','fcMgate2/in2');

    %figure
    %plot(cgraph);

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training TG net %d\n', i);   

    regNet = trainNetwork(X(:, :, i)', Y(:, :, i)', cgraph, sOptions);
    
end