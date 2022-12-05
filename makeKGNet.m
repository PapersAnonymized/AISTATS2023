function regNet = makeKGNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    oLayers = [
        featureInputLayer(m_in,'Name','inputFeature')
        additionLayer(2,'Name','fcAgate')
        multiplicationLayer(2,'Name','fcMgate')
        fullyConnectedLayer(k_hid1,'Name','fcHidden')
        additionLayer(2,'Name','fcAgate2')
        multiplicationLayer(2,'Name','fcMgate2') 
        fullyConnectedLayer(k_hid2,'Name','fcHidden2')
        fullyConnectedLayer(n_out,'Name','fcOut')
        regressionLayer('Name','regOut')
    ];
    cgraph = layerGraph(oLayers);


    % Allow Linear Transform and Sigmoid activation
    aLayers = [
        fullyConnectedLayer(m_in,'Name','fcSig')
        sigmoidLayer('Name','sigAllow')
    ];
    aLayers2 = [
        fullyConnectedLayer(k_hid1,'Name','fcSig2')
        sigmoidLayer('Name','sigAllow2')
    ];
    cgraph = addLayers(cgraph, aLayers);
    cgraph = addLayers(cgraph, aLayers2);


    % Update Linear Transform and Tanh activation
    dLayers = [
        fullyConnectedLayer(m_in,'Name','fcTanh')
        tanhLayer('Name','tanhUpdate')
    ];
    dLayers2 = [
        fullyConnectedLayer(k_hid1,'Name','fcTanh2')
        tanhLayer('Name','tanhUpdate2')
    ];
    cgraph = addLayers(cgraph, dLayers);
    cgraph = addLayers(cgraph, dLayers2);


    % Update Cacade Linear Transform and Hadfamard join with Tanh
    nLayers = [
        fullyConnectedLayer(m_in,'Name','fcNorm')
        multiplicationLayer(2,'Name','normMgate')
    ];
    nLayers2 = [
        fullyConnectedLayer(k_hid1,'Name','fcNorm2')
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


    %figure
    %plot(cgraph);

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training KG net %d\n', i);   

    regNet = trainNetwork(X(:, :, i)', Y(:, :, i)', cgraph, sOptions);
    
end