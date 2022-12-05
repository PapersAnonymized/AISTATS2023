function regNet = makeLSTMNet(i, k_hid1, k_hid2, mb_size, Xl, Yl)

    sLayers = [
        sequenceInputLayer(1)
        %gruLayer(k_hid1)%, 'OutputMode','last')
        %gruLayer(k_hid2)%, 'OutputMode','last')
        lstmLayer(k_hid1)%, 'OutputMode','last')
        lstmLayer(k_hid2)%, 'OutputMode','last')
        fullyConnectedLayer(1)
        regressionLayer
    ];

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','auto',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training LSTM net %d\n', i);   

    regNet = trainNetwork(Xl(:, i)', Yl(:, i)', sLayers, sOptions);
    
end