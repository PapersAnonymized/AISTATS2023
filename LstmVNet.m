classdef LstmVNet < BaseNetV & RNNInputNet

    properties

    end

    methods
        function net = LstmVNet(m_in, n_out, mb_size, ini_rate, max_epoch)

            net = net@BaseNetV(m_in, n_out, ini_rate, max_epoch);
            net = net@RNNInputNet();

            net.name = "lstmV";


        end


        function [net, Xl, Yl, Bl, k_lob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

            [Xl, Yl, Bl, k_lob] = w_series_generic_train_seq_vtensors(M, l_sess, n_sess, norm_fl);
            net.k_lob = k_lob;
            net.mb_size = 2^floor(log2(k_lob));

            sLayers = [
                sequenceInputLayer(2)
                lstmLayer(net.k_hid1)%, 'OutputMode','last')
                lstmLayer(net.k_hid2)%, 'OutputMode','last')
                fullyConnectedLayer(2)
                vRegression('vRegression', 1)
            ];

            net.lGraph = layerGraph(sLayers);
            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);

                %                'Plots', 'training-progress',...
        end



        function [X2, Y2c, Yh2, Bt, k_tob] = TestTensors(net, M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_seq_vtensors(M, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, net.k_lob, 0);

            Y2c = struct;
            Y2c.Y2s = Y2s;
            Y2c.Y2 = Y2;
        end


        function net = Train(net, i, Xl, Yl)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            %net = Train@RNNInputNet(net, i, X, Y);

            tNet = trainNetwork(Xl(:, :, i), Yl(:, :, i), net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            

        end

        function [X2, Y2] = Predict(net, X2, Y2c, regNets, t_sess, sess_off, k_tob)

            Y2s = Y2c.Y2s;
            Y2 = Y2c.Y2;

            for i = 1:t_sess-sess_off
        
                % Now feeding test data
                for j = 1:k_tob
                    lstmNet = resetState(regNets{i}.trainedNet);
                    % Trick - go through all input sequence to get the first next
                    % prediction outside the seqence, discarding intermediate predictions

                    %Either
                    %for k = 1:net.k_lob+1
                    %    [lstmNet, Y2(1, j, i)] = predictAndUpdateState(lstmNet, X2(k, j, i)');
                    %end
                    %or
                    [lstmNet, Y2s(:, :, j, i)] = predictAndUpdateState(lstmNet, X2(:, :, j, i));
                    Y2(:, 1, j, i) = Y2s(:, end, j, i);

                    % Continue predicting further output points based on previous
                    % predicvtion
                    for l = 2:net.n_out
                        [lstmNet, Y2(:, l, j, i)] = predictAndUpdateState(lstmNet, Y2(:, l-1, j, i));
                    end
                
                end

            end

        end

        function [Y2, Yh2] = ReScale(net, Y2, Yh2, Bt, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off
                for j = 1:k_tob
                    Y2(1, :, j, i) = w_series_generic_minmax_rescale(Y2(1, :, j, i), Bt(1,j,i), Bt(2,j,i));
                    Yh2(1, :, j, i) = w_series_generic_minmax_rescale(Yh2(1, :, j, i), Bt(1,j,i), Bt(2,j,i));
                end
            end
        end


        function [S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2, n_out)
            [S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = w_series_generic_calcv_mape(Y2, Yh2, n_out); 
        end

        function [S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2, n_out) 
            [S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = w_series_generic_calcv_rmse(Y2, Yh2, n_out);
        end

        function Err_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg)
            w_series_generic_errv_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);
        end
   

    end

end