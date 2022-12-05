classdef RNNInputNet < MLPInputNet

    properties
        k_lob = 0;
        k_tob = 0;
    end

    methods
        function net = RNNInputNet()

            net = net@MLPInputNet();

        end


        function [net, Xl, Yl, Bl, k_lob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)
            [Xl, Yl, Bl, k_lob] = w_series_generic_train_seq_tensors(M, l_sess, n_sess, norm_fl);
            net.k_lob = k_lob;
            net.mb_size = 2^floor(log2(k_lob));

        end


        function [X2, Y2c, Yh2, Bt, k_tob] = TestTensors(net, M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_seq_tensors(M, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, net.k_lob, 0);

            Y2c = struct;
            Y2c.Y2s = Y2s;
            Y2c.Y2 = Y2;
        end


        function net = Train(net, i, Xl, Yl)

            tNet = trainNetwork(Xl(:, i)', Yl(:, i)', net.lGraph, net.options);
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
                    [lstmNet, Y2s(:, j, i)] = predictAndUpdateState(lstmNet, X2(:, j, i)');
                    Y2(1, j, i) = Y2s(end, j, i);

                    % Continue predicting further output points based on previous
                    % predicvtion
                    for l = 2:net.n_out
                        [lstmNet, Y2(l, j, i)] = predictAndUpdateState(lstmNet, Y2(l-1, j, i)');
                    end
                
                end

            end

        end
        
    end
end