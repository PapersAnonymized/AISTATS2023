classdef RNNSeqInputNet < MLPInputNet

    properties

    end

    methods
        function net = RNNSeqInputNet()

            net = net@MLPInputNet();

        end


        function [net, X, Yc, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)
            [X, Xc, Xr, Ys, Y, B, XI, C, k_ob] = w_series_generic_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl);
            Yc = struct;
            Yc.Ys = Ys;
            Yc.Y = Y;
            net.mb_size = 2^floor(log2(k_ob));
        end


        function [X2, Y2c, Yh2, Bt, k_tob] = TestTensors(net, M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_tensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, 0);
            Y2c = struct;
            Y2c.Y2s = Y2s;
            Y2c.Y2 = Y2;
        end


        function net = Train(net, i, X, Yc)

            CX = num2cell(X(:, :, i)', 2);
            CY = num2cell(Yc.Y(:, :, i)', 2);

            tNet = trainNetwork(CX, CY, net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            

        end

        function [X2, Y2] = Predict(net, X2, Y2c, regNets, t_sess, sess_off, k_tob)

            Y2s = Y2c.Y2s;
            Y2 = Y2c.Y2;

            for i = 1:t_sess-sess_off

                % Now feeding test data
                for j = 1:k_tob
                    lstmNet = regNets{i}.trainedNet;
                    Y2(:, j, i) = predict(lstmNet, X2(:, j, i)');
                end

            end

        end
        
    end
end