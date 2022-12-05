classdef MLPInputNetV < MLPInputNet

    properties

    end

    methods
        function net = MLPInputNetV()

            net = net@MLPInputNet();

        end

        function net = LatentInit(net, m_ine, n_oute)
            net.k_hid1 = m_ine + 1;
            net.k_hid2 = 2*m_ine + 1;
            net.m_ine = m_ine;
            net.n_oute = n_oute;
            net.n_outv = net.n_oute - net.n_out;
        end

        function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)
            [X, Xc, Xr, Ys, Y, B, XI, C, k_ob, m_ine, n_oute] = w_series_generic_train_vtensors(M, m_in, n_out, l_sess, n_sess, norm_fl);
            net.mb_size = 2^floor(log2(k_ob));

            net = LatentInit(net, m_ine, n_oute);
        end


        function [X2, Y2, Yh2, Bt, k_tob] = TestTensors(net, M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_vtensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl,...
                0, net.m_ine, net.n_oute);
        end


        function net = Train(net, i, X, Y) 

            tNet = trainNetwork(X(:, :, i)', Y(:, :, i)', net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;

        end

        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off
                predictedScores = predict(regNets{i}.trainedNet, X2(:, :, i)');
                Y2(:, :, i) = predictedScores';
            end
        end
        
    end
end