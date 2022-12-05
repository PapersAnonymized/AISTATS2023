classdef CNNInputNetV < CNNInputNet & MLPInputNetV

    properties

    end

    methods
        function net = CNNInputNetV()

            net = net@CNNInputNet();
            net = net@MLPInputNetV();

        end


        function [net, Xc, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)
            [X, Xc, Xr, Ys, Y, B, XI, C, k_ob, m_ine, n_oute] = w_series_generic_train_vtensors(M, m_in, n_out, l_sess, n_sess, norm_fl);
            net.mb_size = 2^floor(log2(k_ob));

            net = LatentInit(net, m_ine, n_oute);
        end


        function [Xc2, Y2, Yh2, Bt, k_tob] = TestTensors(net, M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bt, k_tob] = w_series_generic_test_vtensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl,...
                0, net.m_ine, net.n_oute);
        end


        function net = Train(net, i, X, Y)

            net = Train@CNNInputNet(net, i, X, Y);

        end

        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)

            [X2, Y2] = Predict@CNNInputNet(net, X2, Y2, regNets, t_sess, sess_off);

        end
        
    end
end