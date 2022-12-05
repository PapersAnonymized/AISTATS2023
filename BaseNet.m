classdef BaseNet

    properties
        name = [];

        m_in 
        n_out
        k_hid1
        k_hid2
        ini_rate 
        max_epoch
    
        %layers
        lGraph = [];
        options = [];
        trainedNet = [];
    end

    methods
        function net = BaseNet(m_in, n_out, ini_rate, max_epoch)

            %net.name = [];

            net.m_in = m_in; 
            net.n_out = n_out;

            mult = 1;
            net.k_hid1 = floor(mult * (m_in + 1));
            net.k_hid2 = floor(mult * (2*m_in + 1));
            net.ini_rate = ini_rate;
            net.max_epoch = max_epoch;

            %net.layers = [];
            %net.lGraph = [];
            %net.options = [];
            %net.trainedNet = [];
        end

        function [Y2, Yh2] = ReScale(net, Y2, Yh2, Bt, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off
                for j = 1:k_tob
                    Y2(:, j, i) = w_series_generic_minmax_rescale(Y2(:, j, i), Bt(1,j,i), Bt(2,j,i));
                    Yh2(:, j, i) = w_series_generic_minmax_rescale(Yh2(:, j, i), Bt(1,j,i), Bt(2,j,i));
                end
            end
        end

        function [S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2, n_out)
            [S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = w_series_generic_calc_mape(Y2, Yh2, n_out); 
        end

        function [S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2, n_out) 
            [S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = w_series_generic_calc_rmse(Y2, Yh2, n_out);
        end

        function Err_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg)
            w_series_generic_err_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);
        end

    end
end