function [X2, Y2s, Y2, Yh2, B, k_tob] = w_series_generic_test_seq_tensors(M, n_out, l_sess, l_test, n_sess, sess_off, offset, norm_fl, k_ob, k_tob)

    %% Test regression ANN
    if(k_tob == 0)
        k_tob = ceil(l_sess/n_out);
    end

    X2 = ones([k_ob+1, k_tob, n_sess-sess_off]);
    Y2s = zeros([k_ob+1, k_tob, n_sess-sess_off]);
    Y2 = zeros([n_out, k_tob, n_sess-sess_off]);
    Yh2 = zeros([n_out, k_tob, n_sess-sess_off]);
    B = zeros([2, k_tob, n_sess-sess_off]);

    % Re-format test input into session tensor
    for i = 1:n_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess - k_ob + (j-1)*n_out + 1 + offset;

            Mx = M(idx:idx+k_ob);
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series_generic_minmax_scale(Mx, B(1,j,i), B(2,j,i));
            end
            X2(:, j, i) = Mx(:);

            My = M(idx+k_ob+1:idx+k_ob+n_out);
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
            end
            Yh2(:, j, i) = My(:);
        end
    end

end