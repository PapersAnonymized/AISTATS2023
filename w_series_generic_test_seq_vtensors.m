function [X2, Y2s, Y2, Yh2, B, k_tob] = w_series_generic_test_seq_tensors(M, n_out, l_sess, l_test, n_sess, sess_off, offset, norm_fl, k_ob, k_tob)

    %% Test regression ANN
    if(k_tob == 0)
        k_tob = ceil(l_sess/n_out);
    end

    X2 = zeros([2, k_ob+1, k_tob, n_sess-sess_off]);
    Y2s = zeros([2, k_ob+1, k_tob, n_sess-sess_off]);
    Y2 = zeros([2, n_out, k_tob, n_sess-sess_off]);
    Yh2 = zeros([2, n_out, k_tob, n_sess-sess_off]);
    B = zeros([2, k_tob, n_sess-sess_off]);

    % Re-format test input into session tensor
    for i = 1:n_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess - k_ob + (j-1)*n_out + 1 + offset;

            Mx = M(idx:idx+k_ob);
            Mx2 = M(idx+1:idx+k_ob+1);
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series_generic_minmax_scale(Mx, B(1,j,i), B(2,j,i));
                Mx2 = w_series_generic_minmax_scale(Mx2, B(1,j,i), B(2,j,i));
            end
            Vx = Mx - Mx2;
            X2(1,:, j, i) = Mx(:);
            %X2(2,:, j, i) = Mx2(:);
            X2(2,:, j, i) = Vx(:);

            My = M(idx+k_ob+1:idx+k_ob+n_out);
            My2 = M(idx+k_ob+2:idx+k_ob+n_out+1);
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
                My2 = w_series_generic_minmax_scale(My2, B(1,j,i), B(2,j,i));
            end
            Vy = My - My2;
            Yh2(1,:, j, i) = My(:);
            %Yh2(2,:, j, i) = My2(:);
            Yh2(2,:, j, i) = Vy(:);
        end
    end

end