function [X2, Xc2, Xr2, Y2s, Y2, Yh2, B, k_tob] = w_series_generic_test_vtensors(M, m_in, n_out, l_sess, l_test, n_sess, sess_off, offset, norm_fl, k_tob, m_ine, n_oute)
    %% Test regression ANN
    if(k_tob == 0)
        k_tob = ceil(l_sess/n_out);
    end

    X2 = zeros([m_ine, k_tob, n_sess-sess_off]);
    Xc2 = zeros([m_ine, 1, 1, k_tob, n_sess-sess_off]);
    Xr2 = ones([m_ine+1, k_tob, n_sess]);
    Y2s = zeros([m_ine, k_tob, n_sess-sess_off]);
    Y2 = zeros([n_oute, k_tob, n_sess-sess_off]);
    Yh2 = zeros([n_oute, k_tob, n_sess-sess_off]);
    B = zeros([2, k_tob, n_sess-sess_off]);

    % Re-format test input into session tensor
    for i = 1:n_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*n_out + 1 + offset;

            Mx = M(idx:idx+m_in-1);
            Vx = Mx(1:m_in-1) - Mx(2:m_in);
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series_generic_minmax_scale(Mx, B(1,j,i), B(2,j,i));
                Vx = Mx(1:m_in-1) - Mx(2:m_in);
            end
            %Vx = -Mx(1:m_in-1) + Mx(2:m_in);

            X2(1:m_in, j, i) = Mx(:);
            Xc2(1:m_in, 1, 1, j, i) = Mx(:);
            Xr2(2:m_in+1, j, i) = Mx(:);

            X2(m_in+1:m_ine, j, i) = Vx(:);
            Xc2(m_in+1:m_ine, 1, 1, j, i) = Vx(:);
            Xr2(m_in+2:m_ine+1, j, i) = Vx(:);

            My = M(idx+m_in:idx+m_in+n_out-1);
            Vy = My(1:n_out-1) - My(2:n_out);
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
                Vy = My(1:n_out-1) - My(2:n_out);
            end
            %Vy = -My(1:n_out-1) + My(2:n_out);

            Yh2(1:n_out, j, i) = My(:);
            Yh2(n_out+1:n_oute, j, i) = Vy(:);
        end
    end
end