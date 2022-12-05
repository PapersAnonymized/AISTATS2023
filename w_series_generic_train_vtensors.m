function [X, Xc, Xr, Ys, Y, B, XI, C, k_ob, m_ine, n_oute] = w_series_generic_train_vtensors(M, m_in, n_out, l_sess, n_sess, norm_fl)

    % Number of observations in a session
    k_ob = l_sess - m_in + 1;        

    m_ine = 2*m_in-1;
    n_oute = 2*n_out-1;

    % Re-format input into session tensor
    % ('ones' (not 'zeros') for X are for bias 'trick'
    X = zeros([m_ine, k_ob, n_sess]);
    Xc = zeros([m_ine, 1, 1, k_ob, n_sess]);
    Xr = ones([m_ine+1, k_ob, n_sess]);
    Ys = zeros([m_ine, k_ob, n_sess]);
    Y = zeros([n_oute, k_ob, n_sess]);
    B = zeros([2, k_ob, n_sess]);

    k_iob = k_ob * n_sess;
    XI = zeros([m_in, k_iob]);
    I = zeros([k_iob, 1]);

    for i = 1:n_sess
        for j = 1:k_ob
            % extract and scale observation sequence
            idx = (i-1)*l_sess + j;
            
            Mx = M(idx:idx+m_in-1);
            Vx = Mx(1:m_in-1) - Mx(2:m_in);
            % scale bounds over observation span
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series_generic_minmax_scale(Mx, B(1,j,i), B(2,j,i));
                Vx = Mx(1:m_in-1) - Mx(2:m_in);
            end
            %Vx = -Mx(1:m_in-1) + Mx(2:m_in);

            X(1:m_in, j, i) = Mx(:);
            Xc(1:m_in, 1, 1, j, i) = Mx(:);
            Xr(2:m_in+1, j, i) = Mx(:);

            X(m_in+1:m_ine, j, i) = Vx(:);
            Xc(m_in+1:m_ine, 1, 1, j, i) = Vx(:);
            Xr(m_in+2:m_ine+1, j, i) = Vx(:);


            My = M(idx+1:idx+m_in);
            Vy = My(1:m_in-1) - My(2:m_in);
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
                Vy = My(1:m_in-1) - My(2:m_in);
            end
            %Vy = -My(1:m_in-1) + My(2:m_in);

            Ys(1:m_in, j, i) = My(:);
            Ys(m_in+1:m_ine, j, i) = Vy(:);

            My = M(idx+m_in:idx+m_in+n_out-1);
            Vy = My(1:n_out-1) - My(2:n_out);
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
                Vy = My(1:n_out-1) - My(2:n_out);
            end
            Y(1:n_out, j, i) = My(:);
            Y(n_out+1:n_oute, j, i) = Vy(:);

            % Append V below
            i_idx = (i-1)*k_ob + j;
            XI(1:m_in, i_idx) = Mx(:);
            I(i_idx) = i;
        end
    end

    C = categorical(I);
end