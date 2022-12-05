function [X, Y, B, k_ob] = w_series_generic_train_seq_vtensors(M, l_sess, n_sess, norm_fl)


    % Number of observations in a session (training label(sequence) does
    % not touch test period    
    k_ob = l_sess - 1;

    X = zeros([2, k_ob, n_sess]);
    Y = zeros([2, k_ob, n_sess]);
    %Y2s = zeros([k_ob, n_sess]);
    B = zeros([2, n_sess]);

    % Re-format input into session tensor
    for i = 1:n_sess
        % scale bounds over session scale
        idx = (i-1)*l_sess + 1;

        % extract and scale observation sequence
        Mx = M(idx:idx+k_ob-1);
        Mx2 = M(idx+1:idx+k_ob);
        [B(1,i), B(2,i)] = bounds(Mx);
        if norm_fl
            Mx = w_series_generic_minmax_scale(Mx, B(1,i), B(2,i));
            Mx2 = w_series_generic_minmax_scale(Mx2, B(1,i), B(2,i));
        end
        Vx = Mx - Mx2;
        X(1,:, i) = Mx(:);
        %X(2,:, i) = Mx2(:);
        X(2,:, i) = Vx(:);

        My = M(idx+1:idx+k_ob);
        My2 = M(idx+2:idx+k_ob+1);
        if norm_fl
            My = w_series_generic_minmax_scale(My, B(1,i), B(2,i));
            My2 = w_series_generic_minmax_scale(My2, B(1,i), B(2,i));
        end
        Vy = My - My2;
        Y(1,:, i) = My(:);
        %Y(2,:, i) = My2(:);
        Y(2,:, i) = Vy(:);

    end

end