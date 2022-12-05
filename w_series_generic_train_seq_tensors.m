function [X, Y, B, k_ob] = w_series_generic_train_seq_tensors(M, l_sess, n_sess, norm_fl)


    % Number of observations in a session (training label(sequence) does
    % not touch test period    
    k_ob = l_sess - 1;

    X = zeros([k_ob, n_sess]);
    Y = zeros([k_ob, n_sess]);
    %Y2s = zeros([k_ob, n_sess]);
    B = zeros([2, n_sess]);

    % Re-format input into session tensor
    for i = 1:n_sess
        % scale bounds over session scale
        idx = (i-1)*l_sess + 1;

        % extract and scale observation sequence
        Mx = M(idx:idx+k_ob-1);
        [B(1,i), B(2,i)] = bounds(Mx);
        if norm_fl
            Mx = w_series_generic_minmax_scale(Mx, B(1,i), B(2,i));
        end
        X(:, i) = Mx(:);

        My = M(idx+1:idx+k_ob);
        if norm_fl
            My = w_series_generic_minmax_scale(My, B(1,i), B(2,i));
        end
        Y(:, i) = My(:);
    end

end