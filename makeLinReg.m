function W1 = makeLinReg(i, m_in, n_out, n_sess, X, Y)

    fprintf('Training Lin Reg %d\n', i);  
    
    % Fit transformatin matrix into minimal SSE
    W1 = zeros([n_out, m_in+1, n_sess]);
    parfor i = 1:n_sess
        Xi = X(:, :, i);
        Yi = Y(:, :, i);
        XiT = Xi.';
        W1(:,:,i) = Yi * XiT / (Xi * XiT);
    end
    
end
