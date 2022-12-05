classdef LinRegNet < BaseNet & LinRegInputNet

    properties

    end

    methods
        function net = LinRegNet(m_in, n_out, k_hid1, k_hid2, mb_size, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, k_hid1, k_hid2, ini_rate, max_epoch);
            net = net@LinRegInputNet();

            net.name = "reg";

        end


        %function [net, X, Y, B, k_ob] = TrainTensors(net, M, m_in, n_out, l_sess, n_sess, norm_fl)

        %    [net, X, Y, B, k_ob] = TrainTensors@LinRegInputNet(net, M, m_in, n_out, l_sess, n_sess, norm_fl);

        %end



        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@LinRegInputNet(net, i, X, Y);
        end

        
    end
end