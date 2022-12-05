classdef BaseNetV < BaseNet

    properties
        m_ine
        n_oute
        n_outv
    end

    methods
        function net = BaseNetV(m_in, n_out, ini_rate, max_epoch)

            net = net@BaseNet(m_in, n_out, ini_rate, max_epoch);

            %dummy init, re-write in *InputNetV
            net.m_ine = m_in;
            net.n_oute = n_out;
            net.n_outv = net.n_oute - net.n_out;


        end

        
    end
end