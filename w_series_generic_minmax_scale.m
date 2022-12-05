function Mn = w_series_generic_minmax_scale(M, Min, Max)
    Mn = (M - Min) / (Max - Min);
end