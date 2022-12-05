function M = w_series_generic_minmax_rescale(Mn, Min, Max)
    M = Mn * (Max - Min) + Min;
end