function M = decay(y)
    n = length(y);
    M = zeros(n);
    for i=1:n
        M(i:end,i) = y(i);
    end
end