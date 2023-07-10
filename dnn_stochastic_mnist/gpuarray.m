function y=gpuarray(x)
    global usegpu;
    if usegpu
        y=gpuArray(single(x));
    else
        y=single(x);
    end
end
