function s=sampling(prob,samp_type)
global usegpu;
switch samp_type
    case 'S'  % sampled
        if usegpu
            s=(prob>rand(size(prob),'single','gpuArray'));
        else
            s=(prob>rand(size(prob),'single'));
        end
    case 'B'  % binary
        s=(prob>0.5);
    case 'HP' % high precision
        s=prob;
end