function s=sampling(prob,samp_type)

switch samp_type
    case 'S'  % sampled
        s=(prob>rand(size(prob)));
    case 'B'  % binary
        s=(prob>0.5);
    case 'HP' % high precision
        s=prob;
end