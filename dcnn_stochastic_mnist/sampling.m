function s=sampling(prob,samp_type)

switch samp_type
    case 'S'
        s=(prob>rand(size(prob)));
    case 'B'
        s=(prob>0.5);
    case 'HP'
        s=prob;
end