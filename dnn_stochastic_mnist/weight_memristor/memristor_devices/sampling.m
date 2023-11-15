function s=sampling(prob,act_type)

switch act_type
    case 'S'
        s=(prob>rand(size(prob)));
    case 'B'
        s=(prob>0.5);
    case 'HP'
        s=prob;
end