function x_b=error_binarize(x,type)

switch type
    case 'S'
        x_b=((x>0)-(x<0)); 
    case 'HP'
        x_b=x;
    otherwise
        error('Activation type not defined!')
end

