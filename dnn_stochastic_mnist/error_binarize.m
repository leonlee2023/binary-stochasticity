function x_b=error_binarize(x,type)

switch type
    case 'S' % signed
        x_b=((x>0)-(x<0)); 
    case 'HP' % high precision
        x_b=x;
end

