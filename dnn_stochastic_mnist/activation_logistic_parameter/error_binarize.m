function x_b=error_binarize(x,type)

switch type
    case 'S' % signed
        x_b=((x>0)-(x<0)); % for error small than 'e_drop', consider as no error
    case 'HP' % high precision
        x_b=x;
end

