function [z,dzdy]= activation(y,type,a)
% activation function, and derivative of activation function to its input
    if ~exist('type','var')
        type='logistic';
    end
    if ~exist('a','var')
        a=1;
    end

    switch type
        case "logistic"
            z = 1./(1+exp(-a*y));
            dzdy = a*z.*(1-z);
        case "ReLU"
            z = a*y;
            z(y<0)=0;
            dzdy=zeros(size(y));
            dzdy(y>=0)=a;
        case 'Truncated-ReLU'
            z = a*y;
            z(y<0)=0;
            z(y>1/a)=1;
            dzdy=zeros(size(y));
            dzdy(y>=0 & y<1/a)=a;
        case 'Truncated-ReLU2'
            z = a*y+1/2;
            z(y<-1/(2*a))=0;
            z(y>1/(2*a))=1;
            dzdy=zeros(size(y));
            dzdy(y>=-1/(2*a) & y<1/(2*a))=a;
        case 'logistic-ReLU'
            z = 1./(1+exp(-a*y));
            dzdy=zeros(size(y));
            dzdy(y>=-1/(2*a) & y<1/(2*a))=a;
        case 'ReLU-logistic'
            z = a*y+1/2;
            z(y<-1/(2*a))=0;
            z(y>1/(2*a))=1;
            dzdy = a*1./(1+exp(-a*y)).*(1-1./(1+exp(-a*y)));
    
    end

end