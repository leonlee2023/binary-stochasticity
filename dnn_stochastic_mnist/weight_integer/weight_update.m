function [weight,DLDw] = weight_update(weight,DLDw,dnn)

if dnn.eta*dnn.scalor<1
    dw=fix(DLDw/dnn.batch_size);
    DLDw=DLDw - dw * dnn.batch_size;
    weight=weight - int8(dw);
else
    dw=fix(dnn.eta*dnn.scalor*DLDw/dnn.batch_size);
    DLDw=DLDw - dw * dnn.batch_size/(dnn.eta*dnn.scalor);
    weight=weight - int8(dw);
end

switch dnn.w_type
    case 'INT8'
        weight=weight;
    case 'INT7'
        weight(weight>63)=63;
        weight(weight<-64)=-64;
    case 'INT6'
        weight(weight>31)=31;
        weight(weight<-32)=-32;
    case 'INT5'
        weight(weight>15)=15;
        weight(weight<-16)=-16;
    case 'INT4'
        weight(weight>7)=7;
        weight(weight<-8)=-8;
    case 'INT3'
        weight(weight>3)=3;
        weight(weight<-4)=-4;
    case 'INT2'
        weight(weight>1)=1;
        weight(weight<-2)=-2;
    case 'Ternary'
        weight(weight>1)=1;
        weight(weight<-1)=-1;
    case 'Binary'
        weight(weight>1)=1;
        weight(weight<0)=0;
end
