function DLDw=conv_gradient(x,DLDy,conv)

batch_size = size(x, 4);

DLDw=zeros(size(conv.weight));

for inf=1:conv.input_n_feature
    for outf=1:conv.output_n_feature  
        for bb=1:batch_size
            DLDw(:,:,inf,outf)=DLDw(:,:,inf,outf)+convn(x(:,:,inf,bb),rot90(DLDy(:,:,outf,bb),2),'valid');
        end
    end
end
DLDw=DLDw/batch_size;


