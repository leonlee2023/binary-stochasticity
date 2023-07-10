clc
clear
close all

global usegpu;
if gpuDeviceCount>0
    usegpu=true;
else
    usegpu=false;
end

load('../dataset/cifar-10/cifar_10_2d.mat')
images_tr=single(images_tr)/255;
images_ts=single(images_ts)/255;

images_tr=gpuarray(images_tr);
images_ts=gpuarray(images_ts);
images_tr=dlarray(single(images_tr),'SSCB');  
images_ts=dlarray(single(images_ts),'SSCB');  

filename='trained_dnn_conv_ep100_eta0.01_HP_221202_1843';
load([filename,'.mat'])

images_conv_tr=conv_layers(images_tr,dnn);
images_conv_ts=conv_layers(images_ts,dnn);

save cifar_10_convlayers.mat images_conv_tr images_conv_ts labels_tr labels_ts

function images_conv=conv_layers(images,dnn)
    num_images=size(images,4);
    batch_size=100;
    num_batches=num_images/batch_size;
    images_conv=zeros(dnn.full(1).input_size,num_images);
    for batch=1:num_batches     
        num_imgs=((batch-1)*batch_size+1):batch*batch_size;
        input=images(:,:,:,num_imgs);
    
        x=input;
        for ll=1:dnn.n_conv
            [~,~,~,p,~]=conv_forward(x,dnn.conv(ll));
            if ll<dnn.n_conv
                x=p;
            end
        end
        out=reshape(p,[],batch_size)';
        images_conv(:,num_imgs)=out';
    end
end