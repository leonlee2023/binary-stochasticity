clc
clear
close all

batch_size=10000;
num_batches=5;

images_tr=zeros(32,32,3,num_batches*batch_size);
labels_tr=[];

for i=1:num_batches
   load(['data_batch_',num2str(i),'.mat'])
   images_tr(:,:,:,((i-1)*batch_size+1):i*batch_size)=single(reshape(data',32,32,3,[]))/255;
   labels_tr=[labels_tr;labels];
end
labels_tr=one_hot(labels_tr,10)';

load('test_batch.mat')
images_ts=single(reshape(data',32,32,3,[]))/255;
labels_ts=one_hot(labels,10)';

load("batches.meta.mat")

save cifar10.mat images_tr labels_tr images_ts labels_ts label_names

for i=1:16
    ax=subplot(4,4,i);
    imagesc(rot90(images_tr(:,:,:,i),3));
    title(label_names{labels_tr(:,1)==1});
    ax.XTick=[];ax.YTick=[];
end