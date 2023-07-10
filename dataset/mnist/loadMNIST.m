clc
clear
close all

% LOAD DATASET
images_tr = loadMNISTImages('./train-images-idx3-ubyte');
labels_tr = loadMNISTLabels('./train-labels-idx1-ubyte');
images_ts = loadMNISTImages('./t10k-images-idx3-ubyte');
labels_ts = loadMNISTLabels('./t10k-labels-idx1-ubyte');

%images_tr = images_tr';
%images_ts = images_ts';

labels_tr = one_hot(labels_tr, 10)';
labels_ts = one_hot(labels_ts, 10)';

save MNIST.mat images_tr images_ts labels_tr labels_ts

for i=1:16
    ax=subplot(4,4,i);
    imagesc(images_tr(:,:,i)')
    colormap gray
    label=labels_tr(:,i);
    [~,num]=max(label);
    title(num2str(num-1))
    ax.XTick=[];ax.YTick=[];
end

