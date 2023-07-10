function images = loadMNISTImages(filename,resize)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');

fclose(fp);

images = reshape(images, numCols, numRows, numImages);

% resize with 0.5 scale resulting in 14*14 images
if exist('resize','var') && resize==0.5
    images2=zeros(14,14,numImages);
    for num=1:numImages
        for i=1:14
            for j=1:14
                images2(i,j,num)=max(max(images(2*i-1:2*i,2*j-1:2*j,num)));
            end
        end
    end
    images=images2;
end

% Reshape to #pixels x #examples
% images = reshape(images, [], numImages);
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end
