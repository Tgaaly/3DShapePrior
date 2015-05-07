function high_response = data_visualize(model, layer)
% data-driven visualization of the weights after learning.

addpath ..

debug = 0;
threshold = 0.998;
keep = 100;

kernels;
volume_size = 42;
data_size = volume_size + 2 * 3;
batch_size = 32;

base_path = '/n/fs/modelnet/ShapeMachine/data';
data_list = read_data_list(base_path, data_size, 'train', debug);

new_list = balance_data(data_list,batch_size);
n = length(new_list);
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));

num_filter = model.layers{layer}.layerSize(4);
patches = cell(num_filter,1);
scores = cell(num_filter,1);

for b = 1 : batch_num
    if mod(b,floor(batch_num/10)) == 0
        fprintf('processing %d batch of %d\n', b, batch_num);
    end
    batch_index = (b-1)*batch_size + 1 : b * batch_size;
    batch = read_batch(model, new_list(batch_index), false);
    activations = propagate_batch(model, batch, layer+1);
    
    for f = 1 : num_filter
        idx = find(squeeze(activations(:,:,:,:,f)) > threshold);
        [d,x,y,z] = ind2sub(size(squeeze(activations(:,:,:,:,f))), idx);
        for h = 1 : length(idx)
            temp = squeeze(batch(d(h),rf(model,layer,x(h)),rf(model,layer,y(h)),rf(model,layer,z(h))));
            patches{f}(:,end+1) = int8(temp(:));
            scores{f}(end+1) = activations(d(h),x(h), y(h), z(h), f);
        end
    end
end

for f = 1 : num_filter
    if length(scores{f}) > keep
        [~, idx] = sort(scores{f}, 'descend');
        patches_t{f} = patches{f}(:,idx(1:keep));
    end
end

field_size = length(rf(model,layer, 1));
high_response = zeros([field_size, field_size, field_size, num_filter],'single');
for f = 1 : num_filter
    high_response(:,:,:,f) = reshape(mean(patches_t{f},2),[field_size,field_size,field_size]);
end
high_response = permute(high_response,[4,1,2,3]);

function [field] = rf(model,layer, x)

left_most = x; right_most = x;
for l = layer : -1 : 2
    filterSize = model.layers{l}.kernelSize(1);
    stride = model.layers{l}.stride;
    left_most = stride * (left_most-1) + 1;
    right_most = stride * (right_most-1) + filterSize;
end
field = left_most : right_most;

