function h = visualize_second_filter(model)
% Visualize the second layer filter learned. The visualization is taken as
% a linear combination of first layer.

if isfield(model.layers{2}, 'w')
    model = merge_model(model);
end

pre_filter = reshape(model.layers{2}.uw, size(model.layers{2}.uw,1),[]);
pre_filter_len = size(model.layers{2}.uw,2);

stride = model.layers{2}.stride;
filter_num = size(model.layers{3}.uw,1);
filter_len = size(model.layers{3}.uw,2);

vol_patch_len = pre_filter_len + (filter_len - 1) * stride;
filter_display = zeros(filter_num, vol_patch_len ^ 3);

for f = 1 : filter_num
    vol_patch = zeros(vol_patch_len, vol_patch_len, vol_patch_len);
    the_filter = squeeze(model.layers{3}.uw(f,:,:,:,:));
    for x = 1 : size(the_filter,1)
        for y = 1 : size(the_filter,2)
            for z = 1 : size(the_filter, 3)
	        the_patch = pre_filter' * squeeze(the_filter(x, y, z, :));
	        vol_patch((x - 1) * stride + 1: (x - 1) * stride + pre_filter_len, (y - 1) * stride + 1: (y - 1) * stride + pre_filter_len, (z - 1) * stride + 1: (z - 1) * stride + pre_filter_len)...
	            = vol_patch((x - 1) * stride + 1: (x - 1) * stride + pre_filter_len, (y - 1) * stride + 1: (y - 1) * stride + pre_filter_len, (z - 1) * stride + 1: (z - 1) * stride + pre_filter_len) + ...
	            reshape(the_patch, pre_filter_len, pre_filter_len, pre_filter_len);
            end
        end
    end
    filter_display(f,:) = vol_patch(:);
end

filter_display = reshape(filter_display, filter_num, vol_patch_len, vol_patch_len, vol_patch_len);
visualize_filter(filter_display);
