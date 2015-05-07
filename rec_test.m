function [predicted_label] = rec_test(model)
% Given the full 3D shape, recognition test for multi-class model.
% After the model is (generatively) trained, this function test the
% recognition ability(discriminative) of the model using a seperate test
% data.

if ~isfield('model.layers{2}','uw')
    model = merge_model(model);
end
global kConv_forward kConv_forward_c;

debug = 0;
num_layer = length(model.layers);
batch_size = 32;
data_size = 42 + 3 * 2;

base_path = '/n/fs/modelnet/ShapeMachine/data';
test_list = read_data_list(base_path, data_size, 'test', debug);
s = 0;
for c = 1 : length(test_list)
    s = s + length(test_list{c});
end
new_list = repmat(struct('filename', '', 'label', 0), s, 1);
now = 0;
for c = 1 : length(test_list)
    this_num = length(test_list{c});
    new_list(now+1:now+this_num) = test_list{c}(:);
    now = now + this_num;
end
n = now;
batch_num = ceil(n / batch_size);

predicted_label = zeros(n, model.classes); % prediction results.
for b = 1 : batch_num
    batch_end = min(n, b * batch_size);
	batch_index = (b-1)*batch_size + 1 : batch_end;
    batch_data = read_batch(model, new_list(batch_index), false);
    this_size = size(batch_data, 1);
	
    % propagate/inference bottum up using recognition weight. 
    for l = 2 : num_layer - 1
        if l == 2
            stride = model.layers{l}.stride;
            hidden_presigmoid = myConvolve(kConv_forward, batch_data, model.layers{l}.uw, stride, 'forward');
            hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        elseif strcmp(model.layers{l}.type, 'convolution')
            stride = model.layers{l}.stride;
            hidden_presigmoid = myConvolve(kConv_forward_c, batch_data, model.layers{l}.uw, stride, 'forward');
            hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        else
            batch_data = reshape(batch_data, size(batch_data,1), []);
            hidden_presigmoid = bsxfun(@plus, ...
                batch_data * model.layers{l}.uw, model.layers{l}.c);
        end
        batch_data = 1 ./ ( 1 + exp(-hidden_presigmoid) );
    end
	
    batch_data = reshape(batch_data, this_size, []);
    % calculate the free energy for each label hypothesis
    for c = 1 : model.classes
        try_label = zeros(this_size, model.classes);
        try_label(:, c) = 1;
        predicted_label((b-1) * batch_size+1: batch_end, c) = free_energy(model, [try_label, batch_data], num_layer);
    end
    
end

predicted_label = bsxfun(@rdivide, predicted_label, sum(predicted_label, 2));
[~, predicted_label] = max(predicted_label, [], 2);

test_label = zeros(n,1);
for i = 1 : n
    test_label(i) = new_list(i).label;
end

acc = sum(predicted_label == test_label) / n;
fprintf('acc is : %f\n', acc * 100);
