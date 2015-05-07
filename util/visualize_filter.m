function f = visualize_filter(W,facecolor)
% This function uses vol3d to visualize the first layer 3D filters learned.

if ~exist('facecolor','var'),
    facecolor = 'red';
end
warning off all;

filter_size = size(W,2);
filter_num = size(W,1);

if floor(sqrt(filter_num))^ 2 ~= filter_num
    n = ceil(sqrt(filter_num));
    while (mod(filter_num, n) ~= 0 && n < 1.2 * sqrt(filter_num))
        n = n + 1;
    end
    m = ceil(filter_num / n);
else
    n = sqrt(filter_num);
    m = n;
end

% rescale
%W = W - mean(W(:));
m = 1;
n = min(size(W,1),11);
W = W(1:m*n,:,:,:);
filter_num = m*n;
off = 0.3;

figure;
for w = 1 : filter_num
    filter = squeeze(W(w,:,:,:));
    %clim = max(abs(filter(:)));
    filter = (filter - min(filter(:))) ./ (max(filter(:)) - min(filter(:)));
    
    draw_size = [(filter_size+off) * n - off, (filter_size+off) * m - off,(filter_size+off) * n - off,(filter_size+off) * m - off];
    
    position_vector = [mod(w-1,n) * (filter_size+off) , (m-1-floor((w-1)/n)) * (filter_size+off) , filter_size, filter_size];
    subplot('Position', position_vector ./ draw_size);

    
    vol3d('cdata', filter, 'xdata', [0,1], 'ydata', [0,1], 'zdata', [0,1]);
    view([0.5,1,0.5]);
    colormap('gray');
    alphamap([linspace(0.2, 0, 255)]);
    
    axis vis3d ;
    %axis equal; 
    axis off;
    set(gcf, 'color', 'w');
end

axis image off;
warning on all;
