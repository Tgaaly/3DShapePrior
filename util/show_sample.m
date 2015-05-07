function show_sample(samples)
% a simple visualization tool using isosurface to show each 3D sample.

addpath ../voxelization;

n = size(samples,1);
for i = 1 : n
    the_sample = squeeze(samples(i,:,:,:,:));
    
    figure;
    %plot3D(the_sample);
    p = patch(isosurface(the_sample,0.1));
    set(p,'FaceColor','red','EdgeColor','none');
    daspect([1,1,1])
    view(3); axis tight
    camlight 
    lighting gouraud;
    axis off;
    set(gcf,'Color','white');
    title(i);
    pause;
    close(gcf);
end
