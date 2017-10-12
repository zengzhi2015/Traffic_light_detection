close all
clear
clc
%%
img = imread('dayClip10--00000.png');
img(img==0) = 1;
img = double(img)/255;
%% build filter
red_filter = zeros(size(img));
red_filter(:,:,2) = 1;
%%
pixel_direction = img;
temp_norm = sqrt(pixel_direction(:,:,1).^2+pixel_direction(:,:,2).^2+pixel_direction(:,:,3).^2);
for i=1:3
    pixel_direction(:,:,i) = pixel_direction(:,:,i)./temp_norm;
end

%%

%color_distance = sum(pixel_direction.*red_filter,3).*sum(img.*red_filter,3);
%color_distance = sum(pixel_direction.*red_filter,3).*sum(img.*red_filter,3);
% figure(1)
% imshow(sum(pixel_direction.*red_filter,3))
% figure(2)
% imshow(sum(img.*red_filter,3))
figure(3)
imshow(sum(pixel_direction.*red_filter,3).^2.*sum(img.*red_filter,3))

%% 
color_distance = sum(pixel_direction.*red_filter,3).^2.*sum(img.*red_filter,3);

%%
kernal3 = [-1,-1,-1;
           -1, 8,-1;
           -1,-1,-1;]/8;
kernal6 = [-1,-1,-1,-1,-1,-1;
           -1,-1, 2, 2,-1,-1;
           -1, 2, 2, 2, 2,-1;
           -1, 2, 2, 2, 2,-1;
           -1,-1, 2, 2,-1,-1;
           -1,-1,-1,-1,-1,-1]/24;
ksize = [8,8];
kernel_scaled = imresize(kernal6,ksize,'nearest');
kernel_scaled(kernel_scaled>=0) = kernel_scaled(kernel_scaled>=0)/sum(sum(kernel_scaled(kernel_scaled>=0)));
kernel_scaled(kernel_scaled<0) = -kernel_scaled(kernel_scaled<0)/sum(sum(kernel_scaled(kernel_scaled<0)));

disp(sum(kernel_scaled(kernel_scaled>=0)))
disp(sum(kernel_scaled(kernel_scaled<0)))

%% 