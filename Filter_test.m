close all
clear
clc
%% read and convert image to double tpye
img = imread('dayClip5--00015.png');
img(img==0) = 1;
img = double(img)/255;
%% build filter
red_filter = zeros(size(img));
red_filter(:,:,1) = 1;
%% calculate pixel direction
pixel_direction = img;
temp_norm = sqrt(pixel_direction(:,:,1).^2+pixel_direction(:,:,2).^2+pixel_direction(:,:,3).^2);
for i=1:3
    pixel_direction(:,:,i) = pixel_direction(:,:,i)./temp_norm;
end
%% test combination

figure(1)
imshow(img)
%color_distance = sum(pixel_direction.*red_filter,3).*sum(img.*red_filter,3);
%color_distance = sum(pixel_direction.*red_filter,3).*sum(img.*red_filter,3);
% figure(2)
% imshow(sum(pixel_direction.*red_filter,3))
% figure(3)
% imshow(sum(img.*red_filter,3))
% figure(4)
% imshow(sum(pixel_direction.*red_filter,3).^2.*sum(img.*red_filter,3))

%% 
figure(5)
% color_distance = sum(pixel_direction.*red_filter,3).^2.*sum(img.*red_filter,3);
color_distance = max(0,sum(pixel_direction.*red_filter,3)*2.5-1.5).*sum(img.*red_filter,3);%!!!!!!!!!!!!!!!
color_distance_show = color_distance/max(max(color_distance));
imshow(color_distance)

%% Create kernel
kernal_hole = [ 0, 0,-1,-1,-1,-1, 0, 0;
            0,-1, 0, 0, 0, 0,-1, 0;
           -1, 0, 0, 1, 1, 0, 0,-1;
           -1, 0, 1, 1, 1, 1, 0,-1;
           -1, 0, 1, 1, 1, 1, 0,-1;
           -1, 0, 0, 1, 1, 0, 0,-1;
            0,-1, 0, 0, 0, 0,-1, 0;
            0, 0,-1,-1,-1,-1, 0, 0;];
ksize = [30,30];
kernel_scaled = imresize(kernal_hole,ksize,'nearest');
kernel_scaled(kernel_scaled>=0) = kernel_scaled(kernel_scaled>=0)/sum(sum(kernel_scaled(kernel_scaled>=0)));
kernel_scaled(kernel_scaled<0) = -1.5*kernel_scaled(kernel_scaled<0)/sum(sum(kernel_scaled(kernel_scaled<0))); %!!!!!!

disp(sum(kernel_scaled(kernel_scaled>=0)))
disp(sum(kernel_scaled(kernel_scaled<0)))
%% Convolution
figure(6)
origin = color_distance;
result = conv2(origin,kernel_scaled,'same');
% result = result*1-0; % !!!!!!!!!!!!!!!!
result(result<0) = 0;
disp(min(min(result)))
disp(max(max(result)))
result_show = result/max(max(result));
imshow(result)
