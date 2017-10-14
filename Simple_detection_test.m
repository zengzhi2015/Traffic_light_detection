close all
clear
clc
%% read and convert image to double tpye
img = imread('dayClip5--00015.png');
img(img==0) = 1;
img = double(img)/255;
%% build filters
red_filter = zeros(size(img));
red_filter(:,:,1) = 1;
yellow_filter = zeros(size(img));
yellow_filter(:,:,1) = sqrt(2)/2;
yellow_filter(:,:,2) = sqrt(2)/2;
green_filter = zeros(size(img));
green_filter(:,:,2) = 1;
%% calculate pixel direction
pixel_direction = img;
temp_norm = sqrt(pixel_direction(:,:,1).^2+pixel_direction(:,:,2).^2+pixel_direction(:,:,3).^2);
for i=1:3
    pixel_direction(:,:,i) = pixel_direction(:,:,i)./temp_norm;
end
%% Calculate color distances and weighted color channels
red_channel = max(0,sum(pixel_direction.*red_filter,3)*2.5-1.5).*sum(img.*red_filter,3);
yellow_channel = max(0,sum(pixel_direction.*yellow_filter,3)*2.5-1.5).*sum(img.*yellow_filter,3);
green_channel = max(0,sum(pixel_direction.*green_filter,3)*2.5-1.5).*sum(img.*green_filter,3);
figure(1)
subplot(3,3,2)
imshow(img)
subplot(3,3,4)
imshow(red_channel)
subplot(3,3,5)
imshow(yellow_channel)
subplot(3,3,6)
imshow(green_channel)
%% Create holo kernel
kernal_hole = [ 0, 0,-1,-1,-1,-1, 0, 0;
                0,-1, 0, 0, 0, 0,-1, 0;
               -1, 0, 0, 1, 1, 0, 0,-1;
               -1, 0, 1, 1, 1, 1, 0,-1;
               -1, 0, 1, 1, 1, 1, 0,-1;
               -1, 0, 0, 1, 1, 0, 0,-1;
                0,-1, 0, 0, 0, 0,-1, 0;
                0, 0,-1,-1,-1,-1, 0, 0;];
%% Set scales
scales = 1:0.5:5;
%% Deal with red light detection
best_scale = -1;
max_score = -1;
final_ratio = [];
for s = scales
    % rescale the kernal
    kernel_scaled = imresize(kernal_hole,s);
    % normalize the positive entries
    kernel_scaled(kernel_scaled>=0) = kernel_scaled(kernel_scaled>=0)/sum(sum(kernel_scaled(kernel_scaled>=0)));
    % normalize and PENALIZE the negative entries !!!!!!!!!!!!!
    kernel_scaled(kernel_scaled<0) = -1.5*kernel_scaled(kernel_scaled<0)/sum(sum(kernel_scaled(kernel_scaled<0)));
    % Convolution
    conv_red_temp = conv2(red_channel,kernel_scaled,'same');
    conv_yellow_temp = conv2(yellow_channel,kernel_scaled,'same');
    conv_green_temp = conv2(green_channel,kernel_scaled,'same');
    % Saturate
    conv_red_temp(conv_red_temp<0) = 0;
    conv_yellow_temp(conv_yellow_temp<0) = 0;
    conv_green_temp(conv_green_temp<0) = 0;
    % calculate the everage strength of the pulses
    mask_red_temp = conv_red_temp>0.1;
    mask_yellow_temp = conv_yellow_temp>0.1;
    mask_green_temp = conv_green_temp>0.1;
    strength_red = sum(sum(conv_red_temp(mask_red_temp)))/max(1,sum(sum(double(mask_red_temp))));
    strength_yellow = sum(sum(conv_yellow_temp(mask_yellow_temp)))/max(1,sum(sum(double(mask_yellow_temp))));
    strength_green = sum(sum(conv_green_temp(mask_green_temp)))/max(1,sum(sum(double(mask_green_temp))));
    % disp([strength_red,strength_yellow,strength_green])
    % update registers
    if max([strength_red,strength_yellow,strength_green]) > max_score
        max_score = max([strength_red,strength_yellow,strength_green]);
        best_scale = s;
        final_ratio = [strength_red,strength_yellow,strength_green];
    end
end
%% Show the result
% rescale the kernal
kernel_scaled = imresize(kernal_hole,best_scale);
% normalize the positive entries
kernel_scaled(kernel_scaled>=0) = kernel_scaled(kernel_scaled>=0)/sum(sum(kernel_scaled(kernel_scaled>=0)));
% normalize and PENALIZE the negative entries !!!!!!!!!!!!!
kernel_scaled(kernel_scaled<0) = -1.5*kernel_scaled(kernel_scaled<0)/sum(sum(kernel_scaled(kernel_scaled<0)));
% Convolution
conv_red_temp = conv2(red_channel,kernel_scaled,'same');
conv_yellow_temp = conv2(yellow_channel,kernel_scaled,'same');
conv_green_temp = conv2(green_channel,kernel_scaled,'same');
% Saturate
conv_red_temp(conv_red_temp<0) = 0;
conv_yellow_temp(conv_yellow_temp<0) = 0;
conv_green_temp(conv_green_temp<0) = 0;
subplot(3,3,7)
imshow(conv_red_temp)
subplot(3,3,8)
imshow(conv_yellow_temp)
subplot(3,3,9)
imshow(conv_green_temp)
% calculate the everage strength of the pulses
mask_red_temp = conv_red_temp>0.1;
mask_yellow_temp = conv_yellow_temp>0.1;
mask_green_temp = conv_green_temp>0.1;
strength_red = sum(sum(conv_red_temp(mask_red_temp)))/max(1,sum(sum(double(mask_red_temp))));
strength_yellow = sum(sum(conv_yellow_temp(mask_yellow_temp)))/max(1,sum(sum(double(mask_yellow_temp))));
strength_green = sum(sum(conv_green_temp(mask_green_temp)))/max(1,sum(sum(double(mask_green_temp))));
disp(strength_red)
disp(strength_yellow)
disp(strength_green)
if max([strength_red,strength_yellow,strength_green]) == strength_red
    disp('red light')
elseif max([strength_red,strength_yellow,strength_green]) == strength_yellow
    disp('yellow light')
else
    disp('green light')
end
