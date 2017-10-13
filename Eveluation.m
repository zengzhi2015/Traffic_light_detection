close all
clear
clc
%% read and convert image to double tpye
image = imread('img_10_575.png');
flag = f_detection( image );
%%
imshow(image)
disp(flag)
%% 
folder_path = '';