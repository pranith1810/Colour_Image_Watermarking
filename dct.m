clc;
clear all
close all;
cd CIMG
[J, P]=uigetfile('*.*','select the cover Image');
I=imread(strcat(P,J));
I=imresize(I,[512 512]);
cd ..
IM=rgb2gray(I);
B=dct2(IM);
figure,imshow(B);
