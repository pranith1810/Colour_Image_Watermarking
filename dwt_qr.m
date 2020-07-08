clc;
clear all
close all;
cd CIMG
[J, P]=uigetfile('*.*','select the cover Image');
I=imread(strcat(P,J));
I=imresize(I,[512 512]);
cd ..
[LL1,LH1,HL1,HH1]=dwt2(double(I),'haar');
LL1 = LL1/255;
figure,subplot(221),imshow(LL1,[]);
subplot(222),imshow(LH1);
subplot(223),imshow(HL1);
subplot(224),imshow(HH1);

