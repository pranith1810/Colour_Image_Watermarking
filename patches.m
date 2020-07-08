clc;
clear all
close all;
cd CIMG
[J, P]=uigetfile('*.*','select the cover Image');
I=imread(strcat(P,J));
I=imresize(I,[512 512]);
alp=128;bet=128;
N=size(I,1)/alp; M=size(I,2)/bet;
k=1;
for ii=0:N-1
    for jj=0:M-1
        B{k}=(I(ii*alp+[1:alp],jj*bet+[1:bet]));
        k=k+1;
    end
end
figure,subplot(241),imshow(B{1});
subplot(242),imshow(B{2});
subplot(243),imshow(B{3});
subplot(244),imshow(B{4});
subplot(245),imshow(B{5});
subplot(246),imshow(B{6});
subplot(247),imshow(B{7});
subplot(248),imshow(B{8});

