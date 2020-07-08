clc;
clear all
close all;
cd CIMG
[J, P]=uigetfile('*.*','select the cover Image');
I=imread(strcat(P,J));
I=imresize(I,[512 512]);
cd ..
%=========
cd WIMG
[X,Y]=uigetfile('*.*','select the wateremark Image');
Z=imread(strcat(Y,X));
Z=imresize(Z,[128 128]);
cd ..

IR=I(:,:,1);
IG=I(:,:,2);
IB=I(:,:,3);
AB = zeros(size(I, 1), size(I, 2), 'uint8');
just_red = cat(3, IR , AB , AB);
just_green = cat(3, AB, IG , AB);
just_blue = cat(3, AB, AB, IB);
figure,subplot(232),imshow(I);title('Cover Image');
subplot(234),imshow(just_red);title('red channel');
subplot(235),imshow(just_green);title('green channel');
subplot(236),imshow(just_blue);title('blue channel');
k=1;
alpha=128,beta=128;
N1=size(I,1)/alpha; M1=size(I,2)/beta;
for ii=0:N1-1
    for jj=0:M1-1
        B{k}=(I(ii*alpha+[1:alpha],jj*beta+[1:beta]));
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
alp=16;bet=16;
N=size(I,1)/alp; M=size(I,2)/bet;
k=1;
for ii=0:N-1
    for jj=0:M-1
        BLK{k}=(IR(ii*alp+[1:alp],jj*bet+[1:bet]));
        E(k)=entropy(BLK{k});
        k=k+1;
    end
end
T=sum(E)/(M*N);
dx=find(E<T);
bsz=8;++
NW=size(Z,1)/bsz;MW=size(Z,2)/bsz;
t=1;
for ii=0:NW-1
    for jj=0:MW-1
        BW{t}=im2double(Z(ii*bsz+[1:bsz],jj*bsz+[1:bsz]));
        t=t+1;
    end
end
[LL12,LH12,HL12,HH12]=dwt2(double(B{8}),'haar');
LL12 = LL12/255;
figure,subplot(221),imshow(LL12,[]);
subplot(222),imshow(LH12);
subplot(223),imshow(HL12);
subplot(224),imshow(HH12);

for ii=1:k-1
    if(find(dx==ii)~=0)     
        PAT=BLK{ii};
        [LL1,LH1,HL1,HH1]=dwt2(double(PAT),'haar');
        cmn=czt(LL1);
        [Q,R]=qr(cmn);
        D1=diag(R);        
        %===============
        if ii<=t-1
        [U1,S1,V1]=svd(BW{ii});        
        UF{ii}=U1;
        VF{ii}=V1;
        SF{ii}=S1;
        T1{ii}=D1;
        dxy(ii)=ii; 
        RM=R;
        for pp=1:length(D1)
            RM(pp,pp)=real(R(pp,pp)+S1(pp,pp));
        end
        T2{ii}=RM;            
        CMN=Q*RM;
        LLM1=abs(ifft(CMN));
        RPAT=idwt2(LLM1,LH1,HL1,HH1,'haar');
        RBLK{ii}=RPAT;
        else
        RBLK{ii}=PAT;
        end       
    else
        RBLK{ii}=BLK{ii};
    end
end
k=1;
for ii=0:N-1
    for jj=0:M-1
        RIMG(ii*alp+[1:alp],jj*bet+[1:bet])=RBLK{k};
        k=k+1;
    end
end
RF=uint8(RIMG);   
Watermarked_Image = cat(3, RF , IG , IB);
 %=== decoding 
   k=1;
   IRF=Watermarked_Image;
   for ii=0:N-1
       for jj=0:M-1
           BLKF{k}=(IRF(ii*alp+[1:alp],jj*bet+[1:bet]));
           EF(k)=entropy(BLKF{k});
           k=k+1;
       end
  end
  TF=sum(EF)/(M*N);
  dxf=find(EF<TF);
   for ii=1:t-1
       if(find(dxf==ii)~=0)     
           PATF=BLKF{ii};
           [LL1,LH1,HL1,HH1]=dwt2(double(PATF),'haar');
           cmnf=czt(LL1);
           [Qf,Rf]=qr(cmnf);
%           ===========
           ZZ=zeros(size(Qf));
           DZ=diag(T2{ii})-diag(Rf);
           for pp=1:length(DZ)
               ZZ(pp,pp)=DZ(pp);
           end       
          BFW{ii}=UF{ii}*ZZ*VF{ii}';
       else
         BFW{ii}=BW{ii};        
       end
   end
  t=1;
for ii=0:NW-1
    for jj=0:MW-1
        ZF(ii*bsz+[1:bsz],jj*bsz+[1:bsz])=BFW{t};
        t=t+1;
    end
end
  
figure,subplot(221),imshow(I,[]);title('Cover Image');
subplot(222);imshow(Z,[]);title('Watermark');
subplot(223);imshow(Watermarked_Image,[]);title('Watermarked Image');
subplot(224);imshow(ZF,[]);title('Recovered Image');
figure,subplot(221),imshow(I,[]);title('original cover image');
subplot(222);imhist(IR);title('Histogram of the cover Image');
subplot(223);imshow(Watermarked_Image,[]);title('Watermarked Image');
subplot(224),imhist(RF);title('Histogram of the Watermarked Image');


