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
bsz=8;
NW=size(Z,1)/bsz;MW=size(Z,2)/bsz;
t=1;
for ii=0:NW-1
    for jj=0:MW-1
        BW{t}=im2double(Z(ii*bsz+[1:bsz],jj*bsz+[1:bsz]));
        t=t+1;
    end
end

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
 %=== decoding 
   k=1;
   IRF=RF;
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
  
figure,subplot(221),imshow(IR,[]);title('Cover Image');
subplot(222);imshow(Z,[]);title('Embedding image');
subplot(223);imshow(RF,[]);title('Watermarked Image');
subplot(224);imshow(ZF,[]);title('Recovered Image');
figure,subplot(221),imshow(IR,[]);title('original cover image');
subplot(222);imhist(IR);title('Histogram of the cover Image');
subplot(223);imshow(RF,[]);title('Watermarked Image');
subplot(224),imhist(RF);title('Histogram of the Watermarked Image');
%=======
%=== attacks ===

disp('Enter 1 for JPEG compression attack');
disp('Enter 2 for Rotation attack');
disp('Enter 3 for noise attack');
disp('Enter 4 for Scaling attack');
INN=input('Please Enter your choice');
 if INN==1
    CF=80;
    imwrite(RF,'comp_orig.jpg','Quality',CF);
    Q=imread('Comp_orig.jpg');
end
if INN==2
    Q=imrotate(RF,1);   
    Q=imresize(Q,size(RF));
end
if INN==3    
    Q=imnoise(RF,'gaussian',0.001);
end
if INN==4  
Q=RF.*1.1;     
end

%=== decoding with attack 

   IRF=Q;k=1;
    for ii=0:N-1
       for jj=0:M-1
           BLKF{k}=(IRF(ii*alp+[1:alp],jj*bet+[1:bet]));
           EF(k)=entropy(BLKF{k});
           k=k+1;
       end
    end
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
        ZFA(ii*bsz+[1:bsz],jj*bsz+[1:bsz])=BFW{t};
        t=t+1;
    end
end
 
figure,subplot(221),imshow(Q,[]);title('Attacked Image');
subplot(222);imhist(Q);title('Histogram of the Attacked Image');
subplot(223);imshow(mat2gray(ZFA),[]);title('Recoveredd Image');
subplot(224),imhist(mat2gray(ZFA));title('Histogram of the Recovered Image');

