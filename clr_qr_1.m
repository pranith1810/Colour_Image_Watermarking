clc;
clear all
close all;
cd CIMG
[J, P]=uigetfile('*.*','select the cover Image');
I=imread(strcat(P,J));
I=imresize(I,[128 128]);
cd ..
figure,imshow(I);title('Input Image');
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
for cc=1:3
    k=1;
    IR=I(:,:,cc);
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
        T1{ii}=D1;
        dxy(ii)=ii;       
        for pp=1:length(D1)
            R(pp,pp)=real(R(pp,pp)+S1(pp,pp));
        end
        T2=R;          
        CMN=Q*R;
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
RF(:,:,cc)=uint8(RIMG);
end
        
figure,imshow(RF,[]);title('Encrypted Image');

%=== decoding 
k=1;
IRF=RF(:,:,1);
for ii=0:N-1
    for jj=0:M-1
        BLKF{k}=(IRF(ii*alp+[1:alp],jj*bet+[1:bet]));
        EF(k)=entropy(BLKF{k});
        k=k+1;
    end
end
TF=sum(EF)/(M*N);
dxf=find(EF<TF);
for ii=1:k-1
    if(find(dxf==ii)~=0)     
        PATF=BLKF{ii};
        [LL1,LH1,HL1,HH1]=dwt2(double(PATF),'haar');
        cmnf=czt(LL1);
        [Qf,Rf]=qr(cmnf);
        TFF=T1{ii};
        ZZ=zeros(size(TFF));
        DZ=diag(Rf)-(TFF)
        for pp=1:length(DZ)
            ZZ(pp,pp)=DZ(pp);
        end
        
      BFW{ii}=UF{ii}*ZZ*VF{ii}';
        
    end
end
%end
        
           
%         %===============
%         if ii<=t-1
%         [U1,S1,V1]=svd(BW{ii});
%         T1=R;
%         dxy(ii)=ii;
%        
%         for pp=1:length(D1)
%             R(pp,pp)=real(R(pp,pp)+S1(pp,pp));
%         end
%         T2=R;          
%         CMN=Q*R;
%         LLM1=abs(ifft(CMN));
%         RPAT=idwt2(LLM1,LH1,HL1,HH1,'haar');
%         RBLK{ii}=RPAT;
%         else
%         RBLK{ii}=PAT;
%         end       
%     else
%         RBLK{ii}=BLK{ii};
%     end
% end
% k=1;
% for ii=0:N-1
%     for jj=0:M-1
%         RIMG(ii*alp+[1:alp],jj*bet+[1:bet])=RBLK{k};
%         k=k+1;
%     end
% end
% RF(:,:,cc)=uint8(RIMG);
% end
%         
% figure,imshow(RF,[]);title('Encrypted Image');




    