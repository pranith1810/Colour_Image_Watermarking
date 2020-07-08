clc()
clear(mstring('all'))
close(mstring('all'))
cd(mstring('CIMG'))
[J, P] = uigetfile(mstring('*.*'), mstring('select the cover Image'))
I = imread(strcat(P, J))
I = imresize(I, mcat([512, 512]))
cd(mstring('..'))
#=========
cd(mstring('WIMG'))
[X, Y] = uigetfile(mstring('*.*'), mstring('select the wateremark Image'))
Z = imread(strcat(Y, X))
Z = imresize(Z, mcat([128, 128]))
cd(mstring('..'))

IR = I(mslice[:], mslice[:], 1)
IG = I(mslice[:], mslice[:], 2)
IB = I(mslice[:], mslice[:], 3)
alp = 16
bet = 16; print bet

N = size(I, 1) / alp
M = size(I, 2) / bet; print M

k = 1
for ii in mslice[0:N - 1]:
    for jj in mslice[0:M - 1]:
        BLK(k).lvalue = (IR(ii * alp + mcat([mslice[1:alp]]), jj * bet + mcat([mslice[1:bet]])))
        E(k).lvalue = entropy(BLK(k))
        k = k + 1
    end
end
T = sum(E) / (M * N)
dx = find(E < T)
bsz = 8
NW = size(Z, 1) / bsz
MW = size(Z, 2) / bsz; print MW

t = 1
for ii in mslice[0:NW - 1]:
    for jj in mslice[0:MW - 1]:
        BW(t).lvalue = im2double(Z(ii * bsz + mcat([mslice[1:bsz]]), jj * bsz + mcat([mslice[1:bsz]])))
        t = t + 1
    end
end

for ii in mslice[1:k - 1]:
    if (find(dx == ii) != 0):
        PAT = BLK(ii)
        [LL1, LH1, HL1, HH1] = dwt2(double(PAT), mstring('haar'))
        cmn = czt(LL1)
        [Q, R] = qr(cmn)
        D1 = diag(R)
        #===============
        if ii <= t - 1:
            [U1, S1, V1] = svd(BW(ii))
            UF(ii).lvalue = U1
            VF(ii).lvalue = V1
            SF(ii).lvalue = S1
            T1(ii).lvalue = D1
            dxy(ii).lvalue = ii
            RM = R
            for pp in mslice[1:length(D1)]:
                RM(pp, pp).lvalue = real(R(pp, pp) + S1(pp, pp))
            end
            T2(ii).lvalue = RM
            CMN = Q * RM
            LLM1 = abs(ifft(CMN))
            RPAT = idwt2(LLM1, LH1, HL1, HH1, mstring('haar'))
            RBLK(ii).lvalue = RPAT
        else:
            RBLK(ii).lvalue = PAT
        end
    else:
        RBLK(ii).lvalue = BLK(ii)
    end
end
k = 1
for ii in mslice[0:N - 1]:
    for jj in mslice[0:M - 1]:
        RIMG(ii * alp + mcat([mslice[1:alp]]), jj * bet + mcat([mslice[1:bet]])).lvalue = RBLK(k)
        k = k + 1
    end
end
RF = uint8(RIMG)
#=== decoding 
k = 1
IRF = RF
for ii in mslice[0:N - 1]:
    for jj in mslice[0:M - 1]:
        BLKF(k).lvalue = (IRF(ii * alp + mcat([mslice[1:alp]]), jj * bet + mcat([mslice[1:bet]])))
        EF(k).lvalue = entropy(BLKF(k))
        k = k + 1
    end
end
TF = sum(EF) / (M * N)
dxf = find(EF < TF)
for ii in mslice[1:t - 1]:
    if (find(dxf == ii) != 0):
        PATF = BLKF(ii)
        [LL1, LH1, HL1, HH1] = dwt2(double(PATF), mstring('haar'))
        cmnf = czt(LL1)
        [Qf, Rf] = qr(cmnf)
        #           ===========
        ZZ = zeros(size(Qf))
        DZ = diag(T2(ii)) - diag(Rf)
        for pp in mslice[1:length(DZ)]:
            ZZ(pp, pp).lvalue = DZ(pp)
        end
        BFW(ii).lvalue = UF(ii) * ZZ * VF(ii).cT
    else:
        BFW(ii).lvalue = BW(ii)
    end
end
t = 1
for ii in mslice[0:NW - 1]:
    for jj in mslice[0:MW - 1]:
        ZF(ii * bsz + mcat([mslice[1:bsz]]), jj * bsz + mcat([mslice[1:bsz]])).lvalue = BFW(t)
        t = t + 1
    end
end

figure
subplot(221)
imshow(IR, mcat([]))
title(mstring('Cover Image'))

subplot(222)
imshow(Z, mcat([]))
title(mstring('Embedding image'))

subplot(223)
imshow(RF, mcat([]))
title(mstring('Watermarked Image'))

subplot(224)
imshow(ZF, mcat([]))
title(mstring('Recovered Image'))

figure
subplot(221)
imshow(IR, mcat([]))
title(mstring('original cover image'))

subplot(222)
imhist(IR)
title(mstring('Histogram of the cover Image'))

subplot(223)
imshow(RF, mcat([]))
title(mstring('Watermarked Image'))

subplot(224)
imhist(RF)
title(mstring('Histogram of the Watermarked Image'))



#=======
#=== attacks ===

disp(mstring('Enter 1 for JPEG compression attack'))
INN = input(mstring('Please Enter your choice'))
if INN == 1:
    CF = 10
    imwrite(RF, mstring('comp_orig.jpg'), mstring('Quality'), CF)
    Q = imread(mstring('Comp_orig.jpg'))
end
#=== decoding with attack 
k = 1
IRF = Q
for ii in mslice[0:N - 1]:
    for jj in mslice[0:M - 1]:
        BLKF(k).lvalue = (IRF(ii * alp + mcat([mslice[1:alp]]), jj * bet + mcat([mslice[1:bet]])))
        EF(k).lvalue = entropy(BLKF(k))
        k = k + 1
    end
end

for ii in mslice[1:t - 1]:
    if (find(dxf == ii) != 0):
        PATF = BLKF(ii)
        [LL1, LH1, HL1, HH1] = dwt2(double(PATF), mstring('haar'))
        cmnf = czt(LL1)
        [Qf, Rf] = qr(cmnf)
        #           ===========
        ZZ = zeros(size(Qf))
        DZ = diag(T2(ii)) - diag(Rf)
        for pp in mslice[1:length(DZ)]:
            ZZ(pp, pp).lvalue = DZ(pp)
        end
        BFW(ii).lvalue = UF(ii) * ZZ * VF(ii).cT
    else:
        BFW(ii).lvalue = BW(ii)
    end
end
t = 1
for ii in mslice[0:NW - 1]:
    for jj in mslice[0:MW - 1]:
        ZFA(ii * bsz + mcat([mslice[1:bsz]]), jj * bsz + mcat([mslice[1:bsz]])).lvalue = BFW(t)
        t = t + 1
    end
end


figure
subplot(211)
imshow(Q, mcat([]))
title(mstring('Attacked Image'))

subplot(212)
imshow(mat2gray(ZFA), mcat([]))
title(mstring('Recovered Image'))
M file: 
