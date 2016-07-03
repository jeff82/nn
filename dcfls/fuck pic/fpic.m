%  pic='grid2.bmp';     
%  uv=[144,26;26,198;144,416;308,241;19,29;303,30;307,325]; 
%  gm=[144,29;29,198;144,412;304,241;15,26;306,26;307,327]; 
%  og=[145,241];     
%  imtransform(pic,uv,gm,og);
or1=imread('grid2.jpg');
or1=abs(255-or1);
image(or1)
r=3;%É¢½¹°ë¾¶r
PSF=fspecial('disk',r);   %µÃµ½µãÀ©É¢º¯Êý
I1=imfilter(or1,PSF,'symmetric','conv');
image(I1)
w=double(1.2*I1);
w=w+0*(abs(randn(size(w))));
;
image(uint8(w))