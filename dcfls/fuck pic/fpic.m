%  pic='grid2.bmp';     
%  uv=[144,26;26,198;144,416;308,241;19,29;303,30;307,325]; 
%  gm=[144,29;29,198;144,412;304,241;15,26;306,26;307,327]; 
%  og=[145,241];     
%  imtransform(pic,uv,gm,og);
clc
clear all
or1=imread('grid22.jpg');
or1=abs(250-or1);
%image(or1)
figure
r=1.2; 
PSF=fspecial('disk',r);   %得到点扩散函数
I1=imfilter(or1,PSF,'symmetric','conv');
image(I1)
%w=double(1.2*I1);
%w=w+0*(abs(randn(size(w))));
;
%image(uint8(w))
J=imnoise(I1,'gaussian',0.2,0.003);
J=imnoise(J,'salt & pepper',0.0002);
%J=rgb2gray(J);
image(J)
MyFirstGrayPic = rgb2gray(J);
figure(2);  
imshow(MyFirstGrayPic);
% HSIZE=[3,3];
% SIGMA=0.5;
% H = fspecial('prewitt')
% J=imfilter(J,H ,'symmetric','conv');
%  image(J)
% [rows , cols , colors] = size(J);%得到原来图像的矩阵的参数  
% MidGrayPic = zeros(rows , cols);%用得到的参数创建一个全零的矩阵，这个矩阵用来存储用下面的方法产生的灰度图像  
% MidGrayPic = uint8(MidGrayPic);%将创建的全零矩阵转化为uint8格式，因为用上面的语句创建之后图像是double型的  
% MyFirstGrayPic = rgb2gray(J);%用已有的函数进行RGB到灰度图像的转换     
% for i = 1:rows  
%     for j = 1:cols  
%         sum = 0;  
%         for k = 1:colors  
%             sum = sum + J(i , j , k) / 3;%进行转化的关键公式，sum每次都因为后面的数字而不能超过255  
%         end  
%         MidGrayPic(i , j) = sum;  
%     end  
% end  
imwrite(J , './grid12xx.jpg' , 'jpg');  
%   
%    
% %显示转化之后的灰度图像  
% figure(3);  
% imshow(MidGrayPic);