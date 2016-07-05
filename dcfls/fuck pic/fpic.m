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
PSF=fspecial('disk',r);   %�õ�����ɢ����
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
% [rows , cols , colors] = size(J);%�õ�ԭ��ͼ��ľ���Ĳ���  
% MidGrayPic = zeros(rows , cols);%�õõ��Ĳ�������һ��ȫ��ľ���������������洢������ķ��������ĻҶ�ͼ��  
% MidGrayPic = uint8(MidGrayPic);%��������ȫ�����ת��Ϊuint8��ʽ����Ϊ���������䴴��֮��ͼ����double�͵�  
% MyFirstGrayPic = rgb2gray(J);%�����еĺ�������RGB���Ҷ�ͼ���ת��     
% for i = 1:rows  
%     for j = 1:cols  
%         sum = 0;  
%         for k = 1:colors  
%             sum = sum + J(i , j , k) / 3;%����ת���Ĺؼ���ʽ��sumÿ�ζ���Ϊ��������ֶ����ܳ���255  
%         end  
%         MidGrayPic(i , j) = sum;  
%     end  
% end  
imwrite(J , './grid12xx.jpg' , 'jpg');  
%   
%    
% %��ʾת��֮��ĻҶ�ͼ��  
% figure(3);  
% imshow(MidGrayPic);