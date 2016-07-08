clc
clear all
name='grid11xx';
dir='E:\py\nn\dcfls\fuck pic\subpic'
J=imread([name,'.jpg']);
%type:1��ʾ���ڽ�����2��ʾ˫���Բ�ֵ��
type=1;

mul=.5
[m,n]=size(J);
m1=m*mul;n1=n*mul;
%****************************************************
if type==1
for i=1:m1
for j=1:n1;
b(i,j)=J(round(i/mul),round(j/mul));
end
end
elseif type==2
for i=1:m1-1
for j=1:n1-1;
u0=i/mul;v0=j/mul;
u=round(u0);v=round(v0);
s=u0-u;t=v0-v;
b(i,j)=(J(u+1,v)-J(u,v))*s+(J(u,v+1)-J(u,v))*t+(J(u+1,v+1)+J(u,v)-J(u,v+1)-J(u+1,v))*s*t+J(u,v);
end
end
end
%*****************************************************
b=uint8(b);
imshow(b);
title('�����ͼ��');
y=b;
%clear;  %���������������ʵ��ͼ������
I=J;%����ͼ��
%ͼ������
%  Filename: 'f.jpg'
%       FileModDate: '24-Aug-2008 16:50:30'
%           FileSize: 20372
%             Format: 'jpg'
%      FormatVersion: ''
%              Width: 480
%             Height: 640
%           BitDepth: 8
%          ColorType: 'grayscale'
%    FormatSignature: ''
%    NumberOfSamples: 1
%       CodingMethod: 'Huffman'
%      CodingProcess: 'Sequential'
%            Comment: {}

[rows,cols]=size(I);

K1 = str2double(inputdlg('�����������ű���', 'INPUT scale factor', 1, {'0.6'}));%��Ĭ�ϱ�Ϊԭ����0.6��
K2 = str2double(inputdlg('�����������ű���', 'INPUT scale factor', 1, {'0.4'}));%��Ĭ�ϱ�Ϊԭ����0.4��

width = K1 * rows;                       

height = K2 * cols;

im2 = uint8(zeros(width,height)); %�������ͼ�����

widthScale = rows/width;

heightScale = cols/height;

for x = 6:width - 6         %Ϊ��ֹ���������ѡ��Ĳ���6           

   for y = 6:height - 6

       oldX = x * widthScale; %oldX��oldYΪԭ���꣬x��yΪ������      

       oldY = y * heightScale;

       if (oldX/double(uint16(oldX)) == 1.0) & (oldY/double(uint16(oldY)) == 1.0)      

           im2(x,y) = I(int16(oldX),int16(oldY));

       else                                   

           a = double(round(oldX));             

           b = double(round(oldY)); %���������������������ٽ�ֵ����ȥ

           im2(x,y) = I(a,b);                  

       end

    end

end

imshow(I); %���ԭͼ��

figure;

imshow(im2); %������ź�ͼ��