clc
clear all
name='grid1xx';
dir='E:\py\nn\dcfls\fuck pic\subpic'
J=imread([name,'.jpg']);
%type:1表示最邻近法，2表示双极性插值法
[m,n,p]=size(J);
cv=rand(5,5)
h = fspecial('unsharp')
cv=[-1,2,-1;1,-1,1;-2,1,1]
%cv=[1,0;0,1]
window=len(cv)
for w=window+1:m-window-1
    for h=window+1:n-window-1
        pp(w,h)=sum(sum(conv2(double(cv),double(J([w-window:w+window],[h-window:h+window])))));
%         for t=1:window
%             ppx(w,h)=double(J([w-window:w+window],[h-window:h+window])
    end
    w
end

ppn=pp/max(max(pp))*256;
ppn=uint8(ppn);
imshow(ppn)
figure
% cv=[-1,2,-1;1,-1,1;-2,1,1]
jj=imfilter(J,cv);
imshow(jj)

figure
h = fspecial('unsharp')
j2=imfilter(J,h);
imshow(j2)





% type=1;
% 
% mul=.5
% 
% m1=m*mul;n1=n*mul;
% %****************************************************
% if type==1
% for i=1:m1
% for j=1:n1;
% b(i,j)=J(round(i/mul),round(j/mul));
% end
% end
% elseif type==2
% for i=1:m1-1
% for j=1:n1-1;
% u0=i/mul;v0=j/mul;
% u=round(u0);v=round(v0);
% s=u0-u;t=v0-v;
% b(i,j)=(J(u+1,v)-J(u,v))*s+(J(u,v+1)-J(u,v))*t+(J(u+1,v+1)+J(u,v)-J(u,v+1)-J(u+1,v))*s*t+J(u,v);
% end
% end
% end
% %*****************************************************
% b=uint8(b);
% imshow(b);
% title('处理后图像');
% y=b;
% %clear;  %此题是用最近邻域法实现图像缩放
% I=J;%读入图像
% %图像属性
% %  Filename: 'f.jpg'
% %       FileModDate: '24-Aug-2008 16:50:30'
% %           FileSize: 20372
% %             Format: 'jpg'
% %      FormatVersion: ''
% %              Width: 480
% %             Height: 640
% %           BitDepth: 8
% %          ColorType: 'grayscale'
% %    FormatSignature: ''
% %    NumberOfSamples: 1
% %       CodingMethod: 'Huffman'
% %      CodingProcess: 'Sequential'
% %            Comment: {}
% 
% [rows,cols]=size(I);
% 
% K1 = str2double(inputdlg('请输入行缩放倍数', 'INPUT scale factor', 1, {'0.6'}));%行默认变为原来的0.6倍
% K2 = str2double(inputdlg('请输入列缩放倍数', 'INPUT scale factor', 1, {'0.4'}));%列默认变为原来的0.4倍
% 
% width = K1 * rows;                       
% 
% height = K2 * cols;
% 
% im2 = uint8(zeros(width,height)); %定义输出图像矩阵
% 
% widthScale = rows/width;
% 
% heightScale = cols/height;
% 
% for x = 6:width - 6         %为防止矩阵溢出而选择的参数6           
% 
%    for y = 6:height - 6
% 
%        oldX = x * widthScale; %oldX，oldY为原坐标，x，y为新坐标      
% 
%        oldY = y * heightScale;
% 
%        if (oldX/double(uint16(oldX)) == 1.0) & (oldY/double(uint16(oldY)) == 1.0)      
% 
%            im2(x,y) = I(int16(oldX),int16(oldY));
% 
%        else                                   
% 
%            a = double(round(oldX));             
% 
%            b = double(round(oldY)); %若不是整数四舍五入后把临近值赋过去
% 
%            im2(x,y) = I(a,b);                  
% 
%        end
% 
%     end
% 
% end
% 
% imshow(I); %输出原图像
% 
% figure;
% 
% imshow(im2); %输出缩放后图像