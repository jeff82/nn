
clc
clear all
name='grid1xx';
dir='E:\py\nn\dcfls\fuck pic\subpic'
J=imread([name,'.jpg']);
cell=32;

%  image(J)
[rows , cols , colors] = size(J); 
MidGrayPic = zeros(cell , cell);  
MidGrayPic = uint8(MidGrayPic);
npx=uint16(rows/cell);
npy=uint16(cols/cell);
total=npx+npy;
combine=[];
pos=1;
for i = 1:npx  
    for j = 1:npy   
        %MidGrayPic([1:32] , [1:32],:) = J([i:i+31] , [j:j+31],:); 
        MidGrayPic = uint8(MidGrayPic);
        xx=(i-1)*cell+1;
        yy=(j-1)*cell+1;
        
        MidGrayPic = J([xx:min(xx+cell,rows)] , [yy:min(yy+cell,cols)],:); 
%         subplot(9,9,pos);
%         %set(gca, 'Units', 'normalized', 'Position', [0 0.1 0.2 0.7]);
%         subimage(MidGrayPic)
%         box off
%         axis off
%         set(gca,'xtick',[]);
%         set(gca,'ytick',[]);
%         %figure
       %combine=[combine,MidGrayPic];
      % imwrite(MidGrayPic , ['./',name,'/grid12xx','--',num2str(pos),'.jpg'] , 'jpg');  
      imwrite(MidGrayPic , [dir,'\subgrid11','--',num2str(pos),'.jpg'] , 'jpg');  

        
        pos=pos+1
    end  
end  



imwrite(J , ['E:\py\nn\dcfls\fuck pic','\subpic','\grid12xex.jpg'] , 'jpg');  
