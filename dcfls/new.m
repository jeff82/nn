clear all
<<<<<<< HEAD
wlp=920e-9;wls=1090e-9;tf=1e-3;sgms=2e-25;fpaa=.03;ap=3e-3;as=5e-3;af=5e-11;L=50;pp=30;
=======
close all
clc

wlp=920e-9;wls=1090e-9;tf=1e-3;sgms=2e-25;fpaa=.03;ap=3e-3;as=5e-3;af=5e-11;L=40;pp=10;
>>>>>>> origin/master
h=6.62e-34;
vp=3e8/wlp;
vs=3e8/wls;
ps=h*vs/(sgms*tf)*af;
r1=1;
af1=pi*(100e-12);
dh=.001;
dp=0

oedfunp=@(z,ppz,pmz)    sgms*tf/(h*vp)*fpaa/af*pp*exp(-(fpaa+ap)*z)*ppz/(1+(ppz+pmz)/ps)-as*ppz;
oedfunm=@(z,ppz,pmz)    -sgms*tf/(h*vp)*fpaa/af*pp*exp(-(fpaa+ap)*z)*pmz/(1+(ppz+pmz)/ps)+as*pmz;

<<<<<<< HEAD
p=2;
r2=.04;
dp=0;
m=0;
while 1,

    p=p+dp;
    pmz1=p;ppz1=p;
    for i=0:.5:L
=======

format long
p=1;
r2=.3;
dp=0;
m=0;
while m<15,
        p
       
        p=p+dp;
        pmz1=p;ppz1=p;
    for i=0:.5:40
>>>>>>> origin/master
        L1=0.5*oedfunp(i,ppz1,pmz1);
        m1=0.5*oedfunm(i,ppz1,pmz1);
        L2=0.5*oedfunp(i+0.5/2, ppz1+L1/2.0, pmz1+m1/2.0);
        m2=0.5*oedfunm(i+0.5/2, ppz1+L1/2.0, pmz1+m1/2.0);
        L3=0.5*oedfunp(i+0.5/2, ppz1+L2/2.0, pmz1+m2/2.0);
        m3=0.5*oedfunm(i+0.5/2, ppz1+L2/2.0, pmz1+m2/2.0);
        L4=0.5*oedfunp(i+0.5,ppz1+L3, pmz1+m3);
        m4=0.5*oedfunm(i+0.5,ppz1+L3, pmz1+m3);
        ppz1=ppz1+(1./6.)*(L1+2.0*L2+2.0*L3+L4);	
        pmz1=pmz1+(1./6.)*(m1+2.0*m2+2.0*m3+m4);
        x1(2*i+1)=ppz1;x2(2*i+1)=pmz1;z(2*i+1)=i;
    end
    f3=pmz1/ppz1;
    pmz2=p+dh;ppz2=p+dh;
<<<<<<< HEAD
    for i=0:.5:L
=======
    for i=0:.5:40
>>>>>>> origin/master
        L1=0.5*oedfunp(i,ppz2,pmz2);
        m1=0.5*oedfunm(i,ppz2,pmz2);
        L2=0.5*oedfunp(i+0.5/2., ppz2+L1/2.0, pmz2+m1/2.0);
        m2=0.5*oedfunm(i+0.5/2., ppz2+L1/2.0, pmz2+m1/2.0);
        L3=0.5*oedfunp(i+0.5/2., ppz2+L2/2.0, pmz2+m2/2.0);
        m3=0.5*oedfunm(i+0.5/2., ppz2+L2/2.0, pmz2+m2/2.0);
        L4=0.5*oedfunp(i+0.5,ppz2+L3, pmz2+m3);
        m4=0.5*oedfunm(i+0.5,ppz2+L3, pmz2+m3);
        ppz2=ppz2+(1./6.)*(L1+2.0*L2+2.0*L3+L4);	
        pmz2=pmz2+(1./6.)*(m1+2.0*m2+2.0*m3+m4);
    end
    f4=pmz2/ppz2;
    m=m+1;
    D=(f4-f3)/dh;
<<<<<<< HEAD
    dp=(r2-f3)/D;
    disp([m,dp,ppz1,f3*1e2,p])
    if abs(dp)<1e-8
=======
    dp=1.0*(r2-f3)/D;
    if abs(dp)<1e-9
>>>>>>> origin/master
        break
    end
end
    
k2{1}=x1;k2{2}=x2;k2{3}=z;
save 'k2' k2
  
%===================================


hold on
plot(z,x1)
plot(z,x2)
plot(z,(x1+x2)/2)
hold off
set(gca,'xlim',[0,40]);
set(gca,'ylim',[0,25]);
   