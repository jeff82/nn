function fpz=funppz(z,pmz,ppz)
p=2;
r2=.04;
dp=0
while 1,
    m=0
    p=p+dp;
    pmz1=p,ppz1=p
for i=0:.5:40
    L1=0.5*oedfunp(i,ppz1,pmz1);
    m1=0.5*oedfunm(i,ppz1,pmz1);
    L2=0.5*oedfunp(i+0.5/2, ppz1+L1/2.0, pmz1+m1/2.0);
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
for i=0:.5:40
    L1=0.5*oedfunp(i,ppz2,pmz2);
    m1=0.5*oedfunm(i,ppz2,pmz2);
    L2=0.5*oedfunp(i+0.5/2., ppz2+L1/2.0, pmz2+m1/2.0);
	L2=0.5*oedfunp(i+0.5/2., ppz2+L1/2.0, pmz2+m1/2.0);
	m2=0.5*oedfunm(i+0.5/2., ppz2+L1/2.0, pmz2+m1/2.0);
	L3=0.5*oedfunp(i+0.5/2., ppz2+L2/2.0, pmz2+m2/2.0);
	m3=0.5*oedfunm(i+0.5/2., ppz2+L2/2.0, pmz2+m2/2.0);
	L4=0.5*oedfunp(i+0.5,ppz2+L3, pmz2+m3);
	m4=0.5*oedfunm(i+0.5,ppz2+L3, pmz2+m3);
end
f4=pmz2/ppz2;
m=m+1;
D=(f4-f3)/dh;
dp=(r2-f3)/D
if abs(dp)<1e-8
    break
end
end
    
    

    
        