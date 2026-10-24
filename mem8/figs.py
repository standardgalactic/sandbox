import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.alpha":.25,"figure.figsize":(5.4,2.9),"legend.frameon":False,"pdf.fonttype":42})
C=["#0072B2","#D55E00","#009E73","#CC79A7","#555555"]
rng=np.random.default_rng(8)
def save(n): plt.tight_layout(); plt.savefig(f"figs/{n}.pdf"); plt.close()
def S(psi,q):
    e1=np.sum(abs(psi)**2);e2=np.sum(abs(q)**2)
    return .5*np.sum(abs(psi+q)**2)/(e1+e2)*2*np.sqrt(e1*e2)/(e1+e2)
def unit(n):
    z=rng.normal(size=n)+1j*rng.normal(size=n); return z/np.linalg.norm(z)
# 1 score vs scale
lam=np.logspace(-1.3,1.3,300); rho=2*lam/(1+lam**2)
fig,ax=plt.subplots()
ax.plot(lam,rho*(1+rho)/2,C[0],label="aligned")
ax.plot(lam,rho/2,C[2],label="orthogonal")
ax.plot(lam,rho*(1-rho)/2,C[1],label="antiphase")
ls=np.logspace(-1.2,1.2,14)
for sign,col in ((1,C[0]),(-1,C[1]),(0,C[2])):
    pts=[]
    for l in ls:
        p=unit(8)
        if sign==0:
            v=rng.normal(size=8)+1j*rng.normal(size=8); v=v-np.vdot(p,v).real*p; v/=np.linalg.norm(v)
            # make real inner product zero
            v=v-np.vdot(p,v).real*p; v/=np.linalg.norm(v); q=l*v
        else: q=sign*l*p
        pts.append(S(p,q))
    ax.plot(ls,pts,"o",ms=3,color=col)
ax.axhline(1/8,ls=":",color=C[4],lw=.8); ax.text(0.11,.135,"1/8",fontsize=8)
ax.set_xscale("log"); ax.set_xlabel(r"scale $\lambda=\|q\|/\|\psi\|$"); ax.set_ylabel(r"score $S$"); ax.legend(loc="upper right")
save("score_scale")
# 2 basin
P,N,beta=50,64,20.0
M=np.array([v/np.linalg.norm(v) for v in rng.normal(size=(P,N))])
rows=[]
for s in np.linspace(0,1.1,60):
    for _ in range(6):
        r=M[0]+s*rng.normal(size=N)/np.sqrt(N); r/=max(1,np.linalg.norm(r))
        sc=M@r; D=sc[0]-np.max(sc[1:]); w=np.exp(beta*(sc-sc.max())); w/=w.sum()
        rows.append((D,np.linalg.norm(w@M-M[0])))
rows=np.array(rows); rows=rows[rows[:,0]>0]
fig,ax=plt.subplots(); ax.semilogy(rows[:,0],rows[:,1]+1e-18,".",color=C[0],ms=4,label="observed error")
d=np.linspace(0.02,rows[:,0].max(),200); ax.semilogy(d,np.minimum(2,2*(P-1)*np.exp(-beta*d)),color=C[1],label=r"bound $2(P-1)e^{-\beta\Delta}$")
ax.set_ylim(1e-12,5); ax.set_xlabel(r"margin $\Delta$"); ax.set_ylabel(r"$\|F(r)-m_1\|$"); ax.legend(); save("basin")
# 3 contraction
beta=.8; Mc=M[:12]
def F(r): s=beta*(Mc@r); w=np.exp(s-s.max()); w/=w.sum(); return w@Mc
x=rng.normal(size=N); y=rng.normal(size=N)
for _ in range(400): x=F(x)
ref=x; y=3*rng.normal(size=N); d0=np.linalg.norm(y-ref); err=[d0]
for k in range(25): y=F(y); err.append(np.linalg.norm(y-ref))
k=np.arange(26); fig,ax=plt.subplots(); ax.semilogy(k,err,"o-",ms=3,color=C[0],label="observed distance to fixed point")
ax.semilogy(k,d0*beta**k,color=C[1],label=r"bound $\beta^k d_0$, $\beta=0.8$"); ax.set_xlabel("iteration $k$"); ax.set_ylabel("distance"); ax.legend(); save("contraction")
# 4 cap bound
fig,ax=plt.subplots(); aa=np.linspace(.05,.6,100)
for n,col in zip((16,32,64),C):
    Nn=2*n; T=300000; u=rng.normal(size=(T,Nn)); u/=np.linalg.norm(u,axis=1,keepdims=True)
    v=rng.normal(size=Nn); v/=np.linalg.norm(v); ov=u@v
    a=np.linspace(.1,.55,24); emp=np.array([(ov>=x).mean() for x in a]); m=emp>0
    ax.semilogy(a[m],emp[m],"o",ms=3,color=col); ax.semilogy(aa,(1-aa**2)**(n-1),color=col,label=f"$n={n}$")
ax.set_ylim(1e-7,1); ax.set_xlabel(r"overlap threshold $a$"); ax.set_ylabel(r"$\Pr[\mathrm{Re}\langle u,v\rangle\geq a]$"); ax.legend(); save("capacity_tail")
# 5 two store
g,kap,dl=1.0,.5,.1; a=g+kap; t=np.linspace(0,12,400)
mf=np.exp(-a*t); ms=kap/(a-dl)*(np.exp(-dl*t)-np.exp(-a*t)); ts=np.log(a/dl)/(a-dl)
fig,ax=plt.subplots(); ax.plot(t,mf,color=C[1],label="fast store $m_f$"); ax.plot(t,ms,color=C[0],label="slow store $m_s$")
ax.axvline(ts,ls=":",color=C[4]); ax.text(ts+.15,.6,r"$t^\star$",fontsize=9); ax.set_xlabel("time"); ax.set_ylabel("strength ($m_0=1$)"); ax.legend(); save("two_store")
# 6 kuramoto
Nk=30; K=np.abs(rng.normal(size=(Nk,Nk))); K=(K+K.T)/2; K*=(rng.random((Nk,Nk))<.3); K=np.triu(K,1); K=K+K.T; K*=1.0
kappa=np.max(np.abs(K).sum(1)); dt=.9/kappa; th=rng.uniform(0,2*np.pi,Nk); Vs=[];Rs=[]
def V(t): return -np.sum(np.triu(K,1)*np.cos(t[None,:]-t[:,None]))
for _ in range(400):
    Vs.append(V(th)); Rs.append(abs(np.exp(1j*th).mean()))
    th=th+dt*(K*np.sin(th[None,:]-th[:,None])).sum(1)
Vs=np.array(Vs); assert np.all(np.diff(Vs)<=1e-9)
fig,(a1,a2)=plt.subplots(1,2,figsize=(5.4,2.5)); a1.plot(Vs,color=C[0]); a1.set_xlabel("step"); a1.set_ylabel("$V$")
a2.plot(Rs,color=C[1]); a2.set_xlabel("step"); a2.set_ylabel("order parameter"); a2.set_ylim(0,1.02); save("kuramoto")
# 7 stopping
R,l,c=1.0,.7,.02; k=np.linspace(0,30,300); J=R*l**k+c*k; ks=np.log(R*np.log(1/l)/c)/np.log(1/l)
fig,ax=plt.subplots(); ax.plot(k,J,color=C[0],label=r"$J(k)=R\lambda^k+\varsigma k$"); ax.plot(k,R*l**k,color=C[2],ls="--",label="error bound"); ax.plot(k,c*k,color=C[1],ls="--",label="cost")
ax.plot([ks],[R*l**ks+c*ks],"o",color="k"); ax.annotate(r"$k^\star\approx%.1f$"%ks,(ks,R*l**ks+c*ks),(ks+3,.45),arrowprops=dict(arrowstyle="-",lw=.6)); ax.set_ylim(0,1.05)
ax.set_xlabel("refinement steps $k$"); ax.set_ylabel("total cost"); ax.legend(); save("stopping")
# 8 drift
Pp,Nd,beta,eta=5,32,6.0,.05
Md=np.array([v/np.linalg.norm(v) for v in rng.normal(size=(Pp,Nd))]); cue=Md[0]+Md[1]; cue/=np.linalg.norm(cue)
def spread(X): return np.mean([np.linalg.norm(X[i]-X[j]) for i in range(Pp) for j in range(i)])
X=Md.copy(); sp=[spread(X)]
for _ in range(300):
    s=beta*(X@cue); w=np.exp(s-s.max()); w/=w.sum(); r=w@X; X=X+eta*w[:,None]*(r-X); sp.append(spread(X))
fig,ax=plt.subplots(); ax.plot(sp,color=C[1],label="self-referential update"); ax.plot([sp[0]]*301,color=C[0],label="evidence-gated, no evidence")
ax.set_xlabel("retrievals"); ax.set_ylabel("mean pairwise distance of traces"); ax.set_ylim(0,None); ax.legend(); save("drift")
# 9 noise removal
Nn=128; fig,ax=plt.subplots(); ds=np.array([1,2,4,8,16,32,64,100]); emp=[]
for d in ds:
    B,_=np.linalg.qr(rng.normal(size=(Nn,d))); xi=rng.normal(size=(4000,Nn)); emp.append(np.mean(np.sum((xi@B)**2,1)/np.sum(xi**2,1)))
    Mm=B.T; r=rng.normal(size=Nn)
    def Fz(r): s=4*(Mm@r); w=np.exp(s-s.max()); w/=w.sum(); return w@Mm
    assert np.allclose(Fz(r),Fz(B@(B.T@r)))
ax.plot(ds,ds/Nn,color=C[1],label=r"$\dim V/N$"); ax.plot(ds,emp,"o",color=C[0],ms=4,label="simulated fraction of noise energy")
ax.set_xlabel(r"dimension of trace span $\dim V$ ($N=128$)"); ax.set_ylabel("noise energy retained"); ax.legend(); save("noise_removal")
print("ok",ks)
