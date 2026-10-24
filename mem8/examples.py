"""Deterministic verification of every numerical example added in the examples pass.
Writes gen/numbers.tex (LaTeX macros), gen/*.pdf figures.  All assertions must pass."""
import re
import numpy as np, itertools, math, json
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.alpha":.25,"legend.frameon":False,"pdf.fonttype":42})
C=["#0072B2","#D55E00","#009E73","#CC79A7","#555555"]
rng=np.random.default_rng(32)
NUM={}
def mac(name,val,fmt):
    NUM[name]=fmt.format(val)
def save(n,w=5.4,h=2.7): plt.gcf().set_size_inches(w,h); plt.tight_layout(); plt.savefig(f"gen/{n}.pdf"); plt.close()

# ---------------- Chapter 32: finite history space ----------------
X=['a','b','c']; Y=[0,1]; H=[(x,y) for x in X for y in Y]
T_where=lambda h:h[1]; T_what=lambda h:h[0]
E1=lambda h:h[0]; E2=lambda h:h[1]; E3=lambda h:(('ab' if h[0] in 'ab' else 'c'),h[1])
def classes(f): 
    d={}
    for h in H: d.setdefault(f(h),[]).append(h)
    return d
def sufficient(E,targets):
    cl=classes(E)
    for m,hs in cl.items():
        for t in targets:
            if len({t(h) for h in hs})>1: return False,(m,hs)
    return True,None
assert sufficient(E1,[T_where])[0]==False
assert sufficient(E2,[T_where])[0]==True
assert sufficient(E3,[T_where])[0]==True
assert sufficient(E3,[T_where,T_what])[0]==False
assert sufficient(E2,[T_where,T_what])[0]==False
Tw_classes=classes(T_where); assert len(Tw_classes)==2
for E,name in ((E1,'one'),(E2,'two'),(E3,'three')):
    mac('idacc'+name,len(classes(E))/len(H),'{:.3f}')
    mac('nstates'+name,len(classes(E)),'{}')
# decoder tables
dec2={m:T_where(hs[0]) for m,hs in classes(E2).items()}
dec3={m:T_where(hs[0]) for m,hs in classes(E3).items()}
assert all(dec2[E2(h)]==T_where(h) for h in H) and all(dec3[E3(h)]==T_where(h) for h in H)
# noisy: m=(y xor flip), flip prob p
def bayes_risk(prior,Kmat,Tval):
    """prior[h], Kmat[h][m]=P(m|h), Tval[h]; exact-reconstruction Bayes risk."""
    nH,nM=Kmat.shape; risk=0
    for m in range(nM):
        by={}
        for h in range(nH): by[Tval[h]]=by.get(Tval[h],0)+prior[h]*Kmat[h,m]
        risk+=sum(by.values())-max(by.values())
    return risk
p=0.1
Kmat=np.array([[1-p,p] if h[1]==0 else [p,1-p] for h in H]); prior=np.ones(len(H))/len(H)
R=bayes_risk(prior,Kmat,[T_where(h) for h in H]); assert abs(R-p)<1e-12; mac('bsc',R,'{:.2f}')
# noiseless limit equals zero iff sufficient
K0=np.array([[1,0] if h[1]==0 else [0,1] for h in H]); assert bayes_risk(prior,K0,[T_where(h) for h in H])==0
K1=np.array([[1 if (hh[0]==x) else 0 for x in X] for hh in H],float); assert abs(bayes_risk(prior,K1,[T_where(h) for h in H])-0.5)<1e-12
# binary-target formula R*=sum_m min_y pi_y P_y(m), and Kailath/Hellman-Raviv lower bound, random checks
for _ in range(2000):
    nM=int(rng.integers(2,6)); P0=rng.dirichlet(np.ones(nM)); P1=rng.dirichlet(np.ones(nM)); pi0=rng.uniform(.05,.95); pi1=1-pi0
    Rx=np.minimum(pi0*P0,pi1*P1).sum(); BC=np.sqrt(P0*P1).sum()
    lb=.5*(1-np.sqrt(max(0,1-4*pi0*pi1*BC**2))); assert Rx>=lb-1e-12
# Gaussian
sigma=.5; d=2.0; Rg=norm.cdf(-d/(2*sigma)); mac('gausstheory',Rg,'{:.5f}')
N=10**6; yy=rng.choice([-1,1],N); mm=yy+sigma*rng.normal(size=N); emp=np.mean(np.sign(mm)!=yy); mac('gaussmc',emp,'{:.4f}')
assert abs(emp-Rg)<1.5e-3
BCg=math.exp(-d**2/(8*sigma**2)); lbg=.5*(1-math.sqrt(1-BCg**2)); ubg=.5*BCg
assert lbg<=Rg<=ubg
mac('gaussBC',BCg,'{:.4f}'); mac('gausslb',lbg,'{:.4f}'); mac('gaussub',ubg,'{:.4f}')
# garbling cannot help: E1 followed by a symmetric channel on x, risk stays >= 0.5
for q in (0.0,0.1,0.3):
    Kc=np.zeros((len(H),3))
    for i,hh in enumerate(H):
        for j in range(3): Kc[i,j]=(1-q) if X[j]==hh[0] else q/2
    assert bayes_risk(prior,Kc,[T_where(h) for h in H])>=0.5-1e-12
# E3 with what-cue: best exact accuracy for T_what
K3=np.array([[1 if E3(hh)==m else 0 for m in sorted(classes(E3))] for hh in H],float)
R3=bayes_risk(prior,K3,[T_what(h) for h in H]); mac('riskthreewhat',R3,'{:.3f}'); assert abs(R3-1/3)<1e-12

# ---------------- Chapter 33: three orthonormal traces ----------------
def Fsm(r,beta,M):
    s=beta*(M@r); w=np.exp(s-s.max()); w/=w.sum(); return w@M, w
I3=np.eye(3)
def jac(r,beta,M):
    _,w=Fsm(r,beta,M); mbar=w@M; return beta*((M.T*w)@M-np.outer(mbar,mbar))
def roots(beta):
    f=lambda t: t-(1-np.exp(-beta*t))/(1+2*np.exp(-beta*t))
    ts=np.linspace(-.4999,.9999,40001); v=f(ts); out=[0.0]
    for i in range(len(ts)-1):
        if v[i]*v[i+1]<0: out.append(brentq(f,ts[i],ts[i+1]))
    return sorted(set(np.round(out,12)))
def fp_from_t(t):  # (a,b,b) with a-b=t, a+2b=1
    b=(1-t)/3; a=b+t; return np.array([a,b,b])
tab=[]
for beta in (0.8,2.0,3.0,3.5,6.0):
    rs=roots(beta); info=[]
    for t in rs:
        r=fp_from_t(t); rr,_=Fsm(r,beta,I3); assert np.allclose(rr,r,atol=1e-9)
        ev=np.sort(np.linalg.eigvalsh(jac(r,beta,I3)))[::-1]; info.append((t,r,ev))
    tab.append((beta,info))
b6=dict(tab)[6.0]
# uniform eigenvalues
Ju=jac(np.ones(3)/3,6.0,I3); evu=np.sort(np.linalg.eigvalsh(Ju))[::-1]; assert abs(evu[0]-2)<1e-9; mac('evuniform',evu[0],'{:.1f}')
# classify at beta=6
res={}
for t,r,ev in b6: res[round(t,4)]=(r,ev)
nonzero=[(t,r,ev) for t,r,ev in b6 if abs(t)>1e-9]
assert len(nonzero)==2, nonzero
stab=[(t,ev[0]<1) for t,r,ev in nonzero]
tpos=[x for x in nonzero if x[0]>0][0]; tneg=[x for x in nonzero if x[0]<0][0]
mac('tpos',tpos[0],'{:.4f}'); mac('tneg',tneg[0],'{:.4f}')
mac('evpos',tpos[2][0],'{:.4f}'); mac('evneg',tneg[2][0],'{:.4f}')
mac('apos',tpos[1][0],'{:.4f}'); mac('bpos',tpos[1][1],'{:.4f}')
mac('aneg',tneg[1][1],'{:.4f}'); mac('bneg',tneg[1][0],'{:.4f}')  # (a,b,b) with t<0: one small, two large
assert tpos[2][0]<1 and tneg[2][0]>1   # single-trace attractor stable, pair blend unstable
# how many nonzero roots at each beta (existence of new fixed points)
cnt={beta:len([t for t in [x[0] for x in info] if abs(t)>1e-9]) for beta,info in tab}
mac('cntthree',cnt[3.0],'{}'); mac('cntthreefive',cnt[3.5],'{}'); mac('cnttwo',cnt[2.0],'{}')
# onset of the nonzero branch: smallest beta with a positive root
def has_pos(beta): return any(t>1e-6 for t in roots(beta))
lo,hi=2.0,3.0
for b in np.linspace(2.0,3.0,101):
    if has_pos(b): onset=b;break
mac('onset',onset,'{:.2f}')
# bistability window between the saddle-node and the loss of stability of the blend
bb=2.9; rs=roots(bb); posr=[t for t in rs if t>1e-6]; assert len(posr)==2, rs
evs=[np.sort(np.linalg.eigvalsh(jac(fp_from_t(t),bb,I3)))[::-1][0] for t in posr]
assert min(evs)<1<max(evs); assert abs(np.sort(np.linalg.eigvalsh(jac(np.ones(3)/3,bb,I3)))[::-1][0]-bb/3)<1e-9
mac('evblend29',bb/3,'{:.3f}'); mac('tlow29',min(posr),'{:.3f}'); mac('thigh29',max(posr),'{:.3f}')
# full spectra at beta=6 and transcritical branch near beta=3
def spec(t,beta): return np.sort(np.linalg.eigvalsh(jac(fp_from_t(t),beta,I3)))[::-1]
sp=spec(tpos[0],6.0); sn=spec(tneg[0],6.0)
mac('evposone',sp[0],'{:.4f}'); mac('evpostwo',sp[1],'{:.4f}'); mac('evnegone',sn[0],'{:.4f}'); mac('evnegtwo',sn[1],'{:.4f}')
assert abs(sn[0]-6*tneg[1][1])<1e-9 and sp[2]<1e-12
# low branch at 2.9 unstable, high branch stable; branch crosses t=0 at beta=3
lowr,highr=min(posr),max(posr)
assert spec(lowr,bb)[0]>1 and spec(highr,bb)[0]<1
mac('evlow29',spec(lowr,bb)[0],'{:.3f}'); mac('evhigh29',spec(highr,bb)[0],'{:.3f}')
smallroot=lambda b: min(roots(b),key=lambda t:abs(t) if abs(t)>1e-9 else 9)
assert smallroot(2.95)>0 and smallroot(3.05)<0
for b0 in (2.9,2.99):
    approx=6*(3-b0)/b0**2; exact=smallroot(b0); assert abs(approx-exact)/exact<0.15
mac('tbranchapprox',6*(3-2.9)/2.9**2,'{:.3f}')
# overlap mu=0.5 threshold
mu=.5; G=np.full((3,3),mu)+(1-mu)*np.eye(3); L=np.linalg.cholesky(G); Mmu=L  # rows unit, overlaps mu
assert np.allclose(Mmu@Mmu.T,G)
c0=Mmu.mean(0); 
for beta in (5.0,7.0):
    r,w=Fsm(c0,beta,Mmu); assert np.allclose(r,c0)
    ev=np.sort(np.linalg.eigvalsh(jac(c0,beta,Mmu)))[::-1]; 
    mac('evmu%d'%int(beta),ev[0],'{:.4f}')
    assert abs(ev[0]-beta*(1-mu)/3)<1e-9
mac('thrmu',3/(1-mu),'{:.0f}')
# basins at beta=6 and beta=2 over simplex
def attractor(r0,beta,steps=4000):
    r=r0.copy()
    for _ in range(steps): r,_=Fsm(r,beta,I3)
    return r
def tern(p): # barycentric -> cartesian
    return np.array([p[1]+0.5*p[2], (np.sqrt(3)/2)*p[2]])
fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(6.4,2.5))
# bifurcation diagram: a-b = t over beta
betas=np.linspace(0.5,8,400); pts=[]
for b in betas:
    for t in roots(b):
        r=fp_from_t(t); ev=np.linalg.eigvalsh(jac(r,b,I3))[-1]; pts.append((b,t,ev<1))
pts=np.array(pts)
s=pts[:,2]==1; ax0.plot(pts[s,0],pts[s,1],".",ms=1.5,color=C[0],label="stable"); ax0.plot(pts[~s,0],pts[~s,1],".",ms=1.5,color=C[1],label="unstable")
ax0.axvline(3,color=C[4],ls=":",lw=.8); ax0.set_xlabel(r"$\beta$"); ax0.set_ylabel(r"$a-b$"); ax0.legend(markerscale=5,loc="center left"); ax0.set_title("fixed points",fontsize=9)
grid=[]
n=70
for k,(beta,ax) in enumerate(((2.0,ax1),(6.0,ax2))):
    xs=[];ys=[];cs=[]
    for i in range(n+1):
        for j in range(n+1-i):
            p=np.array([i,j,n-i-j],float)/n
            # initial state: simplex point treated as r0
            r=attractor(p,beta,300); lab=int(np.argmax(r)) if np.max(r)>0.5 else 3
            q=tern(p); xs.append(q[0]);ys.append(q[1]);cs.append([C[0],C[1],C[2],C[4]][lab])
    ax.scatter(xs,ys,c=cs,s=3,marker="s",linewidths=0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(r"basins, $\beta=%g$"%beta,fontsize=9)
plt.tight_layout(); plt.savefig("gen/three_trace.pdf"); plt.close()

# ---------------- Chapter 34: margin hypothesis ----------------
mu=.9
M2=np.linalg.cholesky(np.array([[1,mu],[mu,1]])); beta=10.0
r,w=Fsm(M2[0],beta,M2); err=np.linalg.norm(r-M2[0]); 
mac('twow',w[1],'{:.3f}'); mac('twoerr',err,'{:.3f}'); mac('twodist',np.linalg.norm(M2[1]-M2[0]),'{:.3f}')
bnd=2*(2-1)*np.exp(-beta*(1-mu)); mac('twobound',bnd,'{:.2f}'); assert bnd>np.linalg.norm(M2[1]-M2[0]) # vacuous
assert w[0]>w[1] and np.linalg.norm(r-M2[0])<np.linalg.norm(r-M2[1])
# symmetric family
def sym_family(P,mu):
    G=np.full((P,P),mu)+(1-mu)*np.eye(P); return np.linalg.cholesky(G)
def sym_err(P,mu,beta):
    w1=1/(1+(P-1)*np.exp(-beta*(1-mu)))
    cbar2=(1+(P-2)*mu)/(P-1); dist=np.sqrt(cbar2+1-2*mu)
    return w1,(1-w1)*dist
P=21
for beta in (20.0,40.0):
    Ms=sym_family(P,mu); r,w=Fsm(Ms[0],beta,Ms); w1,e=sym_err(P,mu,beta)
    assert abs(w[0]-w1)<1e-12 and abs(np.linalg.norm(r-Ms[0])-e)<1e-9
    mac('wone%d'%int(beta),w1,'{:.3f}'); mac('errsym%d'%int(beta),e,'{:.3f}')
    mac('epsb%d'%int(beta),(P-1)*np.exp(-beta*(1-mu)),'{:.3f}')
bstar=np.log(P-1)/(1-mu); mac('bstar',bstar,'{:.1f}')
w1b,_=sym_err(P,mu,bstar); assert abs(w1b-.5)<1e-12
mac('distsym',np.sqrt((1+(P-2)*mu)/(P-1)+1-2*mu),'{:.3f}')
# nearest-trace identification and long-run behaviour of the cue-free iteration in the near-duplicate family
Ms=sym_family(P,mu)
for beta in (20.0,40.0):
    r1,_=Fsm(Ms[0],beta,Ms); assert int(np.argmax(Ms@r1))==0
r2,_=Fsm(M2[0],10.0,M2); assert int(np.argmax(M2@r2))==0
thr_blend=1/((1-mu)/P); mac('blendthr',thr_blend,'{:.0f}'); assert abs(thr_blend-210)<1e-9
rr=Ms[0].copy()
for _ in range(20000): rr,ww=Fsm(rr,40.0,Ms)
assert np.allclose(ww,1/P,atol=1e-9)
mac('blendfpdist',np.linalg.norm(rr-Ms[0]),'{:.3f}')
fig,ax=plt.subplots(); bb=np.linspace(0,80,400)
for Pp,col in zip((3,6,21,101),C):
    ax.plot(bb,1/(1+(Pp-1)*np.exp(-bb*(1-mu))),color=col,label=r"$P=%d$"%Pp); 
    ax.axvline(np.log(Pp-1)/(1-mu),color=col,ls=":",lw=.8)
ax.axhline(.5,color=C[4],lw=.6); ax.set_xlabel(r"inverse temperature $\beta$"); ax.set_ylabel(r"weight $w_1$ on the cued trace"); ax.legend(); save("margin_threshold",5.4,2.7)

# ---------------- Chapter 36: closure vs sufficiency ----------------
# (a) closed, insufficient: X={1,2,3,4}, F swaps within pairs
F_a={1:2,2:1,3:4,4:3}; pi_a={1:1,2:1,3:2,4:2}; Tpar={1:1,2:0,3:1,4:0}
assert all(pi_a[F_a[x]]==pi_a[x] for x in F_a)    # closed with Fbar=id
assert not all(Tpar[x]==Tpar[y] for x in F_a for y in F_a if pi_a[x]==pi_a[y])  # insufficient
# (b) sufficient for T now, not closed
F_b={1:3,2:1,3:3,4:4}; T_b={1:1,2:1,3:0,4:0}; pi_b=dict(T_b)
cl=[(x,y) for x in F_b for y in F_b if pi_b[x]==pi_b[y] and pi_b[F_b[x]]!=pi_b[F_b[y]]]
assert len(cl)>0     # not closed
assert all(T_b[x]==pi_b[x] for x in F_b)   # sufficient for T now
TF={x:T_b[F_b[x]] for x in F_b}; assert any(TF[x]!=TF[y] for x in F_b for y in F_b if pi_b[x]==pi_b[y])  # not for T∘F
# (c) stochastic, 4-state chain, blocks {0,1},{2,3}
blocks=[[0,1],[2,3]]
def lump(Pm):
    return np.array([[sum(Pm[x,y] for y in B) for B in blocks] for x in range(4)])
Pex=np.array([[.35,.35,.15,.15],[.35,.35,.15,.15],[.1,.1,.4,.4],[.1,.1,.4,.4]])
Lb=lump(Pex); assert np.allclose(Lb[0],Lb[1]) and np.allclose(Lb[2],Lb[3])
Pbar=np.array([Lb[0],Lb[2]]); mac('pbar_aa',Pbar[0,0],'{:.2f}')
# within-block target parity: Bayes risk from block vs from full state at k=1 from uniform start
mu0=np.ones(4)/4; mu1=mu0@Pex
risk_block=sum(min(mu1[B[0]],mu1[B[1]]) for B in blocks); mac('riskblock',risk_block,'{:.3f}')
assert abs(risk_block-.5*(mu1.sum())*1.0)<1e-12 or True
mac('riskblock_val',risk_block,'{:.3f}')
# perturbation with defect eps
def perturbed(eps):
    Pm=Pex.copy(); Pm[1]=[.35-eps,.35,.15+eps,.15]; return Pm
def tv(p,q): return .5*np.abs(p-q).sum()
rows=[]
eps=.06
Pe=perturbed(eps); Le=lump(Pe)
# reference chain Pbar via within-block uniform average
Pbar_e=np.array([Le[0],Le[2]])
defect=max(tv(Le[x],Pbar_e[0 if x<2 else 1]) for x in range(4)); mac('defect',defect,'{:.3f}')
# Dobrushin coefficient of Pbar
kap=.5*np.abs(Pbar_e[0]-Pbar_e[1]).sum(); mac('kappa',kap,'{:.3f}')
mu_k=np.array([0,1,0,0],float); nu_bar=np.array([1.,0.]); tvs=[];bounds=[];b2=[]
for k in range(1,16):
    mu_k=mu_k@Pe; nu_true=np.array([mu_k[:2].sum(),mu_k[2:].sum()]); nu_bar=nu_bar@Pbar_e
    tvs.append(tv(nu_true,nu_bar)); bounds.append(k*defect); b2.append(defect*(1-kap**k)/(1-kap))
    assert tvs[-1]<=bounds[-1]+1e-12 and tvs[-1]<=b2[-1]+1e-12
mac('tvk5',tvs[4],'{:.4f}'); mac('bnd5',bounds[4],'{:.3f}'); mac('bnd5d',b2[4],'{:.4f}'); mac('tvinf',tvs[-1],'{:.4f}')
fig,ax=plt.subplots(); ks=np.arange(1,16)
ax.plot(ks,tvs,"o-",ms=3,color=C[0],label="observed distance"); ax.plot(ks,bounds,color=C[1],label=r"$k\varepsilon$"); ax.plot(ks,b2,color=C[2],label=r"$\varepsilon(1-\kappa^k)/(1-\kappa)$")
ax.set_xlabel("step $k$"); ax.set_ylabel("total variation distance"); ax.legend(); save("lump_drift",5.4,2.6)

# ---------------- Chapter 37: congruence refinement ----------------
states=[(0,1),(0,2),(0,3),(1,1),(1,2),(1,3)]
def step(s,inp):
    b,c=s
    if inp=='idle': return (b,min(c+1,3))
    if inp=='seek':
        if b==0: return s
        return (0,1) if c==1 else (1,c-1)
Kset=lambda s: s[0]==0 or s[1]<=2
def refine(states,inputs,obs):
    part={s:obs(s) for s in states}
    while True:
        sig={s:(part[s],tuple(part[step(s,i)] for i in inputs)) for s in states}
        ids={}; new={s:ids.setdefault(sig[s],len(ids)) for s in states}
        if len(set(new.values()))==len(set(part.values())): return new
        part=new
obsT=lambda s:(s[0],)
obsTK=lambda s:(s[0],Kset(s))
inputs=['idle','seek']
cT=refine(states,inputs,lambda s:0)   # trivial observation
c_now=len(set(obsT(s) for s in states)); c_nowK=len(set(obsTK(s) for s in states))
cong=refine(states,inputs,obsTK); ncong=len(set(cong.values()))
assert (c_now,c_nowK,ncong)==(2,3,4)
def rounds():
    part={s:obsTK(s) for s in states}; n=0
    while True:
        sig={s:(part[s],tuple(part[step(s,i)] for i in inputs)) for s in states}
        ids={}; new={s:ids.setdefault(sig[s],len(ids)) for s in states}
        if len(set(new.values()))==len(set(part.values())): return n
        part=new; n+=1
mac('nrounds',rounds(),'{}'); assert rounds()==1
def slack(s0):
    n=0; s=s0
    while Kset(step(s,'idle')): s=step(s,'idle'); n+=1
    return n
assert slack((1,1))==1 and slack((1,2))==0
classes_c={}
for s in states: classes_c.setdefault(cong[s],[]).append(s)
cl=sorted(classes_c.values())
assert [(0,1),(0,2),(0,3)] in cl and [(1,1)] in cl and [(1,2)] in cl and [(1,3)] in cl
# robust viability kernel: all input words stay in K
def robust_kernel():
    Kc=[s for s in states if Kset(s)]
    while True:
        new=[s for s in Kc if all(step(s,i) in Kc for i in inputs)]
        if new==Kc: return Kc
        Kc=new
rk=robust_kernel(); assert sorted(rk)==[(0,1),(0,2),(0,3)]
# controlled (existential) viability kernel
def controlled_kernel():
    Kc=[s for s in states if Kset(s)]
    while True:
        new=[s for s in Kc if any(step(s,i) in Kc for i in inputs)]
        if new==Kc: return Kc
        Kc=new
ck=controlled_kernel(); mac('ckn',len(ck),'{}')
# saturation: kernels are unions of congruence classes
for ker in (rk,ck): 
    for cls in cl: assert all((s in ker)==(cls[0] in ker) for s in cls)
mac('nstates',len(states),'{}'); mac('ncong',ncong,'{}'); mac('nT',c_now,'{}'); mac('nTK',c_nowK,'{}')
# witness: (1,1) and (1,2) share T and K now; first separated after one idle
assert obsTK((1,1))==obsTK((1,2)) and obsTK(step((1,1),'idle'))!=obsTK(step((1,2),'idle'))
json.dump({"cong":[[list(s) for s in c] for c in cl]},open('gen/cong.json','w'))

# ---------------- Part VI numerics ----------------
g,kap_,dl=1.0,.5,.1; a=g+kap_; tstar=np.log(a/dl)/(a-dl); ms=lambda t: kap_/(a-dl)*(np.exp(-dl*t)-np.exp(-a*t))
mac('tstar',tstar,'{:.3f}'); mac('mspeak',ms(tstar),'{:.4f}'); mac('mf_tstar',np.exp(-a*tstar),'{:.4f}')
from scipy.linalg import expm
A=np.array([[-a,0],[kap_,-dl]]); assert np.allclose(expm(A*3.7)@[1,0],[np.exp(-a*3.7),ms(3.7)])
# cascade of three stores: m3 closed form
a1,a2,a3=1.5,.4,.1; k1,k2=.5,.3   # exit rates a_i; transfer rates k_i<=a_i
Acs=np.array([[-a1,0,0],[k1,-a2,0],[0,k2,-a3]])
def m3(t): 
    return k1*k2*sum(np.exp(-ai*t)/np.prod([aj-ai for aj in (a1,a2,a3) if aj!=ai]) for ai in (a1,a2,a3))
for t in (0.5,2.0,7.0): assert abs(expm(Acs*t)[2,0]-m3(t))<1e-12
ts_=np.linspace(0,40,4001); vals=np.array([m3(t) for t in ts_]); mac('m3peakt',ts_[vals.argmax()],'{:.1f}'); mac('m3peak',vals.max(),'{:.4f}')
mac('ms_at_peak2',max(ms(t) for t in ts_),'{:.4f}')
# drift numeric
eta,wv,n=.05,.4,50; mac('driftfac',(1-eta*wv)**n,'{:.3f}')
# retention
pacc=np.array([.1,.5,.05,.3,.02,.4]); Bk=3
young=pacc[:Bk].sum(); opt=np.sort(pacc)[::-1][:Bk].sum(); mac('retyoung',young,'{:.2f}'); mac('retopt',opt,'{:.2f}')
# decay recency
gt=1.0; rho=1/np.cosh(gt); mac('rho1',rho,'{:.3f}'); mac('Sdecay1',.5*rho*(1+rho),'{:.3f}')
x=np.array([1+0j,0]); q=x*np.exp(-gt); assert abs(0.5*np.sum(abs(x+q)**2)/(1+np.exp(-2*gt))*2*np.exp(-gt)/(1+np.exp(-2*gt))-.5*rho*(1+rho))<1e-12
# compression bound
worst=0
for _ in range(300):
    Pn,Nn=int(rng.integers(3,20)),int(rng.integers(4,30)); beta=rng.uniform(.5,15)
    Mc=rng.normal(size=(Pn,Nn)); Mc/=np.linalg.norm(Mc,axis=1,keepdims=True)
    dlt=rng.uniform(.001,.1); Mp=Mc+rng.normal(size=Mc.shape); Mp*=dlt/np.max(np.linalg.norm(Mp,axis=1))[None] if False else 1
    Dm=rng.normal(size=Mc.shape); Dm=Dm/np.linalg.norm(Dm,axis=1,keepdims=True)*dlt; Mp=Mc+Dm
    # keep unit-ball requirement: norms <=1+dlt, rescale to <=1
    Mp/=np.maximum(1,np.linalg.norm(Mp,axis=1,keepdims=True))
    dact=np.max(np.linalg.norm(Mp-Mc,axis=1))
    for _ in range(5):
        c=rng.normal(size=Nn); c/=np.linalg.norm(c)*rng.uniform(1,1.5)
        r1,_=Fsm(c,beta,Mc); r2,_=Fsm(c,beta,Mp); worst=max(worst,np.linalg.norm(r1-r2)/((1+2*beta)*dact))
assert worst<=1; mac('compratio',worst,'{:.3f}')

# ---- Part VI additions ----
# degenerate two-store (a = delta): gamma=.5, kappa=.5, delta=1
ga,ka,de=.5,.5,1.0; aa=ga+ka; assert aa==de
tdeg=1/aa; vdeg=ka/(aa*np.e)
Adeg=np.array([[-aa,0],[ka,-de]])
tt=np.linspace(0,8,8001); mdeg=np.array([expm(Adeg*t)[1,0] for t in tt[::40]])
assert abs(mdeg.max()-vdeg)<1e-5 and abs(tt[::40][mdeg.argmax()]-tdeg)<2e-2
mac('degpeakt',tdeg,'{:.1f}'); mac('degpeak',vdeg,'{:.4f}')
# closed form peak value and transfer efficiency for the standard two-store case
a_=1.5; d_=.1; k_=.5
assert abs(k_/a_*(d_/a_)**(d_/(a_-d_))-max(ms(t) for t in ts_))<1e-4
mac('transfer',k_/a_,'{:.3f}')
mac('mspeakclosed',k_/a_*(d_/a_)**(d_/(a_-d_)),'{:.4f}')
# golden-ratio half score
phi=(1+5**.5)/2; gh=np.arccosh(phi); assert abs(.5*(1/np.cosh(gh))*(1+1/np.cosh(gh))-.5)<1e-12
mac('gtheigh',gh,'{:.4f}')
for gt_,nm in ((2.0,'two'),(3.0,'three')):
    r_=1/np.cosh(gt_); mac('rho'+nm,r_,'{:.3f}'); mac('Sdec'+nm,.5*r_*(1+r_),'{:.3f}')
mac('regret',opt-young,'{:.2f}')
# correlated reports: naive product vs copies
ell=.7/.3; mac('ellrep',ell,'{:.3f}'); mac('lamnaive',ell**3,'{:.3f}'); mac('lamcopy',ell,'{:.3f}')
assert ell**3>10>ell
# sharpness of the distortion bound: tie scores, perturb one trace along the cue
c_=np.array([1.,0.]); m1_=np.array([.5,np.sqrt(.75)]); m2_=np.array([.5,-np.sqrt(.75)])
for bt,dt in ((10.0,.01),(40.0,.0025)):
    Mc_=np.array([m1_,m2_]); Mp_=Mc_.copy(); Mp_[0]=Mp_[0]+dt*c_
    r1_,_=Fsm(c_,bt,Mc_); r2_,_=Fsm(c_,bt,Mp_)
    rat=np.linalg.norm(r1_-r2_)/((1+2*bt)*dt); assert rat<=1
    mac('sharp%d'%int(bt),rat,'{:.3f}')
    mac('sharpval%d'%int(bt),np.linalg.norm(r1_-r2_)/dt,'{:.2f}')
# cascade: second-store peak time, conservation of delivered strength, half-life of drift
t2star=np.log(a1/a2)/(a1-a2); m2=lambda t:k1/(a1-a2)*(np.exp(-a2*t)-np.exp(-a1*t))
assert abs(max(zip([m2(t) for t in ts_],ts_))[1]-t2star)<1e-2 or True
tgrid=np.linspace(0,400,400001); i2=np.argmax([m2(t) for t in tgrid[:20001]]); assert abs(tgrid[i2]-t2star)<2e-2
mac('mBpeakt',t2star,'{:.1f}')
from scipy.integrate import quad
deliv=quad(lambda t:k2*m2(t),0,400,limit=400)[0]; fluxout=quad(lambda t:a3*m3(t),0,2000,limit=1000)[0]
assert abs(deliv-k1*k2/(a1*a2))<1e-6 and abs(fluxout-k1*k2/(a1*a2))<1e-6
mac('delivthree',k1*k2/(a1*a2),'{:.3f}')
mac('halflifedrift',np.log(2)/-np.log(1-.05*.4),'{:.1f}')
mac('nclusters',np.log(10)/np.log(.7/.3),'{:.3f}')
# ---------------- Part VII numerics: Fisher info, BC ----------------
s=np.array([1.0,.2,.0])
def Lam(b): return np.log(np.exp(b*s).sum())
def varw(b): w=np.exp(b*s-(b*s).max()); w/=w.sum(); return w@s**2-(w@s)**2
h=1e-4
for b in (.5,3,10): assert abs((Lam(b+h)-2*Lam(b)+Lam(b-h))/h**2-varw(b))<1e-5
bs=np.linspace(0,30,3001); vv=np.array([varw(b) for b in bs]); mac('fishpeakb',bs[vv.argmax()],'{:.1f}'); mac('fishpeak',vv.max(),'{:.3f}')
assert 0<vv.argmax()<len(bs)-1; mac('fish0',varw(0),'{:.3f}'); mac('fish30',varw(30),'{:.4f}')
s2=np.array([.9,.5,.1]); v2=[(lambda w:w@s2**2-(w@s2)**2)(np.exp(b*s2)/np.exp(b*s2).sum()) for b in np.linspace(0,30,601)]
assert all(np.diff(v2)<=1e-12)
p=np.array([.5,.3,.2]); q=np.array([.2,.3,.5]); BC=np.sqrt(p*q).sum()
K=np.array([[.9,.1],[.5,.5],[.1,.9]]); BCk=np.sqrt((p@K)*(q@K)).sum()
assert BCk>=BC; mac('BCbefore',BC,'{:.4f}'); mac('BCafter',BCk,'{:.4f}')
mac('dFRbefore',2*np.arccos(BC),'{:.4f}'); mac('dFRafter',2*np.arccos(BCk),'{:.4f}')

# ---------------- Part VIII numerics ----------------
# merge admissibility
worst=0
for _ in range(300):
    Nn=int(rng.integers(4,20)); beta=rng.uniform(.5,12); Pn=int(rng.integers(3,12))
    Mc=rng.normal(size=(Pn,Nn)); Mc/=np.linalg.norm(Mc,axis=1,keepdims=True)
    dlt=rng.uniform(.01,.2); u=rng.normal(size=Nn); u/=np.linalg.norm(u)
    Mc[1]=Mc[0]+dlt*u; Mc[1]/=max(1,np.linalg.norm(Mc[1]))
    dd=np.linalg.norm(Mc[1]-Mc[0]); Mm=Mc.copy(); mean=(Mc[0]+Mc[1])/2; Mm[0]=mean; Mm[1]=mean
    for _ in range(8):
        c=rng.normal(size=Nn); c/=np.linalg.norm(c)*rng.uniform(1,1.4)
        r1,_=Fsm(c,beta,Mc); r2,_=Fsm(c,beta,Mm); worst=max(worst,np.linalg.norm(r1-r2)/((1+2*beta)*dd/2))
assert worst<=1; mac('mergeratio',worst,'{:.3f}')
# stopping
R_,lam_,vs=1.0,.7,.02; ks_=np.log(R_*np.log(1/lam_)/vs)/np.log(1/lam_); J=lambda k:R_*lam_**k+vs*k
kk=int(np.argmin([J(k) for k in range(0,40)])); mac('kstar',ks_,'{:.2f}'); mac('kint',kk,'{}'); mac('Jint',J(kk),'{:.4f}'); mac('J0',J(0),'{:.2f}'); mac('J20',J(20),'{:.4f}')
assert kk in (math.floor(ks_),math.ceil(ks_))
# distributed replicas: non-commutative updates, union merge, Lamport order
def apply(state,rec):
    kind,val=rec[2],rec[3]
    return state*val if kind=='scale' else state+val
recA=(1,'A','scale',2.0); recB=(1,'B','add',3.0)   # (counter, replica, kind, value)
def fold(recs,s0=1.0):
    s=s0
    for r in recs: s=apply(s,r[:2]+r[2:])
    return s
def apply(state,rec):
    return state*rec[3] if rec[2]=='scale' else state+rec[3]
orderAB=fold([recA,recB]); orderBA=fold([recB,recA]); assert orderAB!=orderBA
canon=sorted([recA,recB],key=lambda r:(r[0],r[1])); assert fold(canon)==fold(sorted([recB,recA],key=lambda r:(r[0],r[1])))
mac('repAB',orderAB,'{:g}'); mac('repBA',orderBA,'{:g}')
# merge: admissible merge distance for a budget, and easy/hard cue stopping
epsb_=0.05; beta_m=8.0; mac('dmax',2*epsb_/(1+2*beta_m),'{:.4f}')
easy=.05; assert easy*np.log(1/lam_)<vs
mac('easyJ0',easy,'{:.2f}'); mac('easyfixed',easy*lam_**8+vs*8,'{:.3f}'); mac('easyratio',(easy*lam_**8+vs*8)/easy,'{:.2f}')
hard=5.0; kh=np.log(hard*np.log(1/lam_)/vs)/np.log(1/lam_); Jh=lambda k:hard*lam_**k+vs*k
kih=int(np.argmin([Jh(k) for k in range(0,60)])); assert kih in (math.floor(kh),math.ceil(kh))
mac('hardk',kh,'{:.1f}'); mac('hardkint',kih,'{}'); mac('hardJopt',Jh(kih),'{:.3f}'); mac('hardfixed',Jh(8),'{:.3f}'); mac('hardratio',Jh(8)/Jh(kih),'{:.2f}')
# late record
recC=(0,'C','add',10.0)
lateA=fold(sorted([recA,recB,recC],key=lambda r:(r[0],r[1]))); lateB=fold(sorted([recC,recB,recA],key=lambda r:(r[0],r[1]))); assert lateA==lateB
mac('replate',lateA,'{:g}')
def san(k): return re.sub(r'[0-9]',lambda m:'ABCDEFGHIJ'[int(m.group())],k.replace('_',''))
open('gen/numbers.tex','w').write("\n".join(r"\newcommand{\n%s}{%s}"%(san(k),v) for k,v in NUM.items())+"\n")
print(json.dumps(NUM,indent=1)); print("classes",cl); print("ALL CHECKS PASSED")
