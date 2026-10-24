"""Deterministic arithmetic for Appendix N (reconciliation with the July 2025 MEM|8 specification).
Every quantity is either derived from the monograph's closed forms or computed directly from numbers
printed in the earlier specification. Nothing here reproduces or validates a result of that specification."""
import math, re, numpy as np
NUM={}
def mac(k,v,fmt): NUM[k]=fmt.format(v)
S=lambda g: (lambda r:.5*r*(1+r))(2*g/(1+g*g))          # resonance score, unit aligned query, trace amplitude g
# 1. emotional amplitude gain (alpha=0.3, beta=0.5 as printed in Section 3.4)
al,be=0.3,0.5; gmax=(1+al)*(1+be); gmin=(1-al)*(1+0)
assert abs(gmax-1.95)<1e-12 and abs(gmin-.7)<1e-12
for g in (.5,.7,1.3,1.95,3.0): assert abs(S(g)-S(1/g))<1e-12          # symmetry g <-> 1/g
assert S(1.0)==1.0 and all(S(g)<1 for g in (.7,1.95))
mac('gmax',gmax,'{:.2f}'); mac('Sgmax',S(gmax),'{:.3f}'); mac('Sgmin',S(gmin),'{:.3f}'); mac('ginv',1/gmax,'{:.3f}')
beta=8.0; wboost=1/(1+math.exp(beta*(S(1.0)-S(gmax)))); mac('wboost',wboost,'{:.3f}')
wprior=gmax/(gmax+1); mac('wprior',wprior,'{:.3f}'); mac('shift',math.log(gmax)/beta,'{:.3f}')
# softmax with log-prior: weight exp(beta*s)*g  equals shifting the score by ln g / beta
s1=s2=1.0; w=np.exp(beta*np.array([s1,s2]))*np.array([gmax,1.0]); w/=w.sum(); assert abs(w[0]-wprior)<1e-12
# 2. recency scale from tau(c)=tau_base R F T with printed ranges
lo=.5*.8*.3; hi=2.0*1.5*1.0; assert abs(lo-.12)<1e-12 and abs(hi-3.0)<1e-12
phi=(1+5**.5)/2; ah=math.acosh(phi)          # half-score age in units of tau
mac('taulo',lo,'{:.2f}'); mac('tauhi',hi,'{:.2f}'); mac('taurange',hi/lo,'{:.0f}')
mac('halfscore',ah,'{:.4f}'); mac('halfloS',ah*lo*5,'{:.2f}'); mac('halfhiS',ah*hi*5,'{:.1f}')
# 3. Custodian repetition cap from the drift contraction factor
eta,wv=.05,.4
for f,nm in ((.5,'half'),(.8,'eight')):
    n=math.log(f)/math.log(1-eta*wv); mac('ncap'+nm,n,'{:.1f}'); assert (1-eta*wv)**math.floor(n)>=f
mac('ncapeightfloor',math.floor(math.log(.8)/math.log(1-eta*wv)),'{}')
# P_break symmetry defect (lambda=0.01, C_threshold=10 chosen only to display the defect)
lam,Cth=.01,10; Pb=lambda C:1-math.exp(-lam*(C-Cth)**2); Pbf=lambda C:1-math.exp(-lam*max(C-Cth,0)**2)
assert abs(Pb(0)-Pb(20))<1e-12 and Pbf(0)==0 and Pb(10)==0
mac('pbzero',Pb(0),'{:.3f}'); mac('pbtwenty',Pb(20),'{:.3f}')
# 4. periodic sampling below the noise floor
p=0.01; n94=math.ceil(math.log(1-.94)/math.log(1-p)); n50=math.ceil(math.log(.5)/math.log(1-p))
assert 1-(1-p)**n94>=.94>1-(1-p)**(n94-1)
mac('nninetyfour',n94,'{}'); mac('nmedian',n50,'{}'); mac('nmean',1/p,'{:.0f}')
rng=np.random.default_rng(25); first=rng.geometric(p,size=200000); assert abs(first.mean()-100)<1.0 and abs(np.median(first)-n50)<=1
# 5. coherence normalisation of Eq. (18)
A=np.array([1.,1.,1.]); ph=np.zeros(3); C=abs((A*np.exp(1j*ph)).sum())**2/(A**2).sum(); R=abs((A*np.exp(1j*ph)).sum())/A.sum()
assert abs(C-3)<1e-12 and abs(R-1)<1e-12
for _ in range(2000):
    n=int(rng.integers(2,9)); a=rng.uniform(.1,2,n); f=rng.uniform(0,2*np.pi,n)
    Cc=abs((a*np.exp(1j*f)).sum())**2/(a**2).sum(); assert Cc<=n+1e-9; assert abs((a*np.exp(1j*f)).sum())/a.sum()<=1+1e-12
mac('cthree',C,'{:.0f}')
# 6. arithmetic on printed figures
pts=256*256*65536; mac('pts',pts,'{:d}'); mac('gb32',pts*32/1e9,'{:.1f}'); mac('gb40',pts*40/1e9,'{:.1f}'); mac('gbc',pts*32/1e9/100,'{:.2f}')
assert abs(pts*32/1e9-137.4)<.1
E=42*12e-6; mac('ejoule',E*1e6,'{:.0f}'); mac('eratio',E/13e-9,'{:.0f}'); mac('opsE',42/83e3*1e6,'{:.0f}')
mac('hours',65536/3600,'{:.1f}'); mac('secms',65536/1000,'{:.1f}')
mac('ins',300e-3/308e-6,'{:.0f}'); mac('ret',3.5e-3/12e-6,'{:.0f}'); mac('sto',512/48,'{:.1f}')
def san(k): return re.sub(r'[0-9]',lambda m:'ABCDEFGHIJ'[int(m.group())],k.replace('_',''))
open('gen/july_numbers.tex','w').write("\n".join(r"\newcommand{\n%s}{%s}"%(san(k),v) for k,v in NUM.items())+"\n")
print(NUM); print("JULY OK")
