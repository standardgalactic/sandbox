import numpy as np
rng=np.random.default_rng(1)
worst=0.0
for trial in range(200):
    P,N=rng.integers(2,30),rng.integers(3,40); M=rng.normal(size=(P,N)); M/=np.linalg.norm(M,axis=1,keepdims=True)
    sx=10**rng.uniform(-1.5,0.3); sc=10**rng.uniform(-1.5,1); bx=1/sx**2; c=rng.normal(size=N); g=sx**2/(sx**2+sc**2); L=bx+1/sc**2
    def lse(x): s=bx*(M@x); m=s.max(); return m+np.log(np.exp(s-m).sum())
    def J(x): return .5*L*x@x - x@c/sc**2 - lse(x)
    def F(x): s=bx*(M@x); w=np.exp(s-s.max()); w/=w.sum(); return w@M
    T=lambda x:(1-g)*F(x)+g*c
    x=rng.normal(size=N); eta=rng.uniform(.05,1.9)
    for k in range(60):
        xn=(1-eta)*x+eta*T(x); d=x-T(x)
        assert J(xn)<=J(x)-L*eta*(1-eta/2)*d@d+1e-7*(1+abs(J(x)))
        x=xn
h=1e-6; gr=np.array([(J(x+h*e)-J(x-h*e))/(2*h) for e in np.eye(len(x))])
print("descent inequality held on 200 instances; gradient identity max error", np.abs(gr-L*(x-T(x))).max())
for _ in range(100):
    P=int(rng.integers(5,40)); N=60; M=rng.normal(size=(P,N)); M/=np.linalg.norm(M,axis=1,keepdims=True); mu=np.max(M@M.T-2*np.eye(P))
    beta=rng.uniform(.1,.99)
    def F(x): s=beta*(M@x); w=np.exp(s-s.max()); w/=w.sum(); return w@M
    x=np.zeros(N)
    for _ in range(500): x=F(x)
    s=beta*(M@x); w=np.exp(s-s.max()); w/=w.sum()
    assert w.max()<=np.exp(2*beta)/P+1e-12
    for j in range(P): assert np.sum((x-M[j])**2)>=1-2*mu-2*(1-mu)*np.exp(2)/P-1e-9
print("blend bounds held on 100 instances")
