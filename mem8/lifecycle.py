"""Miniature Mem8 memory lifecycle with deterministic arithmetic (all stored values are dyadic, so
replay and reversal are bitwise exact). Writes gen/lifecycle_table.tex and gen/life_numbers.tex."""
import re
import numpy as np, copy
FEATS=["Ann","Bo","cafe","library","rain","morning","evening"]
def vec(*names): return np.array([1.0 if f in names else 0.0 for f in FEATS])
beta=8.0; g=0.2; theta_d=0.25; tau_attr=0.5; LAMBDA_STAR=10.0
M=[]; ledger=[]; snapshots=[]
def rec(typ,cites,payload): 
    ledger.append(dict(id=len(ledger)+1,type=typ,cites=list(cites),payload=payload)); return ledger[-1]["id"]
E1=vec("Ann","cafe","morning"); E2=vec("Ann","library","rain"); E3=vec("Bo","cafe","rain")
def replay():
    Mx=[]; byid={r['id']:r for r in ledger}
    ins=[r for r in ledger if r['type']=='inscription']; tr=[r for r in ledger if r['type']=='transformation']
    init={"E1":E1,"E2":E2,"E3":E3}
    for r in ins: Mx.append(init[r['payload']].copy())
    for r in tr: Mx[ins.index(byid[r['cites'][0]])]+=r['payload']
    return Mx
def snap(label,op):
    # invariants after every operation: the ledger only grows (prefix property) and replaying it reproduces the store exactly
    if snapshots: assert [r[0] for r in snapshots[-1]['ledger']]==[r['id'] for r in ledger][:len(snapshots[-1]['ledger'])]
    Mr_=replay(); assert len(Mr_)==len(M) and all(np.array_equal(a,b) for a,b in zip(Mr_,M)),label
    snapshots.append(dict(step=len(snapshots),label=label,op=op,M=[m.copy() for m in M],ledger=[(r["id"],r["type"],tuple(r["cites"])) for r in ledger]))
def retrieve(cue):
    Mm=np.array(M); cos=(Mm@cue)/(np.linalg.norm(Mm,axis=1)*np.linalg.norm(cue)+1e-12)
    s=beta*cos; w=np.exp(s-s.max()); w/=w.sum(); F=w@Mm; r=(1-g)*F+g*cue
    order=np.argsort(-cos); margin=cos[order[0]]-cos[order[1]]
    attributed=[i for i in range(len(M)) if w[i]>=tau_attr]
    details=[j for j in range(len(FEATS)) if r[j]>=theta_d]
    warranted=[j for j in details if any(M[i][j]>=0.5 for i in attributed)]
    omega=len(warranted)/len(details) if details else 1.0
    return dict(w=w,r=r,cos=cos,margin=margin,attributed=attributed,details=details,warranted=warranted,omega=omega)
NUM={}
def mac(k,v,fmt): NUM[k]=fmt.format(v)
# S0
snap("S0","empty store")
# S1, S2 encoding (inscriptions)
M.append(E1.copy()); i1=rec("inscription",[],"E1"); snap("S1","encode E1")
M.append(E2.copy()); i2=rec("inscription",[],"E2"); M.append(E3.copy()); i3=rec("inscription",[],"E3"); snap("S2","encode E2, E3")
M_after_encode=[m.copy() for m in M]
# S3 partial cue
ca=vec("Ann","morning"); out=retrieve(ca)
mac('wA',out['w'][0],'{:.3f}'); mac('wAtwo',out['w'][1],'{:.3f}'); mac('marginA',out['margin'],'{:.3f}'); mac('omegaA',out['omega'],'{:.2f}')
assert out['attributed']==[0] and out['omega']==1.0
rec("interpretation",[i1],"cue Ann,morning"); snap("S3","partial cue (Ann, morning)")
detailsA=[FEATS[j] for j in out['details']]
# S4 misleading cue
cb=vec("Ann","cafe","morning","rain"); outb=retrieve(cb)
mac('wB',outb['w'][0],'{:.3f}'); mac('omegaB',outb['omega'],'{:.2f}'); mac('rainB',outb['r'][FEATS.index("rain")],'{:.3f}')
assert outb['attributed']==[0] and "rain" in [FEATS[j] for j in outb['details']] and FEATS.index("rain") not in outb['warranted']
detailsB=[FEATS[j] for j in outb['details']]; unw=[FEATS[j] for j in outb['details'] if j not in outb['warranted']]
rec("interpretation",[i1],"cue Ann,cafe,morning,rain"); 
# hypothetical ungated write-back
eta=0.5; m1_hyp=M[0]+eta*outb['w'][0]*(outb['r']-M[0]); mac('hypshift',np.linalg.norm(m1_hyp-M[0]),'{:.3f}'); mac('hyprain',m1_hyp[FEATS.index("rain")],'{:.3f}')
snap("S4","misleading cue (adds rain)")
# S5 misleading evidence, gate closed
def Lam(rhos): 
    L=1.0
    for r in rhos: L*=r/(1-r)
    return L
L1=Lam([0.6]); mac('lamone',L1,'{:.2f}'); assert L1<LAMBDA_STAR
ev1=rec("evidence",[i1],"report: E1 was at the library (rho=0.6)"); gate_open=L1>=LAMBDA_STAR; assert not gate_open
M_before=[m.copy() for m in M]; snap("S5","misleading report, gate closed")
# S6 corroborated evidence opens gate; controlled revision
L3=Lam([0.7,0.7,0.7]); mac('lamthree',L3,'{:.2f}'); assert L3>=LAMBDA_STAR
ev2=rec("evidence",[i1],"3 independent reports: E1 was in the evening (rho=0.7 each)")
e_target=vec("Ann","cafe","evening"); Delta=e_target-M[0]
M[0]=M[0]+Delta; t7=rec("transformation",[i1,ev2],Delta.copy()); snap("S6","corroborated evidence, revise m1")
assert np.array_equal(M[0],e_target)
# S7 later recollection
cc=vec("Ann","evening"); outc=retrieve(cc); mac('wC',outc['w'][0],'{:.3f}'); mac('omegaC',outc['omega'],'{:.2f}')
assert outc['attributed']==[0] and outc['omega']==1.0
old=retrieve(ca); mac('wOldCue',old['w'][0],'{:.3f}'); mac('wOldCueE2',old['w'][1],'{:.3f}')
assert old['attributed']==[] # old cue no longer attributes to a single trace
rec("interpretation",[t7],"cue Ann,evening"); snap("S7","later recollection (Ann, evening)")
# S8 replay from ledger
Mr=replay(); assert all(np.array_equal(a,b) for a,b in zip(Mr,M)); snap("S8","replay from ledger (state identical)")
# S9 audit by reversal: compute the state before the revision from the later state and the logged increment (read only)
Mrev=[m.copy() for m in M]; tr=[r for r in ledger if r['type']=='transformation']
for r in reversed(tr): Mrev[0]=Mrev[0]-r['payload']
assert all(np.array_equal(a,b) for a,b in zip(Mrev,M_before)); snap("S9","audit: rewind 8t, read only (store unchanged)")
# S10 controlled reversal: append the inverse increment as a new transformation citing 8t; nothing is deleted
t10=rec("transformation",[i1,t7],-Delta); M[0]=M[0]-Delta
assert all(np.array_equal(a,b) for a,b in zip(M,M_before)); snap("S10","reverse revision: append inverse transformation")
Mr2=replay(); assert all(np.array_equal(a,b) for a,b in zip(Mr2,M))
mac('nledger',len(ledger),'{}'); mac('nmem',len(M),'{}'); mac('ninter',sum(r['type']=='interpretation' for r in ledger),'{}')
mac('Ldist',np.linalg.norm(E1-M[0]),'{:.3f}')
# table
def fmt(m): return "("+",".join(("%g"%x) for x in m)+")"
rows=[]
for sn in snapshots:
    tr=r"\par ".join(r"$m_%d=%s$"%(i+1,fmt(m)) for i,m in enumerate(sn['M'])) if sn['M'] else r"(empty)"
    ids=" ".join(str(r[0])+{"inscription":"i","interpretation":"p","evidence":"e","transformation":"t"}[r[1]] for r in sn['ledger']) or "none"
    rows.append(r"%s & %s & \footnotesize %s & \footnotesize %s \tabularnewline"%(sn['label'],sn['op'],tr,ids))
open('gen/lifecycle_table.tex','w').write("\n".join(rows)+"\n\\bottomrule\n")
def san(k): return re.sub(r'[0-9]',lambda m:'ABCDEFGHIJ'[int(m.group())],k.replace('_',''))
open('gen/life_numbers.tex','w').write("\n".join(r"\newcommand{\n%s}{%s}"%(san(k),v) for k,v in NUM.items())+"\n")
print(NUM); print("details A",detailsA,"details B",detailsB,"unwarranted",unw); print("LIFECYCLE OK")
