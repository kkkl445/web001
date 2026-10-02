import numpy as np
DRAG=1.5
G=9.8; TARGET=5.0; DT=0.005; T=12.0
def simulate(kind, kp=0, ki=0, kd=0, T=T, umax=30.0):
    n=int(T/DT); h=0.0; v=0.0; thrust=0.0; integ=0.0
    delay=int(0.08/DT); hist=[0.0]*(delay+1)
    prev_e=None
    H=np.zeros(n); U=np.zeros(n); E=np.zeros(n); I=np.zeros(n); Dv=np.zeros(n)
    for k in range(n):
        meas=hist[0]
        e=TARGET-meas
        if kind=='bang':
            cmd= umax*0.6 if e>0 else 0.0
            d=0
        else:
            integ+=e*DT
            d=0.0 if prev_e is None else (e-prev_e)/DT
            cmd=kp*e+ki*integ+kd*d
        prev_e=e
        cmd=min(max(cmd,0.0),umax)
        thrust+= (cmd-thrust)*DT/0.12
        a=thrust-G-DRAG*v
        v+=a*DT; h+=v*DT
        if h<0: h=0; v=max(v,0)
        hist.pop(0); hist.append(h)
        H[k]=h; U[k]=thrust; E[k]=e; I[k]=integ; Dv[k]=d
    t=np.arange(n)*DT
    return dict(t=t,h=H,u=U,e=E,i=I,d=Dv)
CASES={
 'bang':dict(kind='bang'),
 'p_low':dict(kind='p',kp=3),
 'p_high':dict(kind='p',kp=12),
 'pi':dict(kind='pi',kp=6,ki=4),
 'pid':dict(kind='pid',kp=6,ki=4,kd=4),
}
if __name__=='__main__':
    for k,c in CASES.items():
        r=simulate(**c); h=r['h']
        print(k,'max',round(h.max(),2),'final',round(h[-1],2),'mean last2s',round(h[-400:].mean(),2), 'std last2s',round(h[-400:].std(),3))
