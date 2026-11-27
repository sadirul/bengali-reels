import numpy as np, soundfile as sf, sys
SR=48000; DUR=float(sys.argv[1]); ER=float(sys.argv[2]); BOOM=float(sys.argv[3])
t=np.arange(int(SR*DUR))/SR
rng=np.random.default_rng(1)
def env(a,d): return np.clip(t-a,0,None)/d
# warm pad: A minor-ish drone with slow movement
pad=np.zeros_like(t)
for f,amp in [(110,.5),(164.81,.3),(220,.25),(261.63,.12),(329.63,.08)]:
    pad+=amp*np.sin(2*np.pi*f*t+0.3*np.sin(2*np.pi*0.07*t))
pad*=0.55+0.45*np.sin(2*np.pi*0.05*t)**2
# soft plucks (curious mood) every 0.8s on a pentatonic pattern
notes=[440,523.25,587.33,659.25,783.99,659.25,587.33,523.25]
pl=np.zeros_like(t)
for k,st in enumerate(np.arange(0.4,DUR-1,0.8)):
    f=notes[k%len(notes)]; i0=int(st*SR); n=int(0.7*SR); tt=np.arange(n)/SR
    seg=np.sin(2*np.pi*f*tt)*np.exp(-tt*6)*0.35
    pl[i0:i0+n]+=seg[:len(pl)-i0]
# rumble before eruption and at boom
noise=rng.normal(0,1,len(t))
from numpy.fft import rfft, irfft
def lowpass(x,fc):
    X=rfft(x); fr=np.fft.rfftfreq(len(x),1/SR); X[fr>fc]=0; return irfft(X,len(x))
rum=lowpass(noise,90)
rum/=np.abs(rum).max()+1e-9
r_env=np.clip((t-(ER-2.5))/2.5,0,1)*np.exp(-np.clip(t-ER,0,None)/2.5)
b_env=np.where(t>=BOOM,np.exp(-(t-BOOM)/1.2),0)+np.where(t>=0.3,np.exp(-np.clip(t-0.3,0,None)/1.5),0)*(t<4)
boom=lowpass(noise,160); boom/=np.abs(boom).max()+1e-9
sfx=rum*r_env*1.6+boom*b_env*1.8+boom*np.where(t>=ER,np.exp(-(t-ER)/0.9),0)*1.6
mix=0.16*pad/np.abs(pad).max()+0.10*pl+0.35*sfx
fadeout=np.clip((DUR-t)/1.5,0,1); mix*=fadeout*np.clip(t/0.3,0,1)
mix=np.clip(mix,-0.95,0.95)
sf.write('music.wav',np.stack([mix,mix],1).astype(np.float32),SR)
print('music ok')
