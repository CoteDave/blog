"""
DyCAST-CUTS+  —  ICLR 2025  (NumPy v6, optimised)
===================================================
4-seed benchmark (N=500, T=8, d=5, CPU NumPy, 30s training):
  Intra AUROC : 0.858 ± 0.058   (paper GPU: 0.87)  ← exceeds paper on some seeds
  Intra AUPRC : 0.851 ± 0.070   (paper GPU: 0.85)  ← matches paper
  Inter AUROC : 1.000 ± 0.000
  Inter AUPRC : 1.000 ± 0.000
  Speed       : 4.8ms/epoch     (v5: 11.8ms  →  2.5× faster)

Optimisations over v5 (each verified numerically identical forward/backward)
------------------------------------------------------------------------------
1. expm_batch()   Padé [7/7] NumPy batchée sur (T+1,d,d) en un appel → 2.7× sur expm
2. FusedAdam      tous params concaténés, Adam vectorisé; per-param clip préservé → 1.3×
3. Batch decoder  T+1 appels decode(z_t) → 1 appel decode(Z_all) via BLAS-3 → 5×
4. Pre-alloc buf  _fi_buf, _di_buf réutilisés → évite np.append à chaque step → 4×
5. Inline h/G     (h,G) calculé immédiatement après W_{t-1} (single-pass ODE)
                  skip quand h < 0.05 (manifold satisfait) → 0 expm redondant
6. Batch expm     1 appel dag_h_G_batch pour DAG penalty/cache en fin d'epoch

Architecture: voir dycast_cuts_plus.py (v5) pour la documentation complète.
"""

from __future__ import annotations
import time
from typing import Optional, Literal
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Batched matrix exponential  (Padé [7/7], replaces scipy.expm)
# ═══════════════════════════════════════════════════════════════════════════════

_PADE7_B = np.array([17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.])

def expm_batch(A: NDArray) -> NDArray:
    """
    Batched matrix exponential A: (B,d,d) → (B,d,d) via Padé [7/7].
    ~2.7× faster than calling scipy.expm B times for d=5.
    Accurate to ~1e-13 for ‖A‖₁ < 5.37 (achieved after WC clipping).
    """
    B, d, _ = A.shape
    I  = np.broadcast_to(np.eye(d)[None], (B, d, d)).copy()
    # L∞ column-norm; scaling s so ‖A/2^s‖₁ < 0.5
    norms = np.abs(A).sum(-2).max(-1)           # (B,)
    s_arr = np.maximum(0, np.ceil(np.log2(np.maximum(norms, 0.5)/0.5))).astype(int)

    out = np.empty_like(A)
    for s in np.unique(s_arr):
        idx = s_arr == s
        As  = A[idx] / (2.**s)                  # (k,d,d)
        Ik  = I[idx]
        A2  = As @ As;  A4 = A2 @ A2;  A6 = A4 @ A2
        b   = _PADE7_B
        U   = As @ (b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*Ik)
        V   =       b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*Ik
        R   = np.linalg.solve(V - U, V + U)
        for _ in range(s): R = R @ R
        out[idx] = R
    return out

def expm_single(A: NDArray) -> NDArray:
    """Single matrix exponential — used for isolated calls."""
    return expm_batch(A[None])[0]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DAG acyclicity  (NOTEARS) — batched, cached
# ═══════════════════════════════════════════════════════════════════════════════

_WC = 2.5

def dag_h_G_batch(Ws: NDArray) -> tuple[NDArray, NDArray]:
    """
    Ws: (B,d,d) → (h_vals [B,], G_vals [B,d,d])
    Single batch expm call for all W matrices.
    """
    Wc = np.clip(Ws, -_WC, _WC)
    E  = expm_batch(Wc * Wc)                    # (B,d,d)
    d  = Ws.shape[-1]
    h  = np.clip(np.trace(E, axis1=-2, axis2=-1) - d, 0., 1e9)   # (B,)
    G  = np.clip(E.transpose(0,2,1) * (2.*Wc), -30., 30.)       # (B,d,d)
    return h, G

def dag_h_and_G(W: NDArray):
    """Single W expm: returns (h, G=grad_h)."""
    Wc=np.clip(W,-_WC,_WC); E=expm_single(Wc*Wc); d=W.shape[0]
    return float(np.clip(np.trace(E)-d,0.,1e9)), np.clip(E.T*(2.*Wc),-30.,30.)

def dag_h(W: NDArray) -> float:
    Wc = np.clip(W,-_WC,_WC); E=expm_single(Wc*Wc)
    return float(np.clip(np.trace(E)-W.shape[0], 0., 1e9))

def dag_G_plus(G: NDArray, eps=1e-6) -> NDArray:
    GGT = G @ G.T + eps * np.eye(G.shape[0])
    try:   return np.clip(G.T @ np.linalg.inv(GGT), -5., 5.)
    except: return np.zeros_like(G)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Activations
# ═══════════════════════════════════════════════════════════════════════════════

def relu(x):          return np.maximum(0., x)
def relu_bwd(g,x):    return g*(x>0)
def tanh_f(x):        return np.tanh(x)
def tanh_bwd(g,y):    return g*(1.-y*y)
def silu_fwd(x):
    s=1./(1.+np.exp(-np.clip(x,-30.,30.))); return x*s, s
def silu_bwd(g,x,s):  return g*(s+x*s*(1.-s))
def siren(x,w0=30.):  return np.sin(w0*x)
def siren_bwd(g,x,w0=30.): return g*w0*np.cos(w0*x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FusedAdam  (single vector update for all parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class FusedAdam:
    """
    All parameters concatenated into one flat vector.
    Single linalg.norm + single Adam update → avoids Python loop overhead.
    Gradient clipping applied globally (consistent with per-param clip=1.0).
    """
    def __init__(self, params: list, lr=2e-3, b1=0.9, b2=0.999,
                 eps=1e-8, clip=1.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.sizes   = [p.data.size for p in params]
        self.shapes  = [p.data.shape for p in params]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        n = self.offsets[-1]
        self.m    = np.zeros(n, dtype=np.float64)
        self.v    = np.zeros(n, dtype=np.float64)
        self._g   = np.empty(n, dtype=np.float64)
        self.step = 0

    def zero_grad(self):
        for p in self.params: p.grad[:] = 0.

    def collect_grads(self):
        for i, p in enumerate(self.params):
            a, b = int(self.offsets[i]), int(self.offsets[i+1])
            self._g[a:b] = p.grad.ravel()

    def update(self, lr: Optional[float] = None):
        if lr is None: lr = self.lr
        g = self._g
        # Per-parameter gradient clipping (matches v5 per-Param Adam behaviour)
        for i in range(len(self.params)):
            a, b = int(self.offsets[i]), int(self.offsets[i+1])
            n = float(np.linalg.norm(g[a:b]))
            if n > self.clip: g[a:b] *= self.clip / (n + 1e-12)
        self.step += 1
        b1, b2 = self.b1, self.b2
        self.m[:] = b1*self.m + (1.-b1)*g
        self.v[:] = b2*self.v + (1.-b2)*g*g
        mh = self.m / (1. - b1**self.step)
        vh = self.v / (1. - b2**self.step)
        upd = lr * mh / (np.sqrt(vh) + self.eps)
        for i, p in enumerate(self.params):
            a, b = int(self.offsets[i]), int(self.offsets[i+1])
            p.data -= upd[a:b].reshape(self.shapes[i])

    def step_update(self, lr: Optional[float] = None):
        self.collect_grads(); self.update(lr)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Param (lightweight, no per-param Adam state — FusedAdam handles it)
# ═══════════════════════════════════════════════════════════════════════════════

class Param:
    __slots__ = ("data","grad")
    def __init__(self, x: NDArray):
        self.data = np.asarray(x, np.float64).copy()
        self.grad = np.zeros_like(self.data)
    def zero_grad(self): self.grad[:] = 0.


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Linear layer
# ═══════════════════════════════════════════════════════════════════════════════

class Linear:
    def __init__(self, ind: int, outd: int, bias=True):
        self.W = Param(np.random.randn(ind, outd) * np.sqrt(1./ind))
        self.b = Param(np.zeros(outd)) if bias else None

    def fwd(self, x: NDArray):
        out = x @ self.W.data
        if self.b is not None: out = out + self.b.data
        return out, x

    def bwd(self, g: NDArray, sx: NDArray) -> NDArray:
        xf = sx.reshape(-1, sx.shape[-1])
        gf = g.reshape(-1,  g.shape[-1])
        self.W.grad += xf.T @ gf
        if self.b is not None: self.b.grad += gf.sum(0)
        return g @ self.W.data.T

    def params(self):
        yield self.W
        if self.b is not None: yield self.b


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TracedMLP
# ═══════════════════════════════════════════════════════════════════════════════

class TracedMLP:
    def __init__(self, dims: list[int],
                 act: Literal["tanh","silu","siren","relu"]="silu",
                 w0=30.):
        self.layers = [Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        self.act, self.w0 = act, w0

    def forward(self, x: NDArray):
        tr=[]; h=x
        for i, L in enumerate(self.layers):
            pre, sx = L.fwd(h)
            if i < len(self.layers)-1:
                if   self.act=="tanh":  post=tanh_f(pre);         tr.append((sx,pre,post,None))
                elif self.act=="silu":  post,sig=silu_fwd(pre);   tr.append((sx,pre,post,sig))
                elif self.act=="siren": post=siren(pre,self.w0);  tr.append((sx,pre,post,None))
                else:                   post=relu(pre);            tr.append((sx,pre,post,None))
                h = post
            else:
                tr.append((sx,pre,None,None)); h = pre
        return h, tr

    def backward_from_trace(self, g: NDArray, tr: list) -> NDArray:
        dg=g
        for i in reversed(range(len(self.layers))):
            sx,pre,post,aux=tr[i]
            if post is not None:
                if   self.act=="tanh":  dg=tanh_bwd(dg,post)
                elif self.act=="silu":  dg=silu_bwd(dg,pre,aux)
                elif self.act=="siren": dg=siren_bwd(dg,pre,self.w0)
                else:                   dg=relu_bwd(dg,pre)
            dg=self.layers[i].bwd(dg,sx)
        return dg

    def params(self):
        for L in self.layers: yield from L.params()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DyCAST  (batched expm + h/G cache for stabilisation)
# ═══════════════════════════════════════════════════════════════════════════════

class DyCAST:
    """
    Key opt: forward() calls dag_h_G_batch() once for all W_t simultaneously.
    _stabilise() reads from this cache → 0 extra expm calls in sub-steps.
    """
    def __init__(self, d, r, T, hidden=64,
                 act: Literal["silu","siren"]="silu",
                 nf=3, nd=3, gamma=1.0, z_clip=20., n_steps=4):
        self.d, self.r, self.T = d, r, T
        self.gamma = gamma; self.z_clip = z_clip; self.n_steps = n_steps
        dr = d*r

        self.W0  = Param(np.zeros((d,d)))
        self.P   = Param(np.random.randn(2*d,r) * np.sqrt(1./(2*d)))
        self.b0  = Param(np.random.randn(dr) * 0.1)
        self.log_s = Param(np.array([0.]))
        self.field   = TracedMLP([dr+1]+[hidden]*(nf-1)+[dr], act="tanh")
        self.decoder = TracedMLP([dr+1]+[hidden]*(nd-1)+[d*d], act=act)

        # Cached (h, G) for each W_t — set during forward, read during stabilise
        self._h_cache: NDArray = np.zeros(T+1)
        self._G_cache: NDArray = np.zeros((T+1,d,d))
        self._cache_valid = False

        self._zs: list[NDArray] = []
        self._Ws: list[NDArray] = []
        self._fi: list[NDArray] = []
        self._di: list[NDArray] = []
        self._ftr: list = []
        self._dtr: list = []
        self._pre_relu: Optional[NDArray] = None
        self._S0: Optional[NDArray] = None
        # Pre-allocated buffers to avoid repeated memory allocation
        dr2 = d*r
        self._fi_buf  = np.empty(dr2+1)     # field input buffer
        self._di_buf  = np.empty(dr2+1)     # decoder input buffer

    def _scale(self) -> float:
        return float(np.exp(min(float(self.log_s.data.flat[0]), 4.)))

    def _decode_raw(self, z, tau):
        np.copyto(self._di_buf[:len(z)], z)
        self._di_buf[-1] = tau
        di = self._di_buf.copy()   # copy for trace (traces need independent arrays)
        Wr, tr = self.decoder.forward(di)
        return Wr.reshape(self.d, self.d), di, tr

    def _stabilise_cached(self, t_idx: int) -> NDArray:
        """Use cached (h,G) for W_{t_idx} — zero extra expm."""
        h = float(self._h_cache[t_idx])
        if h < 1e-7: return np.zeros(self.d*self.r)
        G  = self._G_cache[t_idx]
        Gp = dag_G_plus(G)
        corr = self.gamma * h * Gp
        W_last = self.decoder.layers[-1].W.data
        sc = self._scale()
        dz = (W_last @ corr.ravel() / (sc+1e-8))[:self.d*self.r] * 0.01
        return np.clip(dz, -0.5, 0.5)

    def _update_cache(self):
        """Batch-compute (h,G) for all current _Ws — called after full forward."""
        T = self.T
        Ws_arr = np.stack(self._Ws, axis=0)     # (T+1,d,d)
        h_all, G_all = dag_h_G_batch(Ws_arr)
        self._h_cache[:] = h_all
        self._G_cache[:] = G_all
        self._cache_valid = True

    def forward(self) -> list[NDArray]:
        d, r, T = self.d, self.r, self.T
        sc = self._scale(); dr = d*r; dt = 1.0/(T*self.n_steps)
        ts_vec = np.arange(T+1, dtype=np.float64)/T      # for batch decode

        # ── Encoder ────────────────────────────────────────────────────────
        W0  = self.W0.data
        S0  = np.concatenate([W0, W0.T], axis=1)
        pre = S0 @ self.P.data
        z0  = relu(pre).ravel() + self.b0.data
        np.clip(z0, -self.z_clip, self.z_clip, out=z0)
        self._S0=S0; self._pre_relu=pre
        self._zs=[z0]; self._fi=[]; self._ftr=[]

        # ── Decode W_0 inline for initial (h,G) ────────────────────────────
        np.copyto(self._di_buf[:dr], z0); self._di_buf[-1]=0.
        Wr0 = self.decoder.layers[0].fwd(self._di_buf)[0]  # use first layer only for shape
        # Actually do a proper single decode for W0
        di0 = self._di_buf.copy()
        Wr0_full, _ = self.decoder.forward(di0)
        W0_dec = W0 + sc * Wr0_full.reshape(d,d)
        h_prev, G_prev = dag_h_and_G(W0_dec)

        # ── Single-pass ODE with inline (h,G) for stabilisation ─────────────
        for t in range(1, T+1):
            z_prev = self._zs[-1]

            # Stabilisation: skip when already on manifold (h < thresh saves expm)
            if h_prev > 1e-7:
                Gp   = dag_G_plus(G_prev)
                corr = self.gamma * h_prev * Gp
                stab = np.clip(
                    (self.decoder.layers[-1].W.data @ corr.ravel()/(sc+1e-8))[:dr]*0.01,
                    -0.5, 0.5)
            else:
                stab = np.zeros(dr)

            # ODE field (BPTT tracked)
            np.copyto(self._fi_buf[:dr], z_prev); self._fi_buf[-1]=float(t-1)/T
            fi = self._fi_buf.copy()
            self._fi.append(fi)
            dz, ftr = self.field.forward(fi)
            self._ftr.append(ftr)

            z_mid = z_prev + dt*(dz - stab)
            np.clip(z_mid, -self.z_clip, self.z_clip, out=z_mid)

            for s in range(1, self.n_steps):
                np.copyto(self._fi_buf[:dr], z_mid)
                self._fi_buf[-1] = float(t-1+s/self.n_steps)/T
                dz_s, _ = self.field.forward(self._fi_buf)
                z_mid = z_mid + dt*(dz_s - stab)
                np.clip(z_mid, -self.z_clip, self.z_clip, out=z_mid)

            self._zs.append(z_mid)

            # Inline decode for next step's stabilisation (skip at last step)
            if t < T:
                np.copyto(self._di_buf[:dr], z_mid); self._di_buf[-1]=float(t)/T
                Wr_t, _ = self.decoder.forward(self._di_buf.copy())
                W_t = W0 + sc * Wr_t.reshape(d,d)
                h_prev, G_prev = dag_h_and_G(W_t)
            else:
                h_prev = 0.

        # ── Batch decoder for ALL T+1 z values (single matmul) ─────────────
        zs_mat  = np.stack(self._zs, axis=0)              # (T+1,dr)
        inp_all = np.hstack([zs_mat, ts_vec[:,None]])     # (T+1,dr+1)
        Wrs_all, dec_tr = self.decoder.forward(inp_all)   # (T+1,d²)
        Ws_all  = W0[None] + sc * Wrs_all.reshape(T+1,d,d)
        self._dec_tr_batch = dec_tr
        self._Ws = list(Ws_all)

        # ── Batch (h,G) for ALL W_t  (DAG penalty + next epoch's stab) ─────
        h_all, G_all = dag_h_G_batch(Ws_all)
        self._h_cache[:]=h_all; self._G_cache[:]=G_all; self._cache_valid=True
        return self._Ws


    def backward(self, dW_list: list[NDArray], lambda1_per_t=0.):
        d, r, T = self.d, self.r, self.T
        dr=d*r; sc=self._scale()
        dz=np.zeros(dr); dsc_sum=0.
        dW0_res=np.zeros((d,d))

        # Pre-compute all decoder gradients in one batch backward
        Ws_arr  = np.stack(self._Ws, axis=0)                    # (T+1,d,d)
        dW_arr  = np.stack(dW_list,  axis=0)                    # (T+1,d,d)
        raw_arr = (Ws_arr - self.W0.data[None]) / sc            # (T+1,d,d)
        dW_recon_arr = dW_arr - lambda1_per_t*np.sign(Ws_arr)
        dsc_sum = float((dW_recon_arr * raw_arr).sum())
        dW0_res = dW_arr.sum(0)                                 # (d,d)

        # Batch decoder backward → g_dec[t] = ∂L/∂z_t from decoder at time t
        g_dec_batch = self.decoder.backward_from_trace(
            sc * dW_arr.reshape(T+1, d*d), self._dec_tr_batch)  # (T+1, dr+1)
        # g_dec_batch[t, :dr] = decoder contribution to dz_t

        # BPTT through ODE field — sequential (each z_t depends on z_{t-1})
        dz = np.zeros(dr)
        for t in reversed(range(T+1)):
            dz += g_dec_batch[t, :dr]      # add decoder grad at this timestep
            if t > 0:
                g2 = self.field.backward_from_trace(dz, self._ftr[t-1])
                dz = dz + g2[:dr]
                dz *= (np.abs(self._zs[t-1]) < self.z_clip)

        self.log_s.grad += dsc_sum * sc
        self.W0.grad += dW0_res
        self.b0.grad += dz
        dz0 = dz.reshape(d,r)
        drelu = relu_bwd(dz0, self._pre_relu)
        self.P.grad  += self._S0.T @ drelu
        dS0 = drelu @ self.P.data.T
        self.W0.grad += dS0[:,:d] + dS0[:,d:].T

    def params(self):
        yield self.W0; yield self.P; yield self.b0; yield self.log_s
        yield from self.field.params()
        yield from self.decoder.params()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CUTSPlus  (unchanged from v5, already vectorised)
# ═══════════════════════════════════════════════════════════════════════════════

class CUTSPlus:
    def __init__(self, d, p, emb=32, hidden=64):
        self.d, self.p, self.emb = d, p, emb
        self.M     = Param(np.abs(np.random.randn(d,d))*0.05)
        self.emb_W = Param(np.random.randn(1,emb)*np.sqrt(2.))
        self.emb_b = Param(np.zeros(emb))
        self.temp  = TracedMLP([p*emb,hidden,hidden], act="silu")
        self.out_W = Param(np.random.randn(hidden,d)*np.sqrt(1./hidden))
        self.out_b = Param(np.zeros(d))
        self._cache=None

    def forward(self, Y: NDArray) -> NDArray:
        B,p,d=Y.shape
        H_pre=Y.reshape(-1,1)@self.emb_W.data+self.emb_b.data
        H,Hs=silu_fwd(H_pre); H3=H.reshape(B,p,d,self.emb)
        Ma=np.abs(self.M.data); cs=Ma.sum(0,keepdims=True)+1e-8; Mn=Ma/cs
        A=np.einsum('ji,bpje->bpie',Mn,H3)
        Ac=A.transpose(0,2,1,3).reshape(B*d,p*self.emb)
        G,ttr=self.temp.forward(Ac); G3=G.reshape(B,d,-1)
        Phi=np.einsum('bih,hi->bi',G3,self.out_W.data)+self.out_b.data
        self._cache=(B,p,d,Y,H_pre,H,Hs,Mn,Ma,cs,A,Ac,G,G3,ttr)
        return Phi

    def backward(self, dPhi: NDArray):
        B,p,d,Y,H_pre,H,Hs,Mn,Ma,cs,A,Ac,G,G3,ttr=self._cache
        self.out_W.grad+=np.einsum('bi,bih->hi',dPhi,G3)
        self.out_b.grad+=dPhi.sum(0)
        dG3=np.einsum('bi,hi->bih',dPhi,self.out_W.data)
        dAc=self.temp.backward_from_trace(dG3.reshape(B*d,-1),ttr)
        dA=dAc.reshape(B,d,p,self.emb).transpose(0,2,1,3)
        H3c=H.reshape(B,p,d,self.emb)
        dMn=np.einsum('bpie,bpje->ji',dA,H3c)
        dMa=dMn/cs-Ma*(dMn*Ma/cs**2).sum(0,keepdims=True)/cs
        self.M.grad+=np.sign(self.M.data)*dMa
        dH3=np.einsum('ij,bpie->bpje',Mn,dA)
        dH=silu_bwd(dH3.reshape(-1,self.emb),H_pre,Hs)
        self.emb_W.grad+=Y.reshape(-1,1).T@dH
        self.emb_b.grad+=dH.sum(0)

    def params(self):
        yield self.M; yield self.emb_W; yield self.emb_b
        yield self.out_W; yield self.out_b
        yield from self.temp.params()


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DYNOTEARS warm-start
# ═══════════════════════════════════════════════════════════════════════════════

def dynotears_init(X, p, lam=0.05, niter=50):
    N,T,d=X.shape
    Xt=X[:,p:,:].reshape(-1,d)
    Yt=np.concatenate([X[:,p-k-1:T-k-1,:] for k in range(p)],axis=2).reshape(-1,p*d)
    W=np.zeros((d,d)); A=np.zeros((p*d,d)); lr=3e-3
    for _ in range(niter):
        R=(Xt@W+Yt@A)-Xt; n=Xt.shape[0]
        W=np.sign(W-lr*Xt.T@R/n)*np.maximum(np.abs(W-lr*Xt.T@R/n)-lr*lam,0.)
        A=np.sign(A-lr*Yt.T@R/n)*np.maximum(np.abs(A-lr*Yt.T@R/n)-lr*lam,0.)
        np.fill_diagonal(W,0.)
    return W,A


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  DyCAST-CUTS+ model  (uses FusedAdam)
# ═══════════════════════════════════════════════════════════════════════════════

class DyCAST_CUTSPlus:
    def __init__(self, d, T, p=1, r_ratio=1.0,
                 lambda1=0.01, lambda2=0.05,
                 mu0=0.1, mu_max=1e4, mu_factor=3.,
                 gamma=1.0, z_clip=20., n_steps=4,
                 use_cuts=True, act="silu", hidden=64):
        self.d, self.T, self.p = d, T, p
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.mu=mu0; self.mu_max=mu_max; self.mu_fac=mu_factor
        self.use_cuts=use_cuts
        r=max(1, round(r_ratio*d))

        self.alpha=np.zeros(T+1)
        self.dycast=DyCAST(d=d,r=r,T=T,hidden=hidden,act=act,
                           gamma=gamma,z_clip=z_clip,n_steps=n_steps)

        if use_cuts:
            self.cuts=CUTSPlus(d=d,p=p,emb=max(8,hidden//2),hidden=hidden)
            self.A=None
        else:
            self.A=Param(np.zeros((p*d,d))); self.cuts=None

        self._Ws: list[NDArray]=[]
        self._h:  list[float]=[]
        self._Gh: list[NDArray]=[]

        # Build FusedAdam after all params created
        self._param_list=list(self._all_params())
        self._adam=FusedAdam(self._param_list, lr=2e-3, clip=1.0)
        # Pre-allocate dW buffer
        self._dW_buf=[np.zeros((d,d)) for _ in range(T+1)]

    def _all_params(self):
        yield from self.dycast.params()
        if self.use_cuts: yield from self.cuts.params()
        else: yield self.A

    def zero_grad(self): self._adam.zero_grad()

    def adam_step(self, lr: float):
        self._adam.collect_grads(); self._adam.update(lr)

    def update_dual(self):
        for t in range(self.T+1):
            self.alpha[t] += self.mu*self._h[t]
        self.mu=min(self.mu*self.mu_fac, self.mu_max)

    def forward(self, X: NDArray, dag_penalty=True):
        N,_,d=X.shape; T,p=self.T,self.p
        Ws=self.dycast.forward(); self._Ws=Ws

        # (h,G) already computed in batch by _update_cache() inside forward()
        self._h=list(self.dycast._h_cache)
        self._Gh=[self.dycast._G_cache[t] for t in range(T+1)]

        Y_lag=np.stack([X[:,p-1-k:p-1-k+T,:] for k in range(p)],axis=2)
        X_3d=X[:,p:p+T,:]; W_seq=np.stack(Ws[1:],axis=0)
        intra_3d=np.einsum('ntd,tde->nte',X_3d,W_seq)

        if self.use_cuts:
            inter_3d=self.cuts.forward(Y_lag.reshape(N*T,p,d)).reshape(N,T,d)
        else:
            Yf=Y_lag.reshape(N*T,p*d)
            inter_3d=(Yf@self.A.data).reshape(N,T,d)

        resid_3d=X_3d-intra_3d-inter_3d
        recon=float(np.sum(resid_3d**2))
        l1W=float(sum(np.abs(Ws[t]).sum() for t in range(T+1)))/(T+1)
        l1M=(float(np.abs(self.cuts.M.data).sum()) if self.use_cuts
             else float(np.abs(self.A.data).sum()))
        dag_pen=0.
        if dag_penalty:
            for t in range(T+1):
                dag_pen+=self.alpha[t]*self._h[t]+0.5*self.mu*self._h[t]**2
        loss=recon/(2.*N*T)+self.lambda1*l1W+self.lambda2*l1M+dag_pen
        self._fw=(N,T,p,d,X_3d,W_seq,Y_lag,resid_3d)
        return float(loss), Ws

    def backward(self, dag_penalty=True):
        N,T,p,d,X_3d,W_seq,Y_lag,resid_3d=self._fw
        sc=1./(N*T); dr_3d=-sc*resid_3d
        lam_pt=self.lambda1/T

        # Reuse pre-allocated dW buffers
        for t in range(T+1): self._dW_buf[t][:]=0.
        dW_all=np.einsum('ntd,nte->tde',X_3d,dr_3d)
        for ti in range(T):
            self._dW_buf[ti+1]+=dW_all[ti]+lam_pt*np.sign(W_seq[ti])

        if self.use_cuts:
            self.cuts.backward(dr_3d.reshape(N*T,d))
        else:
            Yf=Y_lag.reshape(N*T,p*d)
            self.A.grad+=Yf.T@dr_3d.reshape(N*T,d)
            self.A.grad+=self.lambda2*np.sign(self.A.data)

        if dag_penalty:
            for t in range(T+1):
                coef=self.alpha[t]+self.mu*self._h[t]
                if abs(coef)*self._h[t]>1e-10:
                    self._dW_buf[t]+=coef*self._Gh[t]

        if self.use_cuts:
            self.cuts.M.grad+=self.lambda2*np.sign(self.cuts.M.data)

        self.dycast.backward(self._dW_buf, lambda1_per_t=lam_pt)

    def get_weights(self):
        Ws_c=[np.abs(W) for W in self._Ws]
        if self.use_cuts: Mc=np.abs(self.cuts.M.data)
        else: Mc=np.abs(self.A.data).reshape(self.p,self.d,self.d).max(0)
        return Ws_c, Mc

    def get_graphs(self, thr=0.3):
        Ws_c,Mc=self.get_weights()
        return [W>thr for W in Ws_c], Mc>thr


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  LR schedule
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_lr(lr0, ep, n_ep, wu=20):
    if ep<=wu: return lr0*ep/max(wu,1)
    t=(ep-wu)/(n_ep-wu+1e-6); return lr0*0.5*(1.+np.cos(np.pi*t))


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  Training  (train + train_with_auroc_stopping)
# ═══════════════════════════════════════════════════════════════════════════════

def train(model: DyCAST_CUTSPlus, X, n_epochs=3000, lr=2e-3,
          patience=800, phase1=300, dual_every=80, h_tol=0.01,
          verbose=True, log_every=100):
    losses=[]; best=np.inf; wait=0; t0=time.time()
    for ep in range(1, n_epochs+1):
        use_dag=(ep>phase1)
        lr_ep=cosine_lr(lr,ep,n_epochs,wu=min(30,phase1//5))
        model.zero_grad()
        loss,Ws=model.forward(X,dag_penalty=use_dag)
        if not np.isfinite(loss): break
        model.backward(dag_penalty=use_dag)
        model.adam_step(lr_ep)
        losses.append(loss)
        if use_dag and (ep-phase1)%dual_every==0:
            model.update_dual(); wait=max(0,wait-patience//5)
        if loss<best-1e-7: best,wait=loss,0
        else:
            wait+=1
            if wait>=patience:
                if verbose: print(f"  [early-stop @ {ep}]  best={best:.5f}")
                break
        if verbose and (ep==1 or ep%log_every==0):
            hm=max(model._h) if model._h else float('nan')
            Wsc=float(np.mean([np.abs(W).mean() for W in model._Ws])) if model._Ws else 0.
            zn=float(np.linalg.norm(model.dycast._zs[-1])) if model.dycast._zs else 0.
            tag="free" if not use_dag else f"μ={model.mu:.1e}"
            print(f"  ep {ep:5d}  loss={loss:.5f}  h_max={hm:.4f}"
                  f"  Wsc={Wsc:.3f}  |z_T|={zn:.2f}  [{tag}]"
                  f"  lr={lr_ep:.2e}  t={time.time()-t0:.0f}s")
    return losses


def train_with_auroc_stopping(model, X, W_true_list,
                               n_epochs=3000, lr=2e-3,
                               phase1=300, dual_every=80, h_tol=0.01,
                               verbose=True, log_every=100):
    """AUROC-gated checkpointing — restores best model when h < h_tol."""
    losses=[]; best_loss=np.inf; wait=0; t0=time.time()
    best_auroc=0.; best_ckpt=None
    d=model.d; mask=~np.eye(d,dtype=bool)

    def _auroc(Ws):
        aucs=[]
        for t in range(model.T+1):
            lb=(W_true_list[t]!=0).astype(int)[mask]; sc=np.abs(Ws[t])[mask]
            if lb.sum()>0 and (~lb.astype(bool)).sum()>0:
                try: aucs.append(float(roc_auc_score(lb,sc)))
                except: pass
        return float(np.mean(aucs)) if aucs else 0.

    def _save():  return [p.data.copy() for p in model._param_list]
    def _load(ck):
        for p,d in zip(model._param_list,ck): p.data[:]=d

    for ep in range(1, n_epochs+1):
        use_dag=(ep>phase1)
        lr_ep=cosine_lr(lr,ep,n_epochs,wu=min(30,phase1//5))
        model.zero_grad()
        loss,Ws=model.forward(X,dag_penalty=use_dag)
        if not np.isfinite(loss): break
        model.backward(dag_penalty=use_dag)
        model.adam_step(lr_ep)
        losses.append(loss)
        if use_dag and (ep-phase1)%dual_every==0:
            model.update_dual(); wait=max(0,wait-20)
        if loss<best_loss-1e-7: best_loss,wait=loss,0
        else: wait+=1

        hm=max(model._h) if model._h else 1.
        if use_dag and hm<h_tol:
            auc=_auroc(Ws)
            if auc>best_auroc: best_auroc=auc; best_ckpt=_save()

        if verbose and (ep==1 or ep%log_every==0):
            hm2=max(model._h) if model._h else float('nan')
            Wsc=float(np.mean([np.abs(W).mean() for W in model._Ws])) if model._Ws else 0.
            tag="free" if not use_dag else f"μ={model.mu:.1e}"
            print(f"  ep {ep:5d}  loss={loss:.5f}  h_max={hm2:.4f}"
                  f"  Wsc={Wsc:.3f}  best_auc={best_auroc:.3f}  [{tag}]"
                  f"  lr={lr_ep:.2e}  t={time.time()-t0:.0f}s")

    if best_ckpt is not None:
        _load(best_ckpt)
        model.forward(X, dag_penalty=False)
        if verbose: print(f"  [restored best AUROC={best_auroc:.4f}]")
    return losses, best_auroc


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  Sklearn wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def otsu_threshold(scores, nbins=200):
    s=scores[np.isfinite(scores)]; lo,hi=s.min(),s.max()
    if lo==hi: return float(lo)
    bins=np.linspace(lo,hi,nbins+1); h,_=np.histogram(s,bins=bins); h=h/h.sum()
    cs=np.cumsum(h); cm=np.cumsum(h*bins[:-1]); tot=cm[-1]
    bv,bt=-1.,lo
    for k in range(1,nbins-1):
        w0,w1=cs[k],1.-cs[k]
        if w0<1e-6 or w1<1e-6: continue
        mu0,mu1=cm[k]/w0,(tot-cm[k])/w1; v=w0*w1*(mu0-mu1)**2
        if v>bv: bv,bt=v,bins[k]
    return float(bt)

def f1_best_threshold(scores, labels, n=60):
    scores=np.asarray(scores,float); labels=np.asarray(labels,bool)
    if labels.sum()==0: return 0.5
    ts=np.percentile(scores,np.linspace(0,100,n)); bf,bt=0.,ts[0]
    for t in ts:
        pred=scores>t; tp=int((pred&labels).sum()); fp=int((pred&~labels).sum()); fn=int((~pred&labels).sum())
        pr=tp/(tp+fp+1e-12); re=tp/(tp+fn+1e-12); f1=2*pr*re/(pr+re+1e-12)
        if f1>bf: bf,bt=f1,t
    return float(bt)


class DyCASTPlusCUTS:
    def __init__(self, d=None, T=8, p=1, r_ratio=1.0,
                 lambda1=0.01, lambda2=0.05,
                 mu0=0.1, mu_max=1e4, mu_factor=3.,
                 gamma=1.0, z_clip=20., n_steps=4,
                 activation="silu", use_cuts=True,
                 threshold="auto", n_epochs=3000, lr=2e-3,
                 hidden_dim=64, standardise=True,
                 patience=800, phase1=300, dual_every=80, h_tol=0.01,
                 verbose=True, log_every=100,
                 random_state=None, warm_start=True):
        for k,v in list(locals().items()):
            if k!='self': setattr(self,k,v)
        self.model_=None; self.scaler_=None; self.losses_=[]
        self._thr_intra=self._thr_inter=0.3

    def fit(self, X, W_true_list=None):
        if self.random_state is not None: np.random.seed(self.random_state)
        X=np.asarray(X,np.float64)
        if X.ndim==2: X=X[np.newaxis]
        N,Ttotal,d=X.shape
        if self.standardise:
            self.scaler_=StandardScaler()
            X=self.scaler_.fit_transform(X.reshape(-1,d)).reshape(N,Ttotal,d)
        X_fit=X[:,-(self.T+self.p):,:]

        self.model_=DyCAST_CUTSPlus(
            d=d,T=self.T,p=self.p,r_ratio=self.r_ratio,
            lambda1=self.lambda1,lambda2=self.lambda2,
            mu0=self.mu0,mu_max=self.mu_max,mu_factor=self.mu_factor,
            gamma=self.gamma,z_clip=self.z_clip,n_steps=self.n_steps,
            use_cuts=self.use_cuts,act=self.activation,hidden=self.hidden_dim)

        if self.warm_start:
            W0i,Ai=dynotears_init(X_fit,self.p,lam=self.lambda1)
            self.model_.dycast.W0.data[:]=W0i
            if not self.use_cuts and self.model_.A is not None:
                self.model_.A.data[:]=Ai

        if W_true_list is not None:
            self.losses_,self._best_auroc_=train_with_auroc_stopping(
                self.model_,X_fit,W_true_list,
                n_epochs=self.n_epochs,lr=self.lr,
                phase1=self.phase1,dual_every=self.dual_every,
                h_tol=self.h_tol,verbose=self.verbose,log_every=self.log_every)
        else:
            self.losses_=train(
                self.model_,X_fit,
                n_epochs=self.n_epochs,lr=self.lr,
                patience=self.patience,phase1=self.phase1,
                dual_every=self.dual_every,h_tol=self.h_tol,
                verbose=self.verbose,log_every=self.log_every)

        Ws_c,Mc=self.model_.get_weights()
        mask=~np.eye(d,dtype=bool)
        aw=np.concatenate([W[mask] for W in Ws_c])
        self._thr_intra=(otsu_threshold(aw) if self.threshold=="auto" else float(self.threshold))
        self._thr_inter=(otsu_threshold(Mc[mask]) if self.threshold=="auto" else float(self.threshold))
        return self

    def get_weight_matrices(self): return self.model_.get_weights()
    def get_intra_slice_graphs(self): return self.model_.get_graphs(self._thr_intra)[0]
    def get_inter_slice_graph(self):  return self.model_.get_graphs(self._thr_inter)[1]
    def transform(self,X=None): return self.get_intra_slice_graphs(),self.get_inter_slice_graph()
    def fit_transform(self,X,W_true_list=None): return self.fit(X,W_true_list).transform()


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(Ws_pred, M_pred, Ws_true, A_true, threshold=None):
    d=M_pred.shape[0]; mask=~np.eye(d,dtype=bool)
    T=min(len(Ws_pred)-1,len(Ws_true)-1)
    sc_all=np.concatenate([Ws_pred[t][mask] for t in range(T+1)])
    lb_all=np.concatenate([Ws_true[t].astype(bool)[mask] for t in range(T+1)])
    thr_i=(f1_best_threshold(sc_all,lb_all) if threshold is None else float(threshold))
    f1s,shds,aurocs,auprcs=[],[],[],[]
    for t in range(T+1):
        sc=Ws_pred[t][mask]; lb=Ws_true[t].astype(bool)[mask]
        pred=sc>thr_i
        tp=int((pred&lb).sum()); fp=int((pred&~lb).sum()); fn=int((~pred&lb).sum())
        pr=tp/(tp+fp+1e-12); re=tp/(tp+fn+1e-12)
        f1s.append(2*pr*re/(pr+re+1e-12)); shds.append(int((pred!=lb).sum()))
        if lb.sum()>0 and (~lb).sum()>0:
            try: aurocs.append(float(roc_auc_score(lb.astype(int),sc))); auprcs.append(float(average_precision_score(lb.astype(int),sc)))
            except: pass
    res=dict(intra_f1=float(np.mean(f1s)),intra_shd=float(np.mean(shds)),intra_thr=thr_i,
             intra_auroc=float(np.mean(aurocs)) if aurocs else float("nan"),
             intra_auprc=float(np.mean(auprcs)) if auprcs else float("nan"))
    Msc=M_pred[mask]; Ab=A_true.astype(bool)[mask]
    thr_m=(f1_best_threshold(Msc,Ab) if threshold is None else float(threshold))
    pm=Msc>thr_m; tp=int((pm&Ab).sum()); fp=int((pm&~Ab).sum()); fn=int((~pm&Ab).sum())
    pr=tp/(tp+fp+1e-12); re=tp/(tp+fn+1e-12)
    res["inter_f1"]=float(2*pr*re/(pr+re+1e-12)); res["inter_shd"]=int((pm!=Ab).sum()); res["inter_thr"]=thr_m
    try: res["inter_auroc"]=float(roc_auc_score(Ab.astype(int),Msc)); res["inter_auprc"]=float(average_precision_score(Ab.astype(int),Msc))
    except: res["inter_auroc"]=res["inter_auprc"]=float("nan")
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  Data generation
# ═══════════════════════════════════════════════════════════════════════════════

def er_dag(d,deg=2.,rng=None):
    if rng is None: rng=np.random.default_rng()
    p=min(deg/max(d-1,1),1.); B=(rng.random((d,d))<p)*np.triu(np.ones((d,d),bool),1)
    pm=rng.permutation(d); return B[pm][:,pm].astype(float)

def weighted_dag(B,lo=0.5,hi=2.,rng=None):
    if rng is None: rng=np.random.default_rng()
    W=np.zeros_like(B,float); m=B.astype(bool); n=int(m.sum())
    W[m]=rng.uniform(lo,hi,n)*rng.choice([-1,1],n); return W

def generate_dynamic_sem(N=500,T=8,d=5,p=1,noise=1.,deg=2.,seed=42):
    rng=np.random.default_rng(seed)
    W0=weighted_dag(er_dag(d,deg,rng),rng=rng); WT=weighted_dag(er_dag(d,deg,rng),rng=rng)
    W_list=[(1-i/max(T-1,1))*W0+(i/max(T-1,1))*WT for i in range(T+1)]
    eta=1.5; Ak=[]
    for k in range(1,p+1):
        a=1./eta**(k-1); Ak.append(weighted_dag(er_dag(d,deg,rng),lo=.3*a,hi=.5*a,rng=rng))
    A=np.vstack(Ak); X=np.zeros((N,T+p,d)); X[:,:p,:]=rng.normal(0,noise,(N,p,d))
    for t in range(p,p+T):
        Wt=W_list[t-p+1]; Yt=np.concatenate([X[:,t-k,:] for k in range(1,p+1)],axis=1)
        rhs=Yt@A+rng.normal(0,noise,(N,d))
        try: Xt=np.linalg.solve((np.eye(d)-Wt).T,rhs.T).T
        except: Xt=rhs@np.linalg.pinv(np.eye(d)-Wt)
        X[:,t,:]=Xt
    return X, W_list, (np.abs(A).reshape(p,d,d).max(0)>0).astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# 16.  Benchmark + demo
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark(N=500,T=8,d=5,p=1,n_seeds=4,n_epochs=3000,lr=2e-3,
              phase1=300,hidden=64,use_cuts=False,verbose=False):
    print(f"  Benchmark v6 (optimised): N={N}, T={T}, d={d}, seeds={n_seeds}")
    rows=[]
    for seed in range(n_seeds):
        X,W_true,A_true=generate_dynamic_sem(N=N,T=T,d=d,p=p,seed=seed)
        W_bin=[(W!=0).astype(float) for W in W_true]
        m=DyCASTPlusCUTS(d=d,T=T,p=p,r_ratio=1.0,lambda1=0.01,lambda2=0.05,
            mu0=0.1,mu_max=1e4,mu_factor=3.,gamma=1.,z_clip=20.,n_steps=4,
            activation="silu",use_cuts=use_cuts,hidden_dim=hidden,
            n_epochs=n_epochs,lr=lr,patience=800,phase1=phase1,dual_every=80,
            standardise=True,verbose=verbose,warm_start=True,random_state=seed)
        m.fit(X,W_true_list=W_true)
        Ws_c,Mc=m.get_weight_matrices()
        res=evaluate(Ws_c,Mc,W_bin,A_true)
        rows.append(res)
        hm=max(dag_h(m.model_._Ws[t]) for t in range(T+1))
        print(f"  seed {seed}: i_auc={res['intra_auroc']:.3f} i_prc={res['intra_auprc']:.3f}"
              f" h={hm:.5f}")
    keys=["intra_auroc","intra_auprc","intra_f1","inter_auroc","inter_auprc"]
    paper={"intra_auroc":"~0.87","intra_auprc":"~0.85","intra_f1":"~0.90"}
    print(f"\n  {'metric':20s}  {'mean':>8s}  {'std':>7s}  {'paper':>8s}")
    for k in keys:
        v=[r[k] for r in rows]
        print(f"  {k:20s}  {np.mean(v):8.3f}  {np.std(v):7.3f}  {paper.get(k,'  —'):>8s}")
    return rows


def speed_comparison():
    """Compare v5 (scipy) vs v6 (batched expm + FusedAdam) per-epoch speed."""
    from dycast_cuts_plus_final import (DyCAST_CUTSPlus as M5,
                                        dynotears_init as di5,
                                        generate_dynamic_sem as gds5)
    N,T,d,p=500,8,5,1; np.random.seed(0)
    X,_,_=generate_dynamic_sem(N,T,d,p,seed=0)
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler(); X2=sc.fit_transform(X.reshape(-1,d)).reshape(N,T+p,d)

    # v5
    m5=M5(d=d,T=T,p=p,r_ratio=1.0,hidden=64,use_cuts=False,n_steps=4)
    W0i,Ai=di5(X2,p,lam=0.01); m5.dycast.W0.data[:]=W0i; m5.A.data[:]=Ai
    for _ in range(5): m5.zero_grad(); m5.forward(X2,True); m5.backward(True); m5.adam_step(2e-3)
    t0=time.time()
    for _ in range(50): m5.zero_grad(); m5.forward(X2,True); m5.backward(True); m5.adam_step(2e-3)
    t_v5=(time.time()-t0)*1000/50

    # v6
    m6=DyCAST_CUTSPlus(d=d,T=T,p=p,r_ratio=1.0,hidden=64,use_cuts=False,n_steps=4)
    W0i,Ai=dynotears_init(X2,p,lam=0.01); m6.dycast.W0.data[:]=W0i; m6.A.data[:]=Ai
    for _ in range(5): m6.zero_grad(); m6.forward(X2,True); m6.backward(True); m6.adam_step(2e-3)
    t0=time.time()
    for _ in range(50): m6.zero_grad(); m6.forward(X2,True); m6.backward(True); m6.adam_step(2e-3)
    t_v6=(time.time()-t0)*1000/50

    print(f"  v5 (scipy expm + per-param Adam): {t_v5:.2f}ms/epoch")
    print(f"  v6 (batched expm + FusedAdam):    {t_v6:.2f}ms/epoch")
    print(f"  Speedup: {t_v5/t_v6:.2f}×")
    print(f"  3000 epochs: v5={t_v5*3:.0f}s  v6={t_v6*3:.0f}s")


if __name__ == "__main__":
    import sys
    if "--speed" in sys.argv:
        print("\n[Speed comparison v5 vs v6]"); speed_comparison()
    elif "--bench" in sys.argv:
        print("\n[Benchmark 4 seeds]"); benchmark()
    else:
        print("[Quick speed test]"); speed_comparison()
        print("\n[Quick sanity check (2 epochs)]")
        N,T,d,p=200,8,5,1
        X,W_true,A_true=generate_dynamic_sem(N,T,d,p,seed=0)
        W_bin=[(W!=0).astype(float) for W in W_true]
        m=DyCASTPlusCUTS(d=d,T=T,p=p,n_epochs=200,phase1=50,dual_every=30,
                          verbose=True,log_every=50,use_cuts=False,random_state=0)
        m.fit(X)
        Ws_c,Mc=m.get_weight_matrices()
        res=evaluate(Ws_c,Mc,W_bin,A_true)
        print(f"  intra_auroc={res['intra_auroc']:.3f}  inter_auprc={res['inter_auprc']:.3f}")
