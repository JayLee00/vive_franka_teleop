#!/usr/bin/env python3
"""네트워크 / 노이즈 스케줄러 / EMA — 학습과 배포가 같은 정의를 import 한다.

아키텍처는 참조 구현(R_Franka_KISTAR_Hand/diffusion/diffusion_train.py)과 동일:
    MLP obs encoder (T_obs 평탄화) → global_cond(256)
    FiLM 조건화 1D U-Net (64,128,256) → 노이즈 예측
    Cosine DDPM T=100 학습 / DDIM 결정적 추론

바뀐 것: obs_dim 을 인자로 받는다(전구 32 → 레몬 46). EMA 추가.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_config import ACTION_DIM, DIFF_STEPS, INFER_STEPS, OBS_DIM, OBS_HORIZON


# ══════════════════════════════════════════════════════════════════════════
# DDPM / DDIM
# ══════════════════════════════════════════════════════════════════════════
class DDPMScheduler:
    """Cosine 노이즈 스케줄 (Nichol & Dhariwal 2021)."""

    def __init__(self, num_steps: int = DIFF_STEPS):
        T = num_steps
        s = 0.008
        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        alpha_bar = (f / f[0]).float()
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.T = T
        self.alpha_bar = alpha_bar
        self.sqrt_ab = alpha_bar.sqrt()
        self.sqrt_1m_ab = (1 - alpha_bar).sqrt()

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_ab = self.sqrt_ab.to(device)
        self.sqrt_1m_ab = self.sqrt_1m_ab.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        ab = self.sqrt_ab.to(x0.device)[t][:, None, None]
        s1 = self.sqrt_1m_ab.to(x0.device)[t][:, None, None]
        return ab * x0 + s1 * noise

    @torch.no_grad()
    def ddim_step(self, pred_noise, t_curr: int, t_prev: int, x_t):
        dev = x_t.device
        ab = self.alpha_bar.to(dev)[t_curr]
        ab_prev = (self.alpha_bar.to(dev)[t_prev] if t_prev >= 0
                   else torch.tensor(1.0, device=dev))
        x0 = (x_t - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0 = x0.clamp(-3.0, 3.0)
        return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred_noise

    def get_infer_timesteps(self, n_steps: int = INFER_STEPS) -> list:
        step = max(1, self.T // n_steps)
        return list(range(self.T - 1, -1, -step))[:n_steps]


# ══════════════════════════════════════════════════════════════════════════
# 블록
# ══════════════════════════════════════════════════════════════════════════
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000)
                         * torch.arange(half, device=t.device, dtype=torch.float32)
                         / (half - 1))
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConvResBlock(nn.Module):
    """1D 잔차 블록 + FiLM(scale/shift) 조건화."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        ng = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else 1)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(ng, out_ch)
        self.norm2 = nn.GroupNorm(ng, out_ch)
        self.act = nn.Mish()
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.act(self.norm1(self.conv1(x)))
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class MLPObsEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, obs_horizon=OBS_HORIZON, out_dim=256,
                 dropout: float = 0.0):
        super().__init__()
        h = max(64, out_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, h), nn.LayerNorm(h), nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim), nn.LayerNorm(out_dim), nn.Mish(),
            nn.Dropout(dropout),
        )

    def forward(self, obs):
        return self.net(obs.flatten(1))


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim=ACTION_DIM, global_cond_dim=256,
                 time_emb_dim=128, channels=(64, 128, 256), kernel_size=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        cond_dim = time_emb_dim + global_cond_dim
        self.stem = nn.Conv1d(action_dim, channels[0], kernel_size,
                              padding=kernel_size // 2)
        n_down = len(channels) - 1
        enc_in = [channels[0]] + list(channels[:-2])
        enc_out = list(channels[:-1])
        self.enc_blocks = nn.ModuleList([
            ConvResBlock(enc_in[i], enc_out[i], cond_dim, kernel_size)
            for i in range(n_down)])
        self.enc_ds = nn.ModuleList([
            nn.Conv1d(enc_out[i], enc_out[i], 4, stride=2, padding=1)
            for i in range(n_down)])
        self.mid_blocks = nn.ModuleList([
            ConvResBlock(channels[-2], channels[-1], cond_dim, kernel_size),
            ConvResBlock(channels[-1], channels[-1], cond_dim, kernel_size)])
        dec_us_in = list(reversed(channels[1:]))
        dec_skip = list(reversed(channels[:-1]))
        dec_out = list(reversed(channels[:-1]))
        self.dec_us = nn.ModuleList([
            nn.ConvTranspose1d(dec_us_in[i], dec_us_in[i], 4, stride=2, padding=1)
            for i in range(n_down)])
        self.dec_blocks = nn.ModuleList([
            ConvResBlock(dec_us_in[i] + dec_skip[i], dec_out[i], cond_dim, kernel_size)
            for i in range(n_down)])
        self.head = nn.Conv1d(channels[0], action_dim, kernel_size,
                              padding=kernel_size // 2)

    def forward(self, noisy_actions, timestep, global_cond):
        x = noisy_actions.permute(0, 2, 1)
        cond = torch.cat([self.time_mlp(timestep), global_cond], dim=-1)
        x = self.stem(x)
        skips = []
        for block, ds in zip(self.enc_blocks, self.enc_ds):
            x = block(x, cond)
            skips.append(x)
            x = ds(x)
        for block in self.mid_blocks:
            x = block(x, cond)
        for us, block in zip(self.dec_us, self.dec_blocks):
            x = us(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, cond)
        return self.head(x).permute(0, 2, 1)


# ══════════════════════════════════════════════════════════════════════════
# DiT 디노이저 (diffusion transformer, adaLN-Zero)
# ══════════════════════════════════════════════════════════════════════════
class DiTBlock(nn.Module):
    """self-attention + MLP, adaLN-Zero 조건화 (Peebles & Xie 2023)."""

    def __init__(self, d_model, n_head, mlp_ratio=4.0, cond_dim=384, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)      # adaLN-Zero: 초기엔 항등사상
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond):
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None] * a
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


class DiTDenoiser(nn.Module):
    """액션 시퀀스 T_pred 개를 토큰으로 보는 transformer 디노이저."""

    def __init__(self, action_dim=ACTION_DIM, global_cond_dim=256, time_emb_dim=128,
                 d_model=256, n_head=4, n_layer=4, max_len=64, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim))
        cond_dim = time_emb_dim + global_cond_dim
        self.inp = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_head, 4.0, cond_dim, dropout) for _ in range(n_layer)])
        self.nf = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * d_model))
        nn.init.zeros_(self.ada_f[1].weight); nn.init.zeros_(self.ada_f[1].bias)
        self.head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, noisy_actions, timestep, global_cond):
        T = noisy_actions.shape[1]
        cond = torch.cat([self.time_mlp(timestep), global_cond], dim=-1)
        x = self.inp(noisy_actions) + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x, cond)
        s, b = self.ada_f(cond).chunk(2, dim=-1)
        return self.head(self.nf(x) * (1 + s[:, None]) + b[:, None])


# ══════════════════════════════════════════════════════════════════════════
# 정책 래퍼
# ══════════════════════════════════════════════════════════════════════════
class DiffusionPolicy(nn.Module):
    """obs encoder + denoiser 를 하나로 묶어 state_dict / EMA 관리를 단순화."""

    def __init__(self, obs_dim=OBS_DIM, obs_horizon=OBS_HORIZON,
                 action_dim=ACTION_DIM, cond_dim=256, denoiser="unet",
                 channels=(64, 128, 256), dropout=0.0,
                 dit_d_model=256, dit_n_head=4, dit_n_layer=4):
        super().__init__()
        self.obs_encoder = MLPObsEncoder(obs_dim, obs_horizon, cond_dim, dropout)
        if denoiser == "unet":
            self.noise_pred_net = ConditionalUnet1D(action_dim, cond_dim,
                                                    channels=tuple(channels))
        elif denoiser == "dit":
            self.noise_pred_net = DiTDenoiser(action_dim, cond_dim,
                                              d_model=dit_d_model, n_head=dit_n_head,
                                              n_layer=dit_n_layer, dropout=dropout)
        else:
            raise ValueError(denoiser)
        self.denoiser_kind = denoiser
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim

    def forward(self, noisy_actions, timestep, obs_seq):
        return self.noise_pred_net(noisy_actions, timestep, self.obs_encoder(obs_seq))

    def compute_loss(self, obs, act, sched, diff_steps):
        B = obs.shape[0]
        t = torch.randint(0, diff_steps, (B,), device=obs.device)
        noise = torch.randn_like(act)
        pred = self.forward(sched.add_noise(act, noise, t), t, obs)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs_seq: torch.Tensor, scheduler: DDPMScheduler,
               pred_horizon: int, infer_steps: int = INFER_STEPS,
               generator: torch.Generator = None) -> torch.Tensor:
        """정규화된 obs_seq (B,T_obs,obs_dim) → 정규화된 action (B,T_pred,action_dim)."""
        B = obs_seq.shape[0]
        dev = obs_seq.device
        cond = self.obs_encoder(obs_seq)
        x = torch.randn(B, pred_horizon, self.action_dim, device=dev,
                        generator=generator)
        ts = scheduler.get_infer_timesteps(infer_steps)
        for i, t_curr in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            t_b = torch.full((B,), t_curr, device=dev, dtype=torch.long)
            x = scheduler.ddim_step(self.noise_pred_net(x, t_b, cond),
                                    t_curr, t_prev, x)
        return x


# ══════════════════════════════════════════════════════════════════════════
# EMA
# ══════════════════════════════════════════════════════════════════════════
class EMA:
    """지수이동평균 가중치. warmup 램프로 초기 스텝에서 과도한 관성을 피한다.

    decay = min(max_decay, (1 + step) / (10 + step))
    """

    def __init__(self, model: nn.Module, max_decay: float = 0.9999):
        self.max_decay = max_decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @property
    def decay(self) -> float:
        return min(self.max_decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                s.copy_(m)
        self.step += 1

    def state_dict(self):
        return {"step": self.step, "max_decay": self.max_decay,
                "shadow": self.shadow.state_dict()}

    def load_state_dict(self, sd):
        self.step = sd["step"]
        self.max_decay = sd.get("max_decay", self.max_decay)
        self.shadow.load_state_dict(sd["shadow"])


# ══════════════════════════════════════════════════════════════════════════
# BC 베이스라인 — 확산 없이 액션 청크를 바로 회귀
# ══════════════════════════════════════════════════════════════════════════
class BCPolicy(nn.Module):
    """obs → T_pred x action_dim 결정적 회귀. 확산이 실제로 필요한지 가늠하는 기준선.

    sample() 시그니처를 DiffusionPolicy 와 맞춰 평가·배포 코드를 공유한다.
    """

    def __init__(self, obs_dim=OBS_DIM, obs_horizon=OBS_HORIZON,
                 action_dim=ACTION_DIM, cond_dim=256, pred_horizon=16,
                 hidden=512, dropout=0.0):
        super().__init__()
        self.obs_encoder = MLPObsEncoder(obs_dim, obs_horizon, cond_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.Mish(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Mish(), nn.Dropout(dropout),
            nn.Linear(hidden, pred_horizon * action_dim))
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.denoiser_kind = "bc"

    def predict(self, obs_seq):
        h = self.head(self.obs_encoder(obs_seq))
        return h.view(-1, self.pred_horizon, self.action_dim)

    def compute_loss(self, obs, act, sched=None, diff_steps=None):
        return F.mse_loss(self.predict(obs), act)

    @torch.no_grad()
    def sample(self, obs_seq, scheduler=None, pred_horizon=None,
               infer_steps=None, generator=None):
        out = self.predict(obs_seq)
        if pred_horizon is not None and pred_horizon != self.pred_horizon:
            raise ValueError(f"BC 는 학습 시 pred_horizon={self.pred_horizon} 고정")
        return out


def policy_from_ckpt(ck: dict, device, use_ema: bool = True):
    """체크포인트 하나로 정책을 완전히 재구성 (평가·배포 공용)."""
    cfg = ck["config"]
    m = cfg.get("model", {})
    p, _ = build_policy(
        cfg["obs_dim"], cfg["obs_horizon"], cfg["action_dim"], device,
        algo=cfg.get("algo", "dp_unet"),
        cond_dim=m.get("cond_dim", 256),
        channels=tuple(m.get("channels", (64, 128, 256))),
        dropout=0.0,                      # 추론 시 dropout 끔
        pred_horizon=cfg["pred_horizon"],
        dit_d_model=m.get("dit_d_model", 256),
        dit_n_head=m.get("dit_n_head", 4),
        dit_n_layer=m.get("dit_n_layer", 4))
    sd = ck["ema"]["shadow"] if (use_ema and ck.get("ema")) else ck["policy"]
    p.load_state_dict(sd)
    p.eval()
    return p


def build_policy(obs_dim, obs_horizon, action_dim, device, algo="dp_unet",
                 cond_dim=256, channels=(64, 128, 256), dropout=0.0,
                 pred_horizon=16, dit_d_model=256, dit_n_head=4, dit_n_layer=4):
    if algo == "bc_mlp":
        p = BCPolicy(obs_dim, obs_horizon, action_dim, cond_dim,
                     pred_horizon, dropout=dropout).to(device)
    else:
        p = DiffusionPolicy(obs_dim, obs_horizon, action_dim, cond_dim,
                            denoiser="dit" if algo == "dp_dit" else "unet",
                            channels=channels, dropout=dropout,
                            dit_d_model=dit_d_model, dit_n_head=dit_n_head,
                            dit_n_layer=dit_n_layer).to(device)
    n = sum(q.numel() for q in p.parameters())
    return p, n
