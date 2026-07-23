
import numpy as np
import torch
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler


class ResidualBlock(nn.Module):
    """Residual correction block — learns what XGBoost missed."""
    def __init__(self, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        self.skip = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class CausalAttention(nn.Module):
    """Causal attention head — learns intervention weights."""
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attn(x)
        return x * weights, weights


class CSARNet(nn.Module):
    """
    CSAR-Net: Causal Survival Attention Residual Network.

    Architecture:
        Input → Embedding → ResidualBlock x2
              → CausalAttention → Cox hazard output

    Args:
        in_dim  : input feature dimension (default 12)
        hidden  : hidden layer size (default 128)
    """
    def __init__(self, in_dim=12, hidden=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        self.res1        = ResidualBlock(hidden)
        self.res2        = ResidualBlock(hidden)
        self.causal_attn = CausalAttention(hidden)
        self.cox_head    = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        h, attn_weights = self.causal_attn(h)
        log_hazard = self.cox_head(h)
        return log_hazard, attn_weights


def cox_ph_loss(log_h, t, e):
    """Breslow approximation of Cox partial likelihood."""
    order  = torch.argsort(t, descending=True)
    log_h  = log_h[order]
    e      = e[order]
    lcs    = torch.logcumsumexp(log_h, dim=0)
    return -torch.mean((log_h - lcs) * e)


def train_csar_net(X_tr, t_tr, e_tr,
                   X_va, t_va, e_va,
                   in_dim=12, hidden=128,
                   lr=5e-4, epochs=200,
                   patience=20, device=None):
    """
    Train CSAR-Net with early stopping.

    Returns trained model.
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'cpu')

    model = CSARNet(
        in_dim=in_dim, hidden=hidden
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(
        opt, step_size=50, gamma=0.5)

    def T(a):
        return torch.FloatTensor(a).to(device)

    Xtr=T(X_tr); ttr=T(t_tr); etr=T(e_tr)
    Xva=T(X_va); tva=T(t_va); eva=T(e_va)

    best_loss = np.inf
    best_wts  = None
    p_ctr     = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        lh, _ = model(Xtr)
        loss   = cox_ph_loss(
            lh.squeeze(), ttr, etr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            lh_v, _ = model(Xva)
            vl = cox_ph_loss(
                lh_v.squeeze(), tva, eva)

        if vl.item() < best_loss:
            best_loss = vl.item()
            best_wts  = {
                k: v.clone()
                for k, v in
                model.state_dict().items()
            }
            p_ctr = 0
        else:
            p_ctr += 1
        if p_ctr >= patience:
            break

    model.load_state_dict(best_wts)
    return model


def predict_cindex(model, X_te, t_te, e_te,
                   device=None):
    """Evaluate CSAR-Net and return C-index."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'cpu')
    model.eval()
    with torch.no_grad():
        lh, _ = model(
            torch.FloatTensor(X_te).to(device))
    risk = lh.squeeze().cpu().numpy()
    return concordance_index_censored(
        e_te.astype(bool), t_te, risk)[0]
