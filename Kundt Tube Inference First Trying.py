import sys
print("Python exe:", sys.executable)
import numpy as np
import torch
import sbi
print("ok")

from dataclasses import dataclass

@dataclass
class MeasurementConfig:
    fs: float
    freqs: np.ndarray             # frequenze a cui estrai feature
    mic_positions: np.ndarray     # es. [x1, x2, x3] lungo il tubo
    piston_distances: np.ndarray  # le 3 distanze diverse
    tube_radius: float
    air_temp_C: float
    air_rho: float                # densità aria
    air_c: float                  # velocità del suono

def make_summary_from_measurements(alpha_f: np.ndarray) -> np.ndarray:
    """
    Esempio: prendo α(f) e lo porto in un vettore.
    alpha_f shape suggerita: (n_piston, n_reps, n_freqs)
      - n_piston=3
      - n_reps = 2 campioni * (eventuali medie su 4 sweep) * (eventuali combinazioni mic)
    Output: vettore 1D (feature)
    """
    # opzione semplice: media su repliche, e concateno le 3 condizioni pistone
    alpha_mean = alpha_f.mean(axis=1)  # (n_piston, n_freqs)
    return alpha_mean.reshape(-1)      # (n_piston*n_freqs,)


from torch.distributions import Uniform

# esempio: theta = [log10_sigma, thickness_m, nuisance_gain, nuisance_phase]
# (scegli tu parametri reali del tuo modello)
prior = torch.distributions.Independent(
    torch.distributions.Uniform(
        low=torch.tensor([2.0, 0.01, 0.8, -0.2]),
        high=torch.tensor([6.0, 0.10, 1.2,  0.2])
    ),
    reinterpreted_batch_ndims=1
)

def simulate_alpha_from_theta(theta: np.ndarray, cfg: MeasurementConfig, rng: np.random.Generator):
    """
    TODO: qui inserisci il tuo modello fisico.
    Output consigliato: alpha_f shape (n_piston, n_reps, n_freqs)
    """
    log10_sigma, thickness, gain, phase = theta

    n_piston = len(cfg.piston_distances)
    n_freqs = len(cfg.freqs)

    # --- ESEMPIO PLACEHOLDER ---
    # Produco una α(f) "smooth" tra 0 e 1, dipendente da theta.
    # SOSTITUISCI con il tuo modello ISO 10534 / impedenza / riflessione / ecc.
    base = 1.0 - np.exp(-(cfg.freqs / (10**log10_sigma))**0.7)
    base = np.clip(base, 0.0, 1.0)

    alpha = []
    for _ in range(n_piston):
        # variazione per posizione pistone: piccola perturbazione
        perturb = 0.03 * rng.normal(size=n_freqs)
        a = np.clip(gain * base + perturb, 0.0, 1.0)
        alpha.append(a)

    alpha = np.stack(alpha, axis=0)  # (n_piston, n_freqs)

    # repliche: 2 campioni * (eventuale rumore sweep)
    n_reps = 2
    alpha_reps = alpha[:, None, :] + 0.02 * rng.normal(size=(n_piston, n_reps, n_freqs))
    alpha_reps = np.clip(alpha_reps, 0.0, 1.0)
    return alpha_reps

def simulator_torch(theta: torch.Tensor, cfg: MeasurementConfig, seed: int = 0) -> torch.Tensor:
    """
    Wrapper torch -> torch per sbi.
    theta: (d,) oppure (batch, d)
    """
    theta_np = theta.detach().cpu().numpy()
    rng = np.random.default_rng(seed)

    def one(theta1):
        alpha_f = simulate_alpha_from_theta(theta1, cfg, rng)
        x = make_summary_from_measurements(alpha_f)
        return x.astype(np.float32)

    if theta_np.ndim == 1:
        x = one(theta_np)
        return torch.from_numpy(x)
    else:
        xs = [one(t) for t in theta_np]
        return torch.from_numpy(np.stack(xs, axis=0))
    

from sbi.inference import SNPE
from sbi.utils import process_prior

def train_sbi(cfg: MeasurementConfig, num_simulations: int = 50_000, seed: int = 0):
    torch.manual_seed(seed)
    prior_proc, _, _ = process_prior(prior)

    inference = SNPE(prior=prior_proc)

    # genera dataset simulato
    theta = prior_proc.sample((num_simulations,))
    x = simulator_torch(theta, cfg, seed=seed)

    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior

def infer_posterior(posterior, x_obs: np.ndarray, num_samples: int = 20_000, seed: int = 1):
    torch.manual_seed(seed)
    x_t = torch.tensor(x_obs, dtype=torch.float32)
    samples = posterior.sample((num_samples,), x=x_t)
    return samples.detach().cpu().numpy()


import matplotlib.pyplot as plt

def posterior_predictive_check(samples_theta: np.ndarray, cfg: MeasurementConfig, x_obs: np.ndarray, n_rep: int = 200, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples_theta), size=n_rep, replace=False)
    thetas = samples_theta[idx]

    Xrep = []
    for i, th in enumerate(thetas):
        xrep = simulator_torch(torch.tensor(th, dtype=torch.float32), cfg, seed=seed+i).numpy()
        Xrep.append(xrep)
    Xrep = np.stack(Xrep, axis=0)  # (n_rep, n_features)

    # confronto: media e bande (es. 5–95%)
    q05, q50, q95 = np.quantile(Xrep, [0.05, 0.50, 0.95], axis=0)

    plt.figure()
    plt.plot(x_obs, label="x_obs")
    plt.plot(q50, label="PPC median")
    plt.fill_between(np.arange(len(x_obs)), q05, q95, alpha=0.3, label="PPC 5-95%")
    plt.legend()
    plt.title("Posterior Predictive Check (feature space)")
    plt.xlabel("Feature index")
    plt.ylabel("Value")
    plt.show()

    return Xrep


if __name__ == "__main__":
    # 1) Config minimo (metti valori reali dopo)
    cfg = MeasurementConfig(
        fs=25000.0,
        freqs=np.linspace(100, 2000, 64),          # 64 frequenze esempio
        mic_positions=np.array([0, 0.10, 0.15]),# m (esempio)
        piston_distances=np.array([0.02, 0.05, 0.08]),
        tube_radius=0.05,
        air_temp_C=20.0,
        air_rho=1.204,
        air_c=343.0
    )

    # 2) Crea una "misura osservata" finta (finché non colleghi i dati reali)
    rng = np.random.default_rng(999)
    theta_true = np.array([4.0, 0.05, 1.0, 0.0], dtype=float)
    alpha_true = simulate_alpha_from_theta(theta_true, cfg, rng)
    x_obs = make_summary_from_measurements(alpha_true)

    # 3) Allena (riduci simulazioni per test veloce)
    print("Training SBI...")
    posterior = train_sbi(cfg, num_simulations=2000, seed=0)
    print("Training done.")

    # 4) Inferenza
    print("Sampling posterior...")
    samples = infer_posterior(posterior, x_obs, num_samples=2000, seed=1)
    print("Posterior samples shape:", samples.shape)

    # 5) PPC + plot
    print("Running PPC...")
    _ = posterior_predictive_check(samples, cfg, x_obs, n_rep=200, seed=123)
    print("Done.")