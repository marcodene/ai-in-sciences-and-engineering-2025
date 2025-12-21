# 📐 Guida Completa alle Dimensioni - Task 2 FNO

Una guida visuale per capire TUTTE le dimensioni dei dati, variabili e parametri nel progetto.

---

## 📊 1. Dataset Originali (File .npy)

### Struttura Base
Tutti i dataset hanno la stessa struttura a 3 dimensioni:

```
Shape: (N_trajectories, N_timesteps, Resolution)
         ↓                ↓              ↓
    Numero traiettorie   5 snapshots    Punti spaziali
```

### Dataset Specifici

| File | Shape | Descrizione |
|------|-------|-------------|
| `data_train_128.npy` | **(1024, 5, 128)** | Training: 1024 traiettorie |
| `data_val_128.npy` | **(32, 5, 128)** | Validation: 32 traiettorie |
| `data_test_128.npy` | **(128, 5, 128)** | Test: 128 traiettorie |
| `data_test_32.npy` | **(128, 5, 32)** | Test multi-res: risoluzione 32 |
| `data_test_64.npy` | **(128, 5, 64)** | Test multi-res: risoluzione 64 |
| `data_test_96.npy` | **(128, 5, 96)** | Test multi-res: risoluzione 96 |

### Esempio Visivo: data_train_128.npy

```python
data = np.load("data_train_128.npy")  # Shape: (1024, 5, 128)

# Dimensione 0: Traiettorie (1024)
data[0]      # Prima traiettoria
data[1]      # Seconda traiettoria
data[1023]   # Ultima traiettoria

# Dimensione 1: Timesteps (5)
data[0, 0]   # Traiettoria 0, t=0.0   → u(x, t=0.0)
data[0, 1]   # Traiettoria 0, t=0.25  → u(x, t=0.25)
data[0, 2]   # Traiettoria 0, t=0.50  → u(x, t=0.50)
data[0, 3]   # Traiettoria 0, t=0.75  → u(x, t=0.75)
data[0, 4]   # Traiettoria 0, t=1.0   → u(x, t=1.0)

# Dimensione 2: Spazio (128 punti)
data[0, 0, :]   # Shape: (128,) - valori di u in 128 punti spaziali
```

**Cosa rappresenta:**
- Una "traiettoria" = una soluzione completa della PDE per una condizione iniziale specifica
- Ogni traiettoria ha 5 "snapshot" temporali
- Ogni snapshot ha 128 valori (la funzione u(x) discretizzata in 128 punti)

---

## 🎯 2. Task 2.1: One-to-One Training

### Obiettivo
Imparare: **u(t=0) → u(t=1)**

### Preprocessing dei Dati

#### Input (X)
```python
# PRIMA: Estrai snapshot
X_raw = data[:, 0, :]  # Shape: (N, 128)
                       # Prendi solo t=0 (index 0)

# SECONDA: Aggiungi griglia spaziale
x_grid = linspace(0, 1, 128)  # Shape: (128,)
x_grid_rep = repeat(x_grid, N)  # Shape: (N, 128)

# TERZA: Stack features
X = stack([X_raw, x_grid_rep], dim=-1)
# Shape: (N, 128, 2)
#         ↓   ↓    ↓
#        samples | features
#                128 punti spaziali
#                     2 features: [u(x,t=0), x]
```

#### Output (Y)
```python
Y = data[:, 4, :]  # Shape: (N, 128)
                   # Prendi solo t=1.0 (index 4)

Y = Y.unsqueeze(-1)  # Shape: (N, 128, 1)
                     # Aggiungi dimensione canale
```

### Dimensioni Finali per Training

| Set | N | X Shape | Y Shape | Significato |
|-----|---|---------|---------|-------------|
| **Train** | 1024 | **(1024, 128, 2)** | **(1024, 128, 1)** | 1024 coppie input-output |
| **Val** | 32 | **(32, 128, 2)** | **(32, 128, 1)** | 32 coppie per validazione |
| **Test** | 128 | **(128, 128, 2)** | **(128, 128, 1)** | 128 coppie per test |

### Durante Training (con Batching)

```python
batch_size = 32

# DataLoader crea batch:
for inputs, targets in train_loader:
    # inputs:  (32, 128, 2)  ← batch di 32 samples
    #           ↓   ↓    ↓
    #          batch | 2 features
    #                128 punti spaziali
    
    # targets: (32, 128, 1)  ← outputs corrispondenti
    
    outputs = model(inputs)  # (32, 128, 1)
```

---

## 🔄 3. Task 2.3: All-to-All Training

### Obiettivo
Imparare: **u(t_i) → u(t_j)** per TUTTI i timesteps

### Creazione Coppie (t_i, t_j)

Per ogni traiettoria, creiamo TUTTE le coppie possibili:

```
Timesteps: [0.0, 0.25, 0.50, 0.75, 1.0]

Coppie per UNA traiettoria:
(t=0.0  → t=0.0)   ← autoregressive
(t=0.0  → t=0.25)
(t=0.0  → t=0.50)
(t=0.0  → t=0.75)
(t=0.0  → t=1.0)
(t=0.25 → t=0.25)
(t=0.25 → t=0.50)
(t=0.25 → t=0.75)
(t=0.25 → t=1.0)
... (continua)
(t=1.0  → t=1.0)

Totale: 5 + 4 + 3 + 2 + 1 = 15 coppie
```

### Preprocessing

#### Input (X)
```python
# Per ogni coppia (t_i, t_j):
u_input = trajectory[i]  # (128,) - u(x, t_i)
t_in = time_steps[i]     # scalar - valore di t_i
x_grid = linspace(0, 1, 128)  # (128,) - griglia spaziale

# Ripeti t_in per ogni punto spaziale
t_in_repeated = [t_in] * 128  # (128,)

# Stack features
X = stack([u_input, x_grid, t_in_repeated], dim=-1)
# Shape: (128, 3)
#         ↓    ↓
#        punti spaziali
#             3 features: [u(x,t_i), x, t_i]
```

#### Output (Y)
```python
Y = trajectory[j].unsqueeze(-1)  # (128, 1) - u(x, t_j)
```

### Dimensioni Finali

Per **1 traiettoria** → 15 coppie  
Per **N traiettorie** → N × 15 coppie

| Set | N_traj | N_coppie | X Shape | Y Shape |
|-----|--------|----------|---------|---------|
| **Train** | 1024 | 15,360 | **(15360, 128, 3)** | **(15360, 128, 1)** |
| **Val** | 32 | 480 | **(480, 128, 3)** | **(480, 128, 1)** |
| **Test** | 128 | 1,920 | **(1920, 128, 3)** | **(1920, 128, 1)** |

**Nota:** 1024 × 15 = 15,360 esempi di training!

---

## 🧠 4. Modello FNO1d

### Parametri del Modello

```python
model = FNO1d(
    modes=16,      # Numero di modi Fourier da mantenere
    width=64,      # Dimensione hidden layer
    in_dim=2,      # Canali input (one2one: [u, x])
    in_dim=3,      # Canali input (all2all: [u, x, t])
    out_dim=1,     # Canali output (sempre u)
    padding=0      # Padding (non usato)
)
```

### Flow delle Dimensioni nel Forward Pass

#### One2One (in_dim=2)

```python
Input: (batch, 128, 2)
  ↓
linear_p (2 → 64)
  ↓
(batch, 128, 64)
  ↓
permute → (batch, 64, 128)
  ↓
3x Fourier Layers
  - SpectralConv1d: (batch, 64, 128) → (batch, 64, 128)
  - Conv1d: (batch, 64, 128) → (batch, 64, 128)
  - Tanh activation
  ↓
(batch, 64, 128)
  ↓
permute → (batch, 128, 64)
  ↓
linear_q (64 → 32)
  ↓
(batch, 128, 32)
  ↓
output_layer (32 → 1)
  ↓
Output: (batch, 128, 1)
```

#### All2All (in_dim=3)

```python
Input: (batch, 128, 3)  ← 3 features invece di 2
  ↓
linear_p (3 → 64)  ← Proietta 3 features a 64
  ↓
... (resto identico al one2one)
  ↓
Output: (batch, 128, 1)
```

### Dimensioni Spettro Fourier

```python
# Nel SpectralConv1d:
x_ft = fft.rfft(x)  # (batch, 64, 65)
                    #              ↑
                    # rfft produce 128//2 + 1 = 65 coefficienti

# Tronca ai primi 'modes' modi:
x_ft[:, :, :16]  # (batch, 64, 16)
                 # Mantieni solo 16 modi invece di 65
```

**Perché modes=16?**
- Modi bassi = frequenze basse (smooth)
- Modi alti = frequenze alte (dettagli fini)
- 16 modi è un buon compromesso tra accuratezza e velocità

---

## 📦 5. Batch Processing

### Esempio Concreto di un Batch

```python
batch_size = 32

# Un batch durante training:
inputs.shape:  (32, 128, 2)  o  (32, 128, 3)
               ↓   ↓    ↓
              batch
                  punti spaziali (risoluzione fissa)
                       features (2 per one2one, 3 per all2all)

targets.shape: (32, 128, 1)
               ↓   ↓    ↓
              batch
                  punti spaziali
                       1 output (valore di u)

# Ogni elemento del batch è INDIPENDENTE
inputs[0]:  primo esempio del batch   (128, 2)
inputs[1]:  secondo esempio del batch (128, 2)
...
inputs[31]: 32-esimo esempio           (128, 2)
```

---

## 🎲 6. Calcolo Errore Relativo L²

### Formula

```
Per un singolo sample:
error = ||u_pred - u_true||₂ / ||u_true||₂

Per un batch:
errors = [error_sample_0, error_sample_1, ..., error_sample_31]
```

### Implementazione

```python
def compute_relative_l2_error(pred, true):
    # Input:
    # pred:  (batch_size, resolution, 1)
    # true:  (batch_size, resolution, 1)
    
    pred = pred.squeeze(-1)  # (batch_size, resolution)
    true = true.squeeze(-1)  # (batch_size, resolution)
    
    # Norma L2 per ogni sample (dim=1 = asse spaziale)
    diff_norm = torch.norm(pred - true, dim=1)  # (batch_size,)
    true_norm = torch.norm(true, dim=1)         # (batch_size,)
    
    errors = diff_norm / true_norm  # (batch_size,)
    
    return errors.sum().item()  # Somma → scalar
```

### Aggregazione

```python
# Durante validation/test:
total_error = 0
for batch in dataloader:
    total_error += compute_relative_l2_error(pred, true)
    # Accumula la SOMMA degli errori

# Alla fine:
avg_error = total_error / n_total_samples
```

---

## 📋 7. Tabella Riassuntiva Rapida

### Task 2.1 (One2One)

| Cosa | Dimensione | Note |
|------|-----------|------|
| **Input raw** | (N, 128) | u(x, t=0) |
| **Input + grid** | (N, 128, 2) | [u, x] |
| **Output** | (N, 128, 1) | u(x, t=1) |
| **N train** | 1024 | |
| **N val** | 32 | |
| **N test** | 128 | |
| **Batch** | (32, 128, 2) → (32, 128, 1) | |

### Task 2.3 (All2All)

| Cosa | Dimensione | Note |
|------|-----------|------|
| **Input** | (N, 128, 3) | [u, x, t] |
| **Output** | (N, 128, 1) | u(x, t_out) |
| **N train** | 15,360 | 1024 × 15 |
| **N val** | 480 | 32 × 15 |
| **N test** | 1,920 | 128 × 15 |
| **Batch** | (32, 128, 3) → (32, 128, 1) | |

### Modello FNO

| Layer | Input | Output | Parametri |
|-------|-------|--------|-----------|
| `linear_p` | (B, R, C_in) | (B, R, 64) | C_in × 64 |
| `SpectralConv1d` | (B, 64, R) | (B, 64, R) | 64 × 64 × 16 (complex) |
| `Conv1d` | (B, 64, R) | (B, 64, R) | 64 × 64 |
| `linear_q` | (B, R, 64) | (B, R, 32) | 64 × 32 |
| `output_layer` | (B, R, 32) | (B, R, 1) | 32 × 1 |

**Legenda:** B=batch, R=resolution (128), C_in=numero canali input (2 o 3)

---

## 💡 8. Domande Frequenti

### Q: Perché la risoluzione è sempre 128 nel training?
**A:** Il modello viene addestrato solo su risoluzione 128. Task 2.2 testa se il modello generalizza ad altre risoluzioni (32, 64, 96) **senza riaddestrare**.

### Q: Quale dimensione cambia tra one2one e all2all?
**A:** Solo **in_dim** cambia (2 → 3) perché aggiungiamo il tempo come feature.

### Q: Cosa succede durante il batch?
**A:** Il DataLoader prende N samples e crea "mini-batches" di dimensione batch_size. Il modello processa un batch alla volta.

### Q: Come si calcola il numero di parametri del modello?
**A:** Dipende da `width`, `modes`, e `in_dim`:
- One2one (in_dim=2): ~150K parametri
- All2all (in_dim=3): ~152K parametri (pochissima differenza)

---

## 🎯 Quick Reference

```python
# One2One
X_train:  (1024, 128, 2)   # [u(t=0), x]
Y_train:  (1024, 128, 1)   # u(t=1)

# All2All  
X_train:  (15360, 128, 3)  # [u(t_i), x, t_i]
Y_train:  (15360, 128, 1)  # u(t_j)

# Batch (esempio)
inputs:   (32, 128, 2 o 3)
outputs:  (32, 128, 1)

# FNO flow
(B, R, in_dim) → (B, R, 64) → (B, R, 1)
```

Usa questo documento come riferimento ogni volta che hai dubbi sulle dimensioni! 📐
