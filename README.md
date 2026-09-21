# EEG-MS-MLOps

**Autore**: Luca Pulga  
**Data**: Agosto 2026  

---

## Indice

1. [Panoramica progetto](#panoramica-progetto)
2. [Obiettivi scientifici](#obiettivi-scientifici)
3. [Architettura Pipeline](#architettura-pipeline)
4. [Dataset e dati](#dataset-e-dati)
5. [Modelli utilizzati](#modelli-utilizzati)
6. [Metriche e performance](#metriche-e-performance)
7. [Risultati dettagliati](#risultati-dettagliati)
8. [Come riprodurre il progetto](#come-riprodurre-il-progetto)
9. [Interpretazione dei risultati](#interpretazione-dei-risultati)
10. [Limitazioni e futuri sviluppi](#limitazioni-e-futuri-sviluppi)

---

## Overview

### Che cosa è stato realizzato?

Questo progetto è un **pipeline end-to-end di machine learning per la classificazione di soggetti con Sclerosi Multipla (MS) vs Controlli Sani (HC)** utilizzando dati EEG (elettroencefalografia).
Il progetto segue le caratteristiche della metodologia MLOps:

- **Validazione riproducibile** mediante nested repeated cross-validation
- **Tracciamento dei dati** con audit completo
- **Configurazione centralizzata** per facilità di modifica e esperimenti
- **Pipeline modulare** in Python seguendo le best practices MLOps
- **Metriche clinicamente rilevanti** (sensibilità, specificità, AUC-ROC)

### Stack tecnologico

| Componente | Libreria | Versione |
|-----------|----------|----------|
| Elaborazione dati | `pandas` | ≥2.2 |
| Calcoli numerici | `numpy` | ≥2.0 |
| Segnali | `scipy` | ≥1.14 |
| ML | `scikit-learn` | ≥1.6 |
| Gradient Boosting | `xgboost` | ≥3.0 |
| I/O | `pyarrow` | ≥17.0 |
| Visualizzazione | `matplotlib`, `seaborn` | ≥3.9, ≥0.13 |
| Config | `pyyaml` | ≥6.0 |
| Python | ≥3.11 | |

---

## Scientific goals

### 1. Caratterizzazione Biomarker EEG

Le misure provenienti dall'elaborazione dei segnali EEG in questione sono le seguenti descritte. La pipleine in realtà fornisce un semplice modo per aggiungere nuove feature che si potranno rendere significative.
In questo progetto sono state prese in considerazione:

#### **PSD (Power Spectral Density)**
- Misure di potenza nelle bande frequenziali standard:
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-100 Hz)
- Calcolata su 5 sottobande per misura
- 27 canali × 18 finestre temporali per condizione

#### **Complessità Spettrale (Spectral Entropy)**
- Misura dell'organizzazione dell'attività cerebrale
- Aiuta identificare disorganizzazione in MS
- 27 canali × 18 finestre temporali per condizione

### 2. Condizioni sperimentali

**Due condizioni di registrazione del segnale EEG**:
- **CE (Closed Eyes)**: baseline a riposo
- **OE (Open Eyes)**: reattività cerebrale (blocco-rallentamente del ritmo alpha)

**Due gruppi**:
- **HC (Healthy Controls)**: soggetti sani di controllo
- **MS (Multiple Sclerosis)**: pazienti diagnosticati con Sclerosi Multipla

### 3. Risoluzione Spaziale

**Doppia rappresentazione**:
- **Canale**: 27 canali individuali
- **ROI (Region Of Interest)**: 6 aree macroscopiche

| Livello | N. Channels | Uso |
|---------|------------|-----|
| Canale | 27 | Localizzazione precisa anomalie |
| ROI | 6 | Semplificazione, stabilità modello |

---

## Architettura della pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (data/raw/)                         │
│   ├─ PSD/                                                       │
│   │  ├─ hc/ (file .mat)                                         │
│   │  └─ ms/ (file .mat)                                         │
│   └─ Complexity/                                                │
│      ├─ hc/ (file .mat)                                         │
│      └─ ms/ (file .mat)                                         │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: AUDIT & DISCOVERY                          │
│  python -m eeg_ms.data.audit                                    │
│  Scoperta ricorsiva file .mat                                │
│  Parsing metadati da percorsi                                │
│  Validazione strutturale                                      │
│  Generazione manifest + variabili inventory                  │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│            INTERIM DATA (data/interim/)                         │
│  ├─ file_manifest.csv (tracciabilità file)                     │
│  ├─ variable_inventory.csv (variabili MATLAB)                  │
│  └─ validation_report.json (esito audit)                       │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: DATASET CANONICO (LONG FORMAT)                  │
│  python -m eeg_ms.data.build_dataset                            │
│  Caricamento dati da .mat                                     │
│  Trasformazione a long format                                 │
│  Arricchimento metadati                                       │
│  Validazione completezza dataset                              │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│           PROCESSED DATA (data/processed/)                      │
│  ├─ canonical_features.parquet (formato long)                  │
│  ├─ subjects.csv (metadata soggetti)                            │
│  └─ splits/                                                     │
│     └─ outer_test_folds.csv (fold CV)                          │
└──────────────────┬──────────────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐
  │STEP 3: QUALITY  │  │STEP 4: FEATURE  │
  │CONTROL          │  │ENGINEERING      │
  │python -m eeg_ms │  │python -m eeg_ms │
  │quality.run_     │  │features.build_  │
  │quality_control  │  │tabular          │
  └────────┬────────┘  └────────┬────────┘
           │                    │
           ▼                    ▼
  reports/tables/      data/processed/
  reports/figures/     subject_features_roi.parquet
         │                    │
         └────────┬───────────┘
                  ▼
       ┌─────────────────────────────┐
       │STEP 5: GENERAZIONE SPLIT CV │
       │python -m eeg_ms.validation. │
       │generate_splits              │
       └────────────┬────────────────┘
                    │
                    ▼
       ┌─────────────────────────────┐
       │STEP 6: TRAINING & TUNING    │
       │python -m eeg_ms.modeling.   │
       │run_experiments              │
       │ - 5 repeat × 4 fold outer   │
       │ - GridSearchCV inner        │
       │ - 5 modelli + baseline      │
       └────────────┬────────────────┘
                    │
                    ▼
       ┌─────────────────────────────┐
       │ARTIFACTS (artifacts/)       │
       │├─ metrics/                  │
       ││  └─ nested_cv_fold_metrics │
       │└─ predictions/              │
       │   └─ nested_cv_predictions  │
       └────────────┬────────────────┘
                    │
                    ▼
       ┌─────────────────────────────┐
       │STEP 7: ANALISI RISULTATI    │
       │python -m eeg_ms.evaluation. │
       │analyze_results              │
       │ - Aggregazione metriche     │
       │ - Bootstrap confidence int. │
       │ - Analisi errori soggetti   │
       └─────────────────────────────┘
```

---

### Pipeline steps

#### **Step 1: Audit & Discovery**
```bash
python -m eeg_ms.data.audit
```

**Scopo**:
- Scopre ricorsivamente tutti i file .mat, estraendo i metadati dai percorsi analoghi a: `PSD/hc/PSDrelative_ID_01_CTR_CE.mat`
  - `family` = psd
  - `group` = hc
  - `subject_id` = hc_01 (costruito come `{group}_{numero}`)
  - `condition` = CE
- Valida struttura directories per non incorrere in errori di file salvati in posizioni errate
- Valida le convenzioni dei nomi
- Controlla la forma e il tipo di matrici MATLAB contenenti i dati estratti dall'EEG


**Input**: File .mat nelle cartelle raw  
**Output**:
- `data/interim/file_manifest.csv` - una riga per file
- `data/interim/variable_inventory.csv` - una riga per variabile
- `data/interim/validation_report.json` - report esito audit

In caso almeno una delle funzioni di audito sopra riportate fallisca, allora la pipeline **fallisce rapidamente su errori strutturali** in modo da non avviare la pipeline con dati non strutturati nel modo opportuno o dati non appartenenti alle famiglie di feature o condizioni corrette.


#### **Step 2: Dataset canonico**
```bash
python -m eeg_ms.data.build_dataset
```

**Input**: Audit completato senza errori  
**Output**: `data/processed/canonical_features.parquet`

**Scopo**:
- Legge tutti i file .mat validati
- Trasforma da formato wide a **long format**:
  ```
  subject_id | condition | feature_family | ... | value
  hc_01      | CE        | psd            | ... | 2.45
  hc_01      | CE        | psd            | ... | 1.89
  ...
  ```
- Dunque il formato rispetta la forma:
   - una riga = una singola misura
- Colonne: subject_id, group, condition, feature_family, measure, band, spatial_level, location_index, window, value

**Perchè il "long format"**:
Esso facilita il plot e le visualizzazioni dei dati, semplificando il filtering (es. "solo alpha in OE").
Permette inoltre di mantenere la tracciabilità dei dati e facilita gli step successivi della pipeline.

#### **Step 3: Quality control**
```bash
python -m eeg_ms.quality.run_quality_control
```

**Scopo**:
- Controlla completezza dati
- Identifica outlier tecnici
- Produce heatmap correlazioni per ROI
- Visualizza distribuzioni per band/condizione

**Output**: 
- `reports/tables/` - CSV con sintesi qualità
- `reports/figures/quality_control/` - grafici diagnostici


#### **Step 4: Feature engineering**
```bash
python -m eeg_ms.features.build_tabular
```

**Scopo**:
- Aggrega da long a **wide format** (una riga = un soggetto)
- Seleziona ROI al posto dei canali (riduzione da 27 a 6)
- Crea colonne feature: `PSD_alpha_ROI_1`, `Entropy_CE_ROI_2`, ecc.
- **Output**: 1 riga per soggetto, ~120 feature

**Input**: `canonical_features.parquet`  
**Output**: `data/processed/subject_features_roi.parquet`


#### **Step 5: Generazione Split CV**
```bash
python -m eeg_ms.validation.generate_splits
```
**Scopo**:
- Crea assignment soggetti a fold esterni
- 5 repeat × 4 fold = 20 scenari di valutazione

**Output**: `data/processed/splits/outer_test_folds.csv`


#### **Step 6: Nested Repeated Cross-Validation**
```bash
python -m eeg_ms.modeling.run_experiments
```

**Input**: Feature tabular + split  
**Output**:
- `artifacts/metrics/nested_cv_fold_metrics.csv`
- `artifacts/predictions/nested_cv_predictions.csv`

**Schema validazione**:
```
┌─ OUTER LOOP (valutazione) ──────────┐
│  5 Repeat × 4 Fold = 20 test-set    │
│  Per ogni outer fold:                │
│  ┌─ INNER LOOP (tuning) ──────────┐  │
│  │ 4-fold GridSearchCV            │  │
│  │ Cerca iperparametri ottimi     │  │
│  │ su dati training               │  │
│  └─────────────────────────────────┤  │
│  Test su hold-out esterno          │  │
└────────────────────────────────────┘
```

---

## Dataset

### Composizione

| Aspetto | Dettagli |
|---------|----------|
| **Soggetti totali** | 32 (8 HC vs 24 MS) |
| **Condizioni** | 2 (CE, OE) |
| **Feature per soggetto** | 27 canali + 6 ROI per famiglia |
| **Finestre temporali** | 18 per condizione |
| **Bande frequenziali** | 5 (delta, theta, alpha, beta, gamma) |
| **Famiglie feature** | 2 (PSD, Entropy/Complessità) |

### Dimensioni Canoniche

Per ogni soggetto-condizione:
- **PSD**: 5 bande × (27 canali + 6 ROI) × 18 finestre = **5,940 righe**
- **Entropy**: (27 canali + 6 ROI) × 18 finestre = **594 righe**
- **Totale per subject-condition**: ~6,534 righe

### Target

```
group=hc  -> target=0 (controllo sano)
group=ms  -> target=1 (paziente MS)
```

---

## Modelli ML

### Baseline: Dummy Classifier

**Scopo**: Benchmark minimo per validare pipeline

```python
DummyClassifier(strategy="prior")
```

- Predice sempre la classe più frequente
- **ROC-AUC atteso**: 0.5 (casuale)
- **Parametri**: Fixed

**Risultati**: non sta imparando nulla

### 1. Logistic Regression con Elastic Net

**Nome**: `logistic_elastic_net`

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=20_000,
    random_state=42
)
```

**Spazio parametri**:
```python
{
    "selector__k": [5, 10, 20],           # N feature selezionate
    "model__C": [0.1, 1.0, 10.0],         # Inverso regularization
    "model__l1_ratio": [0.25, 0.5, 0.75]  # L1/(L1+L2) mix
}
```

**Caratteristiche**:
- Lineare, interpretabile
- Elasticnet = L1 + L2 (LASSO + Ridge)
- `class_weight="balanced"` gestisce squilibrio classi HC MS
- Feature scaling


### 2. Linear Discriminant Analysis (LDA)

**Nome**: `lda`

```python
LinearDiscriminantAnalysis(solver="lsqr")
```

**Spazio parametri**:
```python
{
    "selector__k": [5, 10, 20],
    "model__shrinkage": ["auto", 0.1, 0.5, 0.9]
}
```

**Caratteristiche**:
- Lineare, probabilistico
- Minimizza varianza entro-classe, massimizza tra-classe
- Shrinkage regolarizza in caso di singolarità
- Feature scaling **OBBLIGATORIO**

**Perché**: Modello lineare alternativo, ben-fondato statisticamente e proposto nei paper per EEG MS 

### 3. K-Nearest Neighbors (KNN)

**Nome**: `knn`

```python
KNeighborsClassifier()
```

**Spazio parametri**:
```python
{
    "selector__k": [5, 10, 20],
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]  # Manhattan (1) vs Euclidean (2)
}
```

**Caratteristiche**:
- Non-parametrico
- Lazy learner (nessun training esplicito)
- Sensibile al numero vicini e distanza
- Feature scaling
- Cattura la non-linearità

### 4. XGBoost - balances classes

```python
BalancedXGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1
)
```

**Spazio parametri**:
```python
{
    "selector__k": [10, 20, "all"],
    "model__n_estimators": [100, 300],
    "model__max_depth": [1, 2],
    "model__learning_rate": [0.03, 0.1],
    "model__min_child_weight": [1, 3]
}
```

**Caratteristiche**:
- Ensemble di alberi decisionali
- Gradient boosting sequenziale
- `BalancedXGBClassifier` = wrapping custom per scale_pos_weight automatico
- Non richiede scaling
- Cattura non-linearità complesse


### Full Pipeline

Ogni modello è wrappato in una pipeline scikit-learn:

```python
Pipeline([
    ("selector", SelectKBest(f_classif)),  # Selezione top-k feature
    ("scaler", StandardScaler() if scale_features else None),
    ("model", estimator)
])
```

- Riduce dimensionalità
- Riduce overfitting
- Complessità computazionale minore

---

## Metriche e performance

### Metriche calcolate

Per ogni fold (soggetti test):

| Metrica | Formula | Interpretazione |
|---------|---------|-----------------|
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$ | Discriminazione a qualsiasi threshold; 0.5=casuale, 1.0=perfetto |
| **Average Precision** | Area ROC-PR curve | Qualità ranking predizioni; rilevante con squilibrio |
| **Balanced Accuracy** | $\frac{\text{TPR} + \text{TNR}}{2}$ | Media sensibilità e specificità; insensibile squilibrio |
| **Sensitivity (TPR)** | $\frac{\text{TP}}{\text{TP}+\text{FN}}$ | Percentuale MS corretti (clinicamente critico) |
| **Specificity (TNR)** | $\frac{\text{TN}}{\text{TN}+\text{FP}}$ | Percentuale HC corretti |
| **F1-Score** | $2 \times \frac{\text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}}$ | Media armonica precisione-recall |

### Aggregazione metriche

**Per ogni modello** (su 20 fold: 5 repeat × 4 fold):

| Statistica | Uso |
|-----------|-----|
| **Media (mean)** | Performance centrale |
| **Mediana (median)** | Robustezza a outlier |
| **Std Dev (std)** | Variabilità tra fold |

### Risultati riassuntivi

Basato su `artifacts/metrics/model_comparison_summary.csv`:

```
Model                  ROC-AUC      Balanced Acc  Sensitivity  F1
                       (mean±std)   (mean±std)    (mean±std)   (mean±std)
────────────────────────────────────────────────────────────────────
dummy                  0.50±0.00    0.50±0.00     1.00±0.00    0.86±0.00
                       Baseline: sempre predice "MS" (1)

logistic_elastic_net   0.49±0.23    0.50±0.21     0.63±0.27    0.65±0.23
                       Marginalmente migliore del baseline

lda                    0.53±0.24    0.53±0.18     0.72±0.16    0.73±0.11
                       Leggermente migliore

knn                    0.49±0.23    0.51±0.16     0.82±0.13    0.78±0.09
                       Elevata sensibilità ma squilibrata

xgboost                0.48±0.22    0.49±0.17     0.65±0.22    0.67±0.16
                       Simile al baseline
```

### Analisi metriche per modello

#### **Dummy Classifier**

| Metrica | Media | Std | Mediana |
|---------|-------|-----|---------|
| ROC-AUC | 0.500 | 0.000 | 0.500 |
| Avg Precision | 0.750 | 0.000 | 0.750 |
| Balanced Accuracy | 0.500 | 0.000 | 0.500 |
| Sensitivity | 1.000 | 0.000 | 1.000 |
| Specificity | 0.000 | 0.000 | 0.000 |
| F1 | 0.857 | 0.000 | 0.857 |

**Interpretazione**:
- Predice sempre target=1 (MS)
- Sensitivity=1.0, Specificity=0.0 (ovvero vede tutti come MS)
- F1 artificialmente alto per squilibrio dataset non-balanced

---

#### **Logistic Elastic Net**:  migliore performance

| Metrica | Media | Std | Mediana |
|---------|-------|-----|---------|
| ROC-AUC | 0.492 | 0.232 | 0.500 |
| Avg Precision | 0.807 | 0.109 | 0.827 |
| Balanced Accuracy | 0.500 | 0.208 | 0.500 |
| Sensitivity | 0.625 | 0.270 | 0.667 |
| Specificity | 0.375 | 0.393 | 0.500 |
| F1 | 0.648 | 0.235 | 0.697 |

**Interpretazione**:
- ROC-AUC: circa 0.49 (peggio del baseline!)
- Avg Precision accettabile (0.81)
- Sensibilità 62.5% (manca 37.5% MS)
- Specificità 37.5% (diagnostica falsi positivi 62.5%)
- Riassunto: modello non discriminante

**Iperparametri ottimi (media)**: 
- `C=0.1-10.0` (variatissimo)
- `l1_ratio=0.25-0.75` (mix L1/L2)
- `k=5-20` feature

**Presenza di variabilità alta**: ciò indica instabilità del modello su fold diversi

---

#### **LDA**

| Metrica | Media | Std | Mediana |
|---------|-------|-----|---------|
| ROC-AUC | 0.533 | 0.244 | 0.583 |
| Avg Precision | 0.828 | 0.107 | 0.837 |
| Balanced Accuracy | 0.533 | 0.176 | 0.542 |
| Sensitivity | 0.717 | 0.163 | 0.667 |
| Specificity | 0.350 | 0.366 | 0.500 |
| F1 | 0.735 | 0.108 | 0.748 |

**Interpretazione**:
- ROC-AUC = 0.533 (appena sopra il baseline, ma high variance)
- Migliore del Logistic su ROC-AUC
- Sensibilità 71.7% (buona, ma con alta variabilità)
- Specificità 35% (bassa)
- F1 più stabile di Logistic

**Iperparametri**:
- Shrinkage da "auto" a 0.9
- k da 5 a 20

**Variabilità**: moderata su ROC-AUC (std=0.24), migliore di Logistic su F1

---

#### **K-Nearest Neighbors**

| Metrica | Media | Std | Mediana |
|---------|-------|-----|---------|
| ROC-AUC | 0.485 | 0.228 | 0.521 |
| Avg Precision | 0.788 | 0.102 | 0.782 |
| Balanced Accuracy | 0.508 | 0.162 | 0.500 |
| Sensitivity | 0.817 | 0.131 | 0.833 |
| Specificity | 0.200 | 0.299 | 0.000 |
| F1 | 0.781 | 0.086 | 0.769 |

**Interpretazione**:
- Peggiore su ROC-AUC
- Elevata sensibilità, ovvero cattura bene i MS
- Bassa specificità (20%), ovvero molti falsi positivi HC
- F1 alto (0.78) perché beneficia dallo squilibrio
- Non è molto affidabile clinicamente, ovvero troppi falsi positivi

---

#### **XGBoost**

| Metrica | Media | Std | Mediana |
|---------|-------|-----|---------|
| ROC-AUC | 0.477 | 0.220 | 0.542 |
| Avg Precision | 0.793 | 0.097 | 0.788 |
| Balanced Accuracy | 0.488 | 0.167 | 0.500 |
| Sensitivity | 0.650 | 0.222 | 0.667 |
| Specificity | 0.325 | 0.335 | 0.500 |
| F1 | 0.674 | 0.164 | 0.727 |

**Interpretazione**:
- ROC-AUC = 0.477 (peggiore tra tutti)
- Avg Precision accettabile (0.79)
- Sensibilità-Specificità bilanciate male
- Non apprende pattern discriminanti

**Possibili cause**:
- Alberi poco profondi (max_depth=1-2) limitano non-linearità
- Dataset troppo piccolo per gradient boosting
- Overfitting durante inner tuning

---

## Risultati dettagliati

### Ipotesi sui risultati non ottimali dei modelli usati

1. **Squilibrio classi**
   - Target=0 (HC): 8 campioni
   - Target=1 (MS): 24 campioni
   - Ratio: moderato, ma modelli tendono a predire maggioranza
   - `class_weight="balanced"` mitiga parzialmente

2. **Pochi soggetti**
   - 32 soggetti totali
   - **Troppo pochi** per training robusto di modelli complessi
   - LDA e Logistic (lineari) teoricamente più adatti

4. **Possibile Separabilità Bassa**
   - ROC-AUC si aggira circa su 0.5 (50%) e suggerisce un **overlap distributions**
   - HC e MS **potrebbero non** essere chiaramente separabili da EEG
   - O le feature non catturano le vere differenze biologiche

5. **Variabilità fold eccessiva**
   - std(ROC-AUC) si aggira circa sul 0.22-0.24 per modelli, ciò significa che gli stessi modelli su fold diversi variano molto, dato anche dal dataset piccolo e con non-alta stabilità
   

### Performance migliore ottenuta

**Singolo fold**:
- LDA, Fold 2, Repeat 2: ROC-AUC=0.917, Balanced Acc=0.75, Sensitivity=0.83
- Logistic, Fold 3, Repeat 3: ROC-AUC=0.833, Balanced Acc=0.92, Sensitivity=0.83
- KNN, Fold 4, Repeat 2: ROC-AUC=0.833, Balanced Acc=0.92, Sensitivity=0.83

**Non replicabile tra fold**, overfitting locale e non c'è generalizzazione

---

## Come setuppare il progetto

### 0. Setup iniziale

#### **Prerequisiti**
- Python 3.11+
- Git
- Dataset raw presente in `data/raw/`

#### **Clonazione e setup**

```bash
# Clona repository
git clone <repository-url>
cd eeg-ms-mlops

# Crea virtual environment
python -m venv .venv

# Attiva virtual environment
# Windows:
.venv\Scripts\activate
# MacOS/Linux:
source .venv/bin/activate

# Installa dipendenze
pip install --upgrade pip
pip install -e ".[dev]"
```

Verifica installazione:
```bash
python -c "import eeg_ms; print('Import OK')"
pytest tests/  # test unitari
```

### 1. Audit dataset

```bash
python -m eeg_ms.data.audit
```

**Output atteso**:
```
file_manifest.csv creato (una riga per file)
variable_inventory.csv creato (una riga per matrice)
validation_report.json creato
```

**Verifica**:
```bash
# Controlla numero file scoperti (bash linux)
wc -l data/interim/file_manifest.csv

# Controlla esito audit (bash linux)
cat data/interim/validation_report.json | head -50
```

### 2. Costruzione dataset canonico

```bash
python -m eeg_ms.data.build_dataset
```

**Output atteso**:
```
Righe canoniche: ...
Soggetti: ...
Output: data/processed/canonical_features.parquet
```

**Verifica**:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/canonical_features.parquet')
print(f'Shape: {df.shape}')
print(f'Colonne: {df.columns.tolist()}')
print(df.head())
"
```

**Controllo qualità**:
```bash
python -c "
import pandas as pd
canonical = pd.read_parquet('data/processed/canonical_features.parquet')

# Controlla soggetti
print('Soggetti unici:', canonical['subject_id'].nunique())

# Controlla distribuzioni condizioni
print(canonical['condition'].value_counts())

# Controlla gruppi
print(canonical['group'].value_counts())

# Controlla missingness
print('Missing values:', canonical.isnull().sum().sum())
"
```

### 3. Quality Control

```bash
python -m eeg_ms.quality.run_quality_control
```

**Output**:
- Tabelle in `reports/tables/`
- Grafici in `reports/figures/quality_control/`

**Controlli eseguiti**:
- Completezza dati per ROI
- Identificazione outlier
- Correlazioni tra feature
- Distribuzioni per banda/condizione

### 4. Feature Engineering

```bash
python -m eeg_ms.features.build_tabular
```

**Input**: `canonical_features.parquet`  
**Output**: `data/processed/subject_features_roi.parquet`

**Scopo**:
- Trasforma da long a wide format
- Aggregazione ROI (6 al posto di 27 canali)
- Creazione feature finali per ML

**Verifica**:
```bash
python -c "
import pandas as pd
features = pd.read_parquet('data/processed/subject_features_roi.parquet')
print(f'Shape: {features.shape}')
print(f'Soggetti: {len(features)}')
print(f'Feature: {features.shape[1]}')
print(features.head())
"
```

### 5. Generazione Split Cross-Validation

```bash
python -m eeg_ms.validation.generate_splits
```

**Output**: `data/processed/splits/outer_test_folds.csv`

**Contiene**: assegnamenti soggetti a fold (5 repeat × 4 fold = 20 scenari)

**Verifica**:
```bash
python -c "
import pandas as pd
splits = pd.read_csv('data/processed/splits/outer_test_folds.csv')
print(f'Righe (soggetti × fold): {len(splits)}')
print(f'Repeat unici: {splits[\"repeat\"].nunique()}')
print(f'Fold unici: {splits[\"fold\"].nunique()}')
print(splits.head())
"
```

### 6. Training e tuning (Nested Cross-Validation)

```bash
python -m eeg_ms.modeling.run_experiments
```

**Output**:
```
Avvio modello: dummy
Avvio modello: logistic_elastic_net
Avvio modello: lda
Avvio modello: knn
Avvio modello: xgboost

artifacts/metrics/nested_cv_fold_metrics.csv creato
artifacts/predictions/nested_cv_predictions.csv creato
```

**Processo per ogni modello**:
1. Loop 5 repeat
   - Loop 4 fold esterno
     - Test fold estratto
     - Training fold rimanente
     - **Inner CV**: GridSearchCV 4-fold su training
       - Esplora spazio parametri
       - Seleziona migliori parametri
     - Applica modello migliore su test fold
     - Registra metriche e predizioni

**Verifica**:
```bash
python -c "
import pandas as pd

# Metriche
metrics = pd.read_csv('artifacts/metrics/nested_cv_fold_metrics.csv')
print(f'Numero righe (fold × modelli): {len(metrics)}')
print(metrics.groupby('model')['roc_auc'].agg(['mean', 'std']))

# Predizioni
preds = pd.read_csv('artifacts/predictions/nested_cv_predictions.csv')
print(f'Numero predizioni: {len(preds)}')
print(preds.columns.tolist())
"
```

### 7. Analisi risultati

```bash
python -m eeg_ms.evaluation.analyze_results
```

**Output**:
- `reports/tables/model_evaluation/fold_metric_summary.csv` - metriche aggregate
- `reports/tables/model_evaluation/bootstrap_confidence_intervals.csv` - bootstrap CI
- `reports/tables/model_evaluation/subject_oof_predictions.csv` - predizioni per soggetto
- `reports/tables/model_evaluation/subject_error_analysis.csv` - soggetti più difficili
- `reports/figures/model_evaluation/` - grafici

**Verifica**:
```bash
python -c "
import pandas as pd

summary = pd.read_csv('reports/tables/model_evaluation/fold_metric_summary.csv')
print('Metriche aggregate per modello:')
print(summary)
"
```

---

## Interpretazione dei risultati

1. **Feature Selection non adeguata**
   - Le feature scelte (top-k da F-test) potrebbero non essere discriminanti
   - Prova con altre strategie: mutual_information, recursive elimination

2. **Aggregazione ROI troppo aggressiva**
   - Media 27->6 canali potrebbe perdere pattern locali
   - Prova con ROI diversi o aggregazione soft

---

## Futuri sviluppi 
1. **Deep Learning**
   - CNN 1D su segnali EEG grezzi
   - Input: (batch, n_channels, n_timepoints)
   - Prova PyTorch + TorchEEG

2. **Time-Series Specific Models**
   - LSTM/GRU su tracce temporali
   - Attention mechanisms
   - Transformer per pattern temporali

3. **Multi-modal Fusion**
   - Combina PSD + Entropy + altri biomarker
   - Combina EEG + MRI + clinical scores
   - Fusion architecture (early, late, intermediate)

4. **Probabilistic Models**
   - Gaussian Process per uncertainty
   - Bayesian Neural Networks
   - Variational Autoencoder per outlier detection

5. **Longitudinal Analysis**
   - Se disponibili follow-up: modella evoluzione MS
   - Mixed-effects models per intra-subject variability

---
