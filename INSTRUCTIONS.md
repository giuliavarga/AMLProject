# Istruzioni per Riccardo - Progetto AML

## Cosa è stato aggiunto

Ho aggiunto a tutti e 3 i notebook (DINOv2, DINOv3, SAM):
1. **Light finetuning degli ultimi layer** - attivabile con flag `ENABLE_FINETUNING`
2. **Window soft-argmax** - attivabile con flag `USE_SOFT_ARGMAX`
3. **Notebook orchestratore** - automatizza tutti i notebook e consolida i risultati

## Struttura del Progetto

```
├── DINOv2_Correspondence.ipynb    # ✓ Completo con dataloader + finetuning + soft-argmax
├── DINOv3_Correspondence.ipynb    # ✓ Completo con dataloader + finetuning + soft-argmax
├── SAM_Correspondence.ipynb       # ✓ Completo con dataloader + finetuning + soft-argmax
├── AML_Project_Orchestrator.ipynb # ✓ NUOVO - Automatizza tutti e 3
├── data/
│   └── SPair-71k/                 # Dataset con annotazioni keypoint
│       ├── Layout/trn/            # Training annotations (JSON)
│       ├── Layout/val/            # Validation annotations (JSON)
│       ├── Layout/test/           # Test annotations (JSON)
│       └── JPEGImages/            # Immagini per categoria
└── outputs/                       # Risultati consolidati
```

## SPair-71k Dataloader

Il dataloader SPair-71k è **completamente implementato** in tutti e 3 i notebook:

**Caratteristiche:**
- ✅ Caricamento automatico da JSON annotations
- ✅ Resize e padding per ogni backbone (224 per DINO, 1024 per SAM)
- ✅ Scaling automatico delle coordinate keypoint
- ✅ Normalizzazione ImageNet
- ✅ Gestione bounding box per normalizzazione PCK
- ✅ Train/Val splits separati
- ✅ Supporto per subset (per debug rapidi)

**Utilizzo:**
Il dataloader viene creato automaticamente quando `ENABLE_FINETUNING=True`.

## Come Usare i Notebook Individuali

### Opzione 1: Baseline Training-Free (argmax)

Apri uno dei notebook (es. `DINOv2_Correspondence.ipynb`) e:

1. **Imposta i flag a `False`** nella cella "Configuration Flags":
   ```python
   ENABLE_FINETUNING = False
   USE_SOFT_ARGMAX = False
   ```

2. **Esegui tutte le celle** (Run All)

3. **Risultati**: Otterrai la baseline con argmax standard su SPair-71k

### Opzione 2: Solo Window Soft-Argmax

1. **Imposta i flag**:
   ```python
   ENABLE_FINETUNING = False
   USE_SOFT_ARGMAX = True
   ```

2. **Esegui tutte le celle**

3. **Risultati**: Baseline con sub-pixel refinement (soft-argmax)

### Opzione 3: Solo Light Finetuning

1. **Imposta i flag**:
   ```python
   ENABLE_FINETUNING = True
   USE_SOFT_ARGMAX = False
   ```

2. **Configura hyperparameters** (nella stessa cella):
   ```python
   FINETUNE_K_LAYERS = 2    # Prova {1, 2, 4}
   FINETUNE_LR = 1e-5
   FINETUNE_EPOCHS = 3
   FINETUNE_BATCH_SIZE = 1
   ```

3. **✅ Il dataloader SPair-71k è già implementato!**
   - Il notebook caricherà automaticamente il dataset da `data/SPair-71k/`
   - Assicurati che SPair-71k sia scaricato e nella posizione corretta
   - Layout annotations e immagini devono essere presenti

4. **Esegui tutte le celle**

### Opzione 4: Full Pipeline (Finetuning + Soft-Argmax)

1. **Imposta i flag**:
   ```python
   ENABLE_FINETUNING = True
   USE_SOFT_ARGMAX = True
   ```

2. **Esegui tutte le celle**

3. **Risultati**: Massima performance attesa

---

## Come Usare il Notebook Orchestratore

Il notebook `AML_Project_Orchestrator.ipynb` automatizza l'esecuzione di tutti e 3 i backbone.

### Step 1: Configura

Apri `AML_Project_Orchestrator.ipynb` e imposta:

```python
ENABLE_FINETUNING = False  # Cambia a True per finetuning
USE_SOFT_ARGMAX = False    # Cambia a True per soft-argmax
```

### Step 2: Esegui

Esegui tutte le celle. L'orchestratore:
1. Eseguirà automaticamente DINOv2, DINOv3, SAM con la stessa configurazione
2. Estrarrà i risultati PCK da ogni notebook
3. Creerà tabelle comparative
4. Identificherà il modello migliore
5. Genererà grafici di confronto

### Step 3: Risultati

Troverai in `outputs/`:
- `consolidated_results.csv` - Tabella comparativa
- `consolidated_results.json` - Risultati dettagliati
- `pck_comparison.png` - Grafico PCK
- `DINOv2_executed_*.ipynb` - Notebook eseguiti per ogni backbone

---

## Roadmap Consigliata per Riccardo

### Fase 1: Baseline Training-Free ✅ PRIORITÀ

**Cosa fare:**
1. Apri `DINOv2_Correspondence.ipynb`
2. Imposta `ENABLE_FINETUNING = False`, `USE_SOFT_ARGMAX = False`
3. Esegui tutto
4. Ripeti per DINOv3 e SAM
5. Confronta i risultati PCK@0.10

**Output atteso:**
- Tabella con PCK@{0.05, 0.10, 0.15, 0.20} per ogni backbone
- Identificazione del backbone migliore per la baseline

**Tempo stimato:** 1-2 ore (dipende dai dati)

---

### Fase 2: Window Soft-Argmax ✅ MEDIO

**Cosa fare:**
1. Apri ogni notebook
2. Imposta `ENABLE_FINETUNING = False`, `USE_SOFT_ARGMAX = True`
3. Prova diverse configurazioni:
   - `SOFT_WINDOW = 5, 7, 9`
   - `SOFT_TAU = 0.03, 0.05, 0.1`
4. Confronta con baseline argmax

**Output atteso:**
- Tabella "Argmax vs Soft-Argmax" per ogni backbone
- Miglioramento atteso su PCK@0.05 (sub-pixel accuracy)

**Tempo stimato:** 2-3 ore

---

### Fase 3: Light Finetuning ✅ AVANZATO

**✅ DATALOADER SPair-71k GIÀ IMPLEMENTATO**

**Cosa fare:**
1. Verifica di avere SPair-71k scaricato in `data/SPair-71k/`
   - Il dataloader è già implementato in tutti i notebook
   - Carica automaticamente immagini e annotazioni keypoint
   - Gestisce resize, normalizzazione e scaling delle coordinate

2. Prova diverse configurazioni:
   - `FINETUNE_K_LAYERS = 1, 2, 4`
   - Valuta su `val` split dopo ogni epoca
   - Scegli il k migliore

3. Valuta su `test` con il modello migliore

**Output atteso:**
- Grafico "PCK vs k" su validation
- Risultati finali su test per k ottimale
- Miglioramento atteso su tutte le soglie PCK

**Tempo stimato:** 1-2 giorni (include implementazione dataloader + training)

---

### Fase 4: Ablation Study Completa 🎯 FINALE

**Cosa fare:**
Usa l'orchestratore per eseguire tutte le combinazioni:

```python
# Run 1: Baseline
ENABLE_FINETUNING = False, USE_SOFT_ARGMAX = False

# Run 2: Only Soft-argmax
ENABLE_FINETUNING = False, USE_SOFT_ARGMAX = True

# Run 3: Only Finetuning
ENABLE_FINETUNING = True, USE_SOFT_ARGMAX = False

# Run 4: Full Pipeline
ENABLE_FINETUNING = True, USE_SOFT_ARGMAX = True
```

**Output atteso:**
- Tabella 4×3 (4 configurazioni × 3 backbone) con PCK
- Identificazione della configurazione migliore
- Grafico comparativo finale

---

## Troubleshooting

### Problema: "SPair-71k not found" o "Dataset empty"
**Soluzione:** 
- Verifica che SPair-71k sia scaricato in `data/SPair-71k/`
- Controlla la struttura: deve contenere `Layout/`, `JPEGImages/`, `PairAnnotation/`
- Il dataloader è già implementato e cerca automaticamente i file JSON in `Layout/trn/` e `Layout/val/`
- Su Colab: verifica che il path sia `/content/drive/MyDrive/AMLProject/data/SPair-71k/`

### Problema: "Model weights not found"
**Soluzione:** 
- Verifica che i checkpoint siano in `checkpoints/` su Colab/Drive
- DINOv2: usa `torch.hub`, non serve checkpoint locale
- DINOv3: serve `dinov3_vitb16.pth` in `checkpoints/dinov3/`
- SAM: serve `sam_vit_b_01ec64.pth` in `checkpoints/sam/`

### Problema: "Out of memory"
**Soluzione:** 
- Riduci `FINETUNE_BATCH_SIZE` a 1
- Usa subset per test rapidi: `FINETUNE_TRAIN_SUBSET = 1000`
- Per SAM, riduci risoluzione input se necessario

### Problema: "Notebook execution failed"
**Soluzione:** 
- Esegui i notebook manualmente invece dell'orchestratore
- Controlla gli output di ogni cella per capire dove fallisce
- Verifica che tutti i path siano corretti (Drive mount, DATA_ROOT, etc.)

---

## Deliverable Finali

Al termine, dovresti avere:

1. **Tabella Baseline** (3 backbone × PCK@{0.05,0.10,0.15,0.20})
2. **Tabella Soft-Argmax** (confronto argmax vs soft per ogni backbone)
3. **Grafici Finetuning** (PCK vs k per ogni backbone)
4. **Tabella Ablation** (4 configurazioni × 3 backbone)
5. **Best Model Report** con:
   - Backbone scelto
   - Configurazione ottimale (k, window, tau)
   - PCK finali su test
   - Tempi di training/inference

---

## FAQ

**Q: Devo eseguire tutto su Colab?**
A: Sì, se hai già i dati/pesi su Drive. I notebook rilevano automaticamente Colab e montano Drive.

**Q: Quanto tempo ci vuole?**
A: 
- Baseline: ~30 min per backbone (totale 1.5h)
- Soft-argmax: ~1h per backbone (totale 3h)
- Finetuning: ~2-4h per backbone (totale 6-12h)

**Q: Posso eseguire solo DINOv2 per ora?**
A: Sì! Inizia con DINOv2, completa tutte le fasi, poi passa a DINOv3 e SAM.

**Q: Cosa faccio se non ho SPair-71k completo?**
A: Usa un subset per test rapidi. Cambia `FINETUNE_TRAIN_SUBSET = 500` per debug.

**Q: Come scelgo il migliore tra DINOv2/DINOv3/SAM?**
A: Guarda PCK@0.10 su test (metrica principale). In caso di parità, preferisci:
1. PCK@0.05 più alto (accuratezza sub-pixel)
2. Tempo di inferenza minore

---

## Contatti

Se hai domande o problemi:
1. Controlla prima la sezione Troubleshooting sopra
2. Verifica che i path e i checkpoint siano corretti
3. Prova a eseguire manualmente invece dell'orchestratore
4. Controlla gli output delle celle per capire l'errore esatto

Buon lavoro! 🚀
