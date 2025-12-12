# PASSI ESATTI PER RICCARDO

## Setup Iniziale (1 volta sola)

### Su Google Colab:

1. **Carica i notebook su Colab:**
   - `DINOv2_Correspondence.ipynb`
   - `DINOv3_Correspondence.ipynb`
   - `SAM_Correspondence.ipynb`

2. **Verifica che i dati siano su Drive:**
   - SPair-71k in: `/content/drive/MyDrive/AMLProject/data/SPair-71k/`
   - Checkpoint in: `/content/drive/MyDrive/AMLProject/checkpoints/`

---

## FASE 1: Training-Free Baseline (PRIORITÀ MASSIMA)

### Obiettivo
Valutare DINOv2, DINOv3 e SAM senza finetuning, usando argmax standard.

### Passi per DINOv2

1. **Apri `DINOv2_Correspondence.ipynb` su Colab**

2. **Trova la cella "Configuration Flags" (seconda o terza cella)**

3. **Imposta così:**
   ```python
   ENABLE_FINETUNING = False
   USE_SOFT_ARGMAX = False
   ```

4. **Clicca Runtime → Run All**

5. **Aspetta che finisca** (circa 30-60 minuti)

6. **Cerca l'output PCK** (verso la fine del notebook):
   ```
   PCK@0.05: 0.XXXX
   PCK@0.10: 0.XXXX
   PCK@0.15: 0.XXXX
   PCK@0.20: 0.XXXX
   ```

7. **Copia questi numeri in una tabella Excel/Google Sheets:**
   | Backbone | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
   |----------|----------|----------|----------|----------|
   | DINOv2   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

### Passi per DINOv3

8. **Apri `DINOv3_Correspondence.ipynb`**

9. **Imposta gli stessi flag:**
   ```python
   ENABLE_FINETUNING = False
   USE_SOFT_ARGMAX = False
   ```

10. **Run All**

11. **Copia i risultati PCK nella tabella:**
    | Backbone | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
    |----------|----------|----------|----------|----------|
    | DINOv2   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
    | DINOv3   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

### Passi per SAM

12. **Apri `SAM_Correspondence.ipynb`**

13. **Imposta gli stessi flag:**
    ```python
    ENABLE_FINETUNING = False
    USE_SOFT_ARGMAX = False
    ```

14. **Run All**

15. **Copia i risultati PCK nella tabella:**
    | Backbone | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
    |----------|----------|----------|----------|----------|
    | DINOv2   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
    | DINOv3   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
    | SAM      | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

### Domanda da rispondere
**Quale backbone ha PCK@0.10 più alto?** → Questo è il migliore per la baseline.

### Output da consegnare
- Tabella Excel/Sheets con i numeri sopra
- Screenshot degli output PCK di ogni notebook

---

## FASE 2: Window Soft-Argmax

### Obiettivo
Vedere se soft-argmax (sub-pixel) migliora rispetto ad argmax.

### Passi per DINOv2

1. **Riapri `DINOv2_Correspondence.ipynb`**

2. **Cambia i flag:**
   ```python
   ENABLE_FINETUNING = False
   USE_SOFT_ARGMAX = True    # <-- CAMBIATO
   ```

3. **Run All**

4. **Copia i risultati in una NUOVA tabella:**
   | Backbone | Method      | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
   |----------|-------------|----------|----------|----------|----------|
   | DINOv2   | Argmax      | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
   | DINOv2   | Soft-Argmax | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

### Passi per DINOv3

5. **Riapri `DINOv3_Correspondence.ipynb`**

6. **Cambia i flag:**
   ```python
   ENABLE_FINETUNING = False
   USE_SOFT_ARGMAX = True
   ```

7. **Run All**

8. **Aggiungi alla tabella:**
   | Backbone | Method      | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
   |----------|-------------|----------|----------|----------|----------|
   | DINOv2   | Argmax      | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
   | DINOv2   | Soft-Argmax | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
   | DINOv3   | Argmax      | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
   | DINOv3   | Soft-Argmax | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

### Passi per SAM

9. **Riapri `SAM_Correspondence.ipynb`**

10. **Cambia i flag:**
    ```python
    ENABLE_FINETUNING = False
    USE_SOFT_ARGMAX = True
    ```

11. **Run All**

12. **Completa la tabella**

### Domande da rispondere
1. **Soft-argmax migliora PCK@0.05?** (dovrebbe migliorare di +2-5%)
2. **Quale configurazione (window, tau) funziona meglio?**

### Output da consegnare
- Tabella "Argmax vs Soft-Argmax" completa
- Screenshot degli output

---

## FASE 3: Light Finetuning (AVANZATO)

### ✅ DATALOADER SPair-71k IMPLEMENTATO
Il dataloader SPair-71k con annotazioni keypoint è **già implementato** in tutti e 3 i notebook.

### Cosa fa il dataloader

**Il dataloader fornisce automaticamente:**
```python
for batch in train_loader:
    src_img = batch['src_img']      # [B, 3, H, W] immagine sorgente
    tgt_img = batch['tgt_img']      # [B, 3, H, W] immagine target
    src_kps = batch['src_kps']      # [B, N, 2] keypoint sorgente (x, y)
    tgt_kps = batch['tgt_kps']      # [B, N, 2] keypoint target (x, y)
    src_bbox_wh = batch['src_bbox_wh']  # [B, 2] bounding box (w, h)
    tgt_bbox_wh = batch['tgt_bbox_wh']  # [B, 2] bounding box (w, h)
```

### Requisiti

**Prima di iniziare, verifica di avere:**
- SPair-71k dataset completo in `data/SPair-71k/`
- Struttura corretta: `Layout/`, `JPEGImages/`, `PairAnnotation/`

#### Passi per DINOv2 (k=1)

1. **Apri `DINOv2_Correspondence.ipynb`**

2. **Imposta:**
   ```python
   ENABLE_FINETUNING = True    # <-- CAMBIATO
   USE_SOFT_ARGMAX = False
   FINETUNE_K_LAYERS = 1       # <-- PROVA k=1 prima
   FINETUNE_EPOCHS = 3
   ```

3. **Run All**

4. **Il notebook farà automaticamente:**
   - Carica il dataloader SPair-71k (già implementato)
   - Unfreeze gli ultimi k blocchi del modello
   - Esegue il training loop per FINETUNE_EPOCHS epoche
   - Salva il modello migliore in `checkpoints/`
   - Continua con la valutazione sul test set

5. **Aspetta il training** (può richiedere 1-2 ore)

8. **Copia i risultati PCK:**
   | Backbone | Config           | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
   |----------|------------------|----------|----------|----------|----------|
   | DINOv2   | Baseline         | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
   | DINOv2   | Finetune (k=1)   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

#### Ripeti per k=2 e k=4

9. **Cambia `FINETUNE_K_LAYERS = 2`** e ripeti

10. **Cambia `FINETUNE_K_LAYERS = 4`** e ripeti

11. **Completa la tabella:**
    | Backbone | Config           | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
    |----------|------------------|----------|----------|----------|----------|
    | DINOv2   | Baseline         | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
    | DINOv2   | Finetune (k=1)   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
    | DINOv2   | Finetune (k=2)   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |
    | DINOv2   | Finetune (k=4)   | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   |

### Domanda da rispondere
**Quale k dà i risultati migliori su validation?**

### Output da consegnare
- Tabella "PCK vs k" per ogni backbone
- Grafico (opzionale): PCK@0.10 vs k
- Screenshot degli output

---

## FASE 4: Ablation Study Completa (FINALE)

### Obiettivo
Confrontare tutte le combinazioni per trovare la configurazione ottimale.

### Configurazioni da testare

| Run | ENABLE_FINETUNING | USE_SOFT_ARGMAX | Descrizione         |
|-----|-------------------|-----------------|---------------------|
| 1   | False             | False           | Baseline            |
| 2   | False             | True            | Solo Soft-Argmax    |
| 3   | True              | False           | Solo Finetuning     |
| 4   | True              | True            | Full Pipeline       |

### Passi

1. **Per ogni backbone (DINOv2, DINOv3, SAM):**
   - Esegui Run 1, 2, 3, 4
   - Usa il k migliore dalla Fase 3 per Run 3 e 4

2. **Crea tabella finale:**
   | Backbone | Config              | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 | Time (s) |
   |----------|---------------------|----------|----------|----------|----------|----------|
   | DINOv2   | Baseline            | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XXX      |
   | DINOv2   | +Soft               | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XXX      |
   | DINOv2   | +Finetune           | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XXX      |
   | DINOv2   | +Both               | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XXX      |
   | DINOv3   | Baseline            | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XXX      |
   | DINOv3   | +Soft               | 0.XXXX   | 0.XXXX   | 0.XXXX   | 0.XXXX   | XXX      |
   | ...      | ...                 | ...      | ...      | ...      | ...      | ...      |

### Domanda finale da rispondere
**Quale combinazione (Backbone + Config) dà PCK@0.10 più alto?**

Esempio risposta:
> "DINOv3 + Finetuning (k=2) + Soft-Argmax (window=7, tau=0.05) raggiunge PCK@0.10 = 0.672, il migliore tra tutte le configurazioni testate."

### Output da consegnare
- Tabella ablation completa (12 righe = 3 backbone × 4 config)
- Grafico comparativo PCK@0.10 per tutte le configurazioni
- Report finale con:
  - Best model identificato
  - Parametri ottimali (backbone, k, window, tau)
  - Risultati finali su test
  - Tempi di training/inference

---

## Checklist Riassuntiva

### Fase 1 (Baseline) ✅
- [ ] DINOv2 eseguito con flag False/False
- [ ] DINOv3 eseguito con flag False/False
- [ ] SAM eseguito con flag False/False
- [ ] Tabella PCK creata
- [ ] Backbone migliore identificato

### Fase 2 (Soft-Argmax) ✅
- [ ] DINOv2 eseguito con flag False/True
- [ ] DINOv3 eseguito con flag False/True
- [ ] SAM eseguito con flag False/True
- [ ] Tabella "Argmax vs Soft" creata
- [ ] Miglioramento quantificato

### Fase 3 (Finetuning) ✅
- [✓] Dataloader SPair-71k implementato in tutti i notebook
- [ ] k=1 testato per ogni backbone
- [ ] k=2 testato per ogni backbone
- [ ] k=4 testato per ogni backbone
- [ ] Tabella "PCK vs k" creata
- [ ] k ottimale identificato

### Fase 4 (Ablation) 🎯
- [ ] Tutte le 12 configurazioni eseguite
- [ ] Tabella ablation completa
- [ ] Grafico comparativo creato
- [ ] Best model identificato
- [ ] Report finale scritto

---

## Tempistiche Stimate

| Fase  | Descrizione           | Tempo Stimato |
|-------|-----------------------|---------------|
| 1     | Baseline              | 2-3 ore       |
| 2     | Soft-Argmax           | 2-3 ore       |
| 3     | Finetuning            | 1-2 giorni    |
| 4     | Ablation Study        | 1 giorno      |
| **TOT** | **Progetto Completo** | **3-5 giorni** |

---

## Troubleshooting Veloce

### Errore: "SPair-71k not found"
```python
# Verifica il path
!ls /content/drive/MyDrive/AMLProject/data/SPair-71k/
```
Se vuoto: scarica SPair-71k e caricalo su Drive.

### Errore: "Out of memory"
```python
# Riduci batch size
FINETUNE_BATCH_SIZE = 1
# Usa subset
FINETUNE_TRAIN_SUBSET = 500
```

### Errore: "Model weights not found"
```python
# Per DINOv2: usa torch.hub (già nel notebook)
# Per DINOv3: verifica che dinov3_vitb16.pth sia in checkpoints/
# Per SAM: verifica che sam_vit_b_01ec64.pth sia in checkpoints/sam/
```

### Il notebook si blocca
- Interrompi (Runtime → Interrupt)
- Riavvia (Runtime → Restart)
- Esegui cella per cella invece di Run All

---

## Domande Frequenti

**Q: Posso saltare una fase?**
A: Sì! Inizia con Fase 1 e 2 (più importanti). Fase 3 richiede più tempo.

**Q: Devo eseguire tutti i backbone?**
A: Idealmente sì, ma puoi iniziare con solo DINOv2 per capire il flusso.

**Q: Come scelgo window e tau per soft-argmax?**
A: Usa i valori di default (window=7, tau=0.05). Se vuoi sperimentare:
- window ∈ {5, 7, 9}
- tau ∈ {0.03, 0.05, 0.1}

**Q: Quanto tempo richiede il finetuning?**
A: Con GPU T4 su Colab: ~1-2 ore per backbone (3 epoche).

**Q: Cosa consegno alla fine?**
A:
1. Tabella Fase 1 (Baseline)
2. Tabella Fase 2 (Soft-Argmax)
3. Tabella Fase 3 (Finetuning vs k) - se completata
4. Tabella Fase 4 (Ablation completa) - se completata
5. Report con best model e risultati finali

---

## Contatti

Se hai problemi:
1. Leggi `INSTRUCTIONS_FOR_RICCARDO.md` per dettagli tecnici
2. Controlla `SUMMARY_OF_CHANGES.md` per capire cosa è stato modificato
3. Verifica che tutti i path siano corretti
4. Prova a eseguire le celle una per una per trovare l'errore

**Buon lavoro! 🚀**
