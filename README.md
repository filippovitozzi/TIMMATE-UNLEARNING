## Struttura

- `BASE/base_utils.py`: caricamento dati, split retain/forget/validation, metriche, salvataggio submission.
- `BASE/SSD.py`: solo metodo Selective Synaptic Dampening.
- `utils/model.py`: definizione `DynamicMLP` richiesta dal `model_artifact`.
- `data/`: dati e modello originale necessari per rigenerare la submission.
- `submissions/TIMmate_SSD_V1/`: cartella locale da controllare prima dell'upload.
- `TIMmate_SSD_V1.zip`: zip pronto per upload manuale.

Metriche locali dell'ultima run:

precision_at_10: 0.0389249995
mia_auc: 0.4997952119
mia_resistance: 0.9995904238
score_no_time: 0.4673319405
execution_time: 0.2544761 s
ssd_changed_fraction: 0.0466801335
ssd_min_scale: 0.6999999881