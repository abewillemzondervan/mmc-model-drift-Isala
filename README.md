# mmc-model-drift-Isala

#  MedImaging: Multi-Modal Model Drift Monitoring Pipeline

Dit project bevat een geavanceerde MLOps-pijplijn voor het detecteren, monitoren en diagnosticeren van **model drift** en **data drift** binnen medische beeldvormingssystemen (specifiek gericht op Chest X-Ray pneumonie-detectie). 

De architectuur bouwt voort op het fundamentele framework van Microsoft (`MedImaging-ModelDriftMonitoring`), maar introduceert een modulaire en geautomatiseerde pipeline die visuele, klinische en demografische verschuivingen correleert via **Multi-Modal Concordance (MMC)**.

## Kernfunctionaliteiten

* **VAE Baseline Architectuur**: Training van een Variational Autoencoder (128x128) om de gezonde visuele latente ruimte (baseline ≤ 2015) vast te leggen.
* **Klinische Feature Extractie**: Integratie van een pre-trained DenseNet121 model via `torchxrayvision` om klinische predictiewaarden en pneumonie-waarschijnlijkheden te genereren.
* **Geautomatiseerde Drift Statistiek**: Continue monitoring via rolling windows (3 maanden) waarbij nieuwe productiedata statistisch wordt getoetst via Kolmogorov-Smirnov (continue variabelen) en Chi-Kwadraat (categorische demografie).
* **Multi-Modal Concordance (MMC)**: Aggregatie van individuele Z-scores tot een overkoepelende stabiliteitsindex (waarbij een stabiel model rond de 0 schommelt en negatieve waarden duiden op acute drift).
* **Stress-Testing & Simulatie**: Ingebouwde experimentele modules om fysieke beeldveranderingen (ruis, helderheidstweakings) en demografische verschuivingen te forceren om de fijngevoeligheid van de monitoringpijplijn te valideren.

## Dashboards & Visualisaties

Na het doorlopen van de pijplijn genereert het systeem een interactief Plotly-dashboard (`dashboard_final.html`) met vier kritieke diagnostische lagen:
1. **AUROC vs. MMCw vs. MMC0**: Directe correlatie tussen echte modelprestaties (Ground Truth) en de gewogen/ongewogen drift-signalen.
2. **VAE Soft Predictions**: Trendlijnen van de visuele reconstructie-afwijkingen.
3. **Classifier Predictions**: Verschuivingen in de output-distributie van het deep learning model.
4. **Metadata**: Demografische en technische dataverschuivingen (zoals leeftijd of patiëntengeslacht).

##  Installatie & Configuratie

### 1. Vereisten & Package-omgeving
Installeer de exacte library-afhankelijkheden om versiediscrepanties (zoals Autocast of PyTorch Lightning-fouten) te voorkomen:

```bash
pip install torch torchvision pandas scikit-image torchxrayvision plotly scipy tqdm mlflow pytorch-lightning==1.9.5 azureml-core
