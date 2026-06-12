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

##  Data & Bronvermelding

Dit project werkt met medische gegevens die op de volgende wijze in de pijplijn worden geïntegreerd:

### 1. DICOM Metadata
De metadata (zoals `StudyDate_DICOM`, `PatientBirth`, `PatientSex_DICOM`) is afkomstig uit de DICOM-headers van de röntgenfoto's. Deze data wordt geëxtraheerd en opgeslagen in een `pneumo_dataset_ITI_rev.tsv` bestand.
- **Waarom?** Deze velden zijn essentieel voor het monitoren van *Covariate Shift*: als de populatie in het ziekenhuis (bijv. de gemiddelde leeftijd) verandert, kan de nauwkeurigheid van de AI variëren.

### 2. Medische Beelden
De beelden worden ingeladen vanuit een lokale directory (`./test_beelden/`). 
- **Verwerking**: De beelden worden in de pipeline genormaliseerd naar een consistent formaat (128x128 voor de VAE, 224x224 voor de DenseNet classifier).
- **Privacy**: Om de privacy te waarborgen, worden de originele DICOM-bestanden direct na verwerking uit het geheugen gewist en worden alleen de geanonimiseerde 'latente vectoren' (vanuit de VAE) opgeslagen in de `pipeline_data` map.

### 3. Visualisatie van de Pijplijn
Hieronder zie je hoe de data stroomt van de ruwe bron naar het uiteindelijke dashboard:

##  Databron & Erkenning
Dit project maakt gebruik van de **PadChest** dataset voor de verwerking van röntgenfoto's. 
- **Bron**: [BIMCV-PadChest COVID-19 Repository](https://github.com/BIMCV-CSUSP/BIMCV-COVID-19/tree/master/padchest-covid)
- **Gebruik**: De data wordt in deze pipeline ingezet als testset voor het simuleren van drift-scenario's en het valideren van de modelrobuustheid. Wij danken de onderzoekers van BIMCV en CSUSP voor het beschikbaar stellen van deze waardevolle medische dataset.