Condizione (OE vs CE)
Come riportato dagli studi, si può provare a valutare la reattività cerebrale, nelle 2 condizioni, il blocco del ritmo Alpha a occhi aperti nei soggetti sani hc rispetto ai patologici ms.

Gruppo (HC vs MS)
Identificare differenze spaziali e spettrali.
Tracciare mappe topografiche medie per evidenziare aree di ipoconnettività o rallentamento focale nei pazienti MS.

Risoluzione spaziale (channels e ROI)
Passare dai dettagli puntuali di singolo canale (27) a una visione macroscopica regionale (6) per ridurre la dimensionalità e individuare pattern macro-area.

---

# Architettura pipeline

## Ingestion
individua e carica i dati originali senza modificarli

## Parsing
Trasforma informazioni implicite nel percorso in metadati espliciti:
> PSD/hc/PSDrelative_ID_01_CTR_CE.mat

family       = psd
group        = hc
subject_id   = hc_01
visit        = CTR
condition    = CE

Il gruppo deve provenire dalla cartella hc/ms, non soltanto dal nome del file. Il token CTR/T0 viene usato come controllo incrociato:
- hc deve corrispondere a CTR;
- ms deve corrispondere a T0.
È importante costruire subject_id = hc_01 oppure ms_01, perché ID_01 esiste in entrambi i gruppi e non identifica globalmente un soggetto

## Validation
La validazione controlla almeno quattro livelli:
- struttura delle directory;
- convenzione dei nomi;
- presenza delle variabili MATLAB;
- forma e contenuto numerico delle matrici.
La pipeline dovrebbe fallire presto su un errore strutturale. Addestrare un modello su dati caricati parzialmente è molto più pericoloso di un errore esplicito.


python -m venv .venv
activate .venv
pytest
python -m eeg_ms.data.audit

Cosa dovremmo ottenere
data/interim/
├── file_manifest.csv
├── variable_inventory.csv
└── validation_report.json

file_manifest.csv avrà una riga per file:

relative_path,family,group,subject_id,visit,condition,...
PSD/hc/PSDrelative_ID_01_CTR_CE.mat,psd,hc,hc_01,CTR,CE,...

variable_inventory.csv avrà una riga per matrice MATLAB:

subject_id,family,condition,variable,shape,dtype,n_nan,...
hc_01,psd,CE,PSD_alpha,27x18,float32,0,...

Il prossimo passaggio naturale, dopo che questo audit risulta verde sull’intero dataset, sarà la costruzione del dataset canonico in formato long, mantenendo ancora separate:

identità del soggetto;
gruppo HC/MS;
condizione CE/OE;
famiglia PSD/entropy;
livello canale/ROI;
finestra 1–18;
valore della feature.

---

# Dataset canonico
1. Formato scelto: long format
Ogni riga rappresenta un singolo valore:
soggetto × condizione × feature × posizione × finestra

Il formato long è adatto per:

controlli di qualità;
grafici;
selezione per banda, ROI o condizione;
creazione successiva di dataset wide;
mantenimento della provenienza del dato.

Non è ancora il formato finale per scikit-learn. La trasformazione long → wide verrà fatta successivamente dentro una fase esplicita di feature engineering.

Esecuzione:

python -m eeg_ms.data.build_dataset

Dimensione attesa

Per ogni soggetto e condizione:

PSD=5×2×(27+6)×18=5940
Entropy=(27+6)×18=594
Totale=6534

Con 32 soggetti e 2 condizioni:
32×2×6534=418176
Pertanto, se tutti i file sono presenti:

canonical_features.parquet: 418.176 righe
subjects.csv:                32 righe

Questo conteggio è un controllo molto utile: una differenza indica file, variabili o righe mancanti.



python -m eeg_ms.quality.run_quality_control


I notebook leggono `data/processed/canonical_features.parquet` e
`data/raw/canali.mat`. Le mappe di gruppo aggregano prima le finestre entro
soggetto e solo dopo i soggetti, evitando di attribuire alle 18 finestre il
ruolo di osservazioni indipendenti.

pip install -e ".[dev]"
jupyter lab