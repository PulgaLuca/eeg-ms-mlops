# Commento ai risultati e direzioni di miglioramento

I risultati qui discussi si riferiscono all'ultimo run completato, `run_20260924T124737_646339Z` (nested cross-validation con 5 ripetizioni esterne e 4 fold per ripetizione). I valori principali sono le medie delle metriche sui 20 fold esterni; dove indicato, sono riportati separatamente anche i risultati pooled ottenuti aggregando le predizioni out-of-fold a livello di soggetto.

## Sintesi

I risultati riportati provengono da 32 soggetti e confrontano gli stessi algoritmi su due rappresentazioni spaziali EEG:

- `roi__`: feature aggregate a livello di Region Of Interest;
- `channel__`: feature mantenute a livello dei singoli canali.

**Con i dati disponibili, nessun modello mostra ancora una capacità diagnostica sufficientemente solida.** Le prestazioni variano molto tra i fold e restano vicine a una classificazione non informativa. Nel riepilogo medio dei 20 fold:

- `channel__svm` ha la ROC-AUC media più alta (`0.588`) e la average precision media più alta (`0.858`), ma la specificità media è appena `0.075`;
- `channel__xgboost` ha la balanced accuracy media migliore (`0.538`), con sensibilità `0.675` e specificità `0.400`;
- `roi__svm` raggiunge la sensibilità media più alta (`0.950`), ma specificità `0.025`.


## Lettura delle metriche

### ROC-AUC

La metrica ROC-AUC misura quanto il modello ordina correttamente i soggetti MS sopra i soggetti HC al variare della soglia. Nel riepilogo medio dei fold, il valore maggiore è quello di `channel__svm` (`0.588`, deviazione standard `0.243`), seguito da `channel__xgboost` (`0.544`, DS `0.240`). Per `channel__svm`, la stima pooled per soggetto è `0.594`, con IC bootstrap 95% `0.365-0.807`.

Un valore di `0.5` equivale, praticamente, a un ordinamento casuale. Anche il valore medio migliore è quindi solo moderatamente superiore al caso, e la sua variabilità tra fold è elevata. L'intervallo pooled di `channel__svm` comprende `0.5`, perciò non sostiene una capacità discriminativa affidabile. Valori inferiori a `0.5` in alcune stime aggregate possono riflettere instabilità campionaria o una direzione delle probabilità non adeguata.

### Average precision

L'average precision è influenzata dalla prevalenza della classe positiva. Nel dataset la prevalenza MS è `24/32 = 0.75`, quindi il riferimento non informativo è circa `0.75`.

Nel riepilogo medio dei fold, `channel__svm` ha il valore più alto (`0.858`), seguito da `channel__lda` (`0.828`) e `channel__logistic_elastic_net` (`0.828`). Il valore di `channel__svm` è superiore alla baseline, ma nell'analisi pooled per soggetto è `0.847` (IC bootstrap 95% `0.757-0.932`): il risultato va comunque interpretato insieme alla bassa specificità e alle altre metriche.

Valori elevati di average precision non sono sufficienti da soli, soprattutto in presenza di classi sbilanciate. Vanno confrontati con la baseline di prevalenza e interpretati insieme a sensibilità, specificità e curve precision-recall.

### Balanced accuracy

La balanced accuracy è la media tra sensibilità e specificità:

```text
balanced accuracy =(sensitivity+specificity)/2
```

È più informativa dell'accuracy ordinaria quando le classi sono sbilanciate.

La media fold-wise più alta è quella di `channel__xgboost`:

```text
(0.675+0.400)/2 = 0.5375 circa
```

Il risultato è solo poco superiore a `0.5`, riferimento di una classificazione non informativa in termini bilanciati, e presenta una deviazione standard tra fold di `0.224`. Sulle predizioni pooled, la balanced accuracy di `channel__xgboost` è `0.604` (IC 95% `0.438-0.792`); questa è una stima diversa dalla media fold-wise e non va sostituita ad essa nel confronto tra modelli.

### Sensibilità e specificità

La sensibilità misura quanti soggetti MS vengono riconosciuti correttamente. La specificità misura quanti soggetti HC vengono riconosciuti correttamente.

| Modello | Sensibilità | Specificità | Interpretazione |
|---|---:|---:|---|
| `channel__svm` | 0.917 | 0.075 | Sensibilità alta, ma riconosce correttamente pochissimi HC |
| `roi__svm` | 0.950 | 0.025 | Sensibilità media più alta, specificità quasi nulla |
| `channel__xgboost` | 0.675 | 0.400 | Specificità media più alta, con sensibilità moderata |
| `roi__xgboost` | 0.667 | 0.350 | Sensibilità e specificità entrambe limitate |
| `roi__knn` | 0.758 | 0.175 | Sensibilità elevata, specificità bassa |

Per uno screening preliminare potrebbe essere accettabile privilegiare la sensibilità, ma la scelta deve essere esplicita. Le specificità medie tra `0.025` e `0.400` implicano molti falsi positivi tra i controlli; inoltre le specificità sono medie fold-wise e non rappresentano una singola matrice di confusione.

### F1 score

I valori di F1 sono relativamente alti per alcuni modelli: per esempio `0.831` per `roi__svm` e `0.819` per `channel__svm`, tra i classificatori non dummy. Tuttavia non devono essere letti isolatamente.
Con una prevalenza MS del 75%, un modello può ottenere un F1 positivo discreto concentrandosi sulla classe maggioritaria e trascurando la specificità.

Il caso `dummy` ha F1 `0.857`, sensibilità `1.0` e specificità `0.0`, pur avendo ROC-AUC `0.5` e balanced accuracy `0.5`. Questo dimostra che F1, se calcolato soprattutto sulla classe positiva, può apparire buono anche quando il modello non distingue correttamente HC e MS.


## Confronto ROI contro channel

### Balanced accuracy e specificità

`channel__xgboost` ottiene:

- la migliore balanced accuracy media sui fold (`0.538`);
- la specificità media più alta (`0.400`), a pari merito con `roi__logistic_elastic_net`;
- una balanced accuracy pooled per soggetto pari a `0.604` (IC bootstrap 95% `0.438-0.792`).

Il risultato non è sufficiente per concludere che una rappresentazione sia più stabile o più informativa: la variabilità tra fold è ampia e l'intervallo bootstrap pooled è largo.

### Vantaggi osservati per channel

`channel__svm` ottiene:

- la migliore ROC-AUC media sui fold (`0.588`);
- la migliore average precision media sui fold (`0.858`);
- sensibilità media elevata (`0.917`), ma specificità molto bassa (`0.075`).

Le feature channel potrebbero conservare segnale utile per l'ordinamento, ma il risultato non dimostra un vantaggio generale della rappresentazione channel: la ROC-AUC pooled ha un intervallo ampio e la specificità alla soglia usata è quasi nulla.

### Attenzione al numero di feature

La matrice ROI contiene circa 504 feature, mentre la matrice channel ne contiene circa 2268, a fronte di soli 32 soggetti. Il rapporto feature/soggetti è quindi molto elevato, soprattutto per channel.

Questo crea un rischio concreto di overfitting. Anche se la selezione delle feature avviene dentro la pipeline e dentro i fold, un numero così elevato di variabili rispetto al campione rende le prestazioni potenzialmente instabili.

La differenza tra ROI e channel non va quindi interpretata soltanto come differenza biologica. Può riflettere anche:

- diversa dimensionalità;
- maggiore collinearità tra canali vicini;
- maggiore spazio per trovare associazioni casuali;
- maggiore sensibilità alla scelta degli iperparametri.

## Pericoli metodologici e interpretativi

### Campione molto piccolo

Con 32 soggetti, una variazione di pochi soggetti cambia sensibilmente sensibilità e specificità. Nel calcolo pooled, con 8 soggetti HC, una specificità di `0.375` corrisponde a 3 controlli corretti su 8.

Le metriche hanno quindi una granularità elevata e intervalli di incertezza probabilmente ampi.

### Dataset sbilanciato

Il dataset contiene 24 soggetti MS e 8 HC. Un modello che tende a predire MS può ottenere:

- sensibilità alta;
- F1 positivo relativamente alto;
- average precision apparentemente buona;

pur riconoscendo male i controlli sani. Per questo la balanced accuracy e la specificità sono fondamentali.

### Prestazioni vicine al caso

La maggior parte delle ROC-AUC è vicina a `0.5`. Anche il miglior valore medio sui fold, `0.588` (`channel__svm`), è ancora lontano da una separazione robusta; la stima pooled corrispondente è `0.594` e il relativo intervallo bootstrap è ampio.

Non si dovrebbe descrivere il modello come diagnostico sulla base di questa tabella. Al massimo si può parlare di segnale preliminare da verificare su dati più numerosi e indipendenti.


### Possibile instabilità della selezione delle feature

Le feature vengono selezionate in ciascun fold, correttamente evitando di usare il test fold. Tuttavia, con pochi soggetti, la selezione può cambiare molto da un fold all'altro.

Una feature selezionata una sola volta non dovrebbe essere interpretata come biomarker affidabile. Serve misurare la frequenza di selezione e la sua stabilità tra ripetizioni.


## Miglioramenti prioritari

### 1. Aumentare il numero di soggetti

È il miglioramento più importante. Con 32 soggetti e 8 controlli sani, le metriche sono molto sensibili al caso.

### 2. Ridurre il rischio di overfitting nei channel

Per la rappresentazione channel è opportuno testare:

- valori più restrittivi di `k` nella selezione delle feature;
- regolarizzazione più forte;
- eliminazione di feature fortemente ridondanti;

## Conclusione operativa

`channel__svm` ha i migliori valori medi di ROC-AUC e average precision, mentre `channel__xgboost` ha la balanced accuracy media più alta e la specificità più alta (a pari merito con `roi__logistic_elastic_net`). `roi__svm` ottiene la sensibilità più alta ma quasi non riconosce i sani. 
