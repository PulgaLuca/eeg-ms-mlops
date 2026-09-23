# Commento ai risultati e direzioni di miglioramento

## Sintesi

I risultati riportati provengono da 32 soggetti e confrontano gli stessi algoritmi su due rappresentazioni spaziali EEG:

- `roi__`: feature aggregate a livello di Region Of Interest;
- `channel__`: feature mantenute a livello dei singoli canali.

La conclusione principale è prudente: **nessun modello mostra ancora una capacità diagnostica sufficientemente solida per un uso clinico**. Le prestazioni sono solo moderatamente superiori, o talvolta inferiori, a una classificazione casuale. Il risultato migliore dipende dalla metrica considerata:

- `channel__xgboost` ha la migliore capacità discriminativa probabilistica: ROC-AUC `0.609` e average precision `0.862`;
- `roi__xgboost` ha la migliore balanced accuracy (`0.583`) e la migliore sensibilità (`0.792`);
- entrambi hanno una specificità bassa (`0.375`), quindi classificano erroneamente molti controlli sani come soggetti MS.

Non è quindi corretto indicare un vincitore assoluto senza prima stabilire quale obiettivo sia prioritario: discriminazione generale, sensibilità clinica, specificità o calibrazione delle probabilità.

## Lettura delle metriche

### ROC-AUC

La ROC-AUC misura quanto il modello ordina correttamente i soggetti MS sopra i soggetti HC al variare della soglia.

- `channel__xgboost`: `0.609`, il valore migliore;
- `roi__xgboost`: `0.536`;
- `channel__knn`: `0.464`;
- `roi__logistic_elastic_net`: `0.484`.

Un valore di `0.5` equivale, in prima approssimazione, a un ordinamento casuale. Di conseguenza, `channel__xgboost` mostra un segnale potenzialmente interessante, ma ancora debole. Una AUC di `0.609` non è sufficiente per sostenere che il modello abbia capacità diagnostica affidabile.

I valori inferiori a `0.5`, come quelli di `channel__lda`, non dimostrano automaticamente che il modello sia inutile: possono indicare un ordinamento peggiore del caso, instabilità campionaria oppure una direzione delle probabilità non adeguata. In ogni caso sono un segnale di debolezza del modello nel setup attuale.

### Average precision

L'average precision è influenzata dalla prevalenza della classe positiva. Nel dataset la prevalenza MS è `24/32 = 0.75`, quindi una baseline che predicesse sempre la prevalenza ha average precision circa `0.75`.

Pertanto:

- `channel__xgboost`: `0.862`, miglioramento rispetto alla baseline;
- `roi__xgboost`: `0.772`, miglioramento minimo;
- `channel__knn`: `0.746`, praticamente baseline;
- `roi__lda`: `0.829`, apparentemente buona ma da interpretare insieme alle altre metriche.

Questo è un punto importante: un average precision elevato non è sufficiente da solo, soprattutto in presenza di classi sbilanciate. Va sempre confrontato con la baseline di prevalenza e con sensibilità, specificità e curve precision-recall.

### Balanced accuracy

La balanced accuracy è la media tra sensibilità e specificità:

```text
balanced accuracy = (sensitivity + specificity) / 2
```

È più informativa dell'accuracy ordinaria quando le classi sono sbilanciate.

Il valore migliore è quello di `roi__xgboost`:

```text
(0.792 + 0.375) / 2 = 0.5835 circa
```

Il risultato è soltanto moderatamente superiore a `0.5`, che rappresenta il riferimento di una classificazione non informativa in termini bilanciati. Questo significa che la buona sensibilità del modello è parzialmente compensata dalla bassa specificità.

### Sensibilità e specificità

La sensibilità misura quanti soggetti MS vengono riconosciuti correttamente. La specificità misura quanti soggetti HC vengono riconosciuti correttamente.

| Modello | Sensibilità | Specificità | Interpretazione |
|---|---:|---:|---|
| `roi__xgboost` | 0.792 | 0.375 | Riconosce molti MS, ma genera molti falsi positivi HC |
| `channel__knn` | 0.875 | 0.250 | Sensibilità alta, specificità molto bassa |
| `channel__xgboost` | 0.708 | 0.375 | Compromesso leggermente migliore in termini probabilistici |
| `roi__lda` | 0.708 | 0.250 | Prestazione sbilanciata verso la sensibilità |
| `channel__lda` | 0.750 | 0.250 | Molti MS rilevati, molti HC classificati erroneamente |

Per uno screening preliminare potrebbe essere accettabile privilegiare la sensibilità, ma questa scelta deve essere esplicita. Per un uso diagnostico, una specificità pari a `0.25` o `0.375` è un problema serio: rispettivamente il 75% o il 62.5% dei controlli sani verrebbe classificato come MS nel riepilogo considerato.

### F1 score

I valori di F1 sono relativamente alti per alcuni modelli, per esempio `0.824` per `channel__knn`. Tuttavia non devono essere letti isolatamente: con una prevalenza MS del 75%, un modello può ottenere un F1 positivo discreto concentrandosi sulla classe maggioritaria e trascurando la specificità.

Il caso `dummy` è istruttivo: ha F1 `0.857`, sensibilità `1.0` e specificità `0.0`, pur avendo ROC-AUC `0.5` e balanced accuracy `0.5`. Questo dimostra che F1, se calcolato soprattutto sulla classe positiva, può apparire buono anche quando il modello non distingue correttamente HC e MS.

### Brier score

Il Brier score misura la qualità delle probabilità predette: più è basso, meglio le probabilità sono calibrate rispetto agli esiti osservati.

Il valore più basso è quello dei modelli `dummy` (`0.1875`), ma questo non significa che siano i migliori classificatori. Il dummy sfrutta la prevalenza elevata della classe MS e produce probabilità costanti vicine a `0.75`; può quindi essere ben calibrato globalmente senza discriminare i singoli soggetti.

Il Brier score deve quindi essere interpretato insieme a ROC-AUC, curve di calibrazione e capacità di separare le classi. Non è una metrica sufficiente per scegliere il modello.

## Confronto ROI contro channel

### Vantaggi osservati per ROI

`roi__xgboost` ottiene:

- la migliore balanced accuracy (`0.583`);
- la migliore sensibilità (`0.792`);
- una specificità uguale a quella di `channel__xgboost` (`0.375`).

Questo suggerisce che l'aggregazione spaziale in ROI può ridurre il rumore e rendere più stabile il riconoscimento dei soggetti MS rispetto ai singoli canali, almeno alla soglia utilizzata.

### Vantaggi osservati per channel

`channel__xgboost` ottiene:

- la migliore ROC-AUC (`0.609`);
- la migliore average precision (`0.862`);
- un Brier score migliore di `roi__xgboost` (`0.230` contro `0.232`, differenza comunque piccola).

Questo suggerisce che i canali conservano informazione spaziale più dettagliata, che XGBoost riesce almeno parzialmente a sfruttare.

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

Con 32 soggetti, una variazione di pochi soggetti cambia sensibilmente sensibilità e specificità. Per esempio, se ci sono 8 soggetti HC, una specificità di `0.375` corrisponde a circa 3 controlli corretti su 8.

Le metriche hanno quindi una granularità elevata e intervalli di incertezza probabilmente ampi.

### Dataset sbilanciato

Il dataset contiene 24 soggetti MS e 8 HC. Un modello che tende a predire MS può ottenere:

- sensibilità alta;
- F1 positivo relativamente alto;
- average precision apparentemente buona;

pur riconoscendo male i controlli sani. Per questo la balanced accuracy e la specificità sono fondamentali.

### Prestazioni vicine al caso

La maggior parte delle ROC-AUC è vicina a `0.5`. Anche il miglior risultato, `0.609`, è ancora lontano da una separazione robusta.

Non si dovrebbe descrivere il modello come diagnostico sulla base di questa tabella. Al massimo si può parlare di segnale preliminare da verificare su dati più numerosi e indipendenti.

### Assenza di incertezza statistica nella tabella

La tabella presenta valori aggregati, ma non riporta intervalli di confidenza. La differenza tra `0.609` e `0.536`, oppure tra due balanced accuracy vicine, potrebbe non essere statisticamente significativa.

Un ranking numerico non equivale a una dimostrazione che un modello sia realmente superiore a un altro.

### Possibile instabilità della selezione delle feature

Le feature vengono selezionate in ciascun fold, correttamente evitando di usare il test fold. Tuttavia, con pochi soggetti, la selezione può cambiare molto da un fold all'altro.

Una feature selezionata una sola volta non dovrebbe essere interpretata come biomarker affidabile. Serve misurare la frequenza di selezione e la sua stabilità tra ripetizioni.

### Rischio di leakage a monte

La nested cross-validation protegge il training e la selezione degli iperparametri, ma non può correggere leakage già introdotto prima del training. È necessario verificare che:

- le trasformazioni dipendenti dai dati siano state calcolate senza usare informazioni di tutti i soggetti in modo improprio;
- eventuali normalizzazioni o selezioni non siano state eseguite prima degli split;
- soggetti correlati o visite multiple non siano distribuiti tra train e test;
- le finestre dello stesso soggetto non siano trattate come soggetti indipendenti.

## Miglioramenti prioritari

### 1. Aggiungere intervalli di confidenza

Calcolare intervalli di confidenza per:

- ROC-AUC;
- average precision;
- balanced accuracy;
- sensibilità;
- specificità;
- Brier score.

Il bootstrap deve essere eseguito a livello di soggetto, non a livello di singola finestra o singola predizione duplicata. La scelta migliore tra ROI e channel dovrebbe essere basata anche sulla sovrapposizione degli intervalli.

### 2. Aumentare il numero di soggetti

È il miglioramento più importante. Con 32 soggetti e 8 controlli sani, le metriche sono molto sensibili al caso.

Sarebbe opportuno usare:

- un campione HC più ampio;
- dati provenienti da più acquisizioni;
- un test set esterno indipendente;
- possibilmente dati provenienti da un centro o sessione diversa.

### 3. Ridurre il rischio di overfitting nei channel

Per la rappresentazione channel è opportuno testare:

- valori più restrittivi di `k` nella selezione delle feature;
- regolarizzazione più forte;
- riduzione dimensionale appresa dentro i fold, ad esempio PCA;
- aggregazioni spaziali intermedie;
- eliminazione di feature fortemente ridondanti, sempre dentro la validazione;
- modelli più semplici come baseline primaria.

Il confronto deve mantenere lo stesso protocollo di validazione per ROI e channel.

### 4. Ottimizzare la soglia decisionale

Tutte le metriche threshold-based dipendono dalla soglia `0.5`. Questa soglia non è necessariamente ottimale, soprattutto con classi sbilanciate.

La soglia dovrebbe essere scelta dentro l'inner CV in base all'obiettivo:

- massimizzare sensibilità per screening;
- massimizzare specificità per ridurre falsi positivi;
- ottimizzare balanced accuracy;
- rispettare un vincolo clinico, per esempio sensibilità minima del 90%.

La soglia non deve essere scelta osservando l'outer test set, altrimenti si introduce leakage nella valutazione.

### 5. Valutare meglio la calibrazione

Per le probabilità si dovrebbero produrre:

- reliability diagram;
- calibration slope e intercept;
- Brier score con intervallo di confidenza;
- confronto con la baseline di prevalenza.

Il dummy deve rimanere un riferimento, non un concorrente diagnostico.

### 6. Analizzare la stabilità delle feature

Per ogni feature si può calcolare:

- frequenza di selezione nei fold;
- frequenza separata per ROI e channel;
- distribuzione dell'importanza;
- coerenza del segno dei coefficienti per i modelli lineari.

Le feature più interessanti sono quelle selezionate frequentemente e con importanza relativamente stabile, non quelle che emergono da un singolo fold.

### 7. Valutare la significatività del confronto ROI/channel

Il confronto dovrebbe usare le predizioni sugli stessi soggetti e gli stessi fold. Possibili analisi:

- bootstrap appaiato delle differenze di metriche;
- confronto delle probabilità soggetto per soggetto;
- test di permutazione;
- analisi della differenza di balanced accuracy tra ROI e channel;
- confronto delle curve ROC e Precision-Recall.

Questo è preferibile al semplice confronto tra medie calcolate separatamente.

## Conclusione operativa

Il risultato più promettente è `channel__xgboost` se l'obiettivo è massimizzare la capacità di ranking e la qualità delle probabilità. Il risultato più equilibrato alla soglia corrente è `roi__xgboost`, grazie alla migliore balanced accuracy e sensibilità.

Tuttavia, entrambi mostrano specificità bassa e prestazioni complessive ancora deboli. La conclusione corretta non è che channel sia definitivamente migliore di ROI, né che XGBoost sia già un classificatore clinico affidabile. La conclusione più difendibile è:

> Le feature channel sembrano contenere un segnale discriminativo potenzialmente maggiore, evidenziato dal risultato di `channel__xgboost`, mentre la rappresentazione ROI offre una soluzione più compatta e una migliore sensibilità/balanced accuracy alla soglia corrente. La differenza deve essere verificata con intervalli di confidenza, analisi di stabilità e validazione esterna su un campione più ampio.

Prima di utilizzare questi modelli in un contesto applicativo, le priorità sono aumentare il campione, quantificare l'incertezza, controllare la stabilità delle feature e definire la soglia sulla base dell'obiettivo clinico.
