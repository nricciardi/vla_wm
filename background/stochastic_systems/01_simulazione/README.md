# Simulazione a eventi discreti

La simulazione è uno degli strumenti più potenti nell'ambito del Management delle Operations e della Supply Chain. Possiamo definire la simulazione come il processo di progettazione di un modello di un sistema reale e la conduzione di esperimenti su tale modello. Gli scopi principali di questo processo sono due: comprendere il comportamento intrinseco del sistema e valutare diverse strategie operative per la sua gestione.

In termini più formali, una simulazione è l'imitazione delle operazioni di un sistema del mondo reale nel corso del tempo. Questo processo complesso comporta tre fasi essenziali:

1. La definizione di un insieme di assunzioni matematiche e logiche sul sistema in esame.
2. La generazione di una "storia artificiale" del sistema tramite l'esecuzione del modello.
3. L'osservazione e l'analisi di questa storia artificiale per trarre inferenze (conclusioni) sulle caratteristiche operative del sistema reale.

Gli obiettivi primari di uno studio di simulazione si dividono in due categorie analitiche:

* **Analisi "AS-IS" (osservazione)**: mira a misurare le performance di un sistema per comprenderne il comportamento attuale mentre evolve nel tempo.
* **Analisi "TO-BE" (ottimizzazione)**: mira a incrementare le performance del sistema esplorando domande del tipo "what-if" (cosa succederebbe se...). Ciò si ottiene agendo sui componenti del sistema o modificando le relazioni logiche tra di essi.

Sebbene esistano modelli semplici risolvibili analiticamente tramite metodi matematici esatti (come la teoria della probabilità, il calcolo differenziale o la programmazione matematica), la maggior parte dei sistemi del mondo reale presenta un livello di complessità tale da rendere impossibili queste soluzioni dirette. In questi casi, la costruzione di modelli di simulazione basati su computer diventa l'unica via percorribile. Un esempio classico di applicabilità è un'azienda manifatturiera che valuta un grande ampliamento di un magazzino: la simulazione permette di confrontare i potenziali guadagni operativi con i costi di costruzione attraverso analisi "what-if" prima di spendere risorse reali.

La simulazione offre innumerevoli vantaggi operativi:

- Permette di esplorare nuove policy (procedure operative, regole decisionali) senza interrompere il funzionamento del sistema reale.
- Consente di testare nuovi design di sistema (layout fisico, processi, sistemi di trasporto) senza impegnare risorse finanziarie per la loro acquisizione.
- Permette di testare la fattibilità di ipotesi sul perché o sul come si verificano determinati fenomeni.
- Offre la possibilità di osservare le interazioni tra le variabili del sistema e di comprendere la loro importanza relativa sulle performance finali.
- È lo strumento ideale per condurre l'analisi dei colli di bottiglia (bottleneck analysis), scoprendo quali risorse causano ritardi eccessivi.
- Aiuta a comprendere come il sistema opera *realmente* rispetto al modello mentale preconcetto del management.

Tuttavia, presenta anche degli svantaggi significativi da tenere in considerazione:

- L'interpretazione delle misure di performance può risultare complessa, poiché la casualità (randomness) negli input genera inevitabilmente casualità negli output.
- Gli output di una simulazione non sono mai esatti, ma sono stime o approssimazioni.
- La modellazione e la successiva analisi possono essere processi lunghi e costosi.

La simulazione è lo strumento appropriato quando si devono studiare interazioni interne complesse, valutare l'impatto di cambiamenti organizzativi, ambientali o informativi, o verificare soluzioni analitiche.

Al contrario, *non* è appropriata quando: il problema può essere risolto con il buon senso o analiticamente, fare un esperimento diretto costa meno, non c'è tempo sufficiente, i dati per validare il modello non esistono, o il livello di dettaglio richiesto è eccessivamente complesso.

Gli ambiti di applicazione sono vastissimi: dallo studio della congestione del traffico alla valutazione di nuove tecnologie di automazione, dalla gestione delle interruzioni della domanda o dei guasti per manutenzione, fino alla progettazione della Supply Chain (come posizionare fabbriche, centri di distribuzione o supermercati), la gestione dell'ultimo miglio, valutazioni finanziarie di portafoglio e il controllo ottimo delle policy.


## Sistema

Prima di poter modellare, è necessario definire l'oggetto di studio. Un **sistema** è definito come un gruppo di entità che sono unite insieme in una qualche interazione regolare o interdipendenza per il raggiungimento di un determinato scopo.

Per esempio, in un sistema di produzione che fabbrica automobili, il sistema è l'impianto stesso, le entità sono le macchine, i componenti e i lavoratori, mentre lo scopo è la produzione di veicoli di alta qualità a un determinato ritmo (rate).

Nella pratica, la definizione dei confini del sistema dipende strettamente dagli obiettivi specifici dello studio. È essenziale definire i seguenti concetti:

- **Stato (State)**: la collezione di variabili necessarie per descrivere un sistema in un particolare istante di tempo.
- **Ambiente (Environment)**: tutto ciò che si verifica all'esterno del sistema ma che lo influenza.
- **Confine (Boundary)**: la netta linea di separazione tra il sistema e l'ambiente. Ad esempio, in una banca, il processo di arrivo dei clienti sfugge al controllo della banca ed è quindi parte dell'ambiente; al contrario, le policy di servizio dei cassieri sono decise internamente e fanno parte del sistema.

Per analizzare e modellare un sistema, si devono definire cinque componenti fondamentali:

1. **Entità (Entity)**: un oggetto di interesse nel sistema (es. i clienti in una banca).
2. **Attributi (Attributes)**: le proprietà specifiche di un'entità (es. il saldo del conto, il numero di identità).
3. **Attività (Activity)**: un periodo di tempo di lunghezza specificata (es. il tempo impiegato per effettuare un deposito).
4. **Eventi (Event)**: un'occorrenza istantanea che potrebbe cambiare lo stato del sistema. Possono essere esogeni (es. l'arrivo di un cliente dall'esterno) o endogeni (es. il completamento del servizio di un cliente da parte del cassiere).
5. **Variabili di Stato (State Variables)**: le variabili che descrivono il sistema in ogni momento (es. il numero di cassieri occupati o il numero di clienti in coda).

Un sistema può inoltre essere classificato come **Discreto** (le variabili di stato cambiano in punti discreti nel tempo, come in una banca o un magazzino) o **Continuo** (le variabili di stato cambiano in modo continuo nel tempo, come l'acqua in una diga o l'andamento del mercato azionario).

## Modello

Un **modello** è definito come una rappresentazione di un sistema con lo scopo di studiare tale sistema. Per definizione, un modello è una *semplificazione della realtà*.

Come affermava George Box, "Tutti i modelli sono sbagliati, ma alcuni sono utili". È fondamentale trovare il giusto bilanciamento: un modello troppo semplice (under-elaboration/under-parametrization) non permetterà di trarre conclusioni valide, ma un modello inutilmente complesso (over-elaboration) diventerà incomprensibile tanto quanto il sistema reale che cerca di rappresentare.

Un modello dovrebbe incorporare esclusivamente quegli aspetti del sistema reale che **influenzano direttamente il problema** in esame. Se lo scopo dello studio cambia, il modello deve cambiare di conseguenza.

I modelli di simulazione si classificano secondo diverse dicotomie:

- **Modelli Fisici vs Modelli Matematici**: i primi sono repliche tangibili (in scala), i secondi utilizzano notazioni simboliche, equazioni e relazioni logiche/quantitative.
- **Statici vs Dinamici**: i modelli statici (spesso chiamati simulazioni Monte Carlo) rappresentano il sistema in un preciso istante di tempo, mentre i modelli dinamici ne rappresentano l'evoluzione temporale.
- **Deterministici vs Stocastici**: i modelli deterministici non contengono variabili casuali (a input noti corrispondono output unici); i modelli stocastici, avendo una o più variabili casuali in input (es. arrivi di clienti con distribuzione di Poisson), generano output che sono stime statistiche.
- **Discreti vs Continui**: similmente ai sistemi, ma senza una mappatura obbligatoria. Un sistema continuo (come l'acqua in un tubo) può essere modellato e semplificato attraverso un modello discreto se l'obiettivo dello studio lo consente.

La maggior parte dei modelli di simulazione utilizzati in Operations Management sono **discreti, stocastici e dinamici**.


## Discrete-Event Simulation

La **Discrete-Event Simulation** (D.E.S.) è una specifica tipologia di simulazione che modella un sistema mentre evolve nel tempo, con la caratteristica fondamentale che le variabili di stato cambiano *istantaneamente* in punti *separati* nel tempo.

Questi specifici punti nel tempo sono chiamati **Eventi** e la loro occorrenza altera lo stato del sistema. La D.E.S. utilizza procedure computazionali numeriche per "eseguire" (run) i modelli matematici, generando una storia artificiale dalla quale estrarre stime sulle misure di performance.

Prendiamo l'esempio di una struttura di servizio a singolo server (es. la cassa di una banca). L'obiettivo è stimare il ritardo medio (delay) in coda dei clienti. Le variabili di stato saranno lo stato del server (occupato/libero), la lunghezza della coda e il tempo di arrivo dei clienti. In questo sistema avremo tre eventi discreti principali che alterano lo stato: l'arrivo di un cliente, l'inizio del suo servizio e la sua partenza (che coincide col completamento del servizio da parte del server).

### Step di uno studio di Simulazione

Condurre uno studio di simulazione D.E.S. richiede un approccio ingegneristico strutturato in quattro macro-fasi:

#### Scoperta e orientamento

- **Formulazione del problema**: il problema deve essere compreso chiaramente. Non bisogna creare "una soluzione in cerca di un problema".


- **Definizione degli obiettivi e del piano di progetto**: gli obiettivi delineano le domande a cui rispondere, mentre il piano di progetto alloca le risorse (costi, tempo, personale).



#### Costruzione del modello

- **Concettualizzazione del modello**: si astraggono le caratteristiche essenziali del problema e si selezionano le assunzioni di base. Si parte da un modello semplice per arrivare gradualmente a una maggiore complessità.


- **Raccolta dei dati (Data collection)**: deve iniziare presto poiché richiede molto tempo. Esiste una costante interazione tra la raccolta dati e la complessità del modello.


- **Traduzione del modello**: implementazione software (e.g., utilizzando Python e la libreria Simpy) per gestire la memorizzazione delle informazioni e i calcoli.


- **Verifica e Validazione**: lo step più cruciale. La verifica controlla se il software esegue correttamente ciò per cui è stato programmato; la validazione controlla se il modello rappresenta fedelmente il comportamento del sistema reale. Un modello non valido porterà a decisioni operative errate.



#### Esecuzione

- **Design Sperimentale (Experimental design)**: si definiscono le alternative decisionali da testare. Per ogni scenario bisogna definire il periodo di inizializzazione (warm-up) per eliminare i bias iniziali, la lunghezza delle simulazioni e il numero di repliche necessarie per ottenere stime non distorte e a bassa varianza.


- **Esecuzione e Analisi**: si generano gli output e si interpretano i risultati.



#### Implementazione

- **Documentazione e Reportistica**: la documentazione del codice rende il modello riutilizzabile in futuro, mentre i report tengono traccia dei progressi.


- **Implementazione finale**: si sceglie l'alternativa migliore. Il successo di questa fase finale dipende interamente dalla qualità dell'esecuzione degli step precedenti.
