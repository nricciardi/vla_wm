# World Models

Un **World Model** apprende una rappresentazione della dinamica di un ambiente per prevedere come il mondo può evolvere in seguito alle azioni dell'agente.

La trattazione partirà dalla distinzione tra stato reale, osservazione e stato latente, per arrivare ai modelli che usano le predizioni per pianificare, apprendere una policy o generare nuove esperienze. L'obiettivo non sarà soltanto descrivere come viene predetto il futuro, ma chiarire **quale informazione viene modellata, come viene utilizzata per decidere e quali errori possono compromettere il controllo**.

## Concetti fondamentali

### Definizione e formulazione probabilistica

Introduzione della dinamica $p(s_{t+1}\mid s_t,a_t)$ e, nei contesti parzialmente osservabili, della relazione tra osservazioni $o_t$, stato $s_t$ e azioni $a_t$. Distinzione tra modelli deterministici e stocastici e tra predizione a un passo e rollout su orizzonti lunghi.

### Modelli nello spazio delle osservazioni e nello spazio latente

Confronto tra la predizione diretta di immagini o altre osservazioni ad alta dimensionalità e l'apprendimento di uno **stato latente compatto**. Saranno discussi encoder, transition model, decoder e criteri con cui stabilire se la rappresentazione conserva le informazioni utili al controllo.

### State-space model e memoria

Studio dei modelli ricorrenti e dei **latent state-space model**, con attenzione alla stima dello stato a partire dalla storia delle osservazioni, alla separazione tra dinamica deterministica e variabili stocastiche e alla gestione della partial observability.

### Obiettivi di training

Analisi delle loss di ricostruzione, predizione, reward e continuation, della regolarizzazione dello spazio latente e del bilanciamento tra accuratezza percettiva e utilità decisionale.

## Pianificazione e apprendimento nel modello

### Model Predictive Control

Uso del World Model per simulare sequenze candidate di azioni, valutarne gli esiti e applicare soltanto la prima azione prima di ripianificare. La sezione comprenderà **shooting methods**, Cross-Entropy Method, scelta dell'orizzonte e compromesso tra qualità del piano e costo computazionale.

### Learning in imagination

Apprendimento di actor e critic su traiettorie generate nello spazio latente. Saranno messi a confronto il planning esplicito durante l'inference e l'uso del modello durante il training per migliorare una policy reattiva.

### Reward e rappresentazioni orientate al controllo

Discussione di quando sia necessario ricostruire l'intera osservazione e quando sia preferibile apprendere soltanto le caratteristiche rilevanti per reward, value e controllo.

## Evoluzione dei modelli

### World Models

Il lavoro di Ha e Schmidhuber sarà utilizzato per introdurre la decomposizione tra **vision model, memory model e controller** e gli esperimenti negli ambienti CarRacing e VizDoom.

#### Novelty

La trattazione approfondirà l'idea di comprimere le osservazioni, apprendere le transizioni nello spazio latente e **addestrare il controller nel futuro immaginato dal modello**.

#### Limiti

Saranno discussi la semplicità degli ambienti considerati, gli errori dei rollout e il rischio che il controller sfrutti imperfezioni della dinamica appresa anziché acquisire un comportamento trasferibile all'ambiente reale.

### PlaNet

PlaNet introdurrà il controllo da immagini mediante un **recurrent state-space model** e il planning online nello spazio latente.

#### Novelty

La novelty da approfondire è l'integrazione tra rappresentazione latente deterministica e stocastica e pianificazione model-based senza dover generare ogni possibile futuro nello spazio dei pixel.

#### Limiti

La sezione analizzerà il costo del planning a ogni passo, la sensibilità all'orizzonte e alla qualità del modello e la dipendenza da obiettivi di ricostruzione che non sempre privilegiano le informazioni più utili al controllo.

### Dreamer

La famiglia Dreamer permetterà di studiare l'apprendimento con actor e critic a partire da **rollout immaginati nello spazio latente**. La trattazione seguirà il passaggio da Dreamer alle versioni successive e il loro impiego in domini con osservazioni e action space differenti.

#### Novelty

Sarà evidenziato il passaggio dal planning online all'apprendimento del comportamento attraverso la dinamica differenziabile del modello, insieme alle modifiche che migliorano stabilità, scalabilità e generalità.

#### Limiti

Saranno separati i limiti dovuti all'errore del modello, alla definizione del reward e alla generalizzazione fuori distribuzione, oltre ai costi di addestramento e alla difficoltà di trasferire i risultati dal benchmark al robot reale.

### MuZero

MuZero sarà discusso nel contesto di giochi da tavolo e Atari come esempio di pianificazione con una dinamica interna appresa.

#### Novelty

Il punto centrale sarà l'apprendimento di una rappresentazione sufficiente a predire **reward, policy e value** senza ricostruire esplicitamente le osservazioni o conoscere in anticipo le regole dell'ambiente.

#### Limiti

La trattazione chiarirà il costo della ricerca durante l'inference, la forte dipendenza dal segnale di reward e le difficoltà che emergono quando si passa da action space discreti e simulatori ripetibili al controllo robotico continuo.

### TD-MPC

La famiglia TD-MPC servirà a esaminare il controllo continuo mediante un modello latente addestrato congiuntamente a reward, value e policy.

#### Novelty

Sarà approfondita la combinazione tra **planning locale e temporal-difference learning**, che orienta la rappresentazione verso ciò che è utile per decidere invece che verso la ricostruzione completa dell'osservazione.

#### Limiti

Saranno analizzati il costo del Model Predictive Control, la dipendenza dalle funzioni obiettivo del task e le difficoltà di robustezza, trasferimento ed esplorazione nei sistemi fisici.

## World Models visuali e robotici

### Video prediction action-conditioned

Studio della generazione di futuri visuali condizionati dalle azioni, della multimodalità dei futuri possibili e delle difficoltà introdotte da occlusioni, contatti e deformazioni.

### Pianificazione visuale per la manipolazione

Analisi dei sistemi che selezionano azioni o traiettorie confrontando futuri visuali predetti con un obiettivo, con attenzione alla traduzione tra cambiamenti nei pixel e comandi robotici eseguibili.

### World Models generativi e simulatori appresi

Esame dei modelli generativi su larga scala usati come ambienti interattivi, generatori di dati o simulatori appresi. Saranno distinti i modelli capaci di produrre video plausibili da quelli sufficientemente **controllabili, consistenti e action-conditioned** da supportare decisioni robotiche.

## Integrazione con i VLA

### World Model come planner o critic

Studio di architetture in cui un VLA propone azioni e il World Model ne anticipa le conseguenze, le confronta o assegna loro un valore prima dell'esecuzione.

### Generazione di dati e miglioramento della policy

Uso dei rollout immaginati per ampliare i dati di training, correggere una policy o apprendere dai fallimenti, distinguendo le esperienze realmente raccolte da quelle sintetiche.

### Modelli congiunti di azione e dinamica

Analisi dei sistemi che condividono rappresentazioni tra predizione del futuro e generazione delle azioni, e del confine tra una policy che possiede implicitamente conoscenza della dinamica e un World Model interrogabile in modo esplicito.

## Valutazione e limiti

### Metriche predittive e utilità per il controllo

Confronto tra qualità visiva, errore nello spazio dello stato, reward prediction e successo del task. Una predizione percettivamente realistica non implica necessariamente un modello utile alla pianificazione, e una rappresentazione efficace per il controllo può non ricostruire ogni dettaglio della scena.

### Accumulo dell'errore e distribution shift

Studio degli errori che si amplificano nei rollout autoregressivi, degli stati fuori distribuzione raggiunti dal planner e dei comportamenti che sfruttano imperfezioni del modello.

### Incertezza, causalità e grounding fisico

Discussione dell'incertezza epistemica e aleatorica, della presenza di futuri multipli, della necessità di distinguere correlazione e conseguenza delle azioni e delle difficoltà nel rappresentare contatti, geometria e vincoli fisici.

### Costo computazionale e trasferimento al mondo reale

Analisi della latenza richiesta per generazione e planning, della dipendenza dai dati, del trasferimento tra ambienti ed embodiment e dei requisiti di sicurezza quando le predizioni guidano un robot reale.
