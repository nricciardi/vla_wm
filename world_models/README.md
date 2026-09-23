# World Models

Un **World Model** è un modello predittivo che rappresenta **come un ambiente evolve nel tempo** e come tale evoluzione dipende dalle azioni di un agente.

In forma generale, dato lo stato corrente $s_t$, una sequenza di azioni $a_{t:t+H-1}$ e un orizzonte di predizione $H$, il modello cerca di descrivere una distribuzione sui possibili stati futuri:

$$
p(s_{t+1:t+H}\mid s_t,a_{t:t+H-1})
$$

Nei sistemi embodied lo stato reale non è normalmente accessibile. Il modello riceve osservazioni $o_t$, come immagini, propriocezione o segnali tattili, e può quindi predire direttamente osservazioni future oppure costruire uno **stato latente** che conserva le informazioni necessarie per previsione e controllo. Nei sistemi condizionati dal linguaggio, un'istruzione $l$ può inoltre specificare il task o il futuro desiderato:

$$
p(o_{t+1:t+H}\mid o_{\leq t},a_{t:t+H-1},l)
$$

La predizione non costituisce però il fine ultimo.

Un World Model diventa utile a un agente quando permette di **anticipare le conseguenze delle azioni**, confrontare piani alternativi, apprendere una policy su esperienze immaginate, generare nuovi dati o valutare un comportamento senza eseguirlo immediatamente nel mondo reale.

## Tassonomia

### Spazio dello stato

Saranno confrontati i modelli che **predicono variabili fisiche** strutturate, quelli che **generano immagini o video** e quelli che operano in uno **spazio latente**. La scelta determina quali dettagli vengono conservati, quanto costa effettuare un rollout e quanto facilmente la predizione può essere impiegata per il controllo.

### Modelli deterministici e stocastici

Una singola azione può produrre esiti differenti a causa di informazione incompleta, rumore e interazioni non osservate. Verranno quindi distinti i transition model deterministici dai modelli probabilistici capaci di rappresentare **futuri multimodali**, separando per quanto possibile incertezza aleatorica e incertezza epistemica.

### World Model espliciti, impliciti e orientati al controllo

Un World Model può essere un modulo esplicitamente interrogabile, una dinamica latente usata per il planning oppure una capacità predittiva internalizzata nella policy. Sarà inoltre chiarita la differenza tra ricostruire accuratamente l'ambiente e apprendere soltanto le variabili sufficienti a predire reward, value o conseguenze rilevanti per la decisione.

### Ruoli nel ciclo decisionale

La tassonomia distinguerà quattro impieghi principali: **planning mediante rollout**, apprendimento della policy in imagination, simulazione ed evaluation, generazione di dati sintetici. La stessa architettura può ricoprire più ruoli, ma richiede proprietà diverse a seconda che debba generare immagini realistiche, preservare la causalità delle azioni o classificare correttamente il successo di un comportamento.

## Dinamiche latenti e controllo in imagination

Questa parte ricostruirà il passaggio dai modelli compatti della dinamica ai sistemi che pianificano o addestrano actor e critic interamente su traiettorie latenti. Il confronto chiarirà la differenza tra **planning online**, eseguito a ogni passo, e *learning in imagination*, nel quale il modello viene sfruttato soprattutto durante il training.

### World Models (2018)

**World Models**, di David Ha e Jürgen Schmidhuber, mostra come separare la complessità dell'apprendimento in un grande modello predittivo e in un controller molto piccolo. Le osservazioni RGB vengono compresse da un **Variational Autoencoder (VAE)** in un vettore latente $z_t$; un **Mixture Density Network-Recurrent Neural Network (MDN-RNN)** usa $z_t$, l'azione $a_t$ e la propria memoria $h_t$ per modellare una distribuzione sul latent successivo; un controller lineare trasforma infine $z_t$ e $h_t$ nell'azione.

Il paper considera due ambienti OpenAI Gym simulati, senza robot fisici. In **CarRacing-v0** il modello viene addestrato su 10,000 rollout di una policy casuale e fornisce al controller una rappresentazione spazio-temporale appresa dai pixel. Il sistema completo raggiunge un reward medio di $906\pm21$ su 100 episodi, contro $632\pm251$ quando il controller vede soltanto il latent del VAE.

In **VizDoom: Take Cover** il ruolo del World Model è più radicale: oltre al prossimo latent, l'MDN-RNN predice la terminazione dell'episodio e diventa un ambiente virtuale nel quale il controller viene addestrato interamente con CMA-ES. La policy viene poi trasferita senza ulteriore training al simulatore originale, dove raggiunge $1,092\pm556$ step di sopravvivenza rispetto alla soglia di 750 richiesta dal task.

#### Novelty

Il contributo centrale è la decomposizione **Vision–Memory–Controller** e la dimostrazione che una policy può essere ottimizzata dentro la dinamica appresa e successivamente trasferita nell'ambiente reale del benchmark. Il lavoro rende inoltre esplicita un'idea destinata a diventare centrale: il controller non deve necessariamente ricostruire o comprendere ogni dettaglio dell'osservazione, ma può agire usando una rappresentazione compatta del presente e del futuro probabile.

#### Limiti

VAE, MDN-RNN e controller vengono addestrati separatamente, perciò la rappresentazione visuale non è ottimizzata per preservare ciò che è rilevante al task. I dati provengono da policy casuali e da ambienti semplici; gli esperimenti non affrontano robot reali, contatti complessi o generalizzazione tra task.

Il controller può inoltre trovare **adversarial policy che sfruttano gli errori del World Model** e visitano stati fuori distribuzione. Il paper attenua il problema aumentando la temperatura del modello stocastico, ma non lo risolve in modo generale.

L'[approfondimento su World Models](models/world_models/README.md) sviluppa architettura, obiettivi di training, esperimenti CarRacing e VizDoom e il problema del model exploitation.

### PlaNet (2019)

**PlaNet**, acronimo di *Deep Planning Network*, apprende dai pixel un modello della dinamica e lo usa per scegliere le azioni mediante planning online. A differenza di World Models, non addestra un controller separato: a ogni step valuta sequenze di azioni nel modello appreso, esegue soltanto la prima azione e pianifica nuovamente dopo la successiva osservazione.

Il cuore dell'architettura è il **Recurrent State-Space Model (RSSM)**. Lo stato latente combina una memoria deterministica $h_t$, che conserva informazione nel tempo, e una variabile stocastica $s_t$, che rappresenta incertezza e futuri multipli. Un observation model ricostruisce $o_t$ per fornire un segnale di training ricco, mentre un reward model predice $r_t$ direttamente dal latent. Durante il planning le immagini non vengono decodificate: il **Cross-Entropy Method (CEM)** cerca le azioni usando soltanto transizioni e reward latenti.

PlaNet viene valutato su sei task simulati della **DeepMind Control Suite** osservati attraverso immagini RGB da $64\times64$ pixel: Cartpole Swing Up, Reacher Easy, Cheetah Run, Finger Spin, Cup Catch e Walker Walk. I task comprendono partial observability, reward sparsi, contatti e controllo continuo; non sono coinvolti robot fisici.

L'agente parte da cinque episodi casuali e amplia iterativamente il dataset con traiettorie raccolte dal proprio planner. Dopo 1,000 episodi raggiunge prestazioni comparabili a D4PG addestrato per 100,000 episodi e, nella media riportata dagli autori, richiede circa **200 volte meno interazione con l'ambiente**.

#### Novelty

- Introduce il **planning interamente nello spazio latente da osservazioni visuali**, evitando di generare immagini per valutare le sequenze di azioni.
- Propone l'**RSSM**, che combina una dinamica deterministica ricorrente con stati stocastici e diventerà la base della famiglia Dreamer.
- Formula il **latent overshooting**, una regolarizzazione multi-step che allinea predizioni latenti a lungo orizzonte e posteriori inferiti senza decodificare immagini aggiuntive.
- Integra apprendimento del modello, raccolta online dei dati e Model Predictive Control in un unico ciclo puramente model-based, senza actor o value network.

#### Limiti

Il CEM richiede di simulare migliaia di sequenze a ogni step, introducendo un costo di inferenza elevato nonostante il planning avvenga nel latent. L'orizzonte è fissato a 12 passi del modello e dipende dall'action repeat; conseguenze più lontane non vengono stimate da una value function.

Gli esperimenti rimangono limitati a simulatori con immagini semplici, reward disponibili durante la raccolta e action space continui di dimensione moderata. La ricostruzione può inoltre dedicare capacità a dettagli visivi irrilevanti. Infine, il latent overshooting è un contributo del paper, ma la configurazione RSSM finale non lo utilizza perché nelle ablation non migliora tale architettura.

L'[approfondimento su PlaNet](models/planet/README.md) sviluppa inferenza e dinamica dell'RSSM, obiettivo variazionale, latent overshooting, planning con CEM e risultati sulla DeepMind Control Suite.

### Dreamer, DreamerV2 e DreamerV3 (2020, 2021, 2023)

La famiglia Dreamer mostrerà come sostituire il planning online con l'apprendimento di actor e critic su rollout immaginati. L'approfondimento seguirà l'evoluzione dell'RSSM, degli obiettivi di training e delle tecniche di stabilizzazione, separando i progressi sui benchmark dalla robustezza richiesta in un sistema fisico.

### MuZero (2020)

MuZero offrirà un esempio di **modello orientato alla decisione** che non ricostruisce le osservazioni, ma apprende rappresentazione, dinamica, reward, policy e value necessari alla ricerca. Il capitolo ne chiarirà il rapporto con i World Model generativi e i limiti del trasferimento da action space discreti e simulatori ripetibili al controllo robotico continuo.

### TD-MPC e TD-MPC2 (2022, 2024)

La famiglia TD-MPC permetterà di studiare l'integrazione tra dinamica latente, temporal-difference learning e Model Predictive Control. L'attenzione sarà rivolta alle rappresentazioni *task-oriented*, al campionamento delle traiettorie candidate e al compromesso tra efficienza, generalizzazione multi-task e accuratezza dei rollout.

## Dal benchmark al robot fisico

I risultati in simulazione non garantiscono che un modello sappia gestire rumore sensoriale, latenza, contatti e distribuzioni che cambiano durante l'esecuzione. Questa parte farà da ponte tra il model-based reinforcement learning e il robot learning reale.

### DayDreamer (2023)

DayDreamer sarà approfondito come dimostrazione di **learning in imagination su robot fisici**. Il capitolo esaminerà il training online da immagini e reward, gli embodiment considerati, l'efficienza nell'acquisizione dei dati e i vincoli che rendono l'esplorazione reale più difficile e rischiosa rispetto a quella simulata.

## Predizione visuale e robot imagination

In questa famiglia il futuro viene rappresentato come immagine o video. Una predizione visuale può fungere da goal intermedio, piano interpretabile o sorgente di supervisione, ma deve poi essere collegata ad azioni eseguibili mediante ottimizzazione, inverse dynamics o una policy dedicata.

### Deep Visual Foresight (2017)

Deep Visual Foresight introdurrà la pianificazione robotica basata su video prediction e Model Predictive Control. L'approfondimento ricostruirà il visual model action-conditioned, la definizione del costo nello spazio dei pixel e l'uso del planning closed loop per il pushing, evidenziando i limiti dovuti a orizzonte, occlusioni e accuratezza percettiva.

### UniPi (2023)

UniPi rappresenterà il paradigma **predict-then-act**: un generatore video condizionato dal testo immagina una traiettoria coerente con il task, mentre un inverse dynamics model converte la transizione visuale in azioni. Saranno analizzati il trasferimento dai video alla policy e la possibile incoerenza tra un futuro plausibile e uno realmente eseguibile.

### RoboDreamer (2024)

RoboDreamer permetterà di approfondire la generazione composizionale di futuri per la manipolazione. Il capitolo esaminerà la scomposizione di istruzioni, oggetti e azioni, il ruolo del video come immaginazione per la policy e i limiti di un modello guidato dal task ma non direttamente condizionato da ogni comando motorio.

## World Model integrati nella policy

I sistemi più recenti riducono la separazione tra previsione e controllo. Il World Model può condividere il backbone con la policy, interagire con un action expert o comparire come obiettivo ausiliario che induce rappresentazioni sensibili alla dinamica. La sezione confronterà pipeline disaccoppiate, **single-backbone**, Mixture-of-Experts o Mixture-of-Transformers, VLA unificati e modelli predittivi latenti.

### GR-1 (2024)

GR-1 sarà studiato come uno dei primi foundation model robotici che addestra congiuntamente la predizione delle azioni e delle immagini future in un Transformer GPT-style. L'approfondimento chiarirà quali segnali sono condivisi, come viene sfruttato il video pre-training e in che misura la previsione visuale migliora la policy.

### Unified World Models (2025)

Unified World Models, indicato anche come **UWA**, rappresenterà l'accoppiamento di video diffusion e action diffusion nello stesso Transformer. Il capitolo esaminerà la gestione di timestep specifici per modalità, il modo in cui il modello viene interrogato come policy e il costo di unificare generazione visuale e controllo.

### WorldVLA (2025)

WorldVLA illustrerà l'integrazione autoregressiva di comprensione multimodale, generazione delle azioni e previsione delle immagini future. La trattazione distinguerà il World Model usato come segnale di training dalla generazione esplicita del futuro durante l'inference e analizzerà il problema dell'allineamento causale tra azione e predizione.

### Cosmos Policy (2026)

Cosmos Policy mostrerà come adattare un video diffusion model pre-addestrato al controllo rappresentando azioni, stati futuri e value come ulteriori frame latenti. Il capitolo confronterà l'uso diretto come policy con l'uso per planning e valuterà il vantaggio dei prior appresi dai video rispetto ai costi di inferenza e adattamento.

### VLA-JEPA (2026)

VLA-JEPA rappresenterà l'alternativa alla generazione dei pixel: il modello predice **target latenti futuri** e usa tale obiettivo per apprendere transizioni rilevanti per l'azione. L'approfondimento chiarirà la relazione con la famiglia JEPA, le precauzioni contro la fuga di informazione dai frame futuri e il compromesso tra efficienza e interpretabilità.

## World Model come simulatore ed evaluator

Un World Model può essere separato dalla policy e operare come ambiente appreso. In questo ruolo genera rollout per reinforcement learning o post-training, assegna reward, confronta azioni candidate e stima se una policy avrà successo. La qualità richiesta non coincide con il solo realismo visivo: il simulatore deve reagire correttamente agli interventi della policy e mantenere affidabile il segnale di valutazione.

### IRASim (2025)

IRASim sarà approfondito come video World Model per la manipolazione condizionato da traiettorie robotiche. La trattazione esaminerà il condizionamento dei singoli fotogrammi sulle azioni, la capacità di simulare embodiment e task diversi e la distanza tra metriche video open loop e utilità per una policy closed loop.

### World-Env (2025)

World-Env illustrerà l'uso del World Model come **ambiente virtuale per il post-training dei VLA**. Il capitolo analizzerà generazione delle interazioni, reward e aggiornamento della policy, prestando particolare attenzione al rischio che errori o bias del simulatore vengano amplificati durante l'ottimizzazione.

### Ctrl-World (2026)

Ctrl-World rappresenterà i simulatori visuali controllabili multi-view e a lungo orizzonte. L'approfondimento discuterà memoria temporale, fedeltà alle azioni e uso dei rollout per evaluation e policy improvement, distinguendo la coerenza cinematica dalla plausibilità puramente percettiva.

### VLAW (2026)

VLAW permetterà di studiare la **co-evoluzione iterativa tra policy e World Model**. Il capitolo seguirà il ciclo in cui la policy produce nuove interazioni, il simulatore ne apprende le dinamiche e i rollout sintetici contribuiscono a migliorare nuovamente la policy, includendo i problemi di instabilità e mutuo sfruttamento degli errori.

## Robotic video foundation models e generazione di dati

Questa parte allargherà lo sguardo dai modelli addestrati per un singolo setup ai backbone video riutilizzabili per simulazione, planning, evaluation e data generation. Saranno distinti il condizionamento mediante istruzioni, che descrive **quale futuro è desiderato**, e il condizionamento mediante azioni, che specifica **quale intervento produce il futuro**.

### DreamGen (2025)

DreamGen servirà a esaminare i video World Model come generatori di traiettorie sintetiche. L'approfondimento tratterà l'adattamento al robot target, il recupero delle azioni mediante latent action model o inverse dynamics e la verifica che l'aumento di varietà non introduca supervisione fisicamente inconsistente.

### Cosmos Predict 2.5 (2025)

Cosmos Predict 2.5 rappresenterà i **foundation world model** derivati da grandi video backbone. Il capitolo analizzerà modalità di condizionamento, scalabilità e riuso in sistemi embodied, chiarendo perché una conoscenza video generale non garantisce da sola controllabilità fine, causalità delle azioni o accuratezza nei contatti.

## Altri domini embodied

Manipolazione, navigazione e guida autonoma condividono il bisogno di prevedere il futuro, ma differiscono per action space, geometria, orizzonte e costo dell'errore. Il capitolo userà questi domini come confronto, senza assumere che un'architettura efficace in uno di essi possa essere trasferita direttamente agli altri.

### Navigazione

La sezione discuterà mappe predittive, occupancy, rappresentazioni egocentriche e rollout visuali per scegliere waypoint o traiettorie. Particolare attenzione sarà dedicata alla differenza tra prevedere il movimento dell'agente in una scena relativamente statica e modellare interazioni fisiche con oggetti manipolabili.

### Guida autonoma

La guida autonoma offrirà un caso di world modeling multi-view e multi-agente nel quale occorre prevedere sia il moto dell'ego vehicle sia il comportamento degli altri attori. Saranno confrontate rappresentazioni video, occupancy 3D e token multimodali, insieme ai requisiti di sicurezza e calibrazione dell'incertezza.

## Dati, benchmark e protocolli di valutazione

La valutazione sarà organizzata in base alla funzione del modello. I dataset per robot learning dovranno essere descritti considerando embodiment, camere, propriocezione, action space, linguaggio, fallimenti e copertura dei contatti, non soltanto il numero complessivo di episodi.

### Qualità predittiva open loop

Metriche percettive e distribuzionali misurano la somiglianza dei video generati ai dati osservati, ma possono premiare dettagli visivi irrilevanti o ignorare errori decisivi nella dinamica. Saranno quindi affiancate da misure di state prediction, geometria e coerenza temporale.

### Controllabilità e consistenza fisica

La valutazione dovrà verificare se azioni differenti producono conseguenze differenti e corrette, se il movimento rispetta cinematica e contatti e se il modello conserva identità, geometria e coerenza multi-view durante rollout lunghi.

### Utilità closed loop

Il criterio conclusivo sarà l'effetto sul comportamento: successo del task, qualità del planning, accuratezza nel ranking delle policy, miglioramento ottenuto con dati sintetici e correlazione tra evaluation immaginata ed esecuzione reale. I risultati saranno interpretati tenendo conto delle differenze tra benchmark, embodiment e protocolli.

## Limiti e direzioni di ricerca

Il capitolo conclusivo collegherà i limiti architetturali alle conseguenze operative, evitando di trattare la qualità generativa come sostituto automatico della comprensione fisica.

### Causalità e action conditioning

Un futuro coerente con l'istruzione può dipendere poco dall'azione candidata. Questo **causal conditioning gap** rende il modello inadatto a confrontare interventi, anche quando il video generato appare plausibile.

### Accumulo dell'errore e distribution shift

Nei rollout autoregressivi piccoli errori modificano la distribuzione degli input successivi e possono crescere rapidamente. Planner e policy possono inoltre scoprire regioni dello spazio latente nelle quali il modello è inaccurato e sfruttarne involontariamente le imperfezioni.

### Efficienza e controllo real-time

Video diffusion, sampling di molte traiettorie e Model Predictive Control introducono latenza e consumo di memoria. Saranno confrontate generazione parziale, predizione latente, distillazione e uso del World Model soltanto durante il training.

### Percezione multimodale e grounding fisico

Visione e propriocezione non osservano direttamente proprietà come attrito, rigidezza o stabilità del contatto. L'integrazione di tatto e forza richiede di allineare segnali con frequenze e dimensionalità differenti senza lasciare che la modalità visiva domini la rappresentazione.

### Struttura geometrica e simbolica

Pixel e latent continui possono essere affiancati da oggetti, relazioni, geometria 3D, affordance o predicati. I modelli ibridi promettono rollout più compatti e composizionali, ma richiedono un grounding affidabile tra osservazioni e struttura astratta.

### Sicurezza e integrazione con il controllo classico

L'impiego su robot reali richiede di collegare la capacità predittiva a vincoli cinematici, controllo robusto e meccanismi di verifica. Un World Model appreso può descrivere dinamiche difficili da modellare analiticamente, ma non fornisce automaticamente garanzie di stabilità o sicurezza.
