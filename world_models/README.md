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

**MuZero** combina un modello della dinamica appreso con **Monte Carlo Tree Search (MCTS)**, senza cercare di ricostruire le osservazioni o di imitare lo stato reale dell'ambiente. Una representation function trasforma la storia osservata in uno stato latente; una dynamics function applica un'azione ipotetica e produce il latent successivo e il reward; una prediction function stima policy e value. La rappresentazione viene quindi ottimizzata soltanto per conservare le informazioni utili alla decisione.

Durante l'esecuzione, MCTS costruisce un albero nello spazio latente usando policy e value per guidare la ricerca e i reward predetti per valutarne i rami. Durante il training, le tre funzioni vengono apprese congiuntamente su sequenze reali: policy e value devono riprodurre i target migliorati dalla ricerca, mentre il reward deve corrispondere a quello osservato. Si crea così un ciclo di **search-based policy iteration** nel quale il modello migliora la ricerca e la ricerca fornisce target migliori al modello.

Gli esperimenti coprono Go, scacchi e shogi mediante self-play e tutti i 57 giochi dell'**Arcade Learning Environment** a partire da frame RGB. Non sono coinvolti robot fisici. Nei giochi da tavolo MuZero eguaglia le prestazioni sovrumane di AlphaZero pur sostituendo il simulatore perfetto con una dinamica appresa; su Atari supera i precedenti risultati model-free e model-based del confronto originale. La variante MuZero Reanalyze raggiunge un punteggio mediano normalizzato rispetto all'essere umano del 731.1% usando 200 milioni di frame per gioco.

#### Novelty

- Introduce un **modello value-equivalent**, libero di rappresentare soltanto reward, policy e value necessari al planning anziché predire pixel o stati reali.
- Sostituisce nel tree search le transizioni note di AlphaZero con una **dinamica latente appresa end-to-end** dalle interazioni.
- Estende la search-based policy iteration a domini single-agent con reward intermedi, sconto e osservazioni visuali complesse.
- Unifica representation, dynamics e prediction in un obiettivo multi-step supervisionato dai risultati della ricerca e dai reward osservati.

#### Limiti

Il modello è deterministico e non rappresenta esplicitamente l'incertezza o futuri multimodali. Il suo latent non deve essere interpretabile né fisicamente fedele: è adatto a scegliere azioni secondo il reward di training, ma non costituisce un simulatore generativo interrogabile per altri scopi.

I risultati richiedono grandi quantità di esperienza e calcolo distribuito, mentre MCTS introduce un costo online per ogni decisione. Tutti gli action space sperimentali sono discreti e l'ambiente continua a fornire reward, osservazioni e azioni legali al nodo radice. Il paper non dimostra quindi controllo continuo, trasferimento tra task o uso su robot reali.

L'[approfondimento su MuZero](models/muzero/README.md) sviluppa il modello value-equivalent, le tre funzioni neurali, MCTS, l'obiettivo di training e i risultati nei giochi da tavolo e su Atari.

### TD-MPC e TD-MPC2 (2022, 2024)

**TD-MPC** combina una dinamica latente orientata al task con **temporal-difference learning** e **Model Predictive Control**. Un encoder trasforma l'osservazione in un latent; dynamics e reward model valutano gli effetti a breve termine di sequenze di azioni continue; una funzione $Q$ stima invece il return oltre il breve orizzonte del planner. Una policy appresa non sostituisce il planning, ma propone sequenze promettenti e rende meno costosa l'ottimizzazione online.

Il modello TOLD di TD-MPC è decoder-free: non ricostruisce stato o immagini, ma apprende reward, value e consistenza tra il latent predetto e quello prodotto da un target encoder sull'osservazione successiva. Durante l'esecuzione, MPPI campiona sequenze di azioni, somma i reward immaginati e aggiunge il $Q$-value terminale. Viene eseguita soltanto la prima azione del piano e l'intera procedura viene ripetuta dopo la nuova osservazione.

**TD-MPC2** mantiene questo principio, ma lo rende più stabile e scalabile mediante SimNorm, LayerNorm, regressione discreta di reward e value, un ensemble di funzioni $Q$ e una policy prior stocastica. Task embedding appresi, zero-padding e action masking permettono inoltre a un unico modello di gestire task, embodiment e dimensionalità delle azioni differenti.

TD-MPC viene studiato su 92 configurazioni di controllo continuo in **DMControl** e **Meta-World**, includendo osservazioni di stato, immagini e input multimodali. TD-MPC2 amplia la valutazione a 104 task di DMControl, Meta-World, **ManiSkill2** e **MyoSuite**, comprendendo locomozione, manipolazione robotica simulata e controllo muscoloscheletrico. Il modello multi-task più grande, da 317 milioni di parametri, viene addestrato su 545 milioni di transizioni provenienti dai replay buffer di 240 agenti e controlla 80 task con un'unica rete. Nessuno dei due lavori esegue esperimenti su robot fisici.

#### Novelty

- **TD-MPC:** unisce rollout latenti a breve orizzonte e bootstrap da un $Q$-value terminale, combinando i vantaggi locali del planning con la stima di lungo periodo del TD learning.
- **TD-MPC:** introduce il modello TOLD, addestrato congiuntamente mediante reward prediction, TD objective e consistenza latente senza ricostruire le osservazioni.
- **TD-MPC2:** rende l'algoritmo robusto a scale del reward e task differenti attraverso SimNorm, regressione discreta ed ensemble di value function.
- **TD-MPC2:** scala il modello implicito a centinaia di milioni di parametri e a 80 task eterogenei mediante task embedding, padding e mascheramento delle azioni.

#### Limiti

Entrambi i modelli apprendono dinamiche deterministiche e non stimano esplicitamente l'incertezza. Il latent è specializzato rispetto al reward disponibile: può ignorare proprietà necessarie a nuovi obiettivi e non permette di visualizzare o verificare direttamente il futuro previsto.

Il controllo richiede migliaia di rollout latenti per decisione e rimane progettato per action space continui. TD-MPC2 dipende inoltre da un identificatore del task, non dimostra generalizzazione zero-shot e non applica correzioni specifiche per l'offline RL al dataset multi-task. I risultati provengono interamente da simulatori e non misurano latenza, sicurezza o robustness nel mondo fisico.

L'[approfondimento su TD-MPC e TD-MPC2](models/td_mpc/README.md) sviluppa il modello TOLD, le loss di temporal difference e consistenza, il planner MPPI e le modifiche che rendono possibile lo scaling multi-task.

## Dal benchmark al robot fisico

I risultati in simulazione non garantiscono che un modello sappia gestire rumore sensoriale, latenza, contatti e distribuzioni che cambiano durante l'esecuzione. Questa parte farà da ponte tra il model-based reinforcement learning e il robot learning reale.

### DayDreamer (2023)

**DayDreamer** porta DreamerV2 dal controllo simulato al **reinforcement learning online su robot fisici**. Non introduce un nuovo algoritmo di base: apprende un Recurrent State-Space Model dalle transizioni raccolte dal robot, genera rollout nello spazio latente e usa tali traiettorie per addestrare actor e critic. Durante l'esecuzione l'azione è prodotta direttamente dall'actor, senza effettuare planning iterativo a ogni step.

Il World Model combina uno stato ricorrente deterministico con variabili latenti discrete. Un encoder fonde immagini e propriocezione, la dinamica predice come il latent cambia in funzione delle azioni, mentre decoder e reward model ricostruiscono rispettivamente le osservazioni e i reward. Il decoder fornisce un segnale di training ricco e permette di ispezionare le predizioni, ma non viene eseguito durante i rollout immaginati usati per apprendere il comportamento.

La raccolta dei dati e l'ottimizzazione sono disaccoppiate: un **actor thread** controlla il robot con bassa latenza, mentre un **learner thread** aggiorna in parallelo World Model, actor e critic usando il replay buffer. Il dataset non è quindi preparato in anticipo, ma cresce con l'esperienza reale dell'agente. Tutti gli esperimenti partono da zero, senza dimostrazioni e senza un simulatore, e usano lo stesso insieme di iperparametri.

La valutazione comprende quattro sistemi. Un quadrupede **Unitree A1**, controllato a 20 Hz da propriocezione e azioni continue, impara in circa un'ora a girarsi, alzarsi e camminare; dopo l'introduzione di spinte, adatta il comportamento in altri 10 minuti. Un **UR5** apprende da immagini RGB, propriocezione e reward sparsi un task multi-oggetto di pick-and-place, raggiungendo 2.5 oggetti al minuto dopo 8 ore. Un **XArm a 7 gradi di libertà** usa RGB, profondità e propriocezione e raggiunge 3.1 oggetti al minuto in 10 ore. Infine, uno **Sphero Ollie** impara in meno di 2 ore a raggiungere un goal fisso da sole immagini RGB, senza propriocezione.

#### Novelty

- Dimostra che **DreamerV2 può apprendere direttamente sull'hardware** locomozione, manipolazione e navigazione, senza simulatore o dimostrazioni.
- Introduce un'esecuzione **asincrona actor–learner** che continua ad addestrare le reti mentre il robot interagisce con l'ambiente e rimuove la necessità di fissare una frequenza di training rispetto ai passi reali.
- Valida lo stesso algoritmo e gli stessi iperparametri su azioni continue e discrete, reward densi e sparsi, propriocezione, RGB e profondità.
- Mostra apprendimento rapido di recovery e adattamento online: l'A1 parte sdraiato sulla schiena e l'XArm recupera dopo un forte cambiamento dell'illuminazione.

#### Limiti

L'apprendimento richiede comunque da una a dieci ore per task su un singolo robot e provoca usura, possibili guasti e necessità di supervisione. Il setup dell'A1 usa un filtro Butterworth per proteggere i motori e richiede che un operatore riporti fisicamente il robot nell'area di lavoro; i task di manipolazione includono vincoli e automatismi specifici, mentre lo Sphero viene riposizionato mediante azioni casuali ad alta potenza. **Assenza di simulatore non significa quindi assenza di ingegnerizzazione o intervento umano**.

I risultati coprono scenari controllati, goal e reward definiti manualmente e un numero ridotto di run. Il modello non fornisce garanzie di sicurezza né una stima esplicita dell'incertezza, e le ricostruzioni immaginate mostrano errori sugli oggetti. Inoltre ogni task addestra un agente separato: il lavoro non dimostra trasferimento tra embodiment, generalizzazione a nuovi obiettivi o superiorità sistematica sui metodi model-free, dato che DrQ-v2 ottiene prestazioni simili nella navigazione dello Sphero.

L'[approfondimento su DayDreamer](models/daydreamer/README.md) sviluppa l'RSSM discreto, l'actor–critic addestrato in imagination, l'architettura asincrona e il protocollo sperimentale dei quattro robot.

## Predizione visuale e robot imagination

In questa famiglia il futuro viene rappresentato come immagine o video. Una predizione visuale può fungere da goal intermedio, piano interpretabile o sorgente di supervisione, ma deve poi essere collegata ad azioni eseguibili mediante ottimizzazione, inverse dynamics o una policy dedicata.

### Deep Visual Foresight (2017)

**Deep Visual Foresight** è uno dei primi sistemi a collegare direttamente un modello di video prediction al controllo di un robot reale. Una convolutional LSTM apprende da video non etichettati a prevedere come i pixel si sposteranno in risposta ai comandi del braccio. Al momento del test, l'utente seleziona nell'immagine uno o più pixel appartenenti agli oggetti e le rispettive posizioni desiderate; il controller cerca quindi una sequenza di azioni che renda probabile tale spostamento.

Il dataset comprende 50,000 tentativi di pushing casuale raccolti con dieci bracci a 7 gradi di libertà, centinaia di oggetti e differenti posizioni della camera. Il modello predice trasformazioni di pixel mediante flow e maschere appresi senza annotazioni esplicite. Il **Cross-Entropy Method (CEM)** confronta sequenze candidate su un orizzonte di tre step, esegue la prima azione e ripianifica circa ogni 200 ms usando la nuova immagine. Nei test con oggetti non osservati durante il training, il visual MPC riduce la distanza finale media dal pixel target a 2.52 pixel, rispetto a 3.19 del migliore baseline reattivo.

#### Novelty

- Integra **video prediction action-conditioned e Model Predictive Control** in una pipeline eseguita su un robot reale.
- Formula il goal direttamente come spostamento di pixel, senza segmentazione, stima della posa degli oggetti o modello fisico esplicito.
- Mostra che flow e maschere utili al controllo possono emergere da video robotici non etichettati e trasferirsi a oggetti nuovi.

#### Limiti

Il sistema affronta soltanto pushing non prensile, con immagini da $64\times64$ pixel e un orizzonte di circa 800 ms. Richiede che un utente indichi manualmente pixel iniziali e target e usa optical flow esterno per seguirli durante il replanning. Occlusioni del braccio, errori nella dinamica degli oggetti e predizioni sfocate possono far ottimizzare al planner un futuro che non si realizza.

L'[approfondimento sintetico su Deep Visual Foresight](models/deep_visual_foresight/README.md) descrive il ciclo essenziale che collega dati autonomi, previsione del moto dei pixel e visual MPC.

### UniPi (2023)

**UniPi**, o *Universal Policy*, riformula il controllo come **generazione video condizionata dal testo**. Dall'osservazione iniziale $o_0$ e dall'istruzione $l$, un video diffusion model genera una traiettoria visuale che rappresenta come dovrebbe evolvere la scena; un inverse dynamics model specifico dell'ambiente converte poi i frame prodotti in azioni eseguibili. Video e linguaggio fungono così da interfacce condivise tra task, mentre la dipendenza dall'action space rimane confinata al modulo inverso.

Il generatore viene condizionato esplicitamente sul primo frame e lo replica come contesto durante il denoising di tutti i frame, tecnica chiamata **first-frame tiling**. Il planning è gerarchico: UniPi genera prima keyframe temporalmente radi e usa un secondo diffusion model di temporal super-resolution per interpolare una traiettoria più fine. Le azioni possono essere eseguite open loop oppure in closed loop, rigenerando il piano dopo una nuova osservazione.

Gli esperimenti di controllo sono svolti in simulazione. In un ambiente PyBullet combinatoriale, 200,000 video di un agente scripted insegnano a colorare e disporre blocchi secondo istruzioni linguistiche: UniPi raggiunge il 60.1% sui goal di posizionamento composti non visti e il 46.1% sulle nuove relazioni, contro valori non superiori al 13.2% per i baseline riportati. Un secondo studio usa 200,000 dimostrazioni da task CLIPort e valuta il trasferimento da dieci task di training a tre task esclusi, ottenendo rispettivamente 51.6%, 75.5% e 45.7% di successo.

Per il dominio reale, il generatore viene pre-addestrato su 14 milioni di coppie video–testo, 60 milioni di coppie immagine–testo e LAION-400M, poi adattato su 7,200 video di **Bridge**. Il pretraining migliora FID, FVD e il successo stimato da un classificatore dal 72.6% al 77.1%. Questa parte valuta però video generati: il paper non dimostra l'esecuzione closed loop delle azioni su un robot fisico.

#### Novelty

- Introduce la formulazione **policy-as-video**, usando linguaggio e immagini come spazio comune per goal e traiettorie tra ambienti differenti.
- Separa il planner visuale generale dall'**inverse dynamics model** dipendente dall'embodiment e dall'action space.
- Combina first-frame conditioning, tiling e temporal super-resolution per produrre piani coerenti con la scena iniziale e più lunghi dei singoli segmenti generati.
- Mostra generalizzazione composizionale delle istruzioni e trasferimento multi-task superiori a behavior cloning, Trajectory Transformer e action diffusion nei simulatori considerati.
- Esplora il trasferimento di conoscenza da grandi dataset video e immagine–testo non robotici verso la generazione di piani robotici.

#### Limiti

Generare un video fotorealistico può richiedere circa un minuto, una latenza incompatibile con gran parte del controllo reattivo; gli autori ottengono un'accelerazione preliminare di $16\times$ tramite distillazione, ma non eliminano il costo del sampling. Un video plausibile non è inoltre necessariamente **fisicamente raggiungibile**: il diffusion model non è vincolato dalla cinematica del robot e l'inverse dynamics separato deve tradurre anche transizioni fuori distribuzione.

L'universalità riguarda soprattutto la rappresentazione. Ogni action space richiede dati annotati e un inverse dynamics model dedicato; i risultati di esecuzione provengono da simulatori semplificati, mentre Bridge viene usato soltanto per valutare la generazione mediante metriche percettive e un success classifier. Gli ambienti sono quasi interamente osservabili e il modello può allucinare oggetti o movimenti quando la storia disponibile non determina univocamente il futuro.

L'[approfondimento su UniPi](models/unipi/README.md) sviluppa l'Unified Predictive Decision Process, la diffusione condizionata sul primo frame, il planning gerarchico, l'inverse dynamics e i tre protocolli sperimentali.

### RoboDreamer (2024)

**RoboDreamer** estende il paradigma *policy-as-video* di UniPi per migliorare la generalizzazione composizionale. Invece di condizionare un unico generatore sull'intera istruzione $l$, un parser la scompone in componenti come azione e oggetto, origine e relazione spaziale. Lo stesso video diffusion model viene interrogato separatamente con ciascuna componente e i relativi score vengono combinati durante il denoising. Un comando nuovo può così essere trattato come una combinazione di primitive già apprese.

Il generatore è una Video U-Net con convoluzioni spazio-temporali, temporal attention e cross-attention sulle condizioni. Produce otto frame a $64\times64$ pixel, poi due stadi di super-resolution li portano a $256\times256$. Il primo frame osservato viene replicato come contesto per mantenere stabile la scena. Goal image e sketch possono essere aggiunti alle componenti linguistiche mediante encoder visuali e partecipano alla stessa composizione degli score.

Il training visuale usa circa 70,000 dimostrazioni reali del dataset **RT-1**, con circa 500 task di manipolazione eseguiti dai robot Everyday Robots. La valutazione dei video si basa su giudizi umani: sulle istruzioni composte non viste RoboDreamer raggiunge 81.3 punti, rispetto a 50.1 di HiP, 46.9 di AVDC e 68.8 della propria variante senza parsing. Con un goal sketch il punteggio sale a 94.7 e con una goal image a 95.8, ma queste modalità forniscono informazione aggiuntiva sul risultato desiderato.

Per trasformare l'immaginazione in controllo, un inverse dynamics model riceve coppie di frame generati e lo stato corrente, quindi predice le azioni. L'esecuzione è valutata in simulazione su sei task di **RLBench** con un Franka Panda a 7 gradi di libertà e una sola camera frontale. RoboDreamer raggiunge un successo medio del 49.3%, contro 44.2% di Hiveformer, 41.0% di UniPi e 31.6% di Image-BC. Non vengono invece eseguite azioni sui robot fisici presenti nei video RT-1.

#### Novelty

- Introduce un **World Model video composizionale** che combina score diffusion condizionati su parti linguistiche anziché rappresentare l'intero comando come una condizione monolitica.
- Addestra le singole componenti e loro sottoinsiemi a denoisare la stessa traiettoria, rendendole ricombinabili in istruzioni non presenti nel training.
- Estende la composizione a **goal image e sketch**, che specificano relazioni spaziali difficili da esprimere soltanto mediante testo.
- Mostra un miglioramento sia nell'allineamento dei video a istruzioni nuove sia nel successo closed loop su task simulati di manipolazione.

#### Limiti

La generalizzazione è combinatoriale, non aperta: azioni, oggetti e relazioni individuali devono essere già in distribuzione e il parser rule-based deve scomporre correttamente l'istruzione. Componenti incompatibili possono inoltre produrre score in conflitto, mentre interrogare più volte il diffusion model aumenta il costo del sampling.

Il modello non è condizionato sulle azioni candidate e un video plausibile può quindi essere cinematicamente irrealizzabile. L'inverse dynamics attenua questo divario ma viene addestrato separatamente. Gli autori riportano inoltre difficoltà con viste multiple, camere in movimento e molte immagini reali; l'unica esecuzione quantitativa avviene in RLBench su sei task, usando macro-step e contatti simulati.

L'[approfondimento su RoboDreamer](models/robodreamer/README.md) sviluppa la composizione probabilistica degli score, l'integrazione delle condizioni multimodali, l'architettura diffusion e i protocolli RT-1 e RLBench.

## World Model integrati nella policy

I sistemi più recenti riducono la separazione tra previsione e controllo. Il World Model può condividere il backbone con la policy, interagire con un action expert o comparire come obiettivo ausiliario che induce rappresentazioni sensibili alla dinamica. La sezione confronterà pipeline disaccoppiate, **single-backbone**, Mixture-of-Experts o Mixture-of-Transformers, VLA unificati e modelli predittivi latenti.

### GR-1 (2024)

**GR-1** trasferisce alla manipolazione robotica un principio dei modelli generativi: apprendere prima la struttura temporale di molti video e adattare poi lo stesso backbone alla previsione delle azioni. Un Transformer causale GPT-style riceve l'istruzione $l$, una storia di osservazioni $o_{t-h:t}$ e gli stati del robot $q_{t-h:t}$; da token dedicati predice congiuntamente l'azione $a_t$ e un'immagine futura $o_{t+\Delta}$. La previsione visuale non viene quindi convertita in azioni da un modulo separato, ma agisce come **obiettivo ausiliario interno alla policy**.

Il pretraining usa **Ego4D**, da cui gli autori ricavano 800,000 clip linguaggio–video di tre secondi, pari a circa 8 milioni di frame. GR-1 impara a ricostruire il frame futuro a partire dal testo e dai frame precedenti; durante il fine-tuning robotico conserva questo compito e aggiunge le loss sulle azioni. CLIP codifica il linguaggio, un ViT pre-addestrato con MAE codifica le immagini, un Perceiver Resampler ne riduce i token e due decoder producono rispettivamente patch visuali e comandi continui per il braccio e binari per la pinza.

In simulazione il modello usa il dataset **CALVIN**, che comprende 34 task con un Franka Emika Panda. Sullo split ABCD→D completa almeno il primo task nel 94.9% delle sequenze e raggiunge una lunghezza media di 4.21 task su cinque, contro 88.9% e 3.06 del migliore baseline riportato. Nello split ABC→D, che riserva una scena non vista alla valutazione, ottiene 85.4% e 3.06, mentre i baseline non superano 53.3% sul primo task e 0.93 di lunghezza media. Le ablation mostrano che sia la predizione video durante il fine-tuning sia il pretraining su Ego4D contribuiscono al risultato.

Gli esperimenti fisici impiegano un **Kinova Gen2 a 7 gradi di libertà**, una RealSense sul polso e una Kinect Azure esterna. Su trasporto di oggetti e apertura o chiusura di un cassetto, GR-1 raggiunge rispettivamente il 79% sugli oggetti visti, il 73% su nuove istanze, il 30% su categorie non presenti nei dati robotici e il 75% nella manipolazione articolata. Il forte calo sulle categorie nuove segnala che il prior video migliora la robustezza, ma non risolve la generalizzazione semantica aperta.

#### Novelty

- Introduce un **unico Transformer causale** per pretraining video e fine-tuning della policy, senza un World Model o inverse dynamics separati.
- Mantiene la **predizione dei frame futuri** accanto alla behavior cloning, inducendo rappresentazioni condivise tra dinamica visuale e controllo.
- Trasferisce 800,000 clip egocentriche Ego4D alla manipolazione mediante una sequenza comune di token multimodali.
- Verifica separatamente il contributo dell'obiettivo video e del pretraining attraverso ablation in-distribution, cross-scene e low-data.

#### Limiti

GR-1 predice il futuro osservato nelle dimostrazioni, ma non modella come futuri diversi dipendano da azioni candidate. Non esegue planning, non confronta rollout alternativi e apprende le azioni mediante behavior cloning; chiamarlo World Model richiede quindi di precisare che la componente predittiva è una rappresentazione ausiliaria della policy, non un simulatore controllabile.

La ricostruzione pixel-wise perde dettagli e oggetti occlusi, mentre gli esperimenti fisici coprono soltanto due famiglie di skill in una configurazione controllata. Il modello dipende inoltre da encoder CLIP e MAE pre-addestrati, usa solo clip Ego4D provviste di annotazione linguistica e conserva una finestra temporale locale: non dimostra memoria a lungo termine, adattamento online o trasferimento affidabile a embodiment e ambienti molto differenti.

L'[approfondimento su GR-1](models/gr_1/README.md) ricostruisce la tokenizzazione multimodale, le due fasi di training, le loss congiunte e i protocolli CALVIN e real robot, chiarendo in quale senso la previsione video supporta la policy.

### Unified World Models (2025)

**Unified World Models (UWM)** accoppia action diffusion e video diffusion nello stesso Transformer. Il modello apprende la distribuzione congiunta fra osservazione corrente $o_t$, un chunk di azioni e osservazioni future, ma assegna alle due modalità timestep di rumore indipendenti. Questa scelta consente di decidere durante l'inference quale variabile mantenere pulita, quale denoisare e quale marginalizzare: lo stesso checkpoint può così operare come policy, forward dynamics, inverse dynamics o video predictor.

L'architettura codifica le osservazioni correnti con ResNet-18 e le usa per modulare mediante AdaLN un Diffusion Transformer da 12 layer. Le immagini future vengono compresse nello spazio latente del VAE di SDXL e suddivise in patch spazio-temporali; i chunk di azioni sono convertiti in token da un MLP. Otto register token facilitano lo scambio di informazione fra le due modalità. Durante l'esecuzione, UWM genera 16 azioni con dieci step DDIM, ne applica le prime otto e ripianifica.

Il pretraining reale usa 2,000 traiettorie **DROID** con azioni; il co-training aggiunge altre 2,000 traiettorie private delle annotazioni di azione. Il modello viene poi adattato a cinque task con un Franka Panda e tre camere, usando da 50 a 150 dimostrazioni per task. Mediando le nove condizioni in-distribution e OOD riportate, UWM passa dal 73.3% dopo il solo pretraining all'81.0% con i video action-free, rispetto al 44.1% di Diffusion Policy e al 55.3% di GR-1 senza co-training. In simulazione, il pretraining su 4,500 traiettorie LIBERO-90 seguito dal fine-tuning di cinque task LIBERO-10 produce il 79% medio, contro il 71% di Diffusion Policy.

UWM non utilizza istruzioni linguistiche e i task downstream vengono appresi mediante fine-tuning separato. Il World Model non è inoltre impiegato per scegliere azioni attraverso rollout: la forward dynamics è verificata qualitativamente, mentre l'inverse dynamics viene usata per seguire traiettorie expert. L'unificazione riguarda quindi la rappresentazione e l'inferenza delle distribuzioni azione–video, non un sistema completo di model-based planning.

#### Novelty

- Introduce **timestep di diffusione indipendenti** per azioni e osservazioni future in un unico modello congiunto.
- Ricava policy, forward dynamics, inverse dynamics e video prediction modificando soltanto rumore e condizionamento in inference.
- Integra dati robotici annotati e video action-free con lo stesso obiettivo, mascherando le azioni mancanti mediante rumore completo.
- Mostra che la previsione delle osservazioni future trasferisce meglio della ricostruzione dell'osservazione corrente.

#### Limiti

La flessibilità probabilistica non dimostra una comprensione causale identificata dai dati: il modello viene addestrato su dimostrazioni osservative e non su interventi sistematici. Le predizioni visuali presentano artefatti e non vengono validate come simulatore per planning o policy evaluation.

Il modello è privo di linguaggio, memoria lunga e rappresentazione 3D esplicita. Il co-training principale usa video robotici dello stesso embodiment; i video Internet producono soltanto miglioramenti modesti nell'appendice. L'inferenza diffusion richiede inoltre più passaggi di denoising e gli esperimenti reali coprono cinque task in un unico setup Franka, con protocolli facilitati da più tentativi per configurazione.

L'[approfondimento su Unified World Models](models/unified_world_models/README.md) sviluppa il coupled diffusion objective, la tabella delle quattro modalità di inference, l'architettura latente e i protocolli DROID, LIBERO e video action-free.

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
