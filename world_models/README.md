# World Models

Un **World Model** è un modello predittivo che rappresenta **come un ambiente evolve nel tempo** e come tale evoluzione dipende dalle azioni di un agente.

In forma generale, dato lo stato corrente $s_t$, una sequenza di azioni $a_{t:t+H-1}$ e un orizzonte di predizione $H$, il modello cerca di descrivere una distribuzione sui possibili stati futuri:

$$
p(s_{t+1:t+H}\mid s_t,a_{t:t+H-1})
$$

Nei sistemi embodied lo stato reale non è normalmente accessibile. 

Il modello riceve osservazioni $o_t$, come immagini, propriocezione o segnali tattili, e può quindi predire direttamente **osservazioni future** oppure costruire uno **stato latente** che conserva le informazioni necessarie per previsione e controllo. 

Nei sistemi condizionati dal linguaggio, un'istruzione $l$ può inoltre specificare il task o il futuro desiderato:

$$
p(o_{t+1:t+H}\mid o_{\leq t},a_{t:t+H-1},l)
$$

La predizione non costituisce però il fine ultimo.

Un World Model diventa utile a un agente quando permette di **anticipare le conseguenze delle azioni**, confrontare piani alternativi, apprendere una policy su esperienze immaginate, generare nuovi dati o valutare un comportamento senza eseguirlo immediatamente nel mondo reale.

## Tassonomia

### Spazio dello stato

Un world model può essere classificato in base all'**spazio dello stato** in cui opera, si possono identificare i modelli che:

- **Predicono variabili fisiche** strutturate
- **Generano immagini o video**
- Operano in uno **spazio latente**

La scelta determina quali dettagli vengono conservati, quanto costa effettuare un rollout e quanto facilmente la predizione può essere impiegata per il controllo.

### Modelli deterministici e stocastici

Una singola azione può produrre esiti differenti a causa di informazione incompleta, rumore e interazioni non osservate. Verranno quindi distinti i transition model deterministici dai modelli probabilistici capaci di rappresentare **futuri multimodali**, separando per quanto possibile incertezza aleatorica e incertezza epistemica.

### Ruoli nel ciclo decisionale

Impieghi principali: 

- Planning mediante rollout
- Apprendimento della policy in imagination
- Simulazione ed evaluation
- Generazione di dati sintetici

La stessa architettura può ricoprire più ruoli, ma richiede proprietà diverse a seconda che debba generare immagini realistiche, preservare la causalità delle azioni o classificare correttamente il successo di un comportamento.

## Dinamiche latenti e controllo in imagination

Il confronto chiarirà la differenza tra **planning online**, eseguito a ogni passo, e **learning in imagination**, nel quale il modello viene sfruttato soprattutto durante il training.

### World Models (2018)

**World Models** mostra come separare la complessità dell'apprendimento in un grande **modello predittivo** e in un **controller** molto piccolo. 

1. Le osservazioni RGB vengono compresse da un **Variational Autoencoder** (VAE) in un vettore latente $z_t$
2. Un **Mixture Density Network-Recurrent Neural Network** (MDN-RNN) usa $z_t$, l'azione $a_t$ e la propria memoria $h_t$ per modellare una distribuzione sul latent successivo
3. Un controller lineare trasforma infine $z_t$ e $h_t$ nell'azione

Il paper considera due ambienti OpenAI Gym simulati, senza robot fisici. In **CarRacing-v0** il modello viene addestrato su 10,000 rollout di una policy casuale e fornisce al controller una rappresentazione spazio-temporale appresa dai pixel.

In **VizDoom: Take Cover** il ruolo del World Model è più radicale: oltre al prossimo latent, l'MDN-RNN predice la terminazione dell'episodio e diventa un **ambiente virtuale nel quale il controller viene addestrato** interamente con CMA-ES. La policy viene poi trasferita senza ulteriore training al simulatore originale, dove raggiunge $1,092\pm556$ step di sopravvivenza rispetto alla soglia di 750 richiesta dal task.

#### Novelty

Il contributo centrale è la decomposizione **Vision–Memory–Controller** e la dimostrazione che una policy può essere ottimizzata dentro la dinamica appresa e successivamente trasferita nell'ambiente reale del benchmark. 

Il lavoro rende inoltre esplicita un'idea destinata a diventare centrale: il controller non deve necessariamente ricostruire o comprendere ogni dettaglio dell'osservazione, ma può agire usando una **rappresentazione compatta del presente e del futuro probabile**.

#### Limiti

VAE, MDN-RNN e controller vengono addestrati separatamente, perciò la rappresentazione visuale **non è ottimizzata per preservare ciò che è rilevante al task**. 

I dati provengono da **policy casuali e da ambienti semplici**; gli esperimenti non affrontano robot reali, contatti complessi o generalizzazione tra task.

Il controller può inoltre trovare **adversarial policy che sfruttano gli errori del World Model** e visitano stati fuori distribuzione. Il paper attenua il problema aumentando la temperatura del modello stocastico, ma non lo risolve in modo generale.

L'[approfondimento su World Models](models/world_models/README.md) sviluppa architettura, obiettivi di training, esperimenti CarRacing e VizDoom e il problema del model exploitation.

### PlaNet (2019)

**PlaNet**, acronimo di *Deep Planning Network*, apprende dai pixel un modello della dinamica e lo usa per scegliere le azioni mediante planning online. 

A differenza di World Models, **non addestra un controller separato**: a ogni step valuta sequenze di azioni nel modello appreso, esegue soltanto la prima azione e pianifica nuovamente dopo la successiva osservazione.

Il cuore dell'architettura è il **Recurrent State-Space Model (RSSM)**. Lo stato latente combina una memoria deterministica $h_t$, che conserva informazione nel tempo, e una variabile stocastica $s_t$, che rappresenta incertezza e futuri multipli. 

Un observation model ricostruisce $o_t$ per fornire un segnale di training ricco, mentre un reward model predice $r_t$ direttamente dal latent. Durante il planning le immagini non vengono decodificate: il **Cross-Entropy Method (CEM)** cerca le azioni usando soltanto transizioni e reward latenti.

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

### Dreamer (2020)

**Dreamer** sostituisce il planning online di PlaNet con una **policy appresa in imagination**. 

Un RSSM con stati stocastici gaussiani apprende dinamica e reward dalle immagini; actor e critic vengono poi addestrati su traiettorie latenti generate dal modello. Il critic stima il return oltre l'orizzonte immaginato, mentre l'actor riceve gradienti che attraversano azioni, transizioni, reward e value predetti. Durante l'esecuzione basta aggiornare lo stato latente con l'osservazione e interrogare l'actor.

Il dataset è un **replay buffer raccolto online**, inizializzato con cinque episodi casuali, senza dimostrazioni expert. Il confronto principale copre 20 task visuali della DeepMind Control Suite, con embodiment simulati di locomozione e manipolazione e immagini RGB da $64\times64$ pixel. Dopo 5 milioni di step Dreamer ottiene un return medio di circa 823, rispetto a circa 333 di PlaNet allo stesso budget e 786 di D4PG dopo 100 milioni di step. Non vengono utilizzati robot fisici. [Fonte: Dream to Control](https://arxiv.org/pdf/1912.01603).

#### Novelty

- Apprende actor e critic **interamente da rollout latenti**, sostituendo la ricerca di azioni a ogni decisione.
- Propaga **gradienti analitici del return attraverso la dinamica appresa**, sfruttando la riparametrizzazione delle variabili continue.
- Usa il **value bootstrap e i $\lambda$-return** per apprendere comportamenti con conseguenze oltre l'orizzonte immaginato.

#### Limiti

La policy ottimizza il futuro predetto dal modello e può quindi sfruttarne gli errori. La ricostruzione visuale non garantisce che il latent preservi tutte le proprietà rilevanti per il controllo; manca inoltre una stima esplicita dell'incertezza epistemica. I risultati principali riguardano task simulati e agenti addestrati separatamente, senza dimostrare trasferimento tra embodiment o controllo fisico.

L'[approfondimento su Dreamer](models/dreamer/README.md) sviluppa RSSM gaussiano, training del modello, gradienti in imagination, stima del return e confronto con PlaNet.

### DreamerV2 (2021)

**DreamerV2** mantiene il ciclo modello–imagination–interazione, ma sostituisce i latent gaussiani con **variabili categoriche** e introduce il **KL balancing**: il prior viene addestrato ad avvicinarsi al posterior più intensamente di quanto il posterior venga regolarizzato verso il prior. I campioni discreti usano gradienti straight-through; per l'actor, Atari beneficia invece dell'estimatore REINFORCE con baseline del critic.

Il replay buffer viene raccolto online nei **55 giochi Atari** del protocollo del paper, con sticky actions, osservazioni visuali e 200 milioni di frame per gioco. DreamerV2 supera il livello umano secondo il punteggio mediano normalizzato e i baseline IQN e Rainbow nel confronto su singola GPU. Il lavoro include anche Humanoid Walk e Humanoid Stand della DeepMind Control Suite da pixel: l'embodiment è un umanoide simulato, senza esperimenti su robot reali. [Fonte: Mastering Atari with Discrete World Models](https://arxiv.org/pdf/2010.02193).

#### Novelty

- Introduce nell'RSSM **32 variabili categoriche con 32 classi ciascuna**, apprendibili mediante gradienti straight-through.
- Separa con il **KL balancing** l'apprendimento della dinamica dalla pressione che rende le rappresentazioni predicibili.
- Dimostra prestazioni di livello umano su Atari mediante **behavior learning in un World Model addestrato separatamente dalla policy**.
- Confronta gradienti REINFORCE e gradienti attraverso la dinamica, mostrando che la scelta efficace dipende dal tipo di action space.

#### Limiti

Il risultato aggregato non implica prestazioni umane in ogni gioco, e il budget di 200 milioni di frame resta elevato. Training del modello e rollout immaginati richiedono molto calcolo. Alcuni iperparametri cambiano tra Atari e controllo continuo; gli stati discreti non risolvono da soli errori di dinamica, esplorazione difficile o trasferimento al mondo fisico.

L'[approfondimento su DreamerV2](models/dreamerv2/README.md) approfondisce rappresentazioni discrete, KL balancing, actor-critic e protocollo Atari.

### DreamerV3 (2023)

**DreamerV3** rende l'apprendimento in imagination più robusto a domini con osservazioni, reward e action space differenti. Conserva l'RSSM discreto, ma combina **trasformazioni symlog, predizioni distribuzionali di reward e value, free bits, regolarizzazione delle distribuzioni categoriche e normalizzazione dei return**. L'obiettivo è riutilizzare la stessa configurazione dell'algoritmo senza doverla adattare manualmente a ogni benchmark.

Il paper, introdotto nel 2023 e qui analizzato nella revisione arXiv del 2024, valuta oltre **150 task in otto domini**: Atari, ProcGen, DMLab, Atari100k, controllo propriocettivo e visuale della DeepMind Control Suite, BSuite e Minecraft. I dati vengono raccolti online da ciascun agente; gli embodiment di controllo rimangono simulati. In Minecraft il modello apprende a ottenere diamanti senza dimostrazioni umane né curriculum. **Configurazione condivisa non significa policy universale**: gli agenti vengono addestrati separatamente nei diversi task. [Fonte: Mastering Diverse Domains through World Models, revisione v2](https://arxiv.org/pdf/2301.04104v2).

#### Novelty

- Combina trasformazioni e **loss distribuzionali** per gestire target di ordini di grandezza differenti.
- Integra **KL bilanciata, free bits e uniform mixing** per stabilizzare le rappresentazioni discrete tra domini.
- Normalizza la scala dei return con **percentili e un denominatore minimo**, mantenendo un coefficiente di entropia condiviso.
- Verifica robustezza su oltre 150 task e benefici dello scaling di dimensione del modello e replay ratio.
- Dimostra l'apprendimento del task Minecraft Diamond da interazione, senza dati expert o curriculum.

#### Limiti

La robustezza tra benchmark non equivale a generalizzazione zero-shot, apprendimento multi-task in un unico modello o garanzie sulla dinamica. Il replay ratio e la dimensione delle reti continuano a determinare il costo computazionale. Gli esperimenti non verificano latenza, sicurezza o adattamento sui robot fisici; Minecraft mantiene inoltre un reward definito dall'ambiente, con segnali intermedi legati alla progressione.

L'[approfondimento su DreamerV3](models/dreamerv3/README.md) sviluppa le tecniche di stabilizzazione, i protocolli dei diversi domini e il significato dello scaling.

### MuZero (2020)

**MuZero** combina un modello della dinamica appreso con **Monte Carlo Tree Search (MCTS)**, senza cercare di ricostruire le osservazioni o di imitare lo stato reale dell'ambiente. 

1. Una **representation function** trasforma la storia osservata in uno **stato latente**
2. Una dynamics function **applica un'azione ipotetica e produce il latent successivo e il reward**
3. Una **prediction function stima policy e value**

La rappresentazione viene quindi **ottimizzata soltanto per conservare le informazioni utili alla decisione**.

Durante l'esecuzione, MCTS costruisce un albero nello spazio latente usando policy e value per guidare la ricerca e i reward predetti per valutarne i rami. 

Durante il training, le tre funzioni vengono apprese congiuntamente su sequenze reali: policy e value devono riprodurre i target migliorati dalla ricerca, mentre il reward deve corrispondere a quello osservato. Si crea così un ciclo di **search-based policy iteration** nel quale il modello migliora la ricerca e la ricerca fornisce target migliori al modello.

Gli esperimenti coprono Go, scacchi e shogi mediante self-play e tutti i 57 giochi dell'**Arcade Learning Environment** a partire da frame RGB. Non sono coinvolti robot fisici. Nei giochi da tavolo MuZero eguaglia le prestazioni sovrumane di AlphaZero pur sostituendo il simulatore perfetto con una dinamica appresa; su Atari supera i precedenti risultati model-free e model-based del confronto originale. La variante MuZero Reanalyze raggiunge un punteggio mediano normalizzato rispetto all'essere umano del 731.1% usando 200 milioni di frame per gioco.

#### Novelty

- Introduce un **modello value-equivalent**, libero di rappresentare soltanto reward, policy e value necessari al planning anziché predire pixel o stati reali.
- Sostituisce nel tree search le transizioni note di AlphaZero con una **dinamica latente appresa end-to-end** dalle interazioni.
- Estende la search-based policy iteration a domini single-agent con reward intermedi, sconto e osservazioni visuali complesse.
- Unifica representation, dynamics e prediction in un obiettivo multi-step supervisionato dai risultati della ricerca e dai reward osservati.

#### Limiti

Il modello è **deterministico e non rappresenta esplicitamente l'incertezza o futuri multimodali**. Il suo latent non deve essere interpretabile né fisicamente fedele: è adatto a scegliere azioni secondo il reward di training, ma non costituisce un simulatore generativo interrogabile per altri scopi.

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

**WorldVLA** unifica policy e World Model in un Transformer autoregressivo inizializzato da **Chameleon**. Immagini, testo e azioni vengono discretizzati in un vocabolario condiviso: il modello genera azioni dall'istruzione $l$ e dalla storia visuale, oppure predice il prossimo frame a partire da immagini e azioni. Le due modalità condividono i parametri e vengono addestrate con loss di cross-entropy sui rispettivi token. Una maschera di attenzione impedisce alle azioni successive di un chunk di dipendere dalle azioni precedenti, riducendo la propagazione degli errori.

Il training robotico utilizza dimostrazioni **LIBERO**, con un Franka Emika Panda simulato e azioni a sette componenti per posa relativa dell'end-effector e pinza. Il paper non usa pretraining su grandi dataset robotici e non presenta esperimenti su robot fisici. A risoluzione $512\times512$ raggiunge l'81.8% medio sulle quattro suite LIBERO, contro il 76.5% di OpenVLA e il 95.4% di OpenVLA-OFT nel confronto riportato. Le ablation a backbone condiviso mostrano un beneficio del World Model per la policy e della maschera per l'action chunking. [Fonte: WorldVLA](https://arxiv.org/pdf/2506.21539v1).

#### Novelty

- Unifica **comprensione e generazione di immagini e azioni** in un backbone autoregressivo con token condivisi.
- Addestra congiuntamente policy e **predizione visuale condizionata dalle azioni**, verificandone i benefici reciproci.
- Introduce una **maschera di attenzione per gli action chunk** che limita la dipendenza dagli errori delle azioni precedenti.
- Confronta l'utilità della forward prediction action-conditioned con la video prediction condizionata soltanto dal task.

#### Limiti

Il World Model migliora il training della policy, ma gli esperimenti non dimostrano un planner che confronti online futuri alternativi mediante reward o value. Le metriche video non garantiscono causalità fisica e non migliorano uniformemente a tutti gli orizzonti. Tokenizzazione, costo autoregressivo e bassa reattività dei chunk lunghi restano vincoli; i risultati riguardano soltanto manipolazione simulata.

L'[approfondimento su WorldVLA](models/worldvla/README.md) sviluppa tokenizzazione, training congiunto, maschere di attenzione e risultati LIBERO, distinguendo policy diretta e planning potenziale.

### Cosmos Policy (2026)

**Cosmos Policy** adatta **Cosmos-Predict2-2B**, un video diffusion model pre-addestrato, al controllo robotico mediante **latent frame injection**. Propriocezione, chunk di azioni e value vengono normalizzati e inseriti come frame nella sequenza latente accanto alle immagini multi-view. Il modello apprende con lo stesso denoising a generare azioni, osservazioni future e return, senza aggiungere un action expert o un inverse dynamics model.

Il post-training usa 2,000 dimostrazioni **LIBERO**, 1,200 dimostrazioni umane **RoboCasa** e, per gli esperimenti fisici, 185 dimostrazioni distribuite su quattro task **ALOHA**. I benchmark simulati usano un Franka Panda; ALOHA impiega due bracci ViperX 300 S, tre camere e controllo a 25 Hz. Come policy diretta, Cosmos Policy raggiunge il 98.5% su LIBERO, il 67.1% su RoboCasa e uno **score medio di completamento del 93.6%** su ALOHA. Quest'ultima metrica comprende completamento parziale e non coincide con una percentuale di episodi interamente riusciti. [Fonte: Cosmos Policy](https://arxiv.org/pdf/2601.16163v1).

Il **planning online best-of-N** richiede un ulteriore adattamento su 648 rollout, inclusi fallimenti: un checkpoint propone azioni e un secondo ne predice futuro e value. Su condizioni più difficili di due task ALOHA, il confronto passa dal 59.5 al 72.0 di score medio, un aumento di 12.5 punti.

#### Novelty

- Trasforma azioni, propriocezione e value in **frame latenti compatibili con un video backbone pre-addestrato**.
- Unifica policy, forward dynamics e value modificando il condizionamento nel denoising.
- Separa l'uso come **policy diretta** dall'uso per **planning con futuri e value predetti**.
- Integra rollout della policy, oltre alle dimostrazioni, per apprendere esiti e fallimenti utili alla selezione delle azioni.

#### Limiti

Il planner esplora un solo livello di chunk e non una sequenza profonda di decisioni. Il training da sole dimostrazioni non copre adeguatamente i fallimenti. Negli esperimenti di planning, otto H100 producono un chunk in circa 4.9 secondi; anche la policy diretta su ALOHA introduce una pausa di circa 0.95 secondi. Il modello usa osservazioni correnti e un futuro all'estremità del chunk, senza memoria esplicita di una lunga storia.

L'[approfondimento su Cosmos Policy](models/cosmos_policy/README.md) sviluppa latent injection, obiettivi congiunti, adattamento sui rollout, best-of-N e protocolli di valutazione.

### VLA-JEPA (2026)

**VLA-JEPA** usa la **predizione di stati latenti futuri** per pre-addestrare una policy, evitando la ricostruzione dei pixel. Un encoder **V-JEPA2 congelato** produce i target; **Qwen3-VL-2B** riceve osservazione corrente e istruzione $l$ e genera latent action token che condizionano un World Model temporale. I frame futuri supervisionano la predizione, ma non entrano nel VLM che produce tali token. Nel fine-tuning robotico, una testa flow matching apprende i comandi continui mantenendo l'obiettivo di allineamento latente.

Il pretraining combina circa **220,000 video Something-Something-v2** senza comandi robotici e **76,000 traiettorie DROID** con azioni. Il fine-tuning usa LIBERO, Fractal e BridgeV2; i test coprono LIBERO e LIBERO-Plus con Franka Panda simulato, SimplerEnv con Google Robot e WidowX, e un **Franka Research 3 con Robotiq 2F-85** fisico, addestrato su 100 dimostrazioni. Il paper riporta il 97.2% medio su LIBERO e il 79.5% su LIBERO-Plus, contro il 96.1% e il 62.9% della variante senza video umani. [Fonte: VLA-JEPA, revisione v2](https://arxiv.org/pdf/2602.10098v2).

#### Novelty

- Apprende rappresentazioni di transizione mediante **allineamento con stati V-JEPA2**, senza decoder visuale.
- Separa il percorso che produce i latent action token dai frame futuri, riducendo gli shortcut da **information leakage**.
- Integra video umani action-free e dati robotici annotati in un pretraining con obiettivi compatibili.
- Mantiene la supervisione dinamica durante il fine-tuning di una testa di azione flow matching.

#### Limiti

Il latent action non è un comando fisico identificato e il World Model non viene valutato come planner online o simulatore di interventi robotici arbitrari. I video umani non migliorano ogni benchmark: su SimplerEnv Google Robot la variante senza video raggiunge il 78.4%, contro il 65.2% del modello completo. Sul robot reale il successo scende dal 70% in-distribution al 17% sui task OOD e al 47% sulle nuove disposizioni, indicando limiti nel grounding linguistico e nel trasferimento dei task.

L'[approfondimento su VLA-JEPA](models/vla_jepa/README.md) chiarisce isolamento dei target futuri, teacher forcing, flow matching, protocolli e differenze fra robustezza percettiva e controllo causale.

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
