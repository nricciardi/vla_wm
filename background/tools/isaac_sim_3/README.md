# Simulazione robotica con Isaac Lab e Isaac Sim

La **simulazione robotica** permette di studiare l'interazione tra un robot e il suo ambiente attraverso un modello computazionale di corpi, giunti, contatti e sensori. Nel robot learning costituisce un laboratorio controllabile: si possono ripetere condizioni iniziali, raccogliere traiettorie e valutare policy prima di trasferirle su hardware fisico.

**Isaac Lab** organizza questi esperimenti in ambienti di apprendimento, mentre **Isaac Sim** mette a disposizione un'applicazione di simulazione con strumenti per scene 3D, sensori e rendering. Comprendere la distinzione evita di confondere il mondo simulato, l'interfaccia che lo espone alla policy e l'algoritmo che apprende dall'interazione.

Il percorso parte dai concetti di base e non richiede familiarità con ManiSkill o altri simulatori. I capitoli sul [controllo robotico](../../robot_control/README.md) e sul [Reinforcement Learning](../../reinforcement_learning/README.md) forniscono ulteriori approfondimenti su controller, reward e ottimizzazione delle policy.

## Componenti e ciclo di interazione

Un esperimento collega una descrizione della scena, un motore fisico, un sistema di osservazione e una policy. Il motore aggiorna lo stato del mondo in risposta ai comandi; i sensori ne rendono disponibile una parte; la policy usa queste osservazioni per scegliere l'azione successiva. Il renderer produce le immagini, ma non determina il movimento dei corpi.

Il [capitolo sui concetti fondamentali](concepts/README.md) introduce questa architettura, chiarisce il ruolo di USD e distingue stato, osservazione, azione e tempi di simulazione.

## Installazione ed esecuzione locale

La scelta dell'ambiente software dipende dalle funzionalità richieste. Un esperimento basato su stati articolari ha esigenze diverse da una policy che riceve immagini RTX o manipola un oggetto deformabile. La disponibilità di una GPU, da sola, non assicura la compatibilità con ogni combinazione di motore fisico e rendering.

Il [capitolo sull'installazione locale](local_installation/README.md) spiega dipendenze, configurazione e verifiche progressive, fino all'avvio di un ambiente di controllo semplice.

## Interfaccia Gymnasium e ambienti paralleli

Gymnasium descrive l'interazione tramite inizializzazione e avanzamento dell'ambiente. In Isaac Lab molte copie dello stesso task vengono eseguite insieme: osservazioni, azioni e reward contengono quindi una dimensione di batch. Questa organizzazione aumenta la raccolta di esperienza, ma richiede di distinguere la fine di un singolo episodio dall'arresto dell'intera simulazione.

Il [capitolo su Gymnasium](gymnasium/README.md) sviluppa il ciclo `reset`/`step`, il significato delle terminazioni e la lettura degli stati di un Cartpole, costituito da un carrello con un pendolo articolato.

## Scene, oggetti e proprietà fisiche

Una geometria visibile non è necessariamente un oggetto fisico. Per ottenere una scena coerente occorre specificare collisioni, comportamento statico o dinamico, massa e materiali di contatto. Queste scelte determinano quali interazioni siano possibili e quali comportamenti la policy possa apprendere.

Il [capitolo su scene e oggetti](scenes_and_objects/README.md) usa un tavolo e tre cubi per spiegare costruzione della scena, unità di misura, attrito e reset degli stati.

## Progettazione di un task

Un task aggiunge alla scena una definizione del problema: quali informazioni riceve l'agente, quali comandi può impartire, che cosa viene premiato e quando un episodio termina. La correttezza di questa specifica conta quanto quella della fisica, perché una policy può ottimizzare una reward senza realizzare l'obiettivo desiderato.

Il [capitolo sugli ambienti personalizzati](custom_tasks/README.md) introduce i workflow Direct e manager-based, interpreta l'esempio di reward modificata e descrive il passaggio verso un task di manipolazione.

## Materiali deformabili

Gomma, tessuti e cavi richiedono rappresentazioni più ricche della posa di un corpo rigido. Discretizzazione, modello del materiale e solver diventano parte sostanziale dell'esperimento: una resa visiva plausibile non basta a garantire una risposta fisica adeguata.

Il [capitolo sui deformabili](deformables/README.md) distingue solidi, superfici e cavi, chiarisce i parametri principali e presenta un percorso di verifica basato sulle demo ufficiali. Include inoltre una scena Python con tavolo e oggetti configurabili per forma o asset USD, massa, attrito e proprietà elastiche.

## Apprendimento e scalabilità

Durante il training la policy raccoglie esperienza e aggiorna i propri parametri. Il parallelismo degli ambienti accelera questa raccolta; il training distribuito aggiunge invece processi su più GPU e sincronizzazione degli aggiornamenti. Maggiore throughput non implica automaticamente maggiore qualità della policy.

Il [capitolo sul training](training/README.md) collega rollout, PPO, checkpoint e metriche di prestazione, distinguendo velocità di simulazione ed efficacia dell'apprendimento.

## Camere e osservazioni visuali

Una policy visuomotoria deve ricevere effettivamente le immagini del sensore. La finestra usata per osservare la simulazione può avere una prospettiva diversa e non coincide necessariamente con l'input del modello. Risoluzione, frequenza, calibrazione e randomizzazione visuale influenzano sia il costo sia il trasferimento al mondo reale.

Il [capitolo su camere e rendering](cameras_and_rendering/README.md) sviluppa questa distinzione e la collega alle esigenze dei [VLA](../../../vla/README.md).

## Valutazione e registrazione

La valutazione misura il comportamento su episodi e condizioni dichiarati, mentre un video documenta soltanto una particolare esecuzione. Ritorno, durata e successo devono essere raccolti separatamente, soprattutto quando gli ambienti terminano in momenti diversi.

Il [capitolo sull'evaluation](evaluation/README.md) presenta metriche episodiche, protocolli di confronto e registrazione da visualizzatore o sensore, distinguendo gli esempi con azioni nulle dalle policy addestrate.

## Esecuzione su cluster e accesso alle API

Su un cluster il processo deve essere eseguito sulle risorse assegnate dallo scheduler, con dipendenze, asset e risultati organizzati in modo riproducibile. Il [capitolo HPC](hpc/README.md) introduce SLURM, storage e container, accompagnando il template batch presente nella directory.

Quando il comportamento osservato è inatteso, occorre poter seguire i dati dal comando allo stato fisico e all'osservazione. Il [capitolo sulle API e sul debug](api_and_debug/README.md) spiega accesso ai tensori, scrittura degli stati e diagnosi dei problemi più comuni.

## Versioni ed esempi di accompagnamento

Il riferimento software è la **documentazione Isaac Lab 3.0**; per il percorso con l'applicazione completa, la [guida di installazione NVIDIA](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation) indica Isaac Sim 6.2. Il nome della directory `isaac_sim_3` è mantenuto come percorso della raccolta: non identifica una release Isaac Sim 3.

Gli esempi Python sono conservati nelle sottocartelle `lab/` dei rispettivi capitoli: [Gymnasium](gymnasium/lab/), [scene e oggetti](scenes_and_objects/lab/), [task personalizzati](custom_tasks/lab/), [deformabili](deformables/lab/) ed [evaluation](evaluation/lab/). Codice, commenti e messaggi degli script sono in inglese; la spiegazione nel capitolo è in italiano.

Gli esempi sono materiale da verificare nell'ambiente Isaac corrispondente; **la riscrittura dei capitoli non costituisce una validazione runtime su GPU**. I comandi dipendenti dai task e dai preset vanno confrontati con la checkout effettivamente utilizzata.

Per la riproducibilità occorre conservare anche la revisione Git del simulatore e del progetto, le configurazioni e le dipendenze risolte. La sola indicazione della famiglia 3.0 non identifica tutte le condizioni dell'esperimento.
