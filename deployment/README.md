# Deployment e adattamento dei sistemi robotici

Il **deployment** è il passaggio da un modello addestrato a un sistema che deve operare nelle condizioni effettive di utilizzo. Per una policy robotica, questo significa affrontare osservazioni, configurazioni e interazioni fisiche che possono differire da quelle presenti nei dati di training.

Una variazione dell'illuminazione può alterare la percezione; un offset negli encoder può rendere imprecisa la propriocezione; una presa mancata può portare il robot in uno stato dal quale la policy non riesce a proseguire. La robustezza dipende quindi anche da **come il sistema riconosce e corregge gli errori durante l'esecuzione**.

Questa raccolta collega le [policy VLA](../vla/README.md) e i [World Models](../world_models/README.md) attraverso quattro lavori sull'adattamento durante il deployment. Il README ne presenta il ruolo e le differenze; gli approfondimenti disponibili sono collegati nelle rispettive sezioni. Per i capitoli ancora da sviluppare viene indicata la sottodirectory prevista.

La distinzione principale riguarda **che cosa cambia quando il robot incontra una difficoltà**. Può cambiare il modello predittivo, il contesto che condiziona la policy, la sequenza di azioni scelta da un planner oppure l'organizzazione dell'esecuzione e della memoria.

| Direzione | Lavoro | Elemento adattato durante il deployment |
| :--- | :--- | :--- |
| [Adattamento del modello predittivo](#adattamento-del-modello-predittivo) | AdaJEPA | Parametri del world model mediante feedback auto-supervisionato |
| [Adattamento al disallineamento hardware](#adattamento-al-disallineamento-hardware) | Self-Adaptive VLA | Contesto della policy ricavato dai tentativi precedenti |
| [Monitoraggio e pianificazione correttiva](#monitoraggio-e-pianificazione-correttiva) | Back on Track | Azioni eseguite durante interventi selettivi del planner |
| [Revisione dell'esecuzione e memoria esterna](#revisione-dellesecuzione-e-memoria-esterna) | RoboDoctor | Allocazione del controllo, interventi e metodi conservati in memoria |

**Adattamento online e training online non sono sinonimi**: soltanto alcune strategie aggiornano i pesi durante l'esecuzione. Anche un sistema con modelli congelati può cambiare comportamento accumulando contesto o scegliendo interventi diversi. Occorre inoltre distinguere la preparazione offline del meccanismo adattivo dal suo utilizzo al deployment.

## Adattamento del modello predittivo

### AdaJEPA: An Adaptive Latent World Model (2026)

**AdaJEPA** integra la *test-time adaptation*, cioè l'adattamento durante l'utilizzo, nel **Model Predictive Control (MPC)**. Il controller pianifica nel modello, esegue il primo segmento di azioni, confronta la previsione con la nuova osservazione e aggiorna alcuni parametri prima di ripianificare. Il segnale auto-supervisionato è l'errore di predizione nello spazio latente. Fonte: [paper AdaJEPA](https://arxiv.org/pdf/2606.32026).

Gli esperimenti usano traiettorie offline di **PushT, PushObj e PointMaze**, con variazioni di aspetto, geometria, dinamica e layout. Si tratta di ambienti simulati di pushing e navigazione, senza validazione su robot fisici.

#### Novelty

- Inserimento dell'aggiornamento del world model nel ciclo **pianifica–esegui–adatta–ripianifica**.
- Adattamento dalle transizioni appena osservate, senza nuove dimostrazioni esperte o etichette di reward.
- Miglioramento del planning anche con un solo passo di gradiente per ripianificazione.

#### Limiti

La correzione resta vincolata alla copertura della rappresentazione preaddestrata: caratteristiche assenti dal training possono impedire il recupero completo. Ogni episodio riparte dal modello iniziale, quindi il protocollo non dimostra apprendimento persistente tra episodi. La validazione simulata lascia aperto il trasferimento a manipolazione reale complessa.

L'[approfondimento su AdaJEPA](adajepa/README.md) introduce JEPA e MPC, sviluppa il ciclo di adattamento e analizza dataset, risultati, ablation, costi computazionali e limiti.

## Adattamento al disallineamento hardware

### Self-Adaptive VLA for Robust Robot Deployment (2026)

**Self-Adaptive VLA** usa i rollout precedenti per compensare **bias di attuazione e offset degli encoder articolari**. Un context encoder comprime immagini, propriocezione e azioni in un token che modula la policy attraverso *adaptive layer normalization* (AdaLN). Durante il deployment cambia il contesto, senza richiedere aggiornamenti dei pesi. Fonte: [paper Self-Adaptive VLA](https://arxiv.org/html/2609.30092v1).

Il post-training combina dimostrazioni teleoperate, rollout con perturbazioni hardware introdotte deliberatamente e azioni esperte precompensate. I quattro task reali impiegano bracci **Piper/Piper-X** con gripper paralleli e **Marvin Arms con Wuji Hand**, coprendo manipolazione bimanuale e destrezza.

#### Novelty

- Costruzione di dati contestualizzati associando esperienza sotto perturbazione e azioni esperte compensate.
- Integrazione del contesto nella policy mediante un encoder aggiuntivo e AdaLN.
- Aggregazione dei token di più tentativi per affinare progressivamente la compensazione.

#### Limiti

Il metodo richiede post-training e accesso alla struttura della policy. La valutazione riguarda quattro task e specifiche famiglie di perturbazioni. Sono ammessi fino a **6 tentativi con reset della scena**: il recupero riportato va interpretato rispetto a questo protocollo, senza equipararlo al successo al primo tentativo o al recupero senza reset.

L'approfondimento dedicato a Self-Adaptive VLA è previsto in `self_adaptive_vla/README.md`.

## Monitoraggio e pianificazione correttiva

### Back on Track: Deployment-time Policy Improvement via World Model Planning (2026)

**Back on Track** affianca a una policy congelata un monitor e un planner. Una **liveness value**, appresa per stimare la possibilità di completamento sotto la policy nominale, rileva la perdita di progresso e guida interventi brevi nel world model; poi il controllo torna alla policy. È sufficiente accesso *black-box*, tramite interrogazioni del modello. Fonte: [paper Back on Track](https://arxiv.org/html/2610.19579v1).

Il training usa rollout della policy etichettati per esito e dati di *play*, interazioni che ampliano la copertura delle transizioni. La valutazione comprende **LIBERO-Goal perturbato con $\pi_{0.5}$** e un **Franka Panda reale con diffusion policy** per aprire un contenitore.

#### Novelty

- Miglioramento esterno della policy senza modificarne pesi o architettura.
- Uso della stessa liveness value per decidere quando intervenire e valutare i piani correttivi.
- Pianificazione latente compatta compatibile con il ciclo di controllo sperimentale.

#### Limiti

Servono dati etichettati della policy nominale e training offline dei moduli ausiliari. Errori del monitor possono disturbare esecuzioni riuscite; errori predittivi possono favorire piani inefficaci. La validazione fisica riguarda un solo task e non stabilisce trasferimento generale tra attività.

L'approfondimento dedicato a Back on Track è previsto in `back_on_track/README.md`.

## Revisione dell'esecuzione e memoria esterna

### RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies (2026)

**RoboDoctor** studia il miglioramento dell'esecuzione mantenendo congelati policy e agente supervisore. Il sistema rivede tre livelli: l'**harness**, cioè l'organizzazione del controllo tra policy e agente; la decisione di intervenire; il metodo fisico, come presa o allineamento. Una memoria esterna collega diagnosi, tentativi, effetti osservati e condizioni di riuso. Fonte: [report RoboDoctor, versione 1](https://arxiv.org/pdf/2610.35604v1).

La valutazione usa un archivio selezionato di **54 voci di task di RoboDojo-Sim**, con **PrismWAM** come policy e manipolazione bimanuale simulata. I dati sono registrazioni di interazioni, senza nuovo training dei modelli.

#### Novelty

- Distinzione tra revisione della strategia di controllo, intervento locale e metodo operativo.
- Conservazione di esperienza che associa le correzioni alle evidenze fisiche osservate.
- Verifica degli effetti per correggere il comportamento preservando le capacità già utili.

#### Limiti

Selezione retrospettiva dei risultati, storie diverse e budget non uniformi impediscono di isolare causalmente il beneficio della memoria tra episodi. Tutti i risultati sono simulati. Diagnosi scorrette possono persistere nella memoria; l'assenza di training non elimina i costi di inferenza e interazione.

L'approfondimento dedicato a RoboDoctor è previsto in `robodoctor/README.md`.
