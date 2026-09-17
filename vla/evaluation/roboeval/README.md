# RoboEval

**RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation** parte da una critica precisa: il *success rate* risponde alla domanda «il task è stato completato?», ma comprime in un singolo bit tutta la struttura dell'esecuzione. Due policy possono ottenere lo stesso numero di successi pur differendo radicalmente per durata, percorso seguito, fluidità, collisioni, stabilità della presa e sincronizzazione dei bracci. Un benchmark che registra soltanto l'esito finale non permette quindi di stabilire **come** la policy ha avuto successo, né di localizzare il punto in cui un tentativo fallito si è interrotto.

RoboEval affronta il problema combinando tre elementi: una suite simulata di task bimanuali con variazioni controllate, un dataset di dimostrazioni umane e una strumentazione uniforme di metriche comportamentali e di outcome. Il contributo non consiste nel sostituire il successo binario con un nuovo punteggio unico. Al contrario, mantiene separate misure che descrivono proprietà differenti, perché aggregarle prematuramente nasconderebbe gli stessi trade-off che il benchmark vuole rendere osservabili.

![Struttura generale di RoboEval](../figures/roboeval_overview.webp)

*RoboEval combina task a difficoltà crescente, variazioni strutturate, dimostrazioni teleoperate e metriche che separano outcome e qualità del comportamento. Fonte: [paper RoboEval](https://arxiv.org/abs/2507.00435).*

## Obiettivo e principi di progetto

Il framework è costruito attorno a **diversità, interpretabilità ed estensibilità**. La diversità riguarda sia la struttura temporale dei task sia le capacità motorie richieste: azioni a un solo braccio, coordinazione stretta tra due bracci, trasferimenti, sollevamento, rotazione e attività composte. L'interpretabilità deriva dalla possibilità di associare un fallimento a uno stadio del task o a una proprietà misurabile del moto. L'estensibilità è ottenuta separando definizione del task, generazione delle varianti e logica di valutazione, così da poter aggiungere ambienti e metriche senza cambiare l'intero protocollo.

Questi principi chiariscono anche il significato di «scalabile». Nel lavoro, la scalabilità non indica soltanto la possibilità di aumentare il numero di episodi. Indica soprattutto l'esistenza di interfacce comuni con cui strumentare task diversi, generare perturbazioni confrontabili e raccogliere gli stessi segnali diagnostici per policy eterogenee.

## Formalizzazione dei task

Un task è rappresentato dalla tupla

$$
\mathcal{T}=(\mathcal{S},\mathcal{A},P,\mathcal{G},\rho_0,\mathcal{S}_{\mathrm{success}}),
$$

dove $\mathcal{S}$ è lo spazio degli stati, comprendente configurazione del robot, pose degli oggetti e contesto ambientale; $\mathcal{A}$ è lo spazio delle azioni continue; $P$ descrive la dinamica del simulatore; $\mathcal{G}$ specifica il goal; $\rho_0$ è la distribuzione degli stati iniziali; infine $\mathcal{S}_{\mathrm{success}}\subset\mathcal{S}$ raccoglie gli stati che soddisfano le condizioni geometriche o di contatto usate per dichiarare il successo.

L'azione $a_t\in\mathcal{A}$ può essere espressa come target articolare oppure come spostamento dell'end-effector, a seconda della policy e dell'interfaccia di controllo. Lo stato $s_t\in\mathcal{S}$ è disponibile al simulatore per calcolare condizioni di successo e metriche, mentre l'osservazione $o_t$ fornita alla policy può includere immagini e propriocezione. Questa distinzione evita di confondere le informazioni impiegate per valutare un rollout con quelle effettivamente accessibili al modello durante l'inferenza.

Il dataset esperto associato a un task può essere scritto come

$$
\mathcal{D}_{\mathcal{T}}
=
\left\{(s_0,a_0,\ldots,s_T)^{(i)}\right\}_{i=1}^{N},
$$

dove $N$ è il numero di dimostrazioni e $T$ può variare tra episodi. Ogni task base genera inoltre una famiglia $\mathcal{T}_{\boldsymbol{\eta}}$, nella quale il vettore $\boldsymbol{\eta}\in\mathcal{H}$ controlla perturbazioni come posizione e orientamento degli oggetti. La notazione $\boldsymbol{\eta}$ distingue il parametro della variante sia dall'istruzione linguistica $q$ sia dalla configurazione articolare, indicata nel seguito con $\boldsymbol{\theta}_t$.

## Task, variazioni ed embodiment

La prima release comprende **otto task di manipolazione bimanuale**:

- **Lift Tray**, che richiede di afferrare e sollevare un vassoio mantenendo coordinati i due lati;

- **Stack Two Cubes**, dedicato alla presa e all'impilamento di due cubi;

- **Stack Single Book Shelf**, un task più lungo in cui un libro deve essere collocato sullo scaffale;

- **Rod Handover**, che valuta il passaggio di un oggetto allungato tra i bracci;

- **Lift Pot**, basato sul sollevamento coordinato di un recipiente;

- **Pack Box**, attività multistadio che combina prese, trasferimenti e chiusura della scatola;

- **Pick Book From Table**, che richiede di localizzare, afferrare e rimuovere un libro dal piano;

- **Rotate Valve**, centrato sulla rotazione controllata di una valvola.

![Gli otto task bimanuali di RoboEval](../figures/roboeval_tasks.webp)

*Gli otto task richiedono combinazioni differenti di presa, trasferimento, rotazione e coordinazione tra i due bracci. Le istruzioni riportate nella figura sono quelle associate agli ambienti nel paper. Fonte: [paper RoboEval, figura 2](https://arxiv.org/abs/2507.00435).*

I task coprono contesti tabletop, di servizio e industriali e sono eseguiti con un **embodiment simulato a due bracci**. Il paper descrive uno spazio di controllo continuo compatibile con comandi articolari e delta cartesiani, senza fondare il benchmark sulla replica dichiarata di uno specifico robot commerciale. Questa scelta rende più appropriato parlare di setup bimanuale simulato che attribuire i risultati a un hardware reale non valutato nel lavoro.

Le varianti sono costruite in modo sistematico. **Static** mantiene quasi invariata la configurazione, **Pos** perturba la posizione, **Rot** l'orientamento e **PR** combina entrambi i cambiamenti. Lift Tray, Stack Two Cubes, Rod Handover, Lift Pot, Pack Box e Pick Book From Table includono tutte e quattro le condizioni; Stack Single Book Shelf e Rotate Valve non includono la variante di sola rotazione. Ogni task dispone quindi di tre o quattro livelli di variazione spaziale che preservano la semantica dell'attività.

Questa struttura permette di chiedere se una metrica rimanga informativa quando la configurazione diventa più difficile. Non misura però una generalizzazione aperta: illuminazione, distrattori, texture, massa, attrito e deformabilità non sono assi sistematici della release iniziale.

## Dataset di dimostrazioni

RoboEval distribuisce **oltre 3.000 dimostrazioni esperte**, raccolte con teleoperazione in realtà virtuale. La ripartizione riportata è di 543 traiettorie per Lift Tray, 492 per Stack Two Cubes, 202 per Stack Single Book Shelf, 408 per Rod Handover, 176 per Lift Pot, 394 per Pack Box, 366 per Pick Book From Table e 349 per Rotate Valve. La somma delle singole voci è 2.930; il paper usa «3.000» o «3.000+» come descrizione arrotondata della scala complessiva.

Le lunghezze medie variano sensibilmente: Lift Pot richiede circa 53 step, Lift Tray circa 68, Rod Handover circa 94, mentre Stack Single Book Shelf supera in media 172 step. Questa differenza è importante perché impedisce di interpretare allo stesso modo durata e lunghezza del percorso su task con orizzonti intrinsecamente differenti.

Le traiettorie registrano osservazioni visuali, propriocezione e stati di interazione annotati nella scena. La teleoperazione introduce variazioni naturali nei percorsi e nelle strategie di coordinazione, utili sia per l'imitation learning sia come fascia di riferimento per le metriche. Le dimostrazioni umane non vanno però considerate un limite superiore per ogni misura: correzioni discrete dell'operatore possono aumentare il jerk, facendo apparire una policy molto filtrata più liscia di un umano pur essendo meno efficiente e meno efficace.

## Architettura della valutazione

RoboEval distingue **metriche comportamentali** e **metriche di outcome**. Le prime sono organizzate lungo efficienza, sicurezza e stabilità, e coordinazione. Le seconde descrivono avanzamento e successo. Non tutte devono essere minimizzate nello stesso modo e nessuna, presa isolatamente, rappresenta la competenza complessiva.

### Efficienza temporale e spaziale

L'efficienza temporale viene misurata mediante numero di step e tempo di completamento. Quella spaziale usa la lunghezza cumulativa del percorso nello spazio articolare, cartesiano e delle orientazioni. Indicando con $\boldsymbol{\theta}_t$ la configurazione articolare e con $x_t\in\mathbb{R}^3$ la posizione cartesiana dell'end-effector al tempo $t$, le prime due grandezze sono

$$
L_{\mathrm{joint}}
=
\sum_{t=1}^{T-1}
\left\|\boldsymbol{\theta}_{t+1}-\boldsymbol{\theta}_t\right\|_2,
$$

$$
L_{\mathrm{cart}}
=
\sum_{t=1}^{T-1}
\left\|x_{t+1}-x_t\right\|_2.
$$

Una quantità minore indica un percorso più diretto, ma soltanto a parità di task e di esito. Una traiettoria breve che interrompe precocemente il tentativo non è preferibile a una traiettoria più lunga che completa l'attività. Per questo le metriche di efficienza devono essere lette insieme a progressione e successo.

### Sicurezza, stabilità e fluidità

La fluidità viene caratterizzata tramite il **jerk**, cioè la derivata terza della posizione rispetto al tempo, approssimata con differenze finite. Per la traiettoria cartesiana:

$$
J_{\mathrm{cart}}
=
\frac{1}{T-3}
\sum_{t=1}^{T-3}
\left\|
\frac{x_{t+3}-3x_{t+2}+3x_{t+1}-x_t}{(\Delta t)^3}
\right\|_2,
$$

dove $\Delta t$ è l'intervallo di controllo. La stessa costruzione applicata a $\boldsymbol{\theta}_t$ produce $J_{\mathrm{joint}}$. Valori bassi indicano un segnale meno brusco, ma possono dipendere anche da smoothing, action chunking o frequenza del controller; non costituiscono da soli una prova di destrezza.

Alle misure cinematiche si aggiungono tre conteggi di contatto: **self-collision** tra link del robot, collisioni con l'ambiente e **slip**, cioè perdita involontaria del contatto tra gripper e oggetto afferrato. Questi segnali distinguono, per esempio, una soluzione efficace ma aggressiva da una soluzione altrettanto efficace e stabile.

### Coordinazione tra i bracci

Per misurare l'accoppiamento spaziale, il benchmark calcola la discrepanza verticale media tra end-effector sinistro e destro. Se $x_t^{(L)},x_t^{(R)}\in\mathbb{R}^3$ sono le rispettive posizioni,

$$
\Delta z
=
\frac{1}{T}
\sum_{t=1}^{T}
\left|x_t^{(L)}[z]-x_t^{(R)}[z]\right|.
$$

Per l'accoppiamento temporale considera invece la divergenza delle velocità. Posto

$$
v_t^{(L)}=\frac{x_{t+1}^{(L)}-x_t^{(L)}}{\Delta t},
\qquad
v_t^{(R)}=\frac{x_{t+1}^{(R)}-x_t^{(R)}}{\Delta t},
$$

si ottiene

$$
\Delta v
=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
\left\|v_t^{(L)}-v_t^{(R)}\right\|_2.
$$

Valori ridotti di $\Delta z$ e $\Delta v$ indicano rispettivamente maggiore allineamento verticale e maggiore sincronizzazione delle velocità. L'interpretazione resta task-dependent: bracci che svolgono ruoli volutamente asimmetrici non devono necessariamente mantenere la stessa quota o velocità.

### Progressione ed esito

Ogni attività è scomposta in stadi verificabili, registrati tramite flag binari. La **task progression** indica la porzione di procedura raggiunta, mentre il successo resta la proporzione di rollout conclusi. Questa scomposizione rende distinguibili una policy che non localizza o non afferra l'oggetto e una che completa quasi tutta la sequenza ma fallisce l'ultimo passaggio.

Il vantaggio è particolarmente netto nei regimi a basso successo. Se più modelli ottengono zero, il risultato binario produce un pareggio non informativo; la progressione rivela invece capacità parziali e colli di bottiglia differenti. La definizione degli stadi incorpora però conoscenza specifica del task e richiede lavoro manuale quando si aggiunge una nuova attività.

## Protocollo sperimentale

Il paper confronta **ACT**, **Diffusion Policy**, **GR00T N1.6**, **X-VLA** e **$\pi_0.5$**, includendo per quest'ultimo sia full fine-tuning sia LoRA. ACT e Diffusion Policy sono specialisti single-task con encoder ResNet-18 e action horizon di 16 step. GR00T N1.6 è un VLA da 3 miliardi di parametri di cui vengono adattati projector e action head; X-VLA usa un backbone Florence-2 e una testa flow-matching; $\pi_0.5$ combina SigLIP, Gemma-2B e un action expert Gemma da 300 milioni di parametri.

Le policy sono valutate sugli otto task e sulle rispettive tre o quattro varianti spaziali. Le dimostrazioni umane riuscite formano un riferimento per confrontare tempo, percorso, collisioni, slip e fluidità, ma non una baseline addestrata né un oracle direttamente comparabile in ogni dimensione.

L'analisi è organizzata attorno a tre domande: se le metriche forniscano informazione ulteriore rispetto al successo; in quali condizioni siano maggiormente diagnostiche; e quanto il loro potere discriminativo dipenda dal task o dalla complessità della variante.

## Risultati principali

Il risultato centrale è che **il modello con il miglior successo guida soltanto 34 delle 104 comparazioni sulle altre metriche**, circa il 33%. Essere il migliore nel completamento non implica quindi esserlo per fluidità, sicurezza, efficienza o coordinazione.

Pack Box rende concreto il punto. $\pi_0.5$ e ACT raggiungono successi sovrapposti, rispettivamente circa 0,54 e 0,48, ma ACT mostra un jerk cartesiano circa quattro volte inferiore e un percorso cartesiano circa 2,7 volte più corto. Il successo da solo li descriverebbe come policy simili; le metriche comportamentali indicano strategie di esecuzione sensibilmente diverse.

Anche i fallimenti acquistano struttura. In Lift Tray tre policy rimangono allo 0% di successo, ma raggiungono circa il 35–48% della progressione. In Pick Book From Table modelli con successo nullo coprono un intervallo di progressione dal 2% al 22%. In Stack Two Blocks, GR00T e Diffusion Policy hanno entrambi successo molto basso e simile, ma GR00T produce molti più episodi di self-collision.

Le correlazioni confermano che le misure non sono copie dello stesso segnale. Le lunghezze dei percorsi articolare, cartesiano e orientazionale formano un gruppo molto correlato; jerk articolare e cartesiano ne formano un altro; durata e numero di step un terzo. Le metriche di coordinazione si collocano invece in rami differenti del clustering, suggerendo che discrepanza verticale e divergenza di velocità catturino aspetti complementari.

Il confronto con gli umani richiede cautela. Le policy sono in genere da tre a dieci volte più lente e percorrono traiettorie da due a nove volte più lunghe, oltre a produrre più collisioni e slip. ACT e, in molti task, $\pi_0.5$ mostrano tuttavia jerk inferiore alle dimostrazioni. Il paper interpreta questo dato come effetto della generazione di azioni smussate, mentre la teleoperazione contiene correzioni discrete: **una migliore metrica locale non equivale automaticamente a una migliore competenza globale**.

Il coefficiente di variazione tra policy mostra infine che il successo rimane il discriminatore singolo più forte, seguito da progressione e self-collision. Ciò non contraddice la motivazione del benchmark: il punto non è eliminare il successo, ma completarlo quando policy vicine nell'outcome differiscono nell'esecuzione. Il potere discriminativo delle singole misure cambia inoltre con il task; collisioni e coordinazione sono particolarmente rivelatrici solo dove la struttura fisica rende quelle proprietà rilevanti.

#### Novelty

RoboEval integra in un solo protocollo **task bimanuali, variazioni controllate, dimostrazioni teleoperate, progressione per stadi e metriche comportamentali standardizzate**. Rispetto a benchmark che pubblicano soltanto un catalogo di ambienti e condizioni di successo, fornisce una lente diagnostica per studiare la qualità del moto e la struttura dei fallimenti.

Un secondo contributo è la validazione empirica delle metriche. Il lavoro non presume che qualsiasi segnale continuo sia utile: ne studia correlazioni, stabilità al crescere della perturbazione e capacità di separare policy. Questa analisi mostra sia la complementarità delle famiglie sia la ridondanza interna di alcune misure.

#### Limiti

La copertura è limitata a otto task simulati e a un embodiment bimanuale. Le varianti agiscono soprattutto su posa e orientamento, lasciando fuori cambiamenti sistematici di apparenza, illuminazione, proprietà fisiche, sensori e dinamiche. Non viene quindi dimostrato che il ranking o la qualità diagnostica delle metriche si trasferiscano invariati su robot reali.

Molte metriche dipendono dal contesto. Una differenza verticale ridotta è desiderabile nel sollevamento di un vassoio, ma non necessariamente in un handover asimmetrico; un percorso breve può indicare efficienza oppure fallimento precoce; un jerk basso può derivare da eccessivo smoothing. Per un confronto corretto occorre condizionare l'analisi almeno su task, successo e frequenza di controllo.

La task progression richiede una decomposizione discreta progettata dagli autori. Stadi troppo grossolani nascondono fallimenti, mentre stadi troppo dettagliati possono incorporare una strategia specifica e penalizzare soluzioni alternative valide. Anche i conteggi di collisione dipendono dalla geometria, dal motore fisico e dalle soglie con cui un contatto viene registrato.

Infine, la molteplicità delle metriche migliora la diagnosi ma complica il ranking. Il benchmark non giustifica una somma pesata universale e non dovrebbe essere ridotto a un nuovo scalar score senza dichiarare preferenze applicative. In scenari safety-critical, per esempio, una collisione può dominare qualsiasi guadagno di velocità; in altri contesti il trade-off può essere diverso.

Il riferimento primario è il [paper RoboEval](https://arxiv.org/abs/2507.00435); il [sito del progetto](https://robo-eval.github.io/) raccoglie le risorse associate alla suite.
