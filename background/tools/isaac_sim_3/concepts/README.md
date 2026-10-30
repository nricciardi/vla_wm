# Fondamenti della simulazione robotica

Un robot interagisce con un ambiente attraverso attuatori e sensori. Gli attuatori producono forze o inseguono riferimenti di movimento; i sensori restituiscono misure parziali di ciò che accade. Un **simulatore** riproduce questa interazione tramite una rappresentazione numerica della scena e delle leggi che ne governano l'evoluzione.

La simulazione permette di sperimentare senza preparare ogni volta un apparato fisico. Questa flessibilità richiede però scelte esplicite: quali fenomeni rappresentare, con quale precisione e a quale costo. Un modello adeguato per bilanciare un pendolo può essere insufficiente per studiare una presa su tessuto.

## Stato, osservazione e azione

Lo **stato** $s_t$ raccoglie le variabili che descrivono il mondo al passo decisionale $t$: per esempio pose e velocità degli oggetti e stato del robot. Quest'ultimo è indicato con $q_t$ e può comprendere posizioni articolari, velocità e informazioni sugli attuatori.

L'**osservazione** $o_t$ è ciò che viene fornito alla policy. Può contenere una parte dello stato o essere prodotta da sensori simulati, come una camera RGB. La distinzione è essenziale: conoscere esattamente la posa di un oggetto nel simulatore non significa che il robot reale possa misurarla direttamente.

Una **policy** $\pi$ sceglie l'azione $a_t$ a partire dall'osservazione. Se il task è specificato da un'istruzione linguistica $l$, può essere espressa come distribuzione condizionata:

$$
a_t \sim \pi(\cdot \mid o_t,l)
$$

Il linguaggio non è richiesto da tutti gli ambienti: Cartpole, usato negli esempi iniziali, è un problema di controllo senza istruzioni testuali. Inoltre, l'azione non coincide necessariamente con una coppia motrice: può rappresentare una posizione desiderata o uno spostamento dell'end-effector, convertito in comandi da un controller.

## Dal comando alla nuova osservazione

Un modello fisico aggiorna lo stato in risposta all'azione. Per descrivere concettualmente questa separazione si possono introdurre una funzione di transizione $F$, una funzione di osservazione $H$ e parametri fisici $\theta$:

$$
s_{t+1} = F(s_t,a_t;\theta)
$$

$$
o_{t+1} = H(s_{t+1})
$$

La prima relazione comprende l'effetto del controller, dei contatti e dell'integrazione numerica nell'intervallo decisionale. La seconda descrive la costruzione delle osservazioni; eventuale rumore sensoriale richiederebbe ulteriori variabili. Un renderer può contribuire a $H$ producendo immagini, senza sostituire la dinamica descritta da $F$.

Un [World Model](../../../../world_models/README.md) può apprendere una rappresentazione predittiva di queste relazioni dai dati. Qui, invece, la dinamica è costruita attraverso modelli fisici e parametri assegnati. Entrambi consentono rollout, ma la loro attendibilità dipende da ipotesi e fonti di errore differenti.

## Isaac Lab, Isaac Sim e backend

**Isaac Lab** fornisce l'organizzazione del problema di robot learning: configurazioni, ambienti, asset, osservazioni, reward e reset. **Isaac Sim** è l'applicazione basata su NVIDIA Omniverse e Kit che offre strumenti di simulazione e authoring della scena, oltre all'integrazione con rendering e sensori.

Un **backend** è l'implementazione concreta di un servizio, per esempio il calcolo della fisica. Un **solver** è il metodo numerico che risolve le equazioni e i vincoli. La disponibilità di un backend non implica che ogni suo solver gestisca ogni materiale o sensore.

La versione 3.0 di Isaac Lab permette di scegliere tra percorsi diversi, inclusi Newton e integrazioni PhysX. La selezione della fisica, del renderer e del visualizzatore è distinta; le combinazioni utilizzabili dipendono dal task. La [guida NVIDIA ai primi esperimenti](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/quickstart) mostra come ispezionare i preset disponibili.

## Descrizione della scena e asset

**USD**, Universal Scene Description, è un sistema di descrizione e composizione di scene. Una **stage** rappresenta la scena composta; un **prim** è un elemento della sua gerarchia. Il percorso `/World/Table` identifica quindi un elemento nella scena, non un file nel filesystem.

Un **asset** è un componente riutilizzabile, come un robot, un cubo o un sensore. Un `RigidObject` rappresenta un corpo rigido, mentre una `Articulation` descrive corpi collegati da giunti. Un robot articolato richiede sia la geometria sia informazioni su connessioni, limiti e attuazione.

`InteractiveScene` organizza gli asset e le copie degli ambienti. Separare la descrizione della scena dal suo stato corrente permette di costruire i componenti una volta e reinizializzarli all'inizio dei diversi episodi.

## Tempo fisico, controllo e tempo reale

Il **passo fisico** $\Delta t_{\mathrm{phys}}$ è l'intervallo di tempo simulato coperto da un aggiornamento numerico. La policy può inviare un nuovo comando meno frequentemente. Se la **decimation** $d$ indica quanti passi fisici vengono eseguiti per decisione, il periodo di controllo è:

$$
\Delta t_{\mathrm{ctrl}} = d\,\Delta t_{\mathrm{phys}}
$$

Con $\Delta t_{\mathrm{phys}}=1/120$ s e $d=2$, il controllo opera a 60 Hz simulati. Questo non indica quanto tempo impiega il computer: un secondo simulato può richiedere più o meno di un secondo reale.

Il **rendering** può avere ancora un'altra frequenza. Un sistema headless non apre una finestra interattiva, ma può comunque calcolare immagini per la policy. Questa distinzione diventa rilevante quando si confrontano costi e velocità di esperimenti basati su stato ed esperimenti visuali.
