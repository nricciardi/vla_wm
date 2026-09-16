# Octo

**Octo** è una **generalist robot policy open-source** progettata per essere pre-addestrata su dati provenienti da robot differenti e successivamente **adattata a nuovi task**, sensori e action space senza dover ricostruire la policy da zero.

Il punto di partenza è simile a RT-X: sfruttare un grande mixture di dati robotici cross-embodiment. L'obiettivo di Octo è però diverso. RT-X studia soprattutto se il co-training su robot differenti produca **positive transfer**. Octo parte da questa evidenza e cerca di costruire un modello che possa essere utilizzato concretamente come **pretrained policy initialization** per nuovi setup robotici.

In forma schematica, il paradigma passa da:

$$
D_{\text{target}}
\rightarrow
\pi_{\text{target}}
$$

a:

$$
D_{\text{Open X}}
\rightarrow
\pi_{\text{Octo}}
\rightarrow
\text{fine-tuning su }D_{\text{target}}
\rightarrow
\pi_{\text{target}}
$$

Il vantaggio cercato non è quindi soltanto ottenere una policy che controlli più robot già presenti nel training set, ma costruire una rappresentazione interna che renda più semplice apprendere un **nuovo robot o una nuova configurazione sensoriale con pochi dati target**.

## Da RT-X a una policy realmente adattabile

RT-X mostra che dati provenienti da embodiment differenti possono essere utilizzati insieme, ma mantiene una limitazione importante: input e output devono essere ricondotti all'interfaccia prevista dall'architettura.

Questo crea un problema pratico. Un nuovo robot potrebbe avere:

- una camera esterna aggiuntiva;
- una wrist camera;
- propriocezione;
- un force-torque sensor;
- un action space in joint position;
- un action space cartesiano;
- un task specificato tramite linguaggio;
- un task specificato tramite immagine goal.

Una policy generalista realmente riutilizzabile deve quindi poter cambiare non soltanto i dati, ma anche **l'interfaccia tra mondo e modello**.

Octo viene progettato esplicitamente attorno a questa esigenza.

La formulazione più generale può essere scritta come:

$$
\pi(a_{t:t+H} \mid o_{\leq t}, q)
$$

dove $o_{\leq t}$ rappresenta la storia recente delle osservazioni, $q$ la specifica del task e $a_{t:t+H}$ un **chunk di azioni future**.

Il task $q$ non deve necessariamente essere una frase. Octo supporta due modalità principali:

$$
q \in
\{
\text{language instruction},
\text{goal image}
\}
$$

e l'insieme delle osservazioni può cambiare tra dataset e durante il fine-tuning.

Questa flessibilità costituisce una delle differenze più importanti rispetto alle precedenti policy cross-embodiment.

## Dataset

Octo viene pre-addestrato su circa **800.000 traiettorie robotiche** provenienti da **25 dataset** inclusi in Open X-Embodiment.

Il mixture contiene robot, scene e task differenti e presenta eterogeneità anche a livello delle modalità disponibili. Alcuni dataset contengono soltanto una camera esterna, altri includono una wrist camera, altri ancora propriocezione o differenti forme di task annotation.

Questo significa che il modello non può assumere che ogni esempio possieda sempre la stessa struttura.

Concettualmente il training set è:

$$
D =
D^{(1)} \cup D^{(2)} \cup \dots \cup D^{(25)}
$$

ma i singoli dataset possono avere interfacce differenti:

$$
D^{(i)}
=
\{
(o_t^{(i)}, q^{(i)}, a_t^{(i)})
\}
$$

con:

$$
o_t^{(i)}
\neq
o_t^{(j)}
$$

e potenzialmente:

$$
\mathcal{A}^{(i)}
\neq
\mathcal{A}^{(j)}
$$

dove $\mathcal{A}^{(i)}$ indica lo spazio delle azioni del dataset $i$.

Il problema di Octo è quindi più ampio della semplice concatenazione di traiettorie: deve costruire una rappresentazione capace di gestire **modalità mancanti, sensori differenti e diversi modi di specificare il task**.

### Differenza rispetto al mixture RT-X

Il training mixture di Octo è più ampio di quello utilizzato negli esperimenti originali di RT-X.

L'aumento dei dati non è però l'unico cambiamento. Octo modifica soprattutto l'architettura affinché l'eterogeneità dei dataset non debba essere completamente eliminata prima del training.

In RT-X l'obiettivo principale era ottenere una rappresentazione sufficientemente uniforme da alimentare una policy comune.

Octo cerca invece di costruire una rete in cui una parte dell'eterogeneità possa essere **rappresentata esplicitamente attraverso token modulari**.

## Architettura

Octo **non è basato né su un Vision Encoder pre-addestrato** (come CLIP o SigLIP) **né su un LLM pre-addestrato** autoregressivo (come Llama o Vicuna).

A differenza dei modelli VLA (Vision-Language-Action) classici come RT-2 o OpenVLA, Octo adotta un'architettura modulare proprietaria progettata specificamente per la robotica:Visione (nessun vision encoder pre-addestrato): 

- I frame delle telecamere e le eventuali immagini obiettivo (goal images) vengono elaborati da un semplice stack **convoluzionale leggero** (CNN/patchified tokens stile ViT) **addestrato da zero** direttamente sui dati robotici (Open X-Embodiment)
- Linguaggio (l'unico componente parzialmente pre-addestrato): Per codificare i comandi testuali delle istruzioni usa un **piccolo encoder linguistico congelato**, T5-base (circa 111M parametri), e non un modello generativo/LLM autoregressivo.
- Backbone: È un **Transformer encoder-decoder** (rilasciato in varianti da 27M e 93M parametri) **addestrato da zero con attenzione causale**/a blocchi per fondere i token di osservazione e di task.  
- Predizione delle azioni (Diffusion Head): L'output non è generato come token discreti da un LLM, ma attraverso una **diffusion head condizionale che predice sequenze continue di traiettorie** (action chunking).
  
L'architettura può essere riassunta come:

$$
\text{task tokenizers}
+
\text{observation tokenizers}
\rightarrow
\text{Octo Transformer}
\rightarrow
\text{readout tokens}
\rightarrow
\text{diffusion action head}
$$

Il principio centrale è che ogni sorgente di informazione viene prima trasformata in una sequenza di token compatibile con il Transformer.

Il backbone non deve quindi conoscere direttamente se un certo blocco provenga da una camera, da un'immagine goal o da un encoder linguistico.

Questa separazione tra **tokenizer di input**, **Transformer condiviso** e **head di output** è ciò che permette di modificare l'interfaccia durante il fine-tuning mantenendo gran parte dei pesi pre-addestrati.

![Architecture](figures/octo_architecture.png)

## Task token

Octo supporta almeno due forme principali di task conditioning:

- istruzione linguistica;
- goal image.

### Language instruction

Le istruzioni linguistiche vengono processate attraverso un encoder **T5-base** pre-addestrato.

Data un'istruzione:

$$
q =
\text{``put the knife on the plate''}
$$

l'encoder produce una sequenza di embedding:

$$
\mathcal{T}_{lang}
=
(t_1,t_2,\dots,t_N)
$$

Questi token vengono forniti al Transformer come contesto del task.

È importante notare che Octo **non è costruito attorno a un grande VLM da miliardi di parametri**, come avverrà invece in OpenVLA.

Il language encoder serve principalmente a rappresentare il comando in una forma semantica utile alla policy, mentre la maggior parte del processing multimodale viene eseguita dal Transformer di Octo.

### Goal image

Il task può essere specificato anche attraverso un'immagine che rappresenta lo stato desiderato.

In questo caso:

$$
q = g
$$

dove $g$ è la goal image.

L'immagine goal viene combinata con l'osservazione visuale prima della patchification tramite **early fusion**.

Invece di dire:

$$
\text{``put the cup near the plate''}
$$

è quindi possibile mostrare al modello un'immagine della configurazione finale desiderata.

Questa modalità contiene potenzialmente informazioni difficili da specificare con una breve frase, come posizione precisa, orientamento e relazione geometrica tra oggetti.

Negli esperimenti su WidowX, il goal conditioning produce prestazioni superiori rispetto al solo language conditioning per alcuni task.

Questo mostra che **linguaggio e goal image non sono semplicemente due codifiche equivalenti del task**: possono fornire quantità e tipi di informazione differenti.

## Observation token

Le osservazioni visuali vengono trasformate in token attraverso **shallow convolutional patch encoders**.

A differenza di architetture che utilizzano un visual encoder molto profondo prima del Transformer, Octo segue una filosofia che gli autori descrivono sostanzialmente come **transformer-first**.

La pipeline è:

$$
o_t^{img}
\rightarrow
\text{shallow CNN}
\rightarrow
\text{patch tokens}
\rightarrow
\text{Transformer}
$$

La CNN esegue quindi una trasformazione relativamente leggera.

La maggior parte della capacità viene concentrata nel **Transformer condiviso, che può elaborare congiuntamente informazioni provenienti dalle diverse modalità**.

Questa scelta è importante perché un encoder visuale molto specializzato elaborerebbe gran parte delle informazioni prima che queste possano interagire con task, history e altri sensori.

Octo sposta invece questa interazione verso il backbone comune.

### Osservazioni multiple

Un timestep può contenere più sorgenti:

$$
o_t
=
\{
o_t^{primary},
o_t^{wrist},
o_t^{proprio},
\dots
\}
$$

Ciascuna viene convertita nei propri token.

Concettualmente:

$$
\mathcal{T}_{o,t}
=
[
\mathcal{T}_{primary,t},
\mathcal{T}_{wrist,t},
\mathcal{T}_{proprio,t},
\dots
]
$$

Non tutti i dataset devono contenere tutte queste modalità.

È proprio questa assenza di uniformità a motivare il particolare attention mask utilizzato dal modello.

## Block-wise masked attention

Il Transformer di Octo non utilizza semplicemente una sequenza piatta in cui tutti i token possono interagire liberamente.

La sequenza è organizzata logicamente come:

$$
[
\mathcal{T}_{T},
\mathcal{T}_{o,0},
\mathcal{T}_{R,0},
\mathcal{T}_{o,1},
\mathcal{T}_{R,1},
\dots
]
$$

dove:

- $\mathcal{T}_{T}$ contiene i task token;
- $\mathcal{T}_{o,t}$ contiene i token delle osservazioni al timestep $t$;
- $\mathcal{T}_{R,t}$ contiene i **readout token**.

L'attention mask impone una struttura causale a blocchi.

I token dell'osservazione al tempo $t$ possono utilizzare:

$$
\mathcal{T}_{T}
$$

e le osservazioni fino al tempo corrente:

$$
\mathcal{T}_{o,0:t}
$$

ma non osservazioni future.

Quindi:

$$
\mathcal{T}_{o,t}
\rightarrow
\{
\mathcal{T}_{T},
\mathcal{T}_{o,\leq t}
\}
$$

Questa struttura mantiene la causalità temporale necessaria a una policy closed-loop.

### Modalità mancanti

Supponiamo che un dataset non disponga di wrist camera.

I token corrispondenti possono essere **mascherati completamente**.

Il Transformer può quindi essere addestrato nello stesso batch su esempi con strutture differenti senza richiedere che ogni traiettoria contenga artificialmente tutte le modalità previste.

Questo elemento è fondamentale per il pre-training su Open X-Embodiment, perché l'eterogeneità dei sensori non viene eliminata imponendo una singola configurazione universale.

## Readout token

Una delle componenti più caratteristiche dell'architettura sono i **readout token**.

Ad ogni timestep vengono aggiunti token appresi:

$$
\mathcal{T}_{R,t}
$$

che possono attendere ai task token e alle osservazioni precedenti:

$$
\mathcal{T}_{R,t}
\leftarrow
\{
\mathcal{T}_{T},
\mathcal{T}_{o,\leq t}
\}
$$

ma gli observation token **non possono utilizzare i readout token per aggiornare la propria rappresentazione**.

Quindi il flusso è sostanzialmente unidirezionale:

$$
\text{task/observation}
\rightarrow
\text{readout}
$$

I readout token funzionano quindi come una **rappresentazione compatta dello stato interno** del Transformer da utilizzare per una particolare prediction head.

### Perché i readout token rendono Octo modulare

Supponiamo di voler sostituire l'action space cartesiano con un controllo in joint position.

Il Transformer non deve necessariamente essere riprogettato.

La pipeline può diventare:

$$
\mathcal{T}_{R,t}
\rightarrow
\text{new action head}
\rightarrow
a_t^{joint}
$$

analogamente, in linea di principio, altri readout potrebbero essere utilizzati per nuove prediction head senza cambiare il modo in cui vengono rappresentati task e osservazioni.


## Diffusion action head

Octo non discretizza ogni componente dell'azione in token come RT-1, RT-2-X o successivamente OpenVLA.

Utilizza invece una **diffusion policy** per generare direttamente azioni continue.

Il modello deve rappresentare una distribuzione:

$$
p(a_{t:t+H} \mid o_{\leq t},q)
$$

potenzialmente multimodale.

Questo è importante perché in robotica possono esistere più azioni corrette per raggiungere lo stesso obiettivo.

Per esempio, per afferrare un oggetto:

$$
a^{(1)}
\neq
a^{(2)}
$$

ma entrambe le traiettorie possono essere valide.

Una regressione deterministica con loss MSE tende invece a produrre una media:

$$
\hat a
\approx
\mathbb{E}[a]
$$

che in presenza di modalità molto differenti può non corrispondere ad alcuna traiettoria realmente utile.

### Processo di diffusion

Durante il training viene applicato rumore alle azioni dimostrate.

Indicando con $a^0$ il chunk originale:

$$
a^0
\rightarrow
a^k
$$

dove $a^k$ è una versione progressivamente corrotta da rumore.

La rete impara a invertire questo processo, condizionandosi sulla rappresentazione prodotta dai readout token:

$$
\epsilon_\theta
(
a^k,
k,
\mathcal{T}_{R,t}
)
$$

Durante l'inferenza si parte da rumore:

$$
a^K \sim \mathcal{N}(0,I)
$$

e si applicano iterativamente step di denoising:

$$
a^K
\rightarrow
a^{K-1}
\rightarrow
\dots
\rightarrow
a^0
$$

ottenendo infine un chunk di azioni eseguibili dal robot.

L'idea centrale è quindi:

$$
\text{Transformer}
\rightarrow
\text{conditioning}
\rightarrow
\text{continuous generative policy}
$$

piuttosto che:

$$
\text{Transformer}
\rightarrow
\text{discrete action tokens}
$$

Questa differenza anticipa una linea di ricerca che verrà ulteriormente sviluppata nei modelli successivi basati su diffusion e flow matching.

## Action chunking

Octo non predice necessariamente una singola azione isolata.

L'action head produce un **chunk di azioni future**:

$$
A_t
=
(
a_t,
a_{t+1},
\dots,
a_{t+H}
)
$$

Questo permette al modello di rappresentare localmente la continuità temporale del movimento.

Rispetto alla predizione indipendente di:

$$
a_t
$$

il chunk contiene informazione su come una breve sequenza motoria debba evolvere nel tempo.

Questo è particolarmente utile per comportamenti che richiedono traiettorie coordinate e continue.

Il robot rimane comunque controllato in closed loop: nuove osservazioni vengono acquisite e nuovi chunk possono essere generati nel corso dell'esecuzione.

## Fine-tuning modulare

La parte più importante di Octo non è soltanto il pre-training, ma **come viene utilizzato dopo il pre-training**.

Gli autori vogliono evitare la situazione:

$$
\text{nuovo sensore}
\Rightarrow
\text{nuova architettura}
$$

oppure:

$$
\text{nuovo action space}
\Rightarrow
\text{pre-training inutilizzabile}
$$

Octo separa invece il backbone dalle componenti che trasformano input e output.

### Aggiungere un nuovo sensore

Supponiamo che il modello pre-addestrato utilizzi:

$$
o_t =
\{
RGB
\}
$$

mentre il nuovo robot disponga anche di un force-torque sensor:

$$
o_t' =
\{
RGB,
F/T
\}
$$

È possibile aggiungere:

$$
F/T
\rightarrow
\text{new tokenizer}
\rightarrow
\text{new observation tokens}
$$

mantenendo il Transformer pre-addestrato.

Devono essere appresi i nuovi parametri necessari a rappresentare la modalità aggiuntiva, ma non è necessario ricostruire l'intero modello.

### Cambiare action space

Analogamente, il pre-training può utilizzare prevalentemente end-effector control:

$$
a_t^{ee}
=
(
x,y,z,roll,pitch,yaw,g
)
$$

mentre un nuovo robot può richiedere:

$$
a_t^{joint}
=
(
\theta_1,\theta_2,\dots,\theta_n
)
$$

In questo caso può essere introdotto un nuovo action head:

$$
\mathcal{T}_{R,t}
\rightarrow
h_{\text{joint}}
\rightarrow
a_t^{joint}
$$

dove $h_{\text{joint}}$ viene adattato al nuovo spazio di controllo.

Questo è un cambiamento concettuale importante rispetto alla strategia RT-X.
RT-X cerca soprattutto di mappare robot differenti verso un **action space condiviso**.

Octo cerca anche di rendere la policy capace di **cambiare action space durante il downstream adaptation**.

## Due modelli: Octo-Small e Octo-Base

Gli autori rilasciano due dimensioni principali:

$$
\text{Octo-Small}
\approx
27M
$$

e:

$$
\text{Octo-Base}
\approx
93M
$$

parametri.

Sono dimensioni molto inferiori rispetto ai VLA da miliardi di parametri come RT-2-X o OpenVLA.

Octo non cerca principalmente di incorporare enormi quantità di conoscenza del mondo attraverso un language model molto grande.

Cerca invece di costruire una **robot policy pre-addestrata direttamente su dati di controllo**, sufficientemente compatta da poter essere fine-tuned in modo pratico.

Il modello può quindi essere adattato su hardware relativamente accessibile, senza richiedere necessariamente l'infrastruttura associata ai VLM da miliardi di parametri.

## Fine-tuning su nuovi setup

L'evaluation più importante riguarda il trasferimento verso setup downstream.

Gli autori considerano sei configurazioni che includono:

- task long-horizon;
- precise manipulation;
- nuovi embodiment;
- nuovi segnali sensoriali;
- nuovi action space.

Per ciascun setup vengono utilizzate circa **100 dimostrazioni target**.

Il confronto principale include:

- training from scratch;
- inizializzazione con feature visuali VC-1;
- fine-tuning da Octo.

> [!NOTE]
> **VC-1**, abbreviazione di *Visual Cortex 1*, è una **pre-trained visual representation** progettata per embodied AI. Il modello è un Vision Transformer pre-addestrato tramite **Masked Auto-Encoding (MAE)**: una parte delle patch dell'immagine viene nascosta e la rete impara una rappresentazione ricostruendo il contenuto mancante. I dati di pre-training comprendono oltre 4.000 ore di video egocentrici provenienti da sette sorgenti, insieme a immagini di ImageNet.

VC-1 è stato selezionato come baseline perché rappresenta un'alternativa plausibile al pre-training end-to-end di Octo. Fornisce infatti feature visive già adatte a locomozione, navigazione e manipolazione, ma **non è una policy robotica**: durante il pre-training non apprende direttamente la relazione tra osservazioni, task e azioni. Nel confronto, i pesi VC-1 inizializzano un encoder ViT, quindi l'intera rete viene fine-tuned sulle dimostrazioni target per predire le azioni con una loss MSE.

Il confronto separa quindi due forme di trasferimento. VC-1 trasferisce soprattutto regolarità percettive apprese da immagini e video; Octo trasferisce anche una rappresentazione visuomotoria ottenuta da traiettorie robotiche, action chunk e task conditioning. 

La media riportata sulle sei evaluation è:

$$
\text{From Scratch}
\approx
0.20
$$

$$
\text{VC-1}
\approx
0.15
$$

$$
\text{Octo}
\approx
\mathbf{0.72}
$$


## Nuovi sensori: force-torque input

Una delle evaluation introduce un **force-torque input** che non fa parte dell'interfaccia standard utilizzata durante il pre-training.

Il task richiede precise manipulation e beneficia di informazioni di contatto che non possono essere ricavate in modo affidabile dalla sola immagine.

L'esperimento verifica quindi direttamente il principio modulare:

$$
\text{pretrained Octo}
+
\text{new observation tokenizer}
\rightarrow
\text{new sensor-aware policy}
$$

Il risultato è importante perché mostra che il backbone non è vincolato in modo rigido alle sole modalità presenti durante il pre-training.

## Nuovi action space: joint position control

Un'altra evaluation sostituisce il controllo cartesiano con **joint position control**.

La configurazione articolare può essere rappresentata come:

$$
\boldsymbol{\theta}_t
=
(
\theta_{1,t},
\dots,
\theta_{n,t}
)
$$

e l'azione diventa un target o una variazione nello spazio dei giunti.

Questa evaluation è particolarmente significativa rispetto a RT-X.

RT-X cerca di costruire un linguaggio comune attraverso l'end-effector.

Octo dimostra che il pre-training può essere utile anche quando il downstream task richiede un'azione che **non coincide con quella canonica utilizzata nel corpus principale**.

Quindi il trasferimento non deve necessariamente avvenire perché:

$$
\mathcal{A}_{source}
=
\mathcal{A}_{target}
$$

ma può avvenire tramite la rappresentazione interna:

$$
D_{source}
\rightarrow
\text{shared representation}
\rightarrow
\mathcal{A}_{target}
$$

Questo è uno dei passaggi più importanti nella progressione dei modelli cross-embodiment.

## Quanto conta la history temporale

Octo può utilizzare una breve storia delle osservazioni:

$$
o_{t-h:t}
$$

invece della sola immagine corrente.

Gli ablation experiment mostrano però che aumentare molto la history non produce necessariamente miglioramenti proporzionali.

Questo suggerisce che, nei task studiati, gran parte dell'informazione richiesta può essere inferita dall'osservazione corrente e da una breve finestra temporale.

La scelta progettuale riflette quindi un trade-off:

$$
\text{longer history}
\rightarrow
\text{more temporal information}
$$

ma anche:

$$
\text{longer history}
\rightarrow
\text{more tokens}
\rightarrow
\text{higher Transformer cost}
$$

Poiché il costo dell'attention cresce rapidamente con il numero di token, una history corta consente di mantenere più efficiente il modello.

## Perché non utilizzare un grande visual encoder

Uno degli ablation study riguarda la distribuzione della capacità tra visual encoder e Transformer.

Un design comune può essere schematizzato come:

$$
\text{deep visual encoder}
\rightarrow
\text{small fusion module}
$$

Octo preferisce:

$$
\text{shallow visual encoder}
\rightarrow
\text{large shared Transformer}
$$

L'idea è che, in una policy multimodale, la parte più utile del computation budget possa essere quella che elabora **congiuntamente** task e osservazioni.

Una feature visuale calcolata quasi completamente prima di vedere l'istruzione deve essere sufficientemente generale da servire tutti i task.

Con un Transformer più centrale, invece, le rappresentazioni possono essere trasformate in funzione del contesto del task.

Questa scelta contribuisce anche alla modularità: le nuove modalità devono soltanto essere proiettate in uno spazio di token compatibile, mentre il processing principale resta condiviso.


## Limiti

Il primo limite è che Octo **non è una policy universale zero-shot per nuovi robot**.

Le **capacità out-of-the-box vengono valutate principalmente su robot e domini rappresentati nel pre-training**. Quando cambia realmente l'embodiment o l'interfaccia, il paradigma previsto è:

$$
\text{pre-training}
+
\text{target demonstrations}
+
\text{fine-tuning}
$$

e non:

$$
\text{completely unseen robot}
\rightarrow
\text{zero-shot control}
$$

Il secondo limite riguarda la copertura dei dati. Nonostante Open X-Embodiment sia molto ampio rispetto ai precedenti dataset robotici, rimane concentrato soprattutto sulla **manipolazione con bracci robotici**. La varietà di embodiment è quindi ancora molto più ristretta rispetto a quella che sarebbe necessaria per parlare di controllo robotico universale.

Il terzo limite riguarda la conoscenza semantica. Octo viene pre-addestrato principalmente su dati robotici e **non eredita la stessa quantità di conoscenza visuale e linguistica di un grande Vision-Language Model**. Questo può limitarne la capacità di comprendere oggetti, concetti o istruzioni lontani dalla distribuzione robotica osservata durante il training.

Un ulteriore limite deriva dalla stessa flessibilità dell'architettura. Aggiungere un nuovo sensore o un nuovo action space è possibile, ma richiede comunque **dati target sufficienti ad apprendere come la nuova interfaccia debba essere utilizzata**. La modularità riduce il costo dell'adattamento, ma non elimina il problema della raccolta di dimostrazioni.

Infine, l'utilizzo di diffusion per le azioni introduce un processo generativo più costoso rispetto a un singolo forward pass deterministico. La maggiore espressività della distribuzione delle azioni viene quindi ottenuta al prezzo di una pipeline di inferenza più complessa.

Octo rappresenta così un passaggio importante nella progressione dei robot foundation model: il cross-embodiment learning non viene più considerato soltanto come **training su più robot**, ma come pre-training di una rappresentazione che possa essere **riutilizzata e riconfigurata per robot successivi**.
