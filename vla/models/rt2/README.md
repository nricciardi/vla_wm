# RT-2 (2023)

**RT-2*** introduce un passaggio fondamentale rispetto a RT-1: invece di addestrare da zero una policy visuomotoria principalmente sui dati raccolti dal robot, viene utilizzato come punto di partenza un **Vision-Language Model (VLM) pre-addestrato su grandi quantità di dati provenienti dal web**, che viene successivamente adattato anche alla predizione delle azioni robotiche.

L'obiettivo è **trasferire al controllo robotico parte delle conoscenze semantiche e visive già apprese** dal VLM.

Il problema affrontato è infatti una limitazione fondamentale di RT-1. RT-1 può generalizzare rispetto alle dimostrazioni robotiche osservate, ma la sua conoscenza del mondo deriva essenzialmente dai dati raccolti dal robot. Un VLM, al contrario, è stato esposto a una quantità enormemente maggiore di concetti, oggetti, relazioni visive e informazioni linguistiche.

RT-2 cerca quindi di combinare:

$$
\underbrace{\text{web-scale vision-language knowledge}}_{\text{conoscenza semantica}}
+
\underbrace{\text{robot demonstrations}}_{\text{grounding fisico}}
$$

in un'unica policy:

$$
\pi(a_t \mid o_t,l)
$$

capace di trasformare direttamente un'osservazione visiva $o_t$ e un'istruzione $l$ in un'azione robotica $a_t$.

È proprio in questo lavoro che gli autori formalizzano esplicitamente la categoria dei **Vision-Language-Action models (VLA)**: modelli nei quali un VLM viene esteso in modo da poter produrre, oltre al linguaggio, anche azioni eseguibili da un robot.


## Idea principale: rappresentare le azioni come linguaggio

Il problema principale nel trasformare un VLM in una robot policy è l'incompatibilità tra i due output.

Un VLM tradizionale produce una sequenza di token:

$$
P(y_1,\dots,y_N \mid image,text)
$$

dove ciascun $y_j$ appartiene al vocabolario linguistico del modello.

Un robot, invece, necessita di quantità numeriche come:

$$
\Delta x,\Delta y,\Delta z,
\Delta roll,\Delta pitch,\Delta yaw
$$

che devono successivamente essere trasformate in movimento fisico.

RT-2 introduce una soluzione molto semplice: **anche le azioni vengono rappresentate come token**.

In questo modo:

$$
\text{language}
\rightarrow
\text{tokens}
$$

e

$$
\text{robot actions}
\rightarrow
\text{tokens}
$$

possono essere trattati dallo **stesso modello autoregressivo e dalla stessa output head**.

Quindi non viene aggiunta al VLM una nuova action head continua: il modello viene addestrato a considerare l'azione robotica come una particolare sequenza appartenente al proprio spazio di output.

Concettualmente:

$$
\text{image}+l
\rightarrow
\text{VLM}
\rightarrow
\text{action tokens}
\rightarrow
\text{de-tokenization}
\rightarrow
a_t
$$

Questa rappresentazione permette di utilizzare quasi direttamente l'architettura di un VLM pre-addestrato per il controllo robotico.

![Overview](figures/rt2_overview.png)

## Dataset

RT-2 viene addestrato **contemporaneamente** su due famiglie di dati profondamente differenti:

$$
D=
D_{\text{web}}
\cup
D_{\text{robot}}
$$

Il fine-tuning avviene sulla stessa output head autoregressiva, quindi il modello impara contemporaneamente a produrre sia linguaggio sia azioni robotiche.

Nel caso di RT-2-PaLI-X, il **dataset robotico viene sovracampionato fino a rappresentare circa il 50%** del training mixture. Quindi nonostante il dataset web sia enormemente più grande in termini assoluti, durante il co-fine-tuning il modello non vede il robot data con la sua frequenza naturale: viene pesato molto di più per evitare che il segnale di controllo venga sommerso dai dati vision-language.

### Robot data

La parte robotica deriva principalmente dal dataset utilizzato per RT-1.

Si tratta quindi di **oltre 130.000 episodi raccolti su robot fisici**, nell'arco di circa **17 mesi**, utilizzando una flotta di **13 Everyday Robots mobile manipulators**.

Le traiettorie vengono associate a istruzioni linguistiche che descrivono il task, generalmente attraverso:

$$
\text{skill verb}+\text{object nouns}
$$

ad esempio:

* `pick 7up can`;
* `open drawer`;
* `place napkin into drawer`.

Si tratta principalmente di manipolazione in **ambienti fisici di tipo office-kitchen**, non di un dataset generato interamente in simulazione.

Il robot principale rimane quindi lo stesso embodiment utilizzato in RT-1: un **Everyday Robots mobile manipulator**.

### Web vision-language data

La differenza fondamentale rispetto a RT-1 è però l'utilizzo contemporaneo dei dataset originali dei VLM.

Questi contengono task come:

* **Visual Question Answering (VQA)**;
* image captioning;
* image-text understanding;
* esempi con immagini e testo interleaved;
* altre forme di vision-language supervision.

Questi dati non contengono necessariamente azioni robotiche.

Servono invece a preservare e sviluppare rappresentazioni semantiche del mondo:

$$
\text{image}
+
\text{text}
\rightarrow
\text{text}
$$

mentre i robot data forniscono il grounding:

$$
\text{robot image}
+
l
\rightarrow
\text{action}
$$

Il punto centrale di RT-2 è che **entrambi i tipi di esempi vengono presentati allo stesso modello durante il training**.


## Backbone

RT-2 non identifica una singola architettura specifica come accade con RT-1.

Gli autori mostrano che la stessa idea può essere applicata a diversi VLM e realizzano principalmente due famiglie di modelli:

$$
\text{RT-2-PaLI-X}
$$

e

$$
\text{RT-2-PaLM-E}
$$

In particolare vengono valutate varianti tra cui:

* **PaLI-X 5B**;
* **PaLI-X 55B**;
* **PaLM-E 12B**.

Questo costituisce un cambiamento enorme rispetto a RT-1 avente 35M parametri contro modelli RT-2 dell'ordine di 5B-55B.

La struttura generale non è quindi più:

$$
\text{FiLM-EfficientNet}
\rightarrow
\text{TokenLearner}
\rightarrow
\text{small Transformer}
$$

ma:

$$
\boxed{
\text{pretrained VLM}
\rightarrow
\text{autoregressive token generation}
}
$$

Il VLM possiede già un encoder visivo e un language model addestrati su dati multimodali su larga scala.

Questo significa che **RT-2 non deve imparare da zero cosa siano gli oggetti rappresentati nelle immagini**: parte da rappresentazioni visive e linguistiche che possiedono già conoscenze semantiche acquisite durante il pre-training.

### PaLI-X

**PaLI-X** è un Vision-Language Model progettato per ricevere contemporaneamente **immagini e testo** e produrre una sequenza testuale in output.

PaLI-X appartiene alla famiglia *Pathways Language and Image*, nella quale vengono combinati due componenti pre-addestrati:

* un **Vision Transformer (ViT)** per la parte visiva;
* un modello linguistico **encoder-decoder** per l'elaborazione multimodale e la generazione dell'output.

La componente linguistica di PaLI-X deriva da **UL2**.

UL2 è un Transformer **encoder-decoder**, quindi non è un puro decoder autoregressivo nello stile di GPT.

UL2 è stato pre-addestrato utilizzando una strategia chiamata **Mixture of Denoisers**, nella quale vengono combinati diversi obiettivi di denoising invece di utilizzare un unico schema di language modeling. L'obiettivo è rendere lo stesso modello efficace sia in task di comprensione sia in task di generazione.

Nel PaLI-X da 55B, la componente linguistica è dell'ordine di **32 miliardi di parametri**, mentre la componente visuale utilizza il ViT-22B.


### PaLM-E

**PaLM-E**  nasce già con l'obiettivo di trasformare un Large Language Model in un modello **multimodale ed embodied**, capace di ricevere non soltanto testo ma anche osservazioni provenienti dal mondo fisico.

Gli input possono quindi includere:

$$
\text{text}
+
\text{images}
+
\text{continuous robot state}.
$$

PaLM-E viene infatti progettato per task come:

* robot planning;
* Visual Question Answering;
* image captioning;
* embodied reasoning.

PaLM usa un grande Transformer linguistico autoregressivo.

La struttura di base è quindi:

$$
x_1,x_2,\dots,x_N
\rightarrow
\text{Transformer}
\rightarrow
x_{N+1}.
$$


Il modello PaLM-E costruisce una sorta di **multimodal sentence**, nella quale **token linguistici e embedding sensoriali vengono interleaved** nella stessa sequenza.

Ad esempio:

$$
[
\text{text},
\text{image embeddings},
\text{text},
\text{state embeddings},
\text{text}
].
$$

Il Transformer può quindi ricevere qualcosa concettualmente simile a:

$$
[
\text{``What should the robot do?''},
v_1,\dots,v_N,
s_1,\dots,s_M
].
$$

dove:

* $v_j$ sono embedding visuali;
* $s_j$ sono embedding dello stato continuo del robot.

Non viene quindi richiesto che ogni input corrisponda a una parola.

È sufficiente che **tutte le modalità vengano proiettate nella stessa dimensionalità** utilizzata dagli embedding del language model:

$$
e_j\in\mathbb{R}^{d_{LM}}.
$$

Una volta effettuata questa proiezione, il Transformer può trattare questi vettori come elementi della propria sequenza di input.


## Input

Durante il controllo robotico il modello riceve principalmente:

$$
(o_t,l)
$$

dove:

* $o_t$ rappresenta l'immagine RGB corrente osservata dal robot;
* $l$ rappresenta l'istruzione linguistica.

I dati robotici vengono convertiti in una struttura compatibile con i task di Visual Question Answering utilizzati dal VLM.

Concettualmente, un esempio può essere rappresentato come:

> `Q: What action should the robot take to <instruction>? A:`

seguito dalla sequenza di token corrispondente all'azione.

Quindi l'action prediction viene formulata nello stesso formato di un normale problema question-answering:

$$
(\text{image},\text{question})
\rightarrow
\text{answer tokens}
$$

con la differenza che, nel caso robotico:

$$
\text{answer tokens}
=
\text{action tokens}.
$$

## Action space

L'action space utilizzato da RT-2 va distinto da quello completo descritto per RT-1.

RT-2 utilizza un comando costituito da:

$$
a_t=
(
a_t^{terminate},
\Delta p_t,
\Delta r_t,
g_t
)
$$

dove:

$$
\Delta p_t=
(\Delta x,\Delta y,\Delta z)
$$

rappresenta lo **spostamento cartesiano relativo dell'end-effector**, mentre:

$$
\Delta r_t=
(\Delta roll,\Delta pitch,\Delta yaw)
$$

rappresenta lo **spostamento relativo dell'orientamento**.

A questi vengono aggiunti:

$$
g_t
$$

che rappresenta il livello di estensione/apertura del gripper e:

$$
a_t^{terminate}
$$

che è un comando discreto utilizzato per indicare la conclusione dell'episodio.

L'action string completa ha quindi la struttura:

$$
[
terminate,
\Delta x,
\Delta y,
\Delta z,
\Delta roll,
\Delta pitch,
\Delta yaw,
gripper
]
$$

È importante notare che le sei componenti dell'end-effector sono **delta**, non pose cartesiane assolute:

$$
p_{target}=p_t+\Delta p_t
$$

e analogamente viene applicato un incremento all'orientamento corrente.

RT-2 non produce quindi direttamente:

* joint torques;
* correnti dei motori;
* joint accelerations;
* coppie articolari.

Produce invece un **comando cartesiano relativo dell'end-effector**, che viene successivamente realizzato dallo stack di controllo del robot.

La catena è quindi:

$$
\text{RT-2}
\rightarrow
\text{relative EEF command}
\rightarrow
\text{robot low-level control stack}
\rightarrow
\text{actuators}
$$

RT-2 rimane quindi una policy di alto livello rispetto ai servo controller fisici, pur essendo una policy di **low-level manipulation** rispetto a sistemi modulari come SayCan.

## Discretizzazione delle azioni

Come RT-1, RT-2 non predice direttamente valori continui.

Ciascuna delle sette dimensioni continue:

$$
[
\Delta x,
\Delta y,
\Delta z,
\Delta roll,
\Delta pitch,
\Delta yaw,
gripper
]
$$

viene discretizzata uniformemente in **256 bin**:

$$
a_j\in\mathbb{R}
\rightarrow
q(a_j)\in\{0,\dots,255\}
$$

Il comando `terminate` rimane invece discreto.

Il risultato può essere espresso come una sequenza di interi, ad esempio:

```text
1 128 91 241 5 101 127 217
```

dove la posizione del token all'interno della sequenza determina quale componente dell'azione rappresenta.

Questa è una distinzione importante.

Il token:

$$
128
$$

non significa intrinsecamente, ad esempio, “movimento lungo $x$”.

È la sua **posizione nella action string** a determinarne la semantica.

### Da bin numerici a token del VLM

Per trasformare effettivamente questi valori in output del language model è necessario associare i 256 bin a token appartenenti al vocabolario del VLM.

Le due famiglie utilizzate in RT-2 gestiscono questo problema in modo differente.

Per **PaLI-X**, il tokenizer dispone già di rappresentazioni adatte per numeri interi, quindi i valori discretizzati possono essere espressi attraverso token numerici.

Per **PaLM-E**, invece, vengono riservati **256 token poco utilizzati del vocabolario**, che vengono reinterpretati come action tokens.

In entrambi i casi si ottiene una corrispondenza:

$$
\{0,\dots,255\}
\longleftrightarrow
\{\tau_0,\dots,\tau_{255}\}
$$

e quindi:

$$
a_t
\rightarrow
(\tau_1,\tau_2,\dots,\tau_8).
$$

Il language model può così utilizzare la propria normale output head per generare sia linguaggio sia azioni.

Questa è una delle idee più importanti di RT-2:

$$
\boxed{\text{robot action}=\text{un'altra ``lingua'' del VLM}}
$$

## Generazione autoregressiva

Le componenti dell'azione vengono generate **autoregressivamente**.

Indicando i token dell'azione con:

$$
\tau_1,\tau_2,\dots,\tau_N
$$

la probabilità dell'azione viene fattorizzata come:

$$
P(a_t\mid o_t,l)
=
\prod_{j=1}^{N}
P(
\tau_j
\mid
o_t,l,\tau_{<j}
)
$$

Quindi il modello non produce necessariamente tutte le dimensioni dell'azione con un'unica operazione parallela.

Predice:

$$
\tau_1
\rightarrow
\tau_2
\rightarrow
\dots
\rightarrow
\tau_N.
$$

Questo permette di riutilizzare direttamente il meccanismo di decoding del VLM, ma introduce anche un'importante limitazione computazionale: **per produrre una singola azione robotica devono essere generati più token attraverso un modello che può contenere decine di miliardi di parametri**.

Questo problema diventerà uno dei principali motivi per cui i VLA successivi esploreranno action head continue, diffusion e flow matching.


### Constrained decoding

Durante un normale task linguistico il VLM può produrre token appartenenti all'intero vocabolario:

$$
\tau \in V.
$$

Durante il controllo robotico, invece, non avrebbe senso permettere al modello di produrre arbitrariamente parole al posto di comandi validi.

Per questo durante l'action decoding lo spazio degli output viene **vincolato ai token compatibili con l'action representation**.

In forma concettuale:

$$
P(\tau\mid o_t,l)
\rightarrow
P(\tau\mid o_t,l,\tau\in V_{action})
$$

con:

$$
V_{action}\subset V.
$$

Il modello conserva quindi un unico spazio rappresentazionale, ma durante l'esecuzione robotica vengono accettati soltanto output validi per l'action space.


## Co-fine-tuning

La seconda innovazione fondamentale di RT-2 è il modo in cui viene effettuato il training.

Una possibilità apparentemente naturale sarebbe:

$$
\text{pretrained VLM}
\xrightarrow{\text{fine-tuning solo robot data}}
\text{robot policy}.
$$

Gli autori mostrano però che questo approccio sfrutta peggio le capacità di generalizzazione del modello.

RT-2 utilizza invece **co-fine-tuning**:

$$
D_{\text{training}}
=
D_{\text{web}}
+
D_{\text{robot}}.
$$

Durante il fine-tuning vengono quindi mantenuti sia gli esempi robotici sia parte dei task vision-language originali utilizzati dal VLM.

Per un esempio web:

$$
(image,text)
\rightarrow
\text{language tokens}
$$

mentre per un esempio robotico:

$$
(o_t,l)
\rightarrow
\text{action tokens}.
$$

Entrambi utilizzano la stessa loss autoregressiva:

$$
\mathcal{L}
=
-\sum_{j}
\log
P(y_j\mid y_{<j},x)
$$

dove \(y_j\) può essere:

* un normale token linguistico;
* un action token.

Il modello viene quindi addestrato contemporaneamente a **parlare** e ad **agire**.


### Perché mantenere i web data?

Mantenere i dati vision-language durante il fine-tuning è importante per evitare che l'apprendimento delle azioni robotiche elimini parte della conoscenza acquisita durante il pre-training.

Il problema può essere visto come:

$$
\text{VLM pretrained}
\rightarrow
\text{robot-only fine-tuning}
\rightarrow
\text{possible catastrophic forgetting}.
$$

Con il co-fine-tuning:

$$
\text{robot data}
+
\text{web data}
$$

la conoscenza semantica continua ad essere esercitata mentre il modello apprende il nuovo “linguaggio” delle azioni.

Le ablation del paper mostrano inoltre che il **co-fine-tuning produce una generalizzazione migliore rispetto al semplice fine-tuning sui robot data**, sia nei modelli da 5B sia in quelli più grandi.

Questo risultato è importante perché indica che il miglioramento non deriva semplicemente dall'utilizzo di una rete più grande: deriva anche dal mantenimento della distribuzione vision-language durante l'apprendimento del controllo.


### Cosa viene realmente trasferito dal web?

Un punto fondamentale è distinguere tra **semantic transfer** e **motor-skill transfer**.

Il web contiene immagini e testo che possono insegnare al modello concetti come:

$$
\text{``dinosaur''},
\quad
\text{``smallest object''},
\quad
\text{``number 3''},
\quad
\text{``object about to fall''}.
$$

Non contiene però dimostrazioni fisiche del robot che insegnino come eseguire nuovi movimenti.

Quindi:

$$
\boxed{
\text{web data}
\rightarrow
\text{semantic/visual knowledge}
}
$$

mentre:

$$
\boxed{
\text{robot data}
\rightarrow
\text{motor capabilities}
}
$$

RT-2 può utilizzare una skill motoria già appresa, come il picking, su un concetto semanticamente nuovo.

Ad esempio, se durante le robot demonstrations il modello ha imparato:

$$
\text{pick}(object)
$$

e dal web conosce il significato di “extinct animal”, può eseguire:

> *pick the extinct animal*

su un dinosauro giocattolo, pur non avendo necessariamente osservato quella specifica combinazione durante il robot training.

Questo **non significa** che RT-2 possa apprendere dal web una nuova dinamica di manipolazione mai presente nelle dimostrazioni robotiche.


## Closed-loop control

Nonostante l'enorme dimensione del modello e l'esecuzione remota, RT-2 viene utilizzato come policy **closed-loop**.

Il ciclo è:

$$
o_t
\rightarrow
\text{RT-2}
\rightarrow
a_t
\rightarrow
\text{robot}
\rightarrow
o_{t+1}
\rightarrow
\text{RT-2}
\rightarrow
a_{t+1}.
$$

Quindi RT-2 non produce semplicemente all'inizio un piano completo che viene successivamente eseguito open-loop.

Dopo l'esecuzione dell'azione, la nuova osservazione visiva viene utilizzata per generare il comando successivo.

Questo distingue RT-2 da un VLM utilizzato esclusivamente come planner:

$$
\text{VLM}
\rightarrow
\text{text plan}
\rightarrow
\text{separate robot policy}.
$$

In RT-2:

$$
\boxed{
\text{VLM}
\rightarrow
\text{robot action}
}
$$

direttamente.

## Evaluation

RT-2 viene valutato attraverso **oltre 6.000 trial robotici**.

La valutazione è costruita in modo da misurare non soltanto la capacità di eseguire task presenti nei robot data, ma soprattutto la capacità di trasferire la conoscenza ottenuta dal pre-training vision-language.

Gli esperimenti principali possono essere separati in:

1. task già osservati;
2. generalizzazione visiva;
3. generalizzazione semantica;
4. emergent capabilities;
5. Language Table;
6. esperimenti di reasoning.

## Chain-of-Thought e planning

Gli autori esplorano inoltre una variante di RT-2 in cui linguaggio e azioni vengono interleaved.

Il normale output è:

$$
\text{Action: }a_t.
$$

Nella variante con Chain-of-Thought viene introdotto prima un passaggio linguistico:

$$
\text{Plan}
\rightarrow
\text{Action}.
$$

Ad esempio, di fronte all'istruzione:

> *I need to hammer a nail, what object from the scene might be useful?*

il modello può produrre concettualmente:

$$
\text{Plan: pick the rock}
$$

seguito dai token dell'azione fisica necessaria.

Questa variante viene ottenuta attraverso ulteriore fine-tuning e costituisce un esperimento aggiuntivo, **non il normale funzionamento di tutte le versioni di RT-2**.

È importante quindi non concludere che il modello base utilizzi sempre Chain-of-Thought interno prima di ciascuna azione.

L'esperimento mostra però che planning semantico e controllo possono potenzialmente essere rappresentati all'interno dello stesso modello:

$$
\text{vision}
+
l
\rightarrow
\text{language reasoning}
\rightarrow
\text{action tokens}.
$$

Questo contrasta con sistemi come SayCan, nei quali:

$$
\text{LLM planner}
\rightarrow
\text{separate skill policy}.
$$
