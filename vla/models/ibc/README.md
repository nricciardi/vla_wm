# Implicit Behavioral Cloning

**Implicit Behavioral Cloning (IBC)** è un metodo di imitation learning che rappresenta una policy mediante un **energy-based model (EBM)**.

Il lavoro non introduce un VLA completo: **manca un task conditioning linguistico** generalista e ogni policy è addestrata per lo specifico dominio sperimentale. Il suo interesse per i VLA risiede nel modo in cui trasforma la predizione delle azioni da regressione esplicita a problema di compatibilità tra osservazioni e azioni.

Il punto di partenza è il behavioral cloning. Dato un dataset di dimostrazioni

$$
\mathcal{D}=\{(o_i,a_i)\}_{i=1}^{N},
$$

dove $o_i$ è un'osservazione, $a_i$ l'azione dell'esperto associata ed $N$ il numero di esempi, una policy esplicita apprende normalmente una funzione $F_\theta$ tale che

$$
\hat{a}=F_\theta(o).
$$

Se la loss è un errore quadratico, il modello viene spinto verso la media condizionale delle azioni osservate. Questo comportamento diventa problematico quando lo stesso stato ammette più strategie valide: la media tra due modalità può corrispondere a un'azione mai eseguita dall'esperto e fisicamente inefficace.

## Policy implicita

IBC *apprende* invece una funzione scalare

$$
E_\theta(o,a)\in\mathbb{R},
$$

che assegna energia bassa alle coppie osservazione-azione compatibili ed energia alta a quelle incompatibili. L'azione non è emessa direttamente dalla rete, ma viene definita implicitamente come

$$
\hat{a}=\arg\min_{a\in\mathcal{A}}E_\theta(o,a),
$$

dove $\mathcal{A}$ indica l'action space. La stessa osservazione può così produrre un paesaggio energetico con più minimi distinti, ciascuno corrispondente a una modalità valida del comportamento.

La distribuzione condizionale associata può essere scritta come

$$
p_\theta(a\mid o)
=
\frac{\exp[-E_\theta(o,a)]}{Z(o,\theta)},
$$

dove $Z(o,\theta)$ è la costante di normalizzazione, generalmente difficile da calcolare esattamente. IBC evita di stimarla sull'intero spazio continuo delle azioni attraverso un insieme finito di esempi negativi.

![Implicit vs explicit policy](features/ibc_implicit_vs_explicit.png)

## Training con azioni negative

Per ogni coppia positiva $(o_i,a_i)$ proveniente dalle dimostrazioni vengono campionate azioni negative $\tilde{a}_{i,j}$. Una loss di tipo **InfoNCE** deve assegnare all'azione dimostrata una probabilità maggiore rispetto ai controesempi:

$$
\mathcal{L}_{\mathrm{IBC}}
=
-\sum_{i=1}^{N}
\log
\frac{\exp[-E_\theta(o_i,a_i)]}
{\exp[-E_\theta(o_i,a_i)]+
\sum_{j=1}^{M}\exp[-E_\theta(o_i,\tilde{a}_{i,j})]},
$$

dove $M$ è il numero di azioni negative considerate per ciascun esempio. Il modello apprende quindi una superficie di compatibilità locale tra osservazione e azione, invece di minimizzare direttamente la distanza numerica da un singolo target.

La scelta e il raffinamento dei negativi sono centrali. Esempi troppo facili forniscono poco segnale; esempi vicini a modalità plausibili obbligano invece la funzione energetica a rappresentare con precisione i confini tra azioni valide e non valide.

## Inferenza

Al momento dell'esecuzione occorre risolvere il problema di minimizzazione nello spazio delle azioni. Il lavoro confronta procedure **derivative-free**, una **variante autoregressiva** che ottimizza le componenti per coordinate e Langevin **sampling basato sul gradiente dell'energia**.

Nella procedura sampling-based vengono generate più azioni candidate entro i limiti di $\mathcal{A}$, valutate dalla rete e progressivamente ricampionate attorno alle candidate con energia minore. Questa ricerca rende la policy più flessibile di un singolo passaggio feed-forward, ma introduce costo computazionale e iperparametri di inferenza.

## Perché le policy implicite aiutano

Una policy esplicita continua deve attraversare *tutti i valori intermedi* quando approssima una funzione discontinua. La composizione tra una funzione energetica continua e l'operatore $\arg\min$ può invece **far passare bruscamente la soluzione da un minimo all'altro**. Questo consente di rappresentare decisioni come *"aggirare l'ostacolo a sinistra oppure a destra"* senza interpolare necessariamente tra le due strategie.

Lo stesso meccanismo permette di descrivere **mapping set-valued**, nei quali più azioni sono simultaneamente corrette. L'energia non è costretta a scegliere in anticipo una sola modalità: può assegnare valori bassi a regioni separate dell'action space e lasciare all'inferenza la selezione finale.

## Dataset ed evaluation

La valutazione comprende sei famiglie di esperimenti. I task **D4RL Human-Experts** includono domini Franka Kitchen e Adroit con action space fino a 30 dimensioni. Seguono un particle integrator, block pushing, planar sweeping e un task simulato di sweeping bimanuale con due KUKA IIWA, nel quale 1.000 dimostrazioni scripted controllano complessivamente 12 DoF cartesiani.

Gli esperimenti real-world usano un *xArm6* dotato di **end-effector cilindrico**. La policy osserva immagini RGB prospettiche a 5 Hz e apprende da dimostrazioni teleoperate quattro task: spingere due blocchi verso target assegnati; risolvere una variante multimodale in cui ordine e target possono cambiare; inserire un blocco con tolleranza di 1 mm; separare blocchi blu e gialli.

I dataset reali contengono rispettivamente 95, 410, 223 e 502 dimostrazioni. Questa distribuzione rende esplicito che IBC non è pre-addestrato su un corpus generalista: ogni esperimento usa dati raccolti per il comportamento target. Il confronto principale mantiene simili gli encoder e contrappone la policy EBM a behavioral cloning esplicito con loss MSE.


#### Novelty

IBC mostra che il behavioral cloning può essere formulato come **conditional energy-based modeling** senza ricorrere a reward o interazioni on-policy. La novità non consiste soltanto nell'usare una loss diversa, ma nel sostituire la mappa esplicita osservazione-azione con una funzione la cui minimizzazione definisce implicitamente la policy.

Questa formulazione gestisce naturalmente azioni multimodali e discontinuità, due proprietà frequenti nei task contact-rich. Gli esperimenti isolano il contributo della rappresentazione confrontando modelli impliciti ed espliciti con architetture per quanto possibile analoghe.

#### Limiti

Training e inferenza richiedono di valutare numerose coppie osservazione-azione. La qualità dipende dalla copertura dei campioni negativi e dall'ottimizzatore usato per trovare i minimi; con action space molto ampi, la ricerca può diventare onerosa o mancare una modalità valida.

IBC non risolve inoltre il covariate shift del behavioral cloning: la policy continua ad apprendere soltanto dagli stati visitati dall'esperto. Gli esperimenti reali sono task-specifici, con una singola piattaforma e senza linguaggio, e non dimostrano trasferimento cross-task o cross-embodiment.
