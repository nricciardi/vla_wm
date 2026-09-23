# Approssimazione delle funzioni valore

Nei metodi classici le funzioni di valore sono rappresentate mediante delle **tabelle di lookup**. In questo tipo di rappresentazione, a ogni stato $s$ viene associato un valore $V(s)$, oppure, nel caso delle funzioni action-value, a ogni coppia stato-azione $(s,a)$ viene associato un valore $Q(s,a)$.

Questo approccio funziona bene quando il numero di stati e di azioni è *limitato*. Tuttavia, molti problemi reali di Reinforcement Learning presentano spazi degli stati estremamente grandi o addirittura continui. Ad esempio, il backgammon possiede circa $10^{20}$ configurazioni possibili, mentre nel gioco del Go il numero di stati può arrivare all’ordine di $10^{170}$. Nel controllo di un elicottero, invece, variabili come posizione, velocità e orientamento assumono valori continui.

In questi casi non è possibile memorizzare esplicitamente il valore di ogni stato. Inoltre, anche disponendo di memoria sufficiente, sarebbe troppo lento visitare e apprendere separatamente il valore di ciascuno stato.

La soluzione consiste nell’utilizzare la **Value Function Approximation**, cioè rappresentare la funzione di valore attraverso una funzione parametrica dipendente da un vettore di parametri $w$:

$$
\hat{v}(s,w) \approx v_\pi(s)
$$

oppure, per la funzione action-value:

$$
\hat{q}(s,a,w) \approx q_\pi(s,a)
$$

L’obiettivo non è più apprendere direttamente un valore distinto per ogni stato, ma **apprendere i parametri** ($w$) della funzione approssimante.

Il vantaggio principale è la **generalizzazione**: aggiornando il valore stimato di uno stato, possiamo modificare anche la stima di altri stati simili. In questo modo l’agente può produrre valutazioni ragionevoli anche per stati che non ha mai osservato direttamente.

## Tipologie di approssimazione

La funzione approssimante può ricevere input differenti.

Nel caso della funzione di valore degli stati la funzione riceve lo stato e restituisce una singola stima:

$$
s \longrightarrow \hat{v}(s,w)
$$


Nel caso della funzione action-value possiamo utilizzare una funzione che riceve sia lo stato sia l’azione:

$$
(s,a) \longrightarrow \hat{q}(s,a,w)
$$

In alternativa, soprattutto quando l’insieme delle azioni è discreto, la funzione può ricevere solamente lo stato e produrre contemporaneamente un valore per ogni azione disponibile:

$$
s \longrightarrow
\left[
\hat{q}(s,a_1,w),
\dots,
\hat{q}(s,a_m,w)

\right]
$$

Quest’ultima struttura è comune nelle reti neurali utilizzate nei Deep Q-Network.

## Scelta del function approximator

Esistono diversi modelli che possono essere impiegati per approssimare una funzione di valore:

* combinazioni lineari di feature;
* reti neurali;
* alberi decisionali;
* metodi nearest-neighbour;
* basi di Fourier o wavelet.

Generalmente vengono considerati principalmente approssimatori **differenziabili**, perché permettono di aggiornare i parametri utilizzando il gradiente.

La scelta dell’algoritmo di apprendimento deve inoltre tenere conto di una caratteristica importante dei dati nel Reinforcement Learning: **le osservazioni non sono necessariamente indipendenti e identicamente distribuite**.

Le esperienze consecutive dell’agente sono infatti fortemente correlate. Inoltre, la distribuzione dei dati cambia durante l’apprendimento, perché la policy dell’agente viene modificata e porta a visitare regioni differenti dello spazio degli stati. I dati sono quindi **non stazionari** e **non i.i.d.**

Inoltre nel Reinforcement Learning il vero valore target di una rete neurale $v_\pi(s)$ **non è disponibile**. Deve quindi essere sostituito con un target ottenuto dall’esperienza, come il ritorno Monte-Carlo o un target Temporal-Difference.

Per utilizzare un approssimatore parametrico, lo stato viene generalmente rappresentato attraverso un vettore di caratteristiche.
Le feature descrivono gli aspetti dello stato ritenuti rilevanti per la previsione del valore.
Ad esempio:

- nel controllo di un robot, le feature possono essere le distanze da alcuni punti di riferimento;
- in un problema finanziario possono rappresentare trend, rendimenti o volatilità;
- negli scacchi possono indicare il numero e la configurazione dei pezzi;
- nella supply chain possono rappresentare livello di inventario, domanda prevista, backlog, lead time e capacità disponibile.

Le feature possono essere progettate manualmente oppure apprese automaticamente, come avviene nelle reti neurali profonde.


## Incremental Prediction Algorithms

Si ricorda che nel Reinforcement Learning il vero valore target di una rete neurale $v_\pi(s)$ **non è disponibile**, ma l’agente osserva solamente stati, azioni e ricompense.

Per questo motivo il valore vero viene sostituito da un **target** costruito a partire dall'esperienza.

La forma *generale* dell’aggiornamento è:

$$
\Delta w = \alpha (target - \hat{v}(s,w)) \nabla_w \hat{v}(s,w)
$$

La scelta del target determina l’algoritmo utilizzato.

### Monte-Carlo con approssimazione della funzione di valore

Nei metodi Monte-Carlo il target è il **ritorno** $G_t$ osservato:

$$
G_t = \alpha (G_t - \hat{v}(s_t,w)) \nabla_w \hat{v}(s_t,w)
$$



Il ritorno $G_t$ è un campione **non distorto**, o unbiased, del vero valore $v_\pi(s_t)$.

Tuttavia presenta generalmente un’elevata varianza, perché dipende da tutte le ricompense casuali osservate fino alla conclusione dell’episodio.

Un episodio può essere interpretato come un insieme di esempi di training:

$$
(s_1,G_1),
(s_2,G_2),
...,
(S_T,G_T)
$$

In questo senso, la valutazione Monte-Carlo con function approximation è simile a un problema di apprendimento supervisionato in cui gli stati sono gli input e i ritorni sono i target.

La valutazione Monte-Carlo converge verso un minimo locale della funzione obiettivo anche con approssimatori non lineari, assumendo condizioni adeguate sul learning rate e sulla copertura degli stati.

### TD(0) con approssimazione della funzione di valore

Nel Temporal-Difference Learning il target è:

$$
target = r_{t+1} + \gamma\hat{v}(s_{t+1},\mathbf{w})
$$

L’aggiornamento è:

$$
\Delta w = \alpha \delta_t \nabla_w \hat{v}(s_t,\mathbf{w})
$$

$$
\delta_t = r_{t+1} + \gamma\hat{v}(s_{t+1},\mathbf{w}) - \hat{v}(s_t,\mathbf{w})
$$

Il target TD è **distorto**, perché contiene una stima $\hat{v}(s_{t+1},\mathbf{w})$ che può essere errata. Tuttavia, tende ad avere varianza inferiore rispetto al ritorno Monte-Carlo.

TD aggiorna inoltre la funzione dopo ogni transizione, senza attendere la fine dell’episodio. Può quindi essere utilizzato anche in problemi continui.

Per l’approssimazione lineare e in condizioni on-policy, TD(0) converge a una soluzione vicina all’ottimo globale. Non coincide necessariamente con la soluzione che minimizza direttamente l’errore quadratico rispetto a $v_\pi$, perché TD converge a un punto fisso proiettato dell’operatore di Bellman.

### TD(lambda) con approssimazione della funzione di valore

Analogamente a TD(0) possiamo utilizzare un target TD(lambda).

Nel caso **forward view**:

$$
\Delta w = \alpha (G_t^\lambda - \hat{v}(s_t,w)) \nabla_w \hat{v}(s_t,w)
$$

Mentre nel caso più utile del **backward view**:

$$
\Delta w = \alpha \delta_t E_t
$$

$$
\delta_t = r_{t+1} + \gamma\hat{v}(s_{t+1},w) - \hat{v}(s_t,w)
$$

$$
E_t = \gamma\lambda E_{t-1} + \nabla_w \hat{v}(s_t,w)
$$

## Incremental Control Algorithms

Nella prediction l’obiettivo è valutare una policy fissata. Nel controllo, invece, dobbiamo contemporaneamente:

1. Stimare la funzione di valore della policy corrente;
2. Migliorare la policy utilizzando tali stime.

Si utilizza quindi una forma approssimata della Generalized Policy Iteration:

$$
\hat{q}(\cdot,\cdot,w) \approx q_\pi
$$

seguita da un miglioramento della policy, ad esempio attraverso una strategia $\varepsilon$-greedy.

La policy $\varepsilon$-greedy sceglie l’azione con valore stimato più elevato con probabilità (1-$\varepsilon$), mentre con probabilità ($\varepsilon$) seleziona un’azione casuale. Questo permette di bilanciare exploitation ed exploration.

Per il controllo è necessario approssimare la funzione action-value:

$$
\hat{q}(S,A,w) \approx q_\pi(S,A)
$$




## Convergenza degli algoritmi di prediction

L’utilizzo di function approximation introduce problemi di stabilità che non si presentano nello stesso modo con le rappresentazioni tabellari.

Questo fenomeno è associato alla cosiddetta **deadly triad**, cioè alla combinazione di:

1. function approximation;
2. bootstrapping;
3. apprendimento off-policy.

La presenza simultanea dei tre elementi può rendere instabile l’apprendimento.


Le garanzie di convergenza dipendono dalla rappresentazione, dall’algoritmo e dal fatto che l’apprendimento sia on-policy oppure off-policy.

Per la policy evaluation:

| Algoritmo                | Table lookup |          Lineare |               Non lineare |
| ------------------------ | -----------: | ---------------: | ------------------------: |
| MC on-policy             |     converge |         converge |       converge localmente |
| TD(0) on-policy          |     converge |         converge | nessuna garanzia generale |
| TD(lambda) on-policy  |     converge |         converge | nessuna garanzia generale |
| MC off-policy            |     converge |         converge |       converge localmente |
| TD(0) off-policy         |     converge | nessuna garanzia |          nessuna garanzia |
| TD(lambda) off-policy |     converge | nessuna garanzia |          nessuna garanzia |

La mancanza di una garanzia teorica non implica che l’algoritmo diverga necessariamente in ogni applicazione. Significa però che esistono problemi per i quali può divergere.

### Gradient TD Learning

Uno dei motivi della possibile divergenza è che il classico aggiornamento TD non corrisponde, in generale, al gradiente esatto di una funzione obiettivo.

L’aggiornamento è definito **semi-gradient**, perché durante la derivazione il target TD viene trattato come se fosse costante, anche se contiene a sua volta la funzione parametrica.

Infatti nel TD(0) l'errore temporale è:

$$
\delta_t = r_{t+1} + \gamma\hat{v}(s_{t+1},w) - \hat{v}(s_t,w)
$$

Supponendo di approssimare $\hat{v}(s_t,w)$ con una combinazione lineare di feature:

$$
\hat{v}(s_t,w) = w^T x(s_t)
$$

$$
\nabla_w \hat{v}(s_t,w) = x(s_t)
$$

Si ottiene che l'aggiornamento dei pesi $w$ risulta essere:

$$
\Delta w = \alpha \delta_t \nabla_w \hat{v}(s_t,w) = \alpha \delta_t x(s_t)
$$

Si noti però che $\delta_t$ dipende da $w$ attraverso $\hat{v}(s_{t+1},w)$, quindi dipende dagli stessi parametri che si vogliono aggiornare.

Gli algoritmi **Gradient TD** vengono costruiti per seguire il vero gradiente di una funzione obiettivo, tipicamente legata all’errore di Bellman proiettato.

Questo permette di ottenere garanzie di convergenza più forti anche in condizioni off-policy e con approssimazione lineare.

La maggiore stabilità ha però un costo: gli algoritmi Gradient TD sono generalmente più complessi e richiedono parametri o stime ausiliarie.

Nel controllo la situazione è ancora più complessa, perché la policy cambia durante l’apprendimento e modifica continuamente la distribuzione dei dati.

Con rappresentazione tabellare, Monte-Carlo Control, Sarsa e Q-learning possiedono garanzie di convergenza sotto opportune condizioni di esplorazione e learning rate, mentre con approssimazione lineare:

- Monte-Carlo Control e Sarsa possono oscillare intorno a una soluzione quasi ottimale
- Il Q-learning lineare non possiede garanzie generali
- Alcune versioni di Gradient Q-learning presentano garanzie migliori

Con approssimatori non lineari, come le reti neurali, le garanzie teoriche classiche sono molto limitate. Il loro successo dipende quindi anche dall’utilizzo di tecniche empiriche di stabilizzazione.


## Batch Reinforcement Learning

Gli algoritmi incrementali aggiornano i parametri dopo ogni nuova esperienza. Sono semplici e hanno un costo ridotto per aggiornamento, ma non sono efficienti nell’utilizzo dei dati.

Una transizione osservata viene normalmente utilizzata una sola volta e poi scartata. Se raccogliere esperienza è costoso, può essere conveniente riutilizzare più volte le stesse osservazioni.

I metodi di **Batch Reinforcement Learning** cercano la funzione di valore che si adatta meglio a un intero dataset di esperienze già raccolte.


### Least Squares Prediction

Supponiamo di disporre di un dataset formato da coppie stato-valore:

$$
\mathcal{D} = \{ (s_1, v_1^\pi) \}, \ldots, \{ (s_T, v_T^\pi) \}
$$

e di voler approssimare:

$$
\hat{v}(s,w) \approx v_\pi(s)
$$

La soluzione **Least Squares** $LS$ è il vettore di parametri che minimizza la somma degli errori quadratici:

$$
LS(w) = \sum_{t=1}^T (v_t^\pi - \hat{v}(s_t, w))^2
$$

Equivalentemente:

$$
LS(w) = \mathbb{E}_{(s,v) \sim \mathcal{D}}[(v - \hat{v}(s, w))^2]
$$
Rispetto agli aggiornamenti online, l’approccio batch considera congiuntamente tutte le osservazioni e cerca il miglior adattamento complessivo al dataset.

### Linear Least Squares Prediction

Si consideri un'approssimazione lineare della funzione di valore:

$$
\hat{v}(s, w) = w^T x(s)
$$

La soluzione least squares può essere calcolata **direttamente**, senza eseguire numerosi aggiornamenti SGD, tramite forma chiusa.

Nel punto di minimo, il gradiente complessivo è uguale a zero:

$$
LS(w) = \sum_{t=1}^T (v_t^\pi - \hat{v}(s_t, w))^2
$$

$$
\nabla_w LS(w) = \sum_{t=1}^T (v_t^\pi - \hat{v}(s_t, w)) \nabla_w \hat{v}(s_t, w) = \sum_{t=1}^T (v_t^\pi - w^T x(s_t)) x(s_t) = 0
$$

Separando i termini si ottiene:

$$
\sum_{t=1}^T v_t^\pi x(s_t) = \sum_{t=1}^T x(s_t)x(s_t)^T w
$$

Definendo:

$$
A = \sum_{t=1}^T x(s_t)x(s_t)^T
$$

$$
b = \sum_{t=1}^T v_t^\pi x(s_t)
$$

La soluzione least squares in forma chiusa è quindi:

$$
w = A^{-1} b
$$

Tuttavia, il calcolo diretto dell’inversa ha complessità dell’ordine di $O(N^3)$, riducibile a $O(N^2)$ con metodi più efficienti.


### Least Squares Monte Carlo

Nel reinforcement learning, il target $v_t^\pi$ non è disponibile. Possiamo però utilizzare il ritorno Monte-Carlo osservato come target:

$$
v_t^\pi \approx G_t
$$

Nel caso *lineare* si ha:

$$
b = \sum_{t=1}^T G_t x(s_t)
$$

$$
\implies w = A^{-1} b = \left[ \sum_{t=1}^T x(s_t)x(s_t)^T \right]^{-1} \sum_{t=1}^T G_t x(s_t)
$$


L’interpretazione è semplice: effettuiamo una regressione lineare che cerca di prevedere il ritorno osservato a partire dalle feature dello stato.

Poiché $G_t$ è un stimatore non distorto del valore atteso, LSMC possiede buone proprietà statistiche. Tuttavia, il ritorno può avere varianza elevata e richiede episodi completi.

LSMC conserva le proprietà del Monte-Carlo ed è utilizzabile sia on-policy sia off-policy, almeno nel caso lineare considerato.

### Least Squares Temporal Difference

In modo analogico, possiamo utilizzare un target TD(0):

$$
v_t^\pi \approx r_{t+1} + \gamma \hat{v}(s_{t+1}, w)
$$

A differenza di una normale regressione, questo target dipende dallo stesso vettore w che vogliamo trovare.

LSTD non tratta il target come un valore fissato. Parte invece dalla condizione secondo cui, alla soluzione, la somma degli aggiornamenti TD deve essere **zero**. Nel caso lineare $\nabla_w \hat{v}(s_t, w) = x(s_t)$:

$$
\sum_{t=1}^T \delta_t \nabla_w \hat{v}(s_t, w) = \sum_{t=1}^T (r_{t+1} + \gamma \hat{v}(s_{t+1}, w) - \hat{v}(s_t, w)) \nabla_w \hat{v}(s_t, w) = 0
$$

Si suppone che la somma debba essere zero perché alla soluzione ottima, l’errore TD medio deve essere nullo. In altre parole, la funzione di valore approssimata non dovrebbe avere un bias sistematico.

Riorganizzando i termini si ottiene:

$$
[ \sum_{t=1}^T x(s_t) [x(s_t) - \gamma x(s_{t+1})]^T ] w = \sum_{t=1}^T r_{t+1} x(s_t)
$$

La soluzione in forma chiusa diventa quindi:

$$
A = \sum_{t=1}^T x(s_t) [x(s_t) - \gamma x(s_{t+1})]^T
$$

$$
b = \sum_{t=1}^T r_{t+1} x(s_t)
$$

$$
w = A^{-1} b
$$

LSTD è particolarmente importante perché, nella policy evaluation lineare, può essere applicato anche in condizioni off-policy nelle quali il TD incrementale standard non possiede garanzie generali.


## Least Squares Policy Iteration

Finora LSTD è stato utilizzato per la prediction, cioè per valutare una policy fissata. Per risolvere un problema di controllo dobbiamo alternare:

1. Policy evaluation
2. Policy improvement

La **Least Squares Policy Iteration** (LSPI) utilizza:

- LSTDQ per valutare la policy
- Una scelta greedy per migliorare la policy


LSPI rappresenta la funzione action-value tramite un'approssimazione lineare:

$$
\hat{q}(s,a,w) = w^T x(s,a)
$$


Le feature quindi *dipendono sia dallo stato sia dall'azione*.

Un metodo comune per azioni discrete consiste nell’utilizzare blocchi separati di feature. Se lo stato possiede il vettore $\phi(s) \in \R^n$ e sono disponibili $m$ azioni discrete, possiamo definire le feature per la coppia stato-azione come:

$$
x(s,a_1) =
\begin{bmatrix}
\phi(s) \\
0 \\
\vdots \\
0
\end{bmatrix} \in \R^{m \times n}
$$
$$
x(s,a_2) =
\begin{bmatrix}
0 \\
\phi(s) \\
\vdots \\
0
\end{bmatrix} \in \R^{m \times n}
$$
$$
x(s,a_m) =
\begin{bmatrix}
0 \\
0 \\
\vdots \\
\phi(s)
\end{bmatrix} \in \R^{m \times n}
$$

LSPI è **off-policy**, infatti può apprendere da un dataset di esperienze raccolte con una policy differente da quella che si sta valutando. Questo lo rende particolarmente utile in scenari in cui l'esplorazione è costosa o rischiosa, come nel controllo robotico o nei sistemi finanziari.

Per ogni transizione osservata, considera:

- L'azione $a_t$ realmente eseguita nello stato $s_t$
- L'azione $A'_{t+1} = \pi(s_{t+1})$ che la policy da valutare sceglierebbe nello stato successivo

Il target TD per LSTDQ diventa quindi:

$$
r_{t+1} + \gamma \hat{q}(s_{t+1}, A'_{t+1}; w)
$$

### Least Squares Temporal Difference Q-learning

LSTDQ è un algoritmo batch che utilizza la soluzione least squares per aggiornare i pesi della funzione action-value. A differenza del TD incrementale, LSTDQ considera l'intero dataset $\mathcal{D}$ di esperienze per calcolare i parametri ottimali in un singolo passo.

$$
\mathcal{D} = \{ (s_t, a_t, r_{t+1}, s_{t+1}) \}_{t=1}^T
$$

L'errore TD per LSTDQ è definito come:

$$
\delta_t = r_{t+1} + \gamma \hat{q}(s_{t+1}, A'_{t+1}; w) - \hat{q}(s_t, a_t; w)
$$

Approssimando la funzione action-value come combinazione lineare di feature, l'aggiornamento dei pesi $w$ può essere espresso in forma chiusa imponendo che la somma degli errori TD ponderati dalle feature sia zero:

$$
\sum_{t=1}^T \delta_t \nabla_w \hat{q}(s_t, a_t; w) = 0
$$

Essendo l'approssimazione lineare, abbiamo:

$$
\hat{q}(s,a,w) = w^T x(s,a)
$$

$$
\nabla_w \hat{q}(s_t, a_t; w) = x(s_t, a_t)
$$

$$
\implies \sum_{t=1}^T \delta_t \nabla_w \hat{q}(s_t, a_t; w) = \sum_{t=1}^T (r_{t+1} + \gamma w^T x(s_{t+1}, A'_{t+1}) - w^T x(s_t, a_t)) x(s_t, a_t) = 0
$$

Sia $x_t = x(s_t, a_t)$ e $x_t' = x(s_{t+1}, \pi(s_{t+1}))$ la condizione di ottimalità diventa:

$$
\sum_{t=1}^T x_t [r_{t+1} + \gamma w^T x_t' - w^T x_t] = 0
$$

Riorganizzando i termini si ottiene:

$$
w = [ \sum_{t=1}^T x_t (x_t - \gamma x_t')^T ]^{-1} \sum_{t=1}^T r_{t+1} x_t
$$

La dipendenza dalla policy si trova nelle feature successive $x_t'$ perché l’azione successiva viene scelta secondo $\pi$.
Cambiando policy, il dataset non cambia, ma cambia la matrice costruita da LSTDQ.


### LSPI Step-By-Step

L'input di LSPI è un dataset di transizioni $\mathcal{D}$, una policy iniziale $\pi_0$ e un numero massimo di iterazioni $N$.

#### Inizializzazione

L'inizializzazione prevede $\pi' = \pi_0$ dove la policy iniziale può essere casuale o definita sulla base di conoscenze preliminari.

#### Policy evaluation

Si utilizza LSTDQ per calcolare i pesi $w$ della funzione action-value approssimata $\hat{q}(s,a,w)$ per la policy corrente $\pi'$.

Allo step $k$:

$$
w_k = LSTDQ(\mathcal{D}, \pi_{k})
$$

Si ottiene quindi la funzione action-value approssimata:

$$
\hat{q}_k(s,a) = w_k^T x(s,a)
$$

#### Policy improvement

Si costruisce una nuova policy *greedy*:

$$
\pi_{k+1}(s) = \arg\max_{a \in \mathcal{A}} \hat{q}_k(s,a)
$$

#### Verifica della convergenza

L’algoritmo termina quando la nuova policy è uguale o sufficientemente simile alla precedente:

$$
\pi_{k+1} \approx \pi_k
$$

In caso contrario, si ripete la valutazione usando sempre lo stesso dataset.

Il punto fondamentale è che LSPI non deve interagire nuovamente con l’ambiente dopo ogni policy improvement. Le transizioni già raccolte vengono reinterpretate in base alla nuova policy.


## Coverage

I metodi batch come LSPI possono riutilizzare sempre gli stessi dati solo entro certi limiti. Infatti migliorando la policy, in generale cambia anche il tipo di esperienza che l’agente raccoglie.

Il punto chiave è distinguere due cose:

- Conoscere meglio l’ambiente
- Valutare meglio una policy usando dati già raccolti

LSPI non migliora la conoscenza dell’ambiente perché non osserva nuove transizioni. Migliora invece il modo in cui *interpreta* le transizioni già disponibili.

Supponiamo che nel dataset ci sia una transizione $(s,a,r,s')$. Questa transizione fornisce l'informazione che eseguendo l'azione $a$ nello stato $s$ si ottiene la ricompensa $r$ e si passa allo stato successivo $s'$.

Per valutare una policy $\pi$, LSPI costruisce un target TD utilizzando l'azione che la policy $\pi$ sceglierebbe nello stato successivo $s'$:

$$
r + \gamma \hat{q}(s', \pi(s'); w)
$$

Se successivamente la policy cambia, la transizione osservata resta la stessa, ma cambia l’azione che la nuova policy sceglierebbe in $s'$. Di conseguenza, il target TD cambia e la stima della funzione action-value viene aggiornata.

Quindi lo stesso dato può essere riutilizzato per valutare policy diverse. Non stiamo imparando una nuova transizione: stiamo propagando in modo diverso il valore futuro attraverso una transizione già nota.

In altre parole, LSPI raggiunge la convergenza alla policy ottima *in base ai dati disponibili*, quindi può convergere a una policy ottima per il dataset, ma non necessariamente per l’ambiente reale. Se la policy ottima per il dataset non è ottima per l’ambiente, l’agente può essere costretto a raccogliere nuove esperienze per migliorare ulteriormente la sua performance.

LSPI converge alla policy ottima in assoluto solo se il dataset contiene transizioni sufficientemente rappresentative di tutte le regioni rilevanti dello spazio degli stati e delle azioni. In altre parole, i dati devono avere una buona **coverage**.
Per ogni coppia stato-azione importante, il dataset dovrebbe contenere abbastanza transizioni da permettere di stimarne le conseguenze.

La debolezza principale degli approcci batch e in generale del reinforcement learning **offline** è proprio che una policy ottimale generata non può fornire stime accurate per regioni dello spazio degli stati e delle azioni che non sono state sufficientemente esplorate nel dataset. Tuttavia, LSPI potrebbe comunque produrre una stima di valore ragionevole anche per stati non osservati, grazie alla generalizzazione delle feature, ma questo non è garantito e sarebbe un'estrapolazione piuttosto che una stima accurata.

Dunque, la behaviour policy usata per costruire il dataset dovrebbe essere **sufficientemente esplorativa**.
Idealmente, dovrebbe garantire che tutte le coppie stato-azione potenzialmente importanti abbiano probabilità non nulla di essere osservate.
