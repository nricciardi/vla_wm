# Processi markoviani e MDP

## Proprietà di Markov

Tutta la teoria dei processi di Markov si basa su un'idea centrale: ***"Il futuro è indipendente dal passato, dato il presente"***.

In termini formali, uno stato $s_t$ gode della proprietà di Markov se la probabilità di transizione al prossimo stato dipende esclusivamente dallo stato attuale e non da tutta la sequenza di eventi (la "storia") che lo ha preceduto. Matematicamente si esprime così:


$$
p(s_{t+1}\mid s_t)=p(s_{t+1}\mid s_0,\ldots,s_t)
$$

Lo stato $s_t$ è quindi una variabile casuale che caratterizza perfettamente dove ci troviamo nell'ambiente e rappresenta una statistica sufficiente del futuro.


## Markov Process (Catene di Markov)

Il modello più basilare è il **Markov Process** (o Catena di Markov). Si tratta di un processo casuale "senza memoria", ovvero una sequenza di stati casuali $s_1, s_2, ...$ che rispettano la proprietà di Markov.

Un processo di Markov è definito da una tupla $(\mathcal{S}, \mathcal{P})$:

- $\mathcal{S}$: un insieme finito di stati.
- $\mathcal{P}$: Una matrice delle probabilità di transizione di stato.


In questo modello **non ci sono ancora né azioni né ricompense**. L'evoluzione del sistema è guidata unicamente dalla dinamica intrinseca dell'ambiente. La probabilità di passare da uno stato $s$ a uno stato successivo $s'$ è definita come:


$$
\mathcal{P}_{ss'} = P(s_{t+1}=s'|s_t=s)
$$

La matrice $\mathcal{P}$ racchiude tutte queste probabilità e definisce l'intera struttura del problema. Se proviamo a "campionare" questo sistema, otterremo degli **episodi**, ovvero sequenze casuali di lunghezza variabile regolate dalle probabilità della matrice.

## Markov Reward Process (MRP)

Il passo successivo è aggiungere dei giudizi di valore al nostro processo, trasformandolo in un **Markov Reward Process (MRP)**.

Un MRP è definito da una tupla $(S, \mathcal{P}, \mathcal{R}, \gamma)$:

- Gli stati $S$ e la matrice di transizione $\mathcal{P}$ rimangono identici.
- $\mathcal{R}$: Una funzione di ricompensa, che indica il valore atteso della ricompensa immediata partendo dallo stato $s$: $\mathcal{R}_s=\mathbb{E}[r_{t+1}|s_t=s]$.
- $\gamma$: Il fattore di sconto (discount factor), un valore compreso tra 0 e 1, $\gamma\in[0,1]$.



### Il Ritorno (Return) e il Fattore di Sconto

In RL non ci interessa solo la ricompensa immediata, ma l'*accumulo totale nel tempo*. L'obiettivo è massimizzare il **Return $G_t$**, ovvero la somma totale delle ricompense scontate a partire dal tempo $t$:


$$
G_t=r_{t+1}+\gamma r_{t+2}+\cdots=\sum_{k=0}^\infty\gamma^k r_{t+k+1}
$$

Il parametro $\gamma$ è fondamentale per valutare il "valore attuale" delle ricompense future. Ricevere una ricompensa $R$ tra $k+1$ passi temporali ha un valore scalato pari a $\gamma^k R$.

- Se $\gamma$ è vicino a 0, l'agente è ***"miope"*** e punta solo al guadagno immediato.
- Se $\gamma$ è vicino a 1, l'agente è ***"lungimirante"***.



Perché scontiamo le ricompense future? Principalmente per una convenienza matematica (evita di sommare all'infinito, garantendo la convergenza della serie). Inoltre, rappresenta l'**incertezza del futuro** o concetti economici concreti, come gli interessi maturati rispetto a guadagni ritardati. Infine, modella in modo realistico il comportamento umano e animale, che mostra una naturale preferenza per i premi immediati.

### State-Value Function

La funzione valore di stato $v(s)$ misura la "bontà" di uno stato a lungo termine. In un MRP, è definita matematicamente come il Ritorno Atteso partendo dallo stato $s$:


$$
v(s)=\mathbb{E}[G_t|s_t=s]
$$


## Equazione di Bellman per gli MRP

Arriviamo a uno dei concetti matematici più potenti: la **scomposizione ricorsiva**.
Il valore di uno stato può essere frammentato in due parti:

1. La ricompensa immediata attesa $r_{t+1}$
2. Il valore scontato del prossimo stato $\gamma v(s_{t+1})$



Seguendo i passaggi algebrici:


$$
v(s) = \mathbb{E}[r_{t+1} + \gamma G_{t+1} | s_t=s]
$$

$$
v(s) = \mathbb{E}[r_{t+1} + \gamma v(s_{t+1}) | s_t=s]
$$

Questa espressione prende il nome di **Equazione di Bellman** e afferma che il valore complessivo di uno stato $s$ è la ricompensa immediata più il valore del prossimo stato $s'$.

Per gli MRP, questa equazione è lineare e può essere scritta in forma matriciale:


$$
v=\mathcal{R}+\gamma\mathcal{P}v
$$

Essendo lineare, possiamo risolverla direttamente per ricavare i valori esatti di ogni stato con un'inversione di matrice:


$$
v=(I-\gamma\mathcal{P})^{-1}\mathcal{R}
$$

Tuttavia, la complessità computazionale di questa operazione è $O(n^3)$ per $n$ stati, rendendo la soluzione diretta **fattibile solo per problemi con pochi stati**. Per sistemi grandi, dovremo affidarci a metodi iterativi come la Programmazione Dinamica (Dynamic Programming), la valutazione Monte-Carlo o l'apprendimento Temporal-Difference.


## Markov Decision Process (MDP)

Negli MRP *non c'è possibilità di scelta*: si è trascinati dalle correnti delle probabilità ambientali. Se aggiungiamo le decisioni (o **azioni**), l'MRP si evolve definitivamente in un **Markov Decision Process (MDP)**. Questo è l'ambiente in cui gli agenti di Reinforcement Learning operano concretamente.

Un MDP è descritto dalla tupla $(S, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- $\mathcal{A}$: Un insieme finito di azioni a disposizione dell'agente.
- La matrice di transizione ora dipende dall'azione intrapresa: $\mathcal{P}_{ss'}^a=P(s_{t+1}=s'|s_t=s, a_t=a)$.
- Anche la funzione di ricompensa dipende dall'azione intrapresa: $\mathcal{R}_s^a=\mathbb{E}[r_{t+1}|s_t=s, a_t=a]$.



## Policy e le Nuove Funzioni Valore

In un MDP, il comportamento dell'agente è regolato dalla **Policy ($\pi$)**, che definisce la distribuzione di probabilità sulle azioni in base allo stato attuale: $\pi(a|s)=P(a_t=a|s_t=s)$. Le policy negli MDP dipendono solo dallo stato corrente (non dalla storia) e sono stazionarie, cioè non cambiano nel tempo in modo imprevedibile.

Poiché le azioni modificano profondamente i ritorni, dobbiamo ridefinire le nostre funzioni di valutazione rispetto a una specifica policy $\pi$:

**State-value function $v_\pi(s)$:** Il ritorno atteso partendo dallo stato $s$ e agendo poi secondo la policy $\pi$.


$$
v_\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]
$$

**Action-value function $q_\pi(s,a)$:** Il ritorno atteso partendo dallo stato $s$, intraprendendo una specifica azione $a$, e *solo da quel momento in poi* seguendo fedelmente la policy $\pi$.


$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s, a_t=a]
$$

### Equazioni di Bellman per l'Aspettativa (Bellman Expectation Equations)

Possiamo scomporre ricorsivamente $v_\pi$ e $q_\pi$ proprio come avevamo fatto negli MRP.
Queste due funzioni sono strettamente interconnesse:

- Il valore di uno stato $v_\pi(s)$ è semplicemente la **media ponderata dei valori di tutte le azioni possibili in quello stato** $q_\pi(s,a)$, moltiplicata per la probabilità che la policy scelga quell'azione $\pi(a|s)$.

$$
v_{\pi}(s) = \sum_{a \in \mathcal{A}} \; \pi(a|s) \, q_{\pi}(s,a)
$$

- Il valore di un'azione $q_\pi(s,a)$ è pari alla ricompensa immediata $\mathcal{R}_s^a$ sommata alla media ponderata dei valori di tutti i possibili stati di arrivo $v_\pi(s')$, governata dalle probabilità dell'ambiente $\mathcal{P}_{ss'}^a$.

$$
q_\pi(s,a) = \mathcal{R}_s^a + \gamma \sum_{s' \in S} \; \mathcal{P}_{ss'}^a \, v_\pi(s')
$$

Sostituendo una nell'altra, otteniamo le espressioni complete delle Bellman Expectation Equations che calcolano iterativamente questi valori.

$$
\implies q_\pi(s,a) = \mathcal{R}_s^a + \gamma \sum_{s' \in S} \; \mathcal{P}_{ss'}^a \, \sum_{a \in \mathcal{A}} \; \pi(a|s) \, q_{\pi}(s,a)
$$

$$
\implies v_{\pi}(s) = \sum_{a \in \mathcal{A}} \; \pi(a|s) \, [\mathcal{R}_s^a + \gamma \sum_{s' \in S} \; \mathcal{P}_{ss'}^a \, v_\pi(s')]
$$

La quale in forma matrice diventa:

$$
v_{\pi} = \mathcal{R}_{\pi} + \gamma \mathcal{P}_{\pi} v_{\pi}
$$

$$
\implies v_{\pi} = (1 - \gamma \mathcal{P}_{\pi})^{-1} \cdot \mathcal{R}_{\pi}
$$

dove $1$ è la matrice identità.


## Ottimalità

Lo scopo finale dell'agente è trovare il comportamento perfetto. Definiamo le funzioni ottimali:

- **Valore di stato ottimale** $v_*(s)$: Il massimo valore ottenibile su tutte le policy possibili per lo stato $s$
- **Valore di azione ottimale** $q_*(s,a)$: Il massimo valore ottenibile su tutte le policy possibili partendo dallo stato $s$ con l'azione $a$

Un MDP è considerato "risolto" quando riusciamo a calcolare la funzione di valore ottimale. **Esiste sempre almeno una policy ottima** $\pi_*$ che è migliore o uguale a tutte le altre policy ed è una policy deterministica: se conosciamo la matrice perfetta di $q_*(s,a)$, la policy ottimale consiste nel selezionare semplicemente l'azione con il valore massimo in ogni stato (*argmax*).

### Bellman Optimality Equations

Le funzioni ottimali seguono una propria equazione di Bellman, basata sulla massimizzazione (max) anziché sulla media.


$$
v_*(s) = \max_{a \in \mathcal{A}} q_*(s,a)
$$

$$
q_*(s,a) = \mathcal{R}_s^a + \gamma \sum_{s'\in S} \mathcal{P}_{ss'}^a \, v_*(s') = \mathcal{R}_s^a + \gamma \sum_{s'\in S} \mathcal{P}_{ss'}^a \, \max_{a' \in \mathcal{A}} q_*(s,a')
$$



A differenza delle equazioni di Expectation, l'Equazione di Bellman Optimality è **non-lineare** a causa della presenza dell'operatore matematico `max`. Di conseguenza, **non esiste una soluzione esatta a forma chiusa** (non possiamo invertirla con le matrici). È esattamente qui che nascono i famosi algoritmi del Reinforcement Learning: metodi di approssimazione iterativa come Value Iteration, Policy Iteration e il celebre Q-Learning per trovare i valori ottimali passo dopo passo.
