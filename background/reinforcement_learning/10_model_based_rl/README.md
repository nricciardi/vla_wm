# Model-Based Reinforcement Learning

Nei precedenti approcci al Reinforcement Learning, ci si è concentrati primariamente sui metodi Model-Free, i quali hanno l'obiettivo di apprendere la funzione di valore o direttamente la policy interagendo con l'ambiente, *senza* averne a disposizione una mappa o un modello matematico.

Nel **Model-Based Reinforcement Learning**, al contrario, l'agente apprende esplicitamente un modello delle dinamiche dell'ambiente a partire dall'esperienza raccolta e, successivamente, impiega tale modello per pianificare la funzione di valore o la policy.

Questo paradigma offre il vantaggio di poter apprendere il modello in modo efficiente sfruttando le classiche tecniche dell'apprendimento supervisionato e permette, inoltre, di ragionare sull'incertezza del modello stesso.

Sussistono tuttavia degli svantaggi: richiedendo di apprendere prima il modello e poi di calcolare la funzione di valore, si introducono due distinte fonti di errore di approssimazione che possono accumularsi.

## Definizione

Formalmente, un modello $\mathcal{M}$ è una rappresentazione di un Processo Decisionale di Markov (MDP), definita da un set di parametri $\eta$.

Assumendo che lo spazio degli stati $\mathcal{S}$ e lo spazio delle azioni $\mathcal{A}$ siano noti a priori, il modello ha lo scopo di approssimare le reali probabilità di transizione di stato ($\mathcal{P}_\eta \approx \mathcal{P}$) e la funzione di ricompensa ($\mathcal{R}_\eta \approx \mathcal{R}$).


Dal modello, quindi, è possibile campionare lo stato e la ricompensa al tempo $t+1$:


$$
s_{t+1} \sim \mathcal{P}_\eta(s_{t+1} \vert{} s_t, a_t)
$$

$$
r_{t+1} \sim p_\eta(r_{t+1}\mid s_t,a_t)
$$


Spesso si introduce un'ipotesi semplificativa, assumendo l'**indipendenza condizionale tra le transizioni di stato e le ricompense**, esprimibile matematicamente come:


$$
P(s_{t+1}, r_{t+1} \vert{} s_t, a_t) = P(s_{t+1} \vert{} s_t, a_t) P(r_{t+1} \vert{} s_t, a_t)
$$


L'apprendimento del modello si configura come un problema di apprendimento supervisionato il cui obiettivo è stimare i parametri $\eta$ da un insieme di transizioni reali $\mathcal{D}=\{(s_t,a_t,r_{t+1},s_{t+1})\}_{t=0}^{T-1}$.

Ogni interazione produce un dato di addestramento del tipo $(s_t,a_t)\rightarrow(r_{t+1},s_{t+1})$. La proprietà di Markov consente di fattorizzare la likelihood attraverso le transizioni condizionate allo stato e all'azione correnti; non implica però che i campioni consecutivi di una traiettoria siano indipendenti e identicamente distribuiti.

Nello specifico, apprendere la mappatura verso la ricompensa ($s, a \rightarrow r$) rappresenta un problema di regressione, mentre apprendere la transizione verso il nuovo stato ($s, a \rightarrow s'$) costituisce un problema di stima della densità. Per risolvere tali problemi si seleziona una funzione di perdita (come l'errore quadratico medio o la divergenza di Kullback-Leibler) e si cercano i parametri $\eta$ che ne minimizzino il valore empirico.

## Table Lookup Model

Tra i vari modelli implementabili, il **Table Lookup Model** è l'approccio non parametrico più elementare: l'agente memorizza esplicitamente un MDP approssimato contando il numero di visite $N(s,a)$ per ogni specifica coppia stato-azione. Le probabilità di transizione e le ricompense medie si stimano tramite la frequenza empirica:


$$
\hat{P}(s'\mid s,a) = \frac{N(s,a,s')}{N(s,a)}
$$

$$
\hat{R}(s,a) = \frac{1}{N(s,a)}\sum_{t:\,(s_t,a_t)=(s,a)} r_{t+1}
$$

In alternativa, in forma non-parametrica pura, l'agente può semplicemente memorizzare tutte le tuple di esperienza e, per simulare una transizione a partire da uno stato e un'azione, estrarre casualmente una delle tuple registrate per quella specifica coppia.

## Sample-Based Planning

Una volta generato il modello, è possibile risolvere l'MDP approssimato tramite i canonici algoritmi di Programmazione Dinamica (es. Value Iteration). Tuttavia, una soluzione estremamente potente ed efficiente prende il nome di **Sample-Based Planning**. In questa modalità, il modello non viene risolto analiticamente, ma viene usato esclusivamente per generare campioni di esperienza simulata, ai quali si applicano i classici algoritmi Model-Free (come Monte-Carlo, Sarsa o Q-learning).

L'efficacia della pianificazione Model-Based è intrinsecamente legata all'accuratezza del modello stimato: **se l'ambiente viene modellato in modo impreciso, l'agente calcolerà una policy subottimale**, convergendo alla soluzione ottima del modello errato ma non dell'ambiente reale.

Per ovviare a ciò, si ricorre alle **Architetture Integrate** che uniscono in un unico sistema sia l'apprendimento puro dall'ambiente sia la pianificazione.

Il framework **Dyna** è il paradigma per eccellenza di questa integrazione. In Dyna, l'esperienza reale viene utilizzata per un duplice scopo: **aggiornare direttamente la funzione di valore/policy** (tramite algoritmi Model-Free diretti) e, contestualmente, addestrare e **migliorare il modello**.

Il modello appena aggiornato genera poi esperienza simulata che viene impiegata in background per pianificare e migliorare ulteriormente la funzione di valore.
L'implementazione algoritmica, nota come **Dyna-Q**, si articola nei seguenti passaggi continui:

1. Si osserva lo stato corrente $S$ e si esegue un'azione $A$ seguendo una policy (es. $\epsilon$-greedy)


2. Si osservano la ricompensa reale $R$ e il nuovo stato $S'$


3. Si effettua un aggiornamento diretto tramite Q-learning:

$$
Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma \max_a Q(S',a) - Q(S,A)]
$$



4. Si salva l'esperienza nel modello: $Model(S,A) \leftarrow R, S'$


5. *Fase di pianificazione (ripetuta $n$ volte)*: si estraggono casualmente uno stato e un'azione precedentemente visitati, si interroga il modello per ottenere la transizione simulata e si esegue un ulteriore aggiornamento Q-learning sulle stime generate dal modello



Negli scenari pratici, come la risoluzione di un labirinto, gli aggiornamenti prodotti da esperienza simulata possono ridurre il numero di interazioni reali necessarie rispetto al RL puramente model-free.

Tuttavia, **in ambienti dinamici** dove gli ostacoli possono cambiare nel tempo, algoritmi di base come Dyna-Q possono faticare ad adattarsi, rendendo **necessarie varianti esplorative** (come Dyna-Q+ e Dyna-AC) che incentivano la scoperta di nuovi percorsi ottimali.

## Simulation-Based Search

Nei problemi in cui lo spazio degli stati è troppo vasto per permettere a framework come Dyna di aggiornare l'intera funzione di valore, si ricorre alla **Simulation-Based Search**.

L'algoritmo di **Forward Search** non cerca di risolvere e mappare l'intero Processo Decisionale di Markov (MDP). Al contrario, costruisce un albero di ricerca esaustivo ponendo lo stato reale corrente, $s_t$, come nodo radice. Sfruttando il modello matematico dell'ambiente, l'agente esegue un lookahead (guarda in avanti) per esplorare le conseguenze delle proprie azioni future a partire esclusivamente dal momento presente.

Si immagini di programmare un agente per giocare a scacchi. Un approccio classico cercherebbe di calcolare il valore di ogni possibile disposizione dei pezzi sulla scacchiera (milioni di stati). La Forward Search, invece, calcola l'albero delle mosse possibili partendo solo dalla specifica disposizione dei pezzi che si ha di fronte in quel preciso turno, risolvendo di fatto un "sotto-MDP" localizzato nel tempo e nello spazio.


### Simple Monte-Carlo Search

Nella sua forma più elementare, la ricerca tramite simulazione può essere implementata come **Simple Monte-Carlo Search**. Dato un modello dell'ambiente e una policy di simulazione fissa $\pi$:

1. Per ogni singola azione $a$ disponibile nello stato reale corrente $s_t$, l'algoritmo simula $K$ episodi completi, interrogando il modello
2. Ogni traiettoria simulata genera un ritorno cumulato $G_t$
3. Le azioni vengono valutate calcolando la media aritmetica dei ritorni ottenuti in queste $K$ simulazioni (Monte-Carlo evaluation):

$$
Q(s_t, a) = \frac{1}{K} \sum_{k=1}^K G_t^k \approx q_\pi(s_t, a)
$$

4. Terminato il calcolo, l'agente seleziona e compie nel mondo reale l'azione che ha registrato il valore medio massimo:

$$
a_t = \argmax_{a \in \mathcal{A}} Q(s_t, a)
$$

Il limite di questo approccio semplice è che non memorizza le informazioni sugli stati intermedi attraversati durante la simulazione, perdendo l'opportunità di migliorare la propria strategia di esplorazione.

### Monte-Carlo Tree Search

**Monte-Carlo Tree Search** è un algoritmo di ricerca euristica per processi decisionali che costruisce iterativamente un albero di ricerca partendo dallo stato corrente.

La sua caratteristica più potente è che la policy di simulazione non è fissa, ma si raffina progressivamente concentrando lo sforzo computazionale sulle azioni più promettenti.

Il funzionamento dell'algoritmo si basa sulla ripetizione continua di un ciclo composto da quattro fasi distinte, che separano il processo decisionale in una fase condotta all'interno dell'albero noto (**in-tree**) e una fase di esplorazione esterna (**out-of-tree**):

1. **Selezione** (Tree policy - Fase in-tree): Partendo dal nodo radice (lo stato corrente), l'algoritmo discende l'albero di ricerca selezionando a ogni passo l'azione ottima in base alle statistiche accumulate nelle iterazioni precedenti. Finché la simulazione attraversa stati già esplorati, l'algoritmo adotta una Tree Policy volta a massimizzare le stime correnti $Q(s,a)$, bilanciando allo stesso tempo lo sfruttamento delle mosse migliori note (exploitation) e l'esplorazione di rami meno visitati (exploration). Questa fase termina quando si raggiunge un nodo foglia, ovvero un nodo che possiede azioni non ancora esplorate
2. **Espansione**: Una volta raggiunto il nodo foglia (che non è un nodo terminale, ma l'ultimo nodo presente nell'albero), l'albero viene espanso aggiungendo uno o più nodi figli, corrispondenti agli stati raggiungibili tramite le azioni non ancora esplorate
3. **Simulazione** o Roll-out (Default policy - Fase out-of-tree): Dal nodo appena espanso, la simulazione "cade" fuori dall'albero di ricerca strutturato. In questa fase, l'algoritmo passa a una Default Policy fissa, spesso basata su scelte puramente casuali (random roll-out) o su euristiche molto leggere. Questa policy non ha lo scopo di giocare in modo ottimale, ma unicamente di portare l'episodio a una conclusione rapida (uno stato terminale) per poterne campionare una ricompensa finale o un risultato (es. vittoria, sconfitta o punteggio).
4. **Retropropagazione** (Evaluation e Aggiornamento): Al termine della simulazione, il ritorno ottenuto (la ricompensa finale $G_u$) viene propagato all'indietro lungo il percorso appena compiuto. Vengono aggiornati tutti i nodi stato-azione attraversati nell'albero durante la fase di Selezione e l'Espansione. Per ogni nodo visitato, si incrementa il contatore delle visite $N(s,a)$ e si aggiorna il valore atteso dell'azione $Q(s,a)$, che diventa la **media dei ritorni ottenuti in tutte le simulazioni** passate per quel nodo:


$$
Q(s,a) = \frac{1}{N(s,a)} \sum_{k=1}^K \sum_{u=t}^T 1(S_u, A_u = s, a) G_u
$$

Ripetendo questo ciclo migliaia o milioni di volte, il controllo Monte-Carlo applicato all'esperienza simulata garantisce la convergenza asintotica verso l'albero di ricerca ottimale. Le stime $Q(s,a)$ si avvicinano progressivamente ai valori reali ottimali $q_*(s,a)$. Al termine del tempo computazionale a disposizione, l'agente seleziona l'azione alla radice che possiede il valore $Q$ più alto (o il maggior numero di visite).

Per comprendere la meccanica, si consideri un agente che deve decidere la prossima mossa in un gioco da tavolo (ad esempio, gli Scacchi o il Tris). Lo stato attuale del tabellone rappresenta il nodo radice. L'algoritmo procede per iterazioni:

- Iterazione 1: Essendo l'albero vuoto, l'algoritmo espande immediatamente una mossa possibile (A). Aggiunge il nodo "Mossa A" all'albero. Da qui, avvia una partita completamente casuale fino alla fine (Simulazione). La partita casuale termina con una vittoria (valore = 1). Il valore 1 viene retropropagato: il nodo "Mossa A" ha ora 1 visita e un valore medio di 1.

- Iterazione 2: Si riparte dalla radice. La Tree policy impone di esplorare azioni non ancora tentate. Si espande la "Mossa B". La simulazione casuale a partire da B si conclude con una sconfitta (valore = 0). Questo risultato viene retropropagato: "Mossa B" ha 1 visita e valore medio 0. Il nodo radice ha ora 2 visite totali.

- Iterazione 3: Partendo dalla radice, l'algoritmo deve scegliere se scendere verso A o verso B. La Tree policy nota che la "Mossa A" ha un valore atteso (1) superiore alla "Mossa B" (0). Pertanto, seleziona la "Mossa A". Da lì, espande una successiva mossa (A1), avvia una nuova simulazione casuale e ne retropropaga l'esito aggiornando i valori di A1, di A e della radice.

- Conclusione: Man mano che le iterazioni si susseguono, le mosse che portano a simulazioni vincenti verranno selezionate e approfondite (espanse) sempre più spesso, mentre i rami che portano a sconfitte verranno progressivamente ignorati. Alla scadenza del tempo limite, l'agente giocherà l'azione radice risultata statisticamente più solida.
