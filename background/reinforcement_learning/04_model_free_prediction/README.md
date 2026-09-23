# Model-Free Prediction

Negli scenari classici, un agente interagisce con un ambiente formalizzato matematicamente come un Processo Decisionale di Markov (MDP).

Mentre la programmazione dinamica è una tecnica utilizzata per risolvere MDP in cui il modello dell'ambiente è ***perfettamente noto***, ossia quando conosciamo le dinamiche dell'ambiente, cioè quando sappiamo esattamente con quale probabilità un'azione ci porterà in un nuovo stato e quale sarà la ricompensa. In questo caso possiamo infatti pianificare il comportamento *ottimale*.

Tuttavia, nel mondo reale, quasi mai possediamo un modello perfetto dell'ambiente ed è qui che entra in gioco il **Reinforcement Learning Model-Free**.

Nella **model-free prediction** l'obiettivo è **stimare la funzione di valore** di una data policy $\pi$ in un MDP del quale non si conoscono le dinamiche, utilizzando l'esperienza raccolta dall'agente.

Il passo successivo sarà il "Controllo" (Control), ovvero come ottimizzare la policy sulla base di queste stime.

## Apprendimento Monte-Carlo (MC)

L'approccio più intuitivo per imparare dall'esperienza è il metodo Monte-Carlo (MC). I metodi MC apprendono direttamente dagli episodi completi di esperienza intercorsi tra l'agente e l'ambiente.
Essendo un metodo model-free, non necessita di alcuna conoscenza preliminare sulle probabilità di transizione o sulle ricompense dell'MDP.

L'idea alla base è la più semplice possibile: **il valore di uno stato corrisponde alla media dei ritorni** (ossia la somma delle ricompense future) **osservati a partire da quello stato**.

C'è però una limitazione strutturale: poiché il metodo si basa sul ritorno totale calcolato alla fine dell'episodio, non fa uso di *bootstrapping* (ovvero non aggiorna le proprie stime basandosi su altre stime). Di conseguenza, l'approccio Monte-Carlo può essere applicato esclusivamente a **MDP episodici**, ovvero ambienti in cui l'interazione ha inevitabilmente una fine (uno stato terminale).

### Valutazione della Politica con Monte-Carlo

Il nostro obiettivo matematico è apprendere la funzione di valore $v_\pi$ a partire da episodi di esperienza generati seguendo una specifica strategia (policy) $\pi$. Un episodio si presenta come una sequenza di stati, azioni e ricompense: $s_1, a_1, r_2, ..., S_k \sim \pi$.

Definiamo il **ritorno** (return) $G_t$ come la somma totale delle ricompense future scontate a partire dall'istante di tempo $t$:


$$
G_t = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{T-1} R_T
$$



La funzione di valore **teorica** è il valore atteso di questo ritorno:

$$
v_\pi = E_\pi[G_t | s_t = s]
$$

Il metodo Monte-Carlo sostituisce questo valore atteso teorico (che richiederebbe la conoscenza del modello) con la **media empirica dei ritorni effettivamente osservati**.

Per calcolare questa media, esistono due varianti principali:

- **First-Visit MC**: Valuta lo stato $s$ considerando solo la *prima volta* che esso viene visitato all'interno di un episodio. Si incrementa il contatore delle visite $N(s) \leftarrow N(s) + 1$, si somma il ritorno totale $S(s) \leftarrow S(s) + G_t$ e si calcola la media $V(s) = S(s) / N(s)$. Per la legge dei grandi numeri, all'infinito questa stima convergerà al valore reale.

- **Every-Visit MC**: A differenza del precedente, aggiorna le statistiche *ogni singola volta* che lo stato $s$ viene attraversato durante l'episodio. I passaggi matematici per l'aggiornamento dei contatori e della media rimangono identici.



#### Esempio Pratico: Il Blackjack

Per concretizzare, consideriamo il gioco del Blackjack. L'obiettivo è avvicinarsi a 21 senza superarlo.

Eseguendo migliaia di episodi (es. 500.000) e applicando Monte-Carlo per valutare una politica semplice (es. fermarsi solo se si ha 20 o 21), l'algoritmo mappa gradualmente il valore di ogni singola combinazione di carte, costruendo una superficie di valore che ci indica quali stati sono vantaggiosi e quali no.


## Temporal-Difference (TD) Learning

L'apprendimento Temporal-Difference è considerato una delle scoperte più centrali e originali del Reinforcement Learning. Anch'esso è model-free e apprende dall'esperienza, ma risolve il limite principale di Monte-Carlo.

Il TD Learning apprende da episodi **incompleti** utilizzando il **bootstrapping**. In parole povere, TD non aspetta la fine dell'episodio per scoprire il ritorno totale $G_t$; piuttosto, aggiorna la sua ipotesi iniziale basandosi su una nuova e (si spera) migliore ipotesi fatta allo step successivo.

### L'Algoritmo TD(0)

L'algoritmo TD più semplice si chiama TD(0). Mentre MC aggiorna il valore verso il ritorno *reale* $G_t$, TD(0) aggiorna il valore verso un ritorno *stimato*, guardando solo uno step avanti nel futuro.

In ogni istante la regola di aggiornamento è:


$$
V(s_t) \leftarrow V(s_t) + \alpha(r_{t+1} + \gamma V(s_{t+1}) - V(s_t))
$$



In questa formula troviamo due concetti cardine:

* **TD Target** $r_{t+1} + \gamma V(s_{t+1})$: È il nostro "nuovo obiettivo". Combinando la ricompensa immediata appena ricevuta con il valore stimato dello stato successivo, otteniamo una stima più fresca e accurata rispetto a quella che avevamo prima.


* **TD Error** $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$: È la differenza tra il nostro nuovo target e la nostra vecchia stima. Ci dice di quanto ci eravamo sbagliati.




## Monte Carlo vs Temporal-Difference: Pro e Contro

- **Tempistiche di apprendimento**: TD apprende step-by-step, online, ancor prima di conoscere l'esito finale. MC deve obbligatoriamente aspettare la fine dell'episodio.
- **Tipologia di ambienti**: TD funziona perfettamente anche in ambienti continui che non terminano mai. MC è limitato ad ambienti episodici.
- Il ritorno di MC è una stima non polarizzata (**zero bias**), ma avendo all'interno moltissime azioni, transizioni e ricompense casuali (fino a fine episodio), ha un'**alta varianza**.
- Il target di TD introduce un certo bias (perché si basa su un'altra stima $V(s_{t+1})$ che all'inizio potrebbe essere errata), ma dipende da un solo step casuale, garantendo quindi una **bassa varianza**. Questo rende TD solitamente più efficiente e veloce a convergere rispetto a MC.





## La Visione Unificata: Bootstrapping e Campionamento

Possiamo classificare i principali algoritmi di Reinforcement Learning analizzando due dimensioni fondamentali dell'aggiornamento (backup):

1. **Bootstrapping (Uso di stime)**: L'aggiornamento si basa su altre stime?
   - MC *non fa* bootstrapping (usa i ritorni reali).
   - DP e TD *fanno* bootstrapping.
2. **Sampling (Campionamento)**: L'aggiornamento campiona un'aspettativa simulando un percorso, o calcola il valore esatto diramandosi su tutte le probabilità?
   - DP esplora analiticamente tutte le possibilità (*full backup*), quindi *non campiona*.
   - MC e TD esplorano un singolo percorso alla volta (*sample backup*), quindi *campionano*.

TD si posiziona al crocevia perfetto per l'apprendimento nel mondo reale: campiona singole esperienze (non necessita del modello) e fa bootstrapping (apprende rapidamente online).

## TD(lambda)

Fino ad ora abbiamo visto due estremi: TD(0) guarda avanti di *un solo step*, mentre MC guarda avanti *fino alla fine dell'episodio*. Generalizzando possiamo guardare nel futuro di un numero arbitrario di passi $n$:

- Ritorno a **1-step** (TD):

$$
G_t^{(1)} = r_{t+1} + \gamma V(s_{t+1})
$$

- Ritorno a **2-step**:

$$
G_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2})
$$

- Ritorno a **$\infty$-step** (MC):

$$
G_t^{(\infty)} = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{T-1} R_T
$$


In generale, il ritorno a n-step definisce l'algoritmo **N-step Temporal-Difference**. Tuttavia quale $n$ scegliere è complicato, quindi invece di selezionarne uno solo, possiamo mediare le stime provenienti da orizzonti temporali diversi (e.g., pesare al 50% il ritorno a 2 step e al 50% quello a 4 step) per ottenere una stima più robusta.

### Forward-View

Per combinare *tutti* gli n-step return in modo matematicamente elegante ed efficiente computazionalmente, introduciamo il **$\lambda$-return** $G_t^\lambda$, il quale assegna un peso a ogni n-step return. I pesi decadono geometricamente all'aumentare di $n$, decrescendo di un fattore $\lambda$ ad ogni passo:

$$
G_t^\lambda = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
$$

Questa prospettiva è definita **Forward-view** (visione in avanti) perché, proprio come Monte-Carlo, deve guardare *verso il futuro* per calcolare il ritorno e necessita di attendere la fine dell'episodio. Dal punto di vista teorico è perfetta, ma perde il vantaggio pratico di poter aggiornare le stime step-by-step.

### Backward-View

Per recuperare i vantaggi dell'apprendimento online mantenendo la potenza matematica del $\lambda$-return, si introduce il meccanismo delle **Tracce di Eleggibilità** $E_t(s)$ ottenendo la **Backward-view** (visione all'indietro).

Nella backward-view, invece di guardare avanti nel futuro, l'algoritmo mantiene una *memoria* di quanto è "fresco" il ricordo di ogni stato visitato. In questo modo, quando si scopre qualcosa di nuovo nel presente, possiamo aggiornare *tutti* gli stati passati in proporzione a quanto sono stati **recenti** o **frequenti**.

Le tracce di eleggibilità $E_t(s)$ combinano entrambe le euristiche (frequenza e recenza) cercando di risolvere il problema del **credit assignment**: a chi attribuire (*a quale azione*) la colpa o il merito per un certo risultato?

Nello specifico, l'algoritmo mantiene in memoria un contatore detta **traccia** per *ogni stato* del sistema inizializzato a zero:

$$
E_{t=0}(s) = 0
$$

- Ogni volta che lo stato viene visitato, la sua traccia viene incrementata di 1.
- Ad ogni istante temporale, le tracce di *tutti* gli stati decadono di un fattore $\gamma \lambda$.

$$
E_t(s) = \gamma \lambda E_{t-1}(s) + 1(s_t=s)
$$

La formula della Backward-view di $TD(\lambda)$ aggiorna il valore di *tutti* gli stati passati in proporzione a quanto è fresco/frequente il loro ricordo (ossia la traccia $E_t(s)$) e al TD-error appena calcolato per lo step corrente ($\delta_t$):


$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

$$
V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)
$$



In pratica, se scopriamo qualcosa di nuovo nel presente (un errore TD positivo o negativo), inviamo un "segnale" all'indietro nel tempo per aggiornare i valori degli stati che ci hanno portato fin lì.

### Equivalenze tra Algoritmi

Il parametro $\lambda$ generalizza e unifica le tecniche studiate.

Se $\lambda = 0$, la traccia di eleggibilità si azzera istantaneamente per gli stati passati. Solo lo stato corrente viene aggiornato. La formula si riduce esattamente al classico **TD(0)**.


Se $\lambda = 1$, il credito viene differito senza decadimento fino alla fine dell'episodio. Se gli aggiornamenti vengono applicati "offline" a fine iterazione, TD(1) accumula un errore totale che è *matematicamente identico* a **Monte-Carlo**. Questo risultato è dimostrabile sfruttando le proprietà algebriche delle serie telescopiche, in cui la somma dei TD-error intermedi collassa annullando i termini contrapposti, restituendo esattamente il ritorno reale $(G_t - V(s_t))$.



L'utilizzo pratico di $TD(\lambda)$ con un parametro intermedio tra $0$ e $1$ permette agli ingegneri di tarare dinamicamente il bilanciamento tra l'efficienza a bassa varianza del TD puro e la precisione a basso bias del Monte-Carlo, trovando lo "sweet spot" per massimizzare la velocità di apprendimento in base all'ambiente specifico.
