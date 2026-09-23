# Policy gradient e actor-critic

L'apprendimento per rinforzo può essere affrontato secondo diversi paradigmi. Nelle trattazioni precedenti ci si è concentrati sui metodi basati sul valore (Value-Based), nei quali l'obiettivo è approssimare la funzione di valore di stato $v_w(s) \approx v_\pi(s)$ o la funzione valore-azione $q_w(s,a) \approx q_\pi(s,a)$ tramite un vettore di parametri $w$. In tale approccio, la policy non viene appresa direttamente, ma viene generata in modo **implicito** a partire dalla funzione di valore stimata, per esempio adottando una strategia $\epsilon$-greedy.

Nel framework dei metodi **Policy-Based**, invece, l'obiettivo è **parametrizzare la policy** in modo *diretto*. La policy parametrizzata viene indicata come $\pi_w(s,a) = P(a\vert{}s,w)$, la quale rappresenta la probabilità di intraprendere l'azione $a$ nello stato $s$ dati i parametri $w$. Anche in questo caso, il focus rimane sull'apprendimento *model-free*, ovvero in assenza di un modello esplicito delle dinamiche dell'ambiente.


## Vantaggi e Svantaggi dei Metodi Policy-Based

L'adozione di metodi che parametrizzano direttamente la policy comporta specifici pro e contro.
Tra i vantaggi principali si riscontrano **proprietà di convergenza migliori**. Inoltre, questi metodi risultano particolarmente **efficaci in ambienti caratterizzati da spazi delle azioni ad alta dimensionalità** o continui, dove calcolare un massimo su tutte le azioni possibili (come richiesto nei metodi Value-Based) risulterebbe computazionalmente proibitivo. Infine, a differenza dei metodi Value-Based che tendono a convergere verso policy deterministiche, i metodi Policy-Based **permettono di apprendere policy stocastiche ottimali**.

Tra gli svantaggi si deve notare che questi algoritmi tendono tipicamente a convergere verso un ottimo locale piuttosto che globale. Inoltre, il processo di valutazione della policy risulta spesso inefficiente ed è caratterizzato da un'elevata varianza.

L'importanza delle policy stocastiche si evince in scenari come il gioco "Sasso-Carta-Forbici" o in ambienti parzialmente osservabili. In "Sasso-Carta-Forbici", qualsiasi policy deterministica verrebbe facilmente sfruttata e battuta da un avversario intelligente; la policy ottimale (l'equilibrio di Nash) è infatti una policy stocastica uniforme.
Un altro esempio notevole è l'"Aliased Gridworld", un ambiente in cui l'agente non è in grado di distinguere visivamente due stati specifici (stati grigi), rendendoli di fatto identici (aliased) rispetto alle feature osservabili. Un agente Value-Based apprenderebbe una policy quasi-deterministica (es. andare sempre a Ovest o sempre a Est in entrambi gli stati grigi), rischiando di rimanere bloccato e non raggiungere mai la ricompensa. Al contrario, un metodo Policy-Based è in grado di apprendere una policy stocastica ottimale che, per esempio, scelga di spostarsi a Est o a Ovest con probabilità del 50% negli stati ambigui, garantendo così il raggiungimento dell'obiettivo in pochi passi con alta probabilità.

## Funzioni Obiettivo e Ottimizzazione della Policy

Per trovare la policy ottimale attraverso i metodi Policy-Based, il problema di Reinforcement Learning viene inquadrato come un problema di ottimizzazione. Affinché un algoritmo possa aggiornare i parametri, ha bisogno di un criterio matematico rigoroso che indichi quanto sia "buona" la policy corrente. Questo criterio prende il nome di **funzione obiettivo**, che verrà indicata con $L(w)$. L'obiettivo dell'apprendimento è quindi trovare l'insieme di parametri $w$ che massimizza il valore di $L(w)$.

La formulazione matematica di $L(w)$ cambia a seconda di come si svolge l'interazione tra l'agente e l'ambiente:

- **Ambienti episodici (con un inizio e una fine definiti):**
In scenari strutturati in episodi (ad esempio, una partita a un gioco da tavolo), il metodo più logico per valutare la qualità di una policy è osservare il guadagno complessivo che l'agente si aspetta di ottenere a partire dall'inizio dell'episodio. Pertanto, la funzione obiettivo corrisponde semplicemente al valore del primissimo stato $s_1$:

$$
L_1(w) = V^{\pi_w}(s_1) = \mathbb{E}_{\pi_w}[v_1]
$$


In altre parole, la policy viene giudicata in base alla somma totale delle ricompense (il ritorno $v_1$) che si prevede di accumulare dallo stato iniziale fino al raggiungimento dello stato terminale.


- **Ambienti continui (senza uno stato terminale):**
In contesti che proseguono indefinitamente (come il controllo di un impianto industriale o di un robot sempre attivo), il concetto di "stato iniziale" perde di importanza. Si valuta invece la performance media dell'agente nel lungo termine. Poiché l'agente transiterà in innumerevoli stati, è necessario pesare il valore di ciascuno di essi in base a quanto spesso viene effettivamente visitato. Questa frequenza di visita a lungo termine è matematicamente espressa dalla **distribuzione stazionaria** $d^{\pi_w}(s)$. Essa rappresenta la probabilità di trovarsi in un dato stato $s$ dopo aver fatto evolvere il sistema per un tempo indefinito seguendo la policy $\pi_w$.


Sulla base di questa distribuzione, è possibile definire la funzione obiettivo in due modi:

- **Valore medio (Average Value):** Si calcola il valore medio su tutti gli stati dell'ambiente, moltiplicando il valore di ogni singolo stato $V^{\pi_w}(s)$ per la sua probabilità di visita a lungo termine $d^{\pi_w}(s)$:

$$
L_{avV}(w) = \sum_s d^{\pi_w}(s) V^{\pi_w}(s)
$$


- **Ricompensa media per time-step (Average Reward):** Si calcola la media delle ricompense immediate $\mathcal{R}_s^a$ ottenibili in ogni istante di tempo. Questa somma pesa tramite $d^{\pi_w}(s)$ ogni possibile ricompensa in base a quanto spesso si visita lo stato $s$ e a quanto è probabile che l'agente intraprenda l'azione $a$ in quello stato, ossia $\pi_w(s,a)$:

$$
L_{avR}(w) = \sum_s d^{\pi_w}(s) \sum_a \pi_w(s,a) \mathcal{R}_s^a
$$


L'apprendimento basato su policy si configura dunque come un problema di ottimizzazione in cui si cerca il parametro $w$ che massimizza $L(w)$. Sebbene esistano approcci senza l'uso del gradiente (come Hill climbing, algoritmi genetici o Nelder Mead), una maggiore efficienza è spesso raggiungibile calcolando la direzione di massima pendenza tramite il gradiente.

L'aggiornamento dei parametri avviene secondo la regola di **gradient ascent**:


$$
\Delta w = \alpha \nabla_w L(w)
$$


dove $\alpha$ è il parametro di step-size (o learning rate) e il gradiente della policy $\nabla_w L(w)$ è il vettore delle derivate parziali rispetto a ciascuna componente di $w$.

Un metodo basilare per stimare tale gradiente è l'approccio alle **Differenze Finite** (Finite Differences). Esso consiste nel perturbare ogni dimensione $k$ dei parametri di una piccola quantità $\epsilon$ per stimare la derivata parziale:


$$
\frac{\partial L(w)}{\partial w_k} \approx \frac{L(w + \epsilon u_k) - L(w)}{\epsilon}
$$


dove $u_k$ è il versore della dimensione $k$. Sebbene richieda $n$ valutazioni per calcolare il gradiente in $n$ dimensioni e risulti **rumoroso e inefficiente**, ha il pregio di funzionare **anche per policy non differenziabili**.

### Approccio Analitico: La Score Function

Se si assume che la policy $\pi_w(s,a)$ sia differenziabile ovunque non sia nulla, il gradiente può essere calcolato analiticamente. Sfruttando la tecnica del *likelihood ratio*, si può riscrivere il gradiente della policy come segue:


$$
\nabla_w \pi_w(s,a) = \pi_w(s,a) \frac{\nabla_w \pi_w(s,a)}{\pi_w(s,a)} = \pi_w(s,a) \nabla_w \log \pi_w(s,a)
$$


Il termine $\nabla_w \log \pi_w(s,a)$ prende il nome di **Score Function**. L'introduzione di questo logaritmo semplifica drasticamente il calcolo del valore atteso, rendendolo trattabile.

Questo passaggio matematico si basa su una nota regola di calcolo differenziale ed è spesso chiamato **"log-derivative trick"** (trucco della derivata del logaritmo), che sfrutta le proprietà del *likelihood ratio*.

Il primo passaggio (trucco algebrico):


$$
\nabla_w \pi_w(s,a) = \pi_w(s,a) \frac{\nabla_w \pi_w(s,a)}{\pi_w(s,a)}
$$


Qui si sta semplicemente moltiplicando e dividendo l'espressione di partenza per la stessa quantità, ovvero $\pi_w(s,a)$. Questa operazione lascia inalterato il valore originario ed è matematicamente valida assumendo che la policy sia differenziabile laddove non è nulla ($\pi_w(s,a) \neq 0$).

Il secondo passaggio (la regola della catena per il logaritmo):


$$
\frac{\nabla_w \pi_w(s,a)}{\pi_w(s,a)} = \nabla_w \log \pi_w(s,a)
$$

Questa equivalenza sfrutta la regola di derivazione delle funzioni composte applicata al logaritmo naturale.
Ricordiamo dal calcolo differenziale che la derivata del logaritmo naturale di una generica funzione $f(x)$ è uguale alla derivata della funzione moltiplicata per l'inverso della funzione stessa:

$$
\frac{d}{dx} \log(f(x)) = \frac{1}{f(x)} \cdot f'(x) = \frac{f'(x)}{f(x)}
$$


Isolare il termine $\pi_w(s,a)$ all'inizio dell'espressione è cruciale per la costruzione dell'algoritmo. Riscrivendo la formula in modo da avere una funzione (in questo caso la *Score Function* $\nabla_w \log \pi_w(s,a)$) pesata per la probabilità $\pi_w(s,a)$ con cui quell'azione viene scelta, si ottiene esattamente la definizione matematica di **valore atteso** (Expected Value). Questo trucco permette quindi di calcolare il gradiente della policy attraverso il semplice campionamento di traiettorie nell'ambiente (usando la Score Function), senza la necessità di conoscere le dinamiche esatte dell'ambiente stesso.


A titolo di esempio, per azioni discrete si può impiegare una **Linear Softmax Policy**. In questo caso, le azioni vengono pesate tramite una combinazione lineare delle feature dello stato $x(s,a)^T w$, e la probabilità dell'azione è proporzionale all'esponenziale del suo peso: $\pi_w(s,a) \propto e^{x(s,a)^T w}$. La relativa Score Function risulta essere:


$$
\nabla_w \log \pi_w(s,a) = x(s,a) - \mathbb{E}_{\pi_w}[x(s,\cdot)]
$$


In questo modo, la funzione penalizza o premia la deviazione tra le feature dell'azione intrapresa e la media delle feature delle azioni possibili. Se un'azione porta ad una buona ricompensa, la policy verrà aggiornata per aumentare la probabilità di quell'azione in futuro.

Negli spazi ad azioni continue, una scelta naturale ricade sulla **Policy Gaussiana**, dove l'azione è campionata da una distribuzione normale $a \sim \mathcal{N}(\mu_w(s), \sigma_w(s)^2)$. La media $\mu_w(s)$ è modellata per esempio come una combinazione lineare di feature $\mu_w(s) = x(s)^T w$, e la varianza può essere fissa o anch'essa parametrizzata. La Score Function in questo scenario diviene:


$$
\nabla_w \log \pi_w(s,a) = \frac{(a - \mu_w(s))x(s)}{\sigma_w(s)^2}
$$


## REINFORCE

Il framework matematico trova la sua massima espressione nel **Policy Gradient Theorem**, che generalizza l'approccio del likelihood ratio a MDP (Processi di Decisione di Markov) multi-step. Il teorema dimostra che, indipendentemente dalla funzione obiettivo adottata (stato iniziale, reward medio o valore medio), il gradiente della policy assume la seguente forma esatta:


$$
\nabla_w L(w) = \mathbb{E}_{\pi_w}[\nabla_w \log \pi_w(s,a) Q_\pi(s,a)]
$$


dove $Q_\pi(s,a)$ è il valore atteso a lungo termine per la coppia stato-azione corrente.

Un'applicazione diretta di questo teorema è l'algoritmo **REINFORCE**, o Monte-Carlo Policy Gradient. Dal momento che la funzione $Q_\pi(s_t,a_t)$ non è nota a priori, si utilizza il ritorno effettivo $G_t$ calcolato a fine episodio come sua stima non distorta. L'aggiornamento dei parametri avviene quindi tramite gradient ascent stocastico:


$$
\Delta w_t = \alpha \nabla_w \log \pi_w(s_t,a_t) G_t
$$

## Actor-Critic Policy Gradient

Nonostante l'algoritmo Monte-Carlo Policy Gradient (REINFORCE) produca una stima unbiased, soffre di un'elevata varianza che può rallentare o instabilizzare l'apprendimento. Per ovviare a questo problema si introduce un'architettura **Actor-Critic**.

In questo schema vengono mantenuti due insiemi distinti di parametri $w$ e $\theta$:

- Il **Critic** aggiorna i parametri $w$ per **stimare la funzione valore-azione** $Q_w(s,a) \approx Q_\pi(s,a)$ attraverso metodi robusti di policy evaluation consolidati (es. Temporal-Difference learning).

- L'**Actor** aggiorna i parametri della policy $\theta$ lungo la direzione suggerita dal Critic.

L'aggiornamento dell'attore avviene seguendo un gradiente della policy approssimato:

$$
\nabla_\theta L(\theta) \approx \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) Q_w(s,a)]
$$


generando di conseguenza la regola:


$$
\Delta \theta = \alpha \nabla_\theta \log \pi_\theta(s,a) Q_w(s,a)
$$

Nel caso dell'algoritmo **Action-Value Actor-Critic** (QAC), si usa un'approssimazione lineare per la funzione valore $Q_w(s,a) = x(s,a)^T w$.

In questo caso il Critic calcola un errore TD (Temporal-Difference) $\delta$ per aggiornare i propri parametri $w$, e l'Actor aggiorna i propri parametri $\theta$ sulla base di $Q_w(s,a)$.

$$
\delta = r + \gamma Q_w(s',a') - Q_w(s,a)
$$

### Riduzione della Varianza

Per abbassare ulteriormente la varianza del Policy Gradient senza alterarne il valore atteso, si può sottrarre alla stima una **baseline function** $B(s)$.

È dimostrabile che sottrarre una quantità che dipende unicamente dallo stato $s$ (e non dall'azione $a$) ha aspettativa nulla rispetto al gradiente della policy. Lo sviluppo matematico è il seguente:


$$
\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) B(s)] = \sum_{s \in S} d^{\pi_\theta}(s) B(s) \nabla_\theta \sum_{a} \pi_\theta(s,a) = 0
$$

Il risultato è zero poiché la somma delle probabilità di tutte le azioni in un dato stato, $\sum_a \pi_\theta(s,a)$, è sempre esattamente pari a 1; di conseguenza, il gradiente di una costante è nullo.

Una scelta eccellente per la baseline è la funzione valore di stato stessa: $B(s) = V^{\pi_\theta}(s)$.

Riscrivendo la formula si introduce così la **Advantage Function** $A^{\pi_\theta}(s,a)$, la quale indica quanto l'azione $a$ sia migliore o peggiore rispetto alla media delle azioni eseguibili nello stato $s$. Il gradiente della policy riformulato è:

$$
A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)
$$

$$
\nabla_\theta L(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) A^{\pi_\theta}(s,a)]
$$

Concettualmente, l'Advantage indica quanto l'intraprendere la specifica azione $a$ sia migliore o peggiore rispetto al valore atteso medio derivante dall'agire secondo la policy corrente nello stato $s$.

Questa formulazione riduce significativamente la varianza. Tuttavia, in tal modo il Critic dovrebbe idealmente stimare due funzioni - $Q^{\pi_\theta}(s,a)$ e $V^{\pi_\theta}(s)$ - per poter computare $A(s,a)$.

Per evitarlo, si sfrutta un'importante proprietà dell'errore TD: per la vera funzione di valore, l'errore TD $\delta^{\pi_\theta}$ costituisce uno stimatore privo di distorsione per la funzione Advantage. Pertanto, il gradiente può essere valutato usando direttamente $\delta$ in luogo dell'Advantage esatto, portando alla regola **TD Actor-Critic**:

$$
Q^{\pi_\theta}(s,a) \approx r + \gamma V^{\pi_\theta}(s') \implies A^{\pi_\theta}(s,a) \approx \delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)
$$

$$
\nabla_\theta L(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) \delta^{\pi_\theta}]
$$

In un'implementazione pratica si andrà quindi a stimare una sola funzione di valore $V_v(s)$ sfruttando la quale si calcolerà un errore TD approssimato.

Infine, si deve sottolineare che sia l'Actor che il Critic possono essere valutati su diverse scale temporali (time-scales). Proprio come nei metodi Value-Based si ricorre ai meccanismi TD($\lambda$) mediante le **Eligibility Traces** (ovvero la combinazione su più step tramite i fattori $\lambda$), lo stesso è fattibile per il gradiente della policy. Impiegando un forward-view e, analogamente per applicabilità online, una backward-view con le eligibility traces, le equazioni di aggiornamento per l'attore divengono:


$$
e_{t+1} = \lambda e_t + \nabla_\theta \log \pi_\theta(s,a)
$$

$$
\Delta \theta = \alpha \delta e_t
$$

In sintesi, il Policy Gradient si presenta in forme equivalenti a seconda dello stimatore utilizzato: dal REINFORCE basato sui ritorni effettivi $G_t$, all'uso dei valori stimati dal critic $Q_w(s,a)$ (Q Actor-Critic), alla forma con l'Advantage $A_w(s,a)$ o con l'errore $\delta$ (TD Actor-Critic), per arrivare a versioni ibridate con eligibility traces come TD($\lambda$) Actor-Critic.