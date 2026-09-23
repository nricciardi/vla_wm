

# Teoria delle code

Nell'ambito delle Operations e del Supply Chain Management, lo studio dei sistemi complessi passa frequentemente attraverso l'analisi dei Modelli di Accodamento (**Queueing Models**).

Tali sistemi descrivono una dinamica ricorrente in cui determinate entità, definite in senso generale come "clienti", arrivano nel sistema di volta in volta, si accodano in una linea di attesa, ricevono un servizio da una determinata risorsa, per poi abbandonare il sistema. Il concetto di "cliente" è applicabile a un ampio spettro di scenari: può trattarsi di pazienti in attesa di cure mediche, pallet gestiti da carrelli elevatori in un magazzino, aeroplani che richiedono l'uso di una pista, o persino di chiamate telefoniche e istruzioni elaborate da una CPU.

Al fine di prevedere e ottimizzare l'efficienza, questi sistemi possono essere risolti per via puramente matematica oppure analizzati mediante l'ausilio di tecniche di simulazione. L'obiettivo principale è la predizione di specifiche misure di performance – tra cui l'utilizzazione dei server, la lunghezza delle code e i ritardi subiti dai clienti – espresse in funzione di parametri di input quali il tasso di arrivo, il tasso di servizio e il numero o la disposizione dei server.

## Popolazione di Origine e Capacità del Sistema

Per analizzare logicamente il problema, è necessario definire la **calling population**, ossia la popolazione dei potenziali clienti del sistema. Questa popolazione può essere concettualizzata come *finita o infinita*:

Nei modelli **a popolazione infinita**, il **tasso di arrivo** dei clienti nel sistema **non viene influenzato** dal numero di clienti che hanno già lasciato la popolazione originaria per entrare nel sistema.

Questa assunzione risulta corretta e preferibile quando il numero di clienti all'interno del sistema rappresenta unicamente una minuscola proporzione della popolazione totale.

Al contrario, nei modelli a **popolazione finita**, il **tasso di arrivo decresce progressivamente** in base al numero di clienti già in fase di servizio o in coda.

Un esempio applicativo critico per i modelli a popolazione finita è il *machine-repair problem*, dove le macchine fungono da clienti da servire e il tempo di funzionamento prima della rottura ("time-to-failure") equivale al tempo che intercorre fino alla necessità del servizio.


Oltre alla popolazione, si definisce la **capacità del sistema**. In molti sistemi di accodamento esiste un limite fisico logico al numero massimo di entità ammesse nella linea di attesa o nel sistema stesso.

Un cliente in arrivo che trova un sistema saturo non vi entra, bensì ritorna immediatamente alla popolazione di origine. Questa dinamica richiede una stretta distinzione tra il **tasso di arrivo** (il numero di arrivi che si presentano al sistema per unità di tempo) e il **tasso di arrivo effettivo** (il numero di clienti che concretamente entrano e permangono nel sistema per unità di tempo).

## Il Processo di Arrivo e la Gestione delle Code

Il processo di ingresso al sistema è delineato attraverso le distribuzioni dei tempi di interarrivo.

Per modelli a **popolazione infinita con arrivi casuali**, il processo matematico più rappresentativo è il processo di **arrivo di Poisson**.

In questo framework, le variabili casuali che rappresentano i tempi intercorsi tra un cliente e l'altro, denominate $A_n$, seguono una **distribuzione esponenziale** avente media pari a $\frac{1}{\lambda}$, dove $\lambda$ costituisce il *tasso di arrivo* espresso in clienti per unità di tempo.

Il numero complessivo di arrivi in un dato intervallo di tempo $\Delta t$ seguirà di conseguenza una distribuzione di Poisson con media $\lambda \Delta t$.

Per quanto concerne il comportamento all'interno della coda, le entità possono manifestare logiche comportamentali specifiche:

- Possono compiere un'azione di **balking**: rifiutarsi di entrare qualora la coda si presenti eccessivamente lunga.
- Possono procedere con il **reneging**: abbandonare la coda dopo un prolungato tempo di attesa.
- Possono applicare il **jockeying**: spostarsi strategicamente da una coda all'altra.



A regolamentare il modo in cui i clienti vengono prelevati dalla coda vi è la disciplina del servizio (**Queue discipline**). Le discipline più consuete includono la politica *FIFO* (First-In-First-Out), la *LIFO* (Last-In-First-Out), la selezione casuale *SIRO* (Service-In-Random-Order), la priorità ai tempi di servizio più brevi *SPT* (Shortest-Processing-Time-First) e le *procedure a priorità* (PR).

## Notazione di Kendall

I tempi necessari per processare ogni operazione, indicati genericamente con la variabile $S_n$, possono avere durata costante o casuale. In caso di andamento probabilistico, le distribuzioni maggiormente impiegate sono l'esponenziale, la Weibull, la gamma e la normale troncata. I meccanismi di servizio possono operare tramite un singolo server ($c=1$), mediante un numero multiplo di server in parallelo ($1 < c < \infty$), oppure con server teoricamente infiniti ($c=\infty$), come accade nelle strutture di tipo self-service.

Per standardizzare univocamente la classificazione di questi assetti, la dottrina si avvale della **Notazione di Kendall**, strutturata formalmente come $A/B/c/N/K$:

- Il parametro $A$ rappresenta la distribuzione del **tempo di interarrivo**.
- Il parametro $B$ identifica la distribuzione del **tempo di servizio**.
- Il simbolo $c$ quantifica il **numero di server** disposti in parallelo.
- Il valore $N$ indica la **capacità massima di accoglienza** del sistema.
- La grandezza $K$ esprime la **dimensione totale della popolazione di origine**.

I simboli codificati per le distribuzioni includono $M$ (indicante un processo esponenziale o di Markov), $D$ (deterministico o costante) e $G$ (distribuzione generale). In virtù di questo standard, un sistema designato come $M/M/1/\infty/\infty$ (usualmente contratto in $M/M/1$) definisce univocamente un ambiente a server singolo, popolazioni e capacità di coda infinite, laddove sia gli arrivi che i processi di servizio seguono distribuzioni esponenziali (Markoviane).



## Legge di Little

La formulazione analitica prevede un rigido apparato di definizioni a tempo continuo e di stato stazionario (steady-state). Considerando $L(t)$ come il numero di clienti presenti nel sistema all'istante $t$, e $T$ la durata del periodo di osservazione, lo stimatore $\hat{L}$ per il **numero medio atteso nel sistema** è formulato tramite la media temporale integrale:

$$
\hat{L} = \frac{1}{T}\int_0^T L(t) dt
$$

Affinché l'analisi sia statisticamente convergente, si calcola il limite per cui, quando la lunghezza temporale della simulazione $T \rightarrow \infty$, lo stimatore $\hat{L}$ si approssima infinitamente al valore reale $L$.

$$
\lim_{T \to \infty} \hat{L} = \lim_{T \to \infty} \frac{1}{T}\int_0^T L(t) dt = L
$$

Discorso perfettamente analogo vale per la formulazione del **numero medio atteso in coda** $L_Q$, che si ottiene integrando lo stato di coda $L_Q(t)$ nel tempo.

I **tempi medi trascorsi per singolo cliente**, contrassegnati con $W$ per la durata globale all'interno del sistema e $W_Q$ per il solo periodo in coda, vengono similmente valutati attraverso medie aritmetiche sulle singole osservazioni, convergendo a valori teorici quando il numero di clienti osservati $N \rightarrow \infty$.

Tutte queste metriche sono collegate da una relazione indissolubile e fondamentale, l'Equazione di Conservazione, nota in letteratura come **Legge di Little** (Little's Law). La legge sancisce che:

$$
L = \lambda W
$$

In termini discorsivi, il numero medio di clienti presenti fisicamente all'interno del sistema, in un qualsiasi momento arbitrario del lungo periodo, è esattamente equivalente al prodotto del tasso medio di arrivo $\lambda$ per il tempo medio $W$ trascorso complessivamente da ciascun cliente nel sistema.

## Utilizzo del Server

L'**utilizzo del server** $\rho$ determina la frazione di tempo teorica o osservata per la quale l'unità di processamento risulta impegnata al lavoro.

Per una configurazione a server singolo ($G/G/1$), assumendo un tasso medio di arrivi $\lambda$ e un tempo di servizio atteso $E[S] = \dfrac{1}{\mu}$, la media di utilizzo a lungo termine converge all'equazione:

$$
\rho = \lambda E[S] = \frac{\lambda}{\mu}
$$

Questo rapporto definisce una demarcazione critica. Affinché un qualunque sistema di accodamento a server singolo possieda una *condizione di stabilità matematica* (steady-state ben definito), è perentorio che valga:

$$
\lambda < \mu \implies \rho < 1
$$

Nel momento in cui il tasso di arrivo eguaglia o eccede il tasso di smaltimento ($\lambda > \mu$), si entra in una fase di instabilità conclamata in cui la linea di attesa inizierà a crescere indefinitamente al tasso di $(\lambda - \mu)$ clienti per unità temporale, inficiando definitivamente le performance.

Qualora il sistema preveda server multipli ($c > 1$), il tasso complessivo di smaltimento si attesta a $c\mu$, producendo un'equazione di utilizzazione:

$$
\rho = \frac{\lambda}{c\mu}
$$

traslando il requisito per la stabilità a $\lambda < c\mu$.

## Modelli Markoviani specifici: $M/G/1$ e $M/M/1$

L'analisi raggiunge un elevato livello di precisione esaminando specifici modelli per i quali è possibile derivare soluzioni in forma chiusa nello steady-state, con l'assunzione che la probabilità $P_n$ di trovare $n$ clienti nel sistema sia *indipendente dal tempo* considerato.

Nel modello $M/G/1$, che prevede un processo degli arrivi esponenziale affiancato ad un servizio dalla distribuzione puramente generica (dotata unicamente di una propria media $\frac{1}{\mu}$ e varianza $\sigma^2$), le performance non dipendono solo dai valori attesi ma profondamente dalla volatilità del servizio. I parametri si ricavano, ad esempio, dalle seguenti identità:

- $L_Q = \frac{\lambda^2(1/\mu^2 + \sigma^2)}{2(1-\rho)} = \frac{\rho^2(1 + \sigma^2\mu^2)}{2(1-\rho)}$

- $W_Q = \frac{\lambda(1/\mu^2 + \sigma^2)}{2(1-\rho)}$


Questa espressione analitica rivela una proprietà gestionale intrinseca di fondamentale importanza: **a parità di tassi medi d'ingresso e servizio**, il sistema caratterizzato da una **deviazione standard** ($\sigma$) **superiore** – in altre parole da una maggiore variabilità nei tempi di processo – sconterà necessariamente **ritardi e congestioni significativamente più alti**, sottolineando come l'affidabilità e la consistenza delle esecuzioni (minore varianza) siano tanto essenziali quanto la pura velocità lorda.

Infine, applicando i vincoli stringenti di distribuzione esponenziale sia agli arrivi sia ai servizi, otteniamo il classico modello $M/M/1$. In questo caso la varianza del servizio collima esattamente con il quadrato della sua media matematica ($\frac{1}{\mu^2}$), riducendo il calcolo delle metriche a formulazioni di assoluta linearità e pulizia:

- $L = \frac{\rho}{1-\rho}$

- $W = \frac{1}{\mu(1-\rho)}$

- $L_Q = \frac{\rho^2}{1-\rho}$

- $P_n = (1-\rho)\rho^n$


L'architettura logica di tali sistemi e le corrispondenti derivazioni matematiche rappresentano, pertanto, l'ossatura teoretica indispensabile tanto per lo studio puramente analitico quanto per l'implementazione pratica in ambito simulativo e di automazione.