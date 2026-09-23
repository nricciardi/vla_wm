# Sistema di accodamento a singolo server

Sebbene rappresenti un sistema molto semplice, la coda a singolo server (**Single-Server Queueing**) è estremamente rappresentativa del funzionamento interno di simulazioni di grande complessità. Questo modello descrive una situazione in cui delle entità (clienti) arrivano, attendono in fila se necessario, ricevono un servizio da un'unica risorsa (server) e poi lasciano il sistema.

## Formulazione del Problema

Il sistema è governato da specifiche regole probabilistiche e logiche:

- **Tempi di interarrivo (Interarrival times)**: il tempo che intercorre tra l'arrivo di due clienti successivi, indicato con $A_1, A_2, \dots$. Queste variabili sono modellate come variabili casuali (random variables) *Indipendenti e Identicamente Distribuite* (IID).


- **Tempi di servizio (Service times o processing times)**: il tempo necessario per servire ogni cliente, indicato con $S_1, S_2, \dots$. Anch'esse sono variabili casuali IID e, fondamentale per la modellazione, sono *indipendenti* dai tempi di interarrivo.


- **Regole di instradamento**: se un cliente arriva e trova il server libero (*idle*), entra immediatamente in servizio senza attendere.


- **Disciplina della coda**: quando il server termina un servizio, seleziona il prossimo cliente dalla coda (se presente) seguendo una logica FIFO (First-In, First-Out), ovvero il primo ad arrivare è il primo ad essere servito.


- **Condizioni iniziali**: la simulazione inizia sempre in uno stato "empty-and-idle" (vuoto e libero).

Al tempo zero ($t_0 = 0$), il sistema inizia ad attendere l'arrivo del primo cliente generando il primo tempo di interarrivo $A_1$ (assumendo $A_1 \neq 0$) per definirne l'istante di ingresso.



## Misure di Performance

L'obiettivo della simulazione è **quantificare l'efficienza** del sistema calcolando tre misure di performance principali:

### Ritardo medio atteso in coda

**Ritardo medio atteso in coda** $d(n)$ rappresenta una misura della performance del sistema *dal punto di vista del cliente*. Poiché il ritardo dipende dalle realizzazioni casuali degli arrivi e dei tempi di servizio, il ritardo medio è a sua volta una variabile casuale di cui calcoliamo il valore atteso.
Lo stimatore per questa misura, basato su $n$ clienti osservati, è:


$$
\hat{d}(n) = \frac{\sum_{i=1}^n D_i}{n}
$$

dove $D_i$ è il ritardo in coda del singolo cliente $i$. È importante notare che la parola "ritardo" non esclude un valore pari a zero; ad esempio, il primo cliente troverà sempre il server libero e avrà $D_1 = 0$.

I singoli termini $D_i$ ***non* sono indipendenti**: se il cliente precedente ha atteso molto, è probabile che anche il successivo attenda, creando **autocorrelazione**.

### Numero medio atteso di clienti in coda

**Numero medio atteso di clienti in coda** $q(n)$ è la *prospettiva del manager* sul sistema. È definita come una media ponderata nel tempo (**time-average**) continuo.

Se $Q(t)$ è il numero di clienti in coda al tempo $t$ e $T(n)$ è il tempo totale di simulazione per osservare $n$ ritardi, il calcolo avviene pesando i possibili stati della coda per la porzione di tempo in cui si verificano.

Matematicamente, se definiamo $T_i$ come il tempo totale durante la simulazione in cui la coda è esattamente di lunghezza $i$, la proporzione di tempo è $\hat{p}_i = T_i / T(n)$.

Il numero medio di clienti è quindi la media ponderata:


$$
q(n) = \frac{\sum_{i=0}^\infty i \cdot T_i}{T(n)} = \sum_{i=0}^\infty i \cdot \hat{p}_i
$$


Il numeratore di questa equazione (somma dei rettangoli di base $T_i$ e altezza $i$) corrisponde all'integrale $\int_0^{T(n)} Q(t) dt$, ossia l'area sottesa dalla curva $Q(t)$ nel grafico dell'evoluzione temporale della coda.

$$
q(n) = \frac{1}{T(n)}\int_0^{T(n)} Q(t) dt
$$

### Utilizzo atteso del server

**Utilizzo atteso del server** $u(n)$ rappresenta la proporzione attesa di tempo in cui il server è occupato a lavorare durante la simulazione.
Definendo una funzione indicatrice $B(t)$ che vale 1 se il server è occupato e 0 se è libero, lo stimatore per l'utilizzo è calcolato come una media temporale continua:


$$
\hat{u}(n) = \frac{\int_0^{T(n)} B(t) dt}{T(n)}
$$


L'interpretazione gestionale di questo valore è critica:

- Un'utilizzazione vicina al 100%, soprattutto se accoppiata a un'alta congestione (code lunghe), è fortemente indicativa della presenza di un ***collo di bottiglia*** (bottleneck).


- Un'utilizzazione bassa (sotto il 90%) indica un ***eccesso di capacità*** (excess capacity). L'eccesso di capacità non è necessariamente positivo se si parla di risorse costose, come un robot in un sistema di produzione, dove un basso utilizzo rappresenta uno spreco di capitale.
