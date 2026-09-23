# Model-Free Control

Mentre nella Model-free Prediction l'obiettivo è stimare la funzione di valore di una data policy senza conoscere le dinamiche dell'ambiente, nel **Model-free Control** l'obiettivo è **ottimizzare la funzione di valore e trovare la policy ottimale** in un ambiente sconosciuto.

Molti problemi del mondo reale possono essere modellati come MDP (gestione di portafogli finanziari, parcheggio autonomo, controllo di inventari, ecc.). Per la maggior parte di questi problemi, il modello dell'MDP è sconosciuto, ma possiamo campionare l'esperienza interagendo con l'ambiente, oppure il modello è noto ma è troppo grande per essere risolto analiticamente. In questi casi, il Model-Free Control è la soluzione.

## Apprendimento On-Policy e Off-Policy

Prima di addentrarci negli algoritmi, è fondamentale distinguere due paradigmi di apprendimento:

- **On-policy learning** ("Imparare lavorando"): L'agente apprende informazioni su una policy $\pi$ a partire dall'esperienza campionata seguendo *esattamente* quella stessa policy $\pi$. È un approccio di auto-apprendimento in cui si valuta e si migliora la strategia che si sta correntemente utilizzando.
- **Off-policy learning** ("Imparare guardando le spalle di qualcun altro"): L'agente apprende informazioni su una policy ottimale $\pi$ (la *target policy*) mentre genera esperienza seguendo una policy diversa $\mu$ (la *behavior policy*). Questo permette, ad esempio, di imparare osservando le azioni di un essere umano o esplorando l'ambiente in modo casuale pur mantenendo traccia di quale sarebbe il comportamento ideale.

## Generalized Policy Iteration (GPI)

Per trovare la policy ottimale, il Reinforcement Learning si basa sul concetto di **Generalized Policy Iteration (GPI)**. Questo framework alterna continuamente due processi fino alla convergenza:

1. **Valutazione della Policy (Evaluation)**: Stimare quanto è buona la policy corrente, calcolando la funzione di valore (ad esempio tramite Monte-Carlo o Temporal-Difference).
2. **Miglioramento della Policy (Improvement)**: Agire in modo *greedy* (avido) rispetto alla funzione di valore appena stimata, scegliendo sempre l'azione che massimizza il valore atteso.

Nel caso in cui il modello sia noto possiamo migliorare la policy in modo greedy basandoci semplicemente sulla funzione valore di stato $V(s)$ tramite la programmazione dinamica. La formula per scegliere l'azione migliore in uno stato $s$ sarebbe:

$$
\pi'(s) = \argmax_{a} \left( R_s^a + \sum_{s'} P(s'\vert{}s,a) V(s') \right)
$$

Tuttavia, questa operazione **richiede la conoscenza del modello dell'MDP**, in particolare le probabilità di transizione $P(s'\vert{}s,a)$ e le ricompense attese $R_s^a$. Essendo in un contesto Model-Free, non possediamo queste informazioni.

La soluzione prevede di **stimare** il valore degli stati $V(s)$. In particolare stimiamo il valore che si ottiene compiendo un'azione $a$ in uno stato $s$, ossia la funzione $Q(s,a)$.

Se possediamo i valori $Q(s,a)$ per ogni azione possibile nello stato $s$, il miglioramento della policy diventa banalmente una ricerca del valore massimo, senza bisogno di conoscere come l'ambiente transiterà nello stato successivo:

$$
\pi'(s) = \argmax_{a} Q(s,a)
$$

Il ciclo GPI viene quindi riadattato: si ottiene $Q = q_\pi$ (tramite Monte-Carlo o TD) e si migliora la policy estraendo le azioni direttamente dalla stima di $Q$ tramite $\argmax$.

## Il Dilemma dell'Esplorazione: La policy epsilon-Greedy

Se scegliessimo sempre l'azione che ci sembra migliore (azione *greedy*), rischieremmo di rimanere bloccati in decisioni sub-ottimali.

Per esempio, considerando di avere due porte: aprendo la sinistra si ottiene $0$. Aprendo la destra si ottiene $1$. Se da questo momento si agisse in modo puramente greedy, verrebbe sempre la destra, ma magari la sinistra (se esplorata ulteriormente) nascondeva una ricompensa di $100$.

Per garantire una **continua esplorazione** dell'ambiente, introduciamo la policy **$\epsilon$-greedy** (epsilon-greedy).
L'idea è semplice:

- Con probabilità $1 - \epsilon$, scegliamo l'azione che massimizza $Q(s,a)$ (**Exploitation**).
- Con probabilità $\epsilon$, scegliamo un'azione in modo completamente casuale tra tutte quelle disponibili, inclusa quella greedy (**Exploration**).

Matematicamente, se abbiamo $m$ azioni possibili, la probabilità di scegliere un'azione è:


$$
\pi(a\vert{}s) =  \begin{cases}  \frac{\epsilon}{m} + 1 - \epsilon & \text{se } a = \arg\max_{a'} Q(s,a') \\ \frac{\epsilon}{m} & \text{altrimenti} \end{cases}
$$

Si può dimostrare matematicamente (Teorema del miglioramento $\epsilon$-greedy) che per qualsiasi policy $\epsilon$-greedy $\pi$, generare una nuova policy $\pi'$ agendo in modo $\epsilon$-greedy rispetto ai valori $q_\pi$ garantisce che la nuova policy sia uguale o migliore della precedente: $v_{\pi'}(s) \ge v_{\pi}(s)$.

## GLIE: Monte-Carlo On-Policy Control

Unendo la valutazione Monte-Carlo dei valori $Q$ con il miglioramento $\epsilon$-greedy, otteniamo l'algoritmo **Monte-Carlo Control**. Dopo ogni episodio di esperienza, stimiamo $Q \approx q_\pi$ calcolando la media dei ritorni, e aggiorniamo la policy in modo $\epsilon$-greedy.

Per garantire la convergenza verso la policy ottimale assoluta, dobbiamo assicurarci di esplorare all'infinito tutte le coppie stato-azione, ma allo stesso tempo vogliamo che la nostra policy diventi sempre più avida per massimizzare il ritorno.
Questo approccio si chiama **GLIE (Greedy in the Limit with Infinite Exploration)**.

Un esempio di GLIE è utilizzare una policy $\epsilon$-greedy dove il valore di $\epsilon$ si riduce gradualmente a zero col passare degli episodi (es. $\epsilon_k = 1/k$). In questo modo, all'inizio si esplora molto e asintoticamente la policy converge verso una strategia puramente deterministica e ottimale.

## SARSA: Temporal-Difference On-Policy Control

A differenza dei metodi Monte-Carlo, in cui l'agente deve completare un intero episodio prima di capire se le sue azioni sono state buone o cattive, il *Temporal-Difference (TD) learning* permette all'agente di imparare "in tempo reale".

SARSA è l'algoritmo principale di questa famiglia. Il suo acronimo non è casuale, ma descrive esattamente la cronologia degli eventi necessari per fare **un singolo aggiornamento** della memoria dell'agente.

Immagina l'agente in un preciso istante di tempo. Per imparare qualcosa, ha bisogno di raccogliere esattamente 5 elementi in sequenza:

1. **$S$ (State):** Lo stato in cui si trova attualmente.
2. **$A$ (Action):** L'azione che decide di compiere in quello stato (basandosi sulla sua policy attuale, ad esempio $\epsilon$-greedy).
3. **$R$ (Reward):** La ricompensa immediata che riceve dall'ambiente per aver compiuto quell'azione.
4. **$S'$ (State'):** Il nuovo stato in cui atterra dopo l'azione.
5. **$A'$ (Action'):** L'azione che *deciderà* di compiere nel nuovo stato, sempre seguendo la sua policy attuale.

Per aggiornare il valore dell'azione appena compiuta, **SARSA ha bisogno di sbirciare un passo nel futuro**. L'agente non si limita ad arrivare nel nuovo stato $S'$, ma "tira i dadi" (consulta la sua policy) per decidere quale sarà la sua prossima azione $A'$, *prima* ancora di eseguirla.

Si noti che $A'$ viene **effettivamente eseguita** (non viene ricalcolata una nuova azione con la nuova policy).

Solo quando ha in mano questa quintupla ($S, A, R, S', A'$), l'agente ha tutte le informazioni per applicare la formula di aggiornamento:

$$
Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma Q(S',A') - Q(S,A)]
$$

Analizziamo i pezzi di questa formula per capirne il senso logico:

- **$Q(S,A)$**: È la nostra stima attuale, ovvero quanto pensiamo sia "buona" l'azione $A$ nello stato $S$.
- **$[R + \gamma Q(S',A')]$**: Questo è il **TD target**, ovvero il nostro "nuovo obiettivo". È composto dalla ricompensa vera appena ottenuta ($R$) più il valore stimato della prossima azione che abbiamo già deciso di fare ($\gamma Q(S',A')$). Rappresenta un'informazione *più fresca e reale* rispetto a quella che avevamo prima.
- **$[R + \gamma Q(S',A') - Q(S,A)]$**: Questo è il **TD Error** (l'errore di predizione). È la differenza tra la nostra vecchia stima e la nuova informazione appena raccolta.
- **$\alpha$ (Learning rate)**: Decide quanto pesantemente correggere la nostra vecchia stima in base all'errore appena calcolato.

Nello specifico l'algoritmo step-by-step:

1. **Inizializzazione:** L'agente si trova nello stato iniziale $S$ e sceglie l'azione $A$ usando la sua policy (ad es. $\epsilon$-greedy).
2. **Esecuzione e Osservazione:** L'agente esegue fisicamente $A$. L'ambiente gli restituisce una ricompensa $R$ e lo sposta nel nuovo stato $S'$.
3. **Pianificazione:** L'agente osserva $S'$ e, usando la *stessa identica policy*, decide la prossima azione $A'$.
4. **Apprendimento (Update):** Ora l'agente ha $S, A, R, S', A'$. Usa la formula per aggiornare la stima di $Q(S,A)$.
5. **Avanzamento:** L'agente "slitta" in avanti. Il nuovo stato diventa lo stato corrente ($S \leftarrow S'$) e la nuova azione diventa l'azione corrente ($A \leftarrow A'$).
6. Il ciclo riparte dal punto 2, finché l'episodio non termina.

## Controllo Off-Policy: Q-Learning

Il passaggio ai metodi **Off-Policy** rappresenta un cambio di paradigma fondamentale nel Reinforcement Learning. Mentre un algoritmo On-Policy (come SARSA) apprende valutando esattamente le regole di comportamento che l'agente sta seguendo in quel momento, l'approccio Off-Policy introduce una separazione netta tra il comportamento fisico nell'ambiente e il processo di apprendimento.

Questa separazione sblocca una possibilità strategica enorme: far muovere l'agente in modo esplorativo, o persino del tutto casuale, pur facendogli imparare in background quale sarebbe la strategia perfetta da adottare.

Per comprendere questa architettura, occorre definire due entità distinte che operano in parallelo:

* La **Behavior Policy ($\mu$)**: È il meccanismo che controlla materialmente l'agente, decidendo quali azioni compiere per esplorare l'ambiente (spesso utilizzando una logica $\epsilon$-greedy per garantire una scoperta continua di nuovi stati).
* La **Target Policy ($\pi$)**: È la strategia ideale e puramente ottimizzante che l'algoritmo sta cercando di imparare, calcolata osservando i risultati ottenuti dalla behavior policy.

Il **Q-Learning** offre un algoritmo di controllo Off-Policy la cui formula di aggiornamento è la seguente:

$$
Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma \max_{a'} Q(S',a') - Q(S,A)]
$$

L'elemento nuovo è il termine $\max_{a'} Q(S',a')$.

A differenza di SARSA, che per completare la formula richiede di sapere quale azione verrà effettivamente compiuta nel *turno successivo*, il Q-Learning guarda al nuovo stato $S'$ e valuta unicamente: *"Qual è l'azione teorica migliore disponibile in questo nuovo stato?"*.

L'algoritmo **finge** che nel turno successivo verrà presa l'azione ottimale, estraendo il valore massimo possibile. Non ha alcuna importanza se, al momento di compiere la mossa successiva, la behavior policy deciderà di intraprendere un'azione completamente diversa e sub-ottimale a scopo esplorativo: l'apprendimento della target policy si baserà sempre e solo sull'ipotesi della mossa perfetta.

I passaggi eseguiti ad ogni step temporale:

1. **Osservazione e Scelta:** L'agente si trova nello stato $S$ e seleziona un'azione $A$ seguendo la sua behavior policy esplorativa $\mu$.
2. **Esecuzione:** L'azione $A$ viene eseguita fisicamente nell'ambiente, restituendo una ricompensa immediata $R$ e portando l'agente in un nuovo stato $S'$.
3. **Calcolo del Target (L'ipotesi dell'ottimo):** Si analizza il nuovo stato $S'$. L'algoritmo individua tutte le possibili azioni $a'$ eseguibili in quello stato e seleziona il valore di quella stimata come la migliore in assoluto ($\max_{a'}$). Questa azione ottimale non viene registrata per essere eseguita, serve esclusivamente per il calcolo matematico.
4. **Apprendimento:** Si applica la formula per aggiornare la stima $Q(S,A)$ relativa all'azione effettivamente completata al punto 1.
5. **Avanzamento:** Il nuovo stato diventa lo stato corrente ($S \leftarrow S'$).
6. Il ciclo riparte dal primo punto. Al momento di compiere la nuova mossa, sarà nuovamente la behavior policy a decidere, estraendo un'azione reale che potrebbe differire totalmente dall'azione ottimale calcolata al punto 3.

Grazie a questa netta distinzione, il Q-Learning risulta estremamente robusto ed efficiente. Permette di raccogliere dati esplorando l'ambiente in modo aggressivo, garantendo al contempo che i valori memorizzati convergano costantemente verso il calcolo del comportamento ottimale.


### Sarsa vs Q-Learning: L'esempio del Dirupo (Cliff Walking)

Supponiamo un agente che deve camminare lungo il bordo di un dirupo per arrivare all'obiettivo. Cadere costa una penalità immensa (-100).

* **Q-Learning (Off-policy)** apprenderà il percorso matematicamente ottimale: camminare esattamente sul bordo del precipizio per minimizzare la distanza. Tuttavia, poiché durante l'addestramento continua ad agire con una certa casualità ($\epsilon$-greedy), cadrà spessissimo, accumulando enormi penalità reali pur avendo calcolato una policy teorica perfetta.
* **SARSA (On-policy)** terrà conto della propria policy esplorativa. Sapendo che c'è un rischio di agire a caso (e cadere), imparerà una funzione valore che ritiene il bordo del dirupo "pericoloso". Di conseguenza, SARSA convergerà su un percorso più lungo e sicuro, tenendosi lontano dal bordo per minimizzare i danni dovuti all'esplorazione.