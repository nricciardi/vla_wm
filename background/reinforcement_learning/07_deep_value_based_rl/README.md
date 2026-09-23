# Deep Value-Based Reinforcement Learning

I metodi value-based con approssimatori non lineari combinano gli aggiornamenti del Reinforcement Learning con la capacità rappresentativa delle reti neurali. Questa combinazione permette di trattare osservazioni ad alta dimensionalità, ma rende il training instabile perché target, distribuzione dei dati e funzione approssimata cambiano contemporaneamente.

Il capitolo riprende dal percorso generale sull'[approssimazione delle funzioni](../06_approssimazione_funzioni/README.md) e concentra l'attenzione su replay buffer e Deep Q-Network.

## Batch Reinforcement Learning

Gli algoritmi incrementali aggiornano i parametri dopo ogni nuova esperienza. Sono semplici e hanno un costo ridotto per aggiornamento, ma non sono efficienti nell’utilizzo dei dati.

Una transizione osservata viene normalmente utilizzata una sola volta e poi scartata. Se raccogliere esperienza è costoso, può essere conveniente riutilizzare più volte le stesse osservazioni.

I metodi di **Batch Reinforcement Learning** cercano la funzione di valore che si adatta meglio a un intero dataset di esperienze già raccolte.


### SGD con Experience Replay

Un modo semplice per approssimare la soluzione least squares consiste nell’utilizzare l’**experience replay**.

A ogni iterazione:

1. Si estrae casualmente una coppia dal dataset

$$
(s, v^{\pi}) \sim \mathcal{D}
$$

2. Si esegue un aggiornamento dei parametri $w$ tramite Stochastic Gradient Descent

$$
\Delta w = \alpha (v^{\pi} - \hat{v}(s, w)) \nabla_w \hat{v}(s, w)
$$

Il campionamento casuale ha anche la funzione di **ridurre la correlazione tra osservazioni consecutive**. Se utilizzassimo direttamente le transizioni nell’ordine in cui sono state generate, campioni adiacenti sarebbero molto simili e violerebbero fortemente l’ipotesi di indipendenza utilizzata dagli algoritmi di ottimizzazione.


### Deep Q-Network

I Deep Q-Network, o DQN, combinano Q-learning e reti neurali profonde.

Durante l’interazione:

1. L’agente sceglie $a_t$ con una policy $\varepsilon$-greedy
2. Osserva ricompensa e stato successivo
3. Memorizza la transizione $(s_t, a_t, r_{t+1}, s_{t+1})$ in una *replay memory* $\mathcal{D}$
4. Estrae casualmente un mini-batch di transizioni dalla memoria
5. Aggiorna i pesi della rete neurale minimizzando l'errore tra il Q-value corrente e il target Q-learning

La loss è definita come:

$$
L(w) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} Q(s', a'; w^-) - Q(s, a; w)\right)^2\right]
$$

Come ulteriore elemento di stabilizzazione si utilizzano i **fixed Q-targets** ottenuti da una copia periodica della rete neurale principale. Il vettore $w^-$ indica i parametri della target network, mentre $w$ indica quelli della rete aggiornata a ogni passo di ottimizzazione.

La rete che produce i target viene aggiornata meno frequentemente, riducendo la correlazione tra il target e i parametri che si stanno aggiornando.
Senza una target network, la rete cercherebbe di inseguire un target che cambia immediatamente dopo ogni aggiornamento.

