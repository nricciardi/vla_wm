# Reinforcement Learning

Il **Reinforcement Learning (RL)** studia agenti che imparano a prendere decisioni attraverso l'interazione con un ambiente. A differenza dell'apprendimento supervisionato, il segnale di training non specifica direttamente quale azione sarebbe corretta in ogni situazione: l'agente osserva gli effetti delle proprie decisioni e cerca di massimizzare il ritorno cumulativo atteso.

Questo capitolo organizza i concetti necessari per comprendere il ruolo del RL nei Vision-Language-Action model e nei World Model. Il percorso parte dalla formalizzazione dell'interazione agente–ambiente, introduce Markov Decision Process e funzioni valore, quindi distingue metodi di prediction, controllo, esplorazione e pianificazione.

## Dal problema decisionale all'apprendimento

Al tempo $t$, l'ambiente si trova in uno stato $s_t$, l'agente riceve un'osservazione $o_t$ e seleziona un'azione $a_t$. La dinamica produce lo stato successivo e una ricompensa, mentre la policy determina come l'agente sceglie le azioni sulla base dell'informazione disponibile.

Nel caso più semplice, il comportamento è descritto da una policy stocastica $π(a_t\mid s_t)$. In un sistema parzialmente osservabile, più vicino alla robotica reale, la decisione dipende invece dall'osservazione, dalla storia o da uno stato interno. Quando il task è condizionato dal linguaggio, l'istruzione $l$ entra a sua volta nel contesto della policy:

$$
\pi_\theta(a_t\mid o_{\leq t},l)
$$

L'obiettivo consiste nel massimizzare il ritorno scontato atteso:

$$
J(\pi_\theta) =
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_{t+1}
\right]
$$

Il parametro $\theta$ raccoglie i parametri della policy, $\tau$ indica una traiettoria, $T$ è l'orizzonte temporale, $r_{t+1}$ è la ricompensa ricevuta dopo l'azione $a_t$ e $\gamma\in[0,1]$ è il fattore di sconto.

L'[introduzione ai fondamenti](01_fondamenti/README.md) sviluppa il ciclo agente–ambiente, reward, stato, osservabilità e le principali tassonomie degli agenti.

## Processi decisionali di Markov e funzioni valore

I **Markov Decision Process (MDP)** forniscono la formulazione matematica standard del problema. Separano la dinamica dell'ambiente, descritta dalle probabilità di transizione, dalla policy dell'agente. Le funzioni valore quantificano il ritorno atteso associato a uno stato o a una coppia stato–azione, mentre le equazioni di Bellman ne esprimono la struttura ricorsiva.

Il capitolo sui [processi markoviani](02_processi_markoviani/README.md) introduce catene di Markov, Markov Reward Process, MDP, policy, funzioni valore e condizioni di ottimalità.

## Planning con un modello noto

Quando dinamica e ricompensa sono note, il problema può essere risolto mediante **planning**. La programmazione dinamica alterna valutazione e miglioramento della policy oppure combina i due passaggi nella value iteration. Questi algoritmi rendono esplicito il principio di Generalized Policy Iteration che ricompare anche nei metodi model-free.

L'approfondimento sulla [programmazione dinamica](03_programmazione_dinamica/README.md) tratta iterative policy evaluation, policy iteration, value iteration e aggiornamenti asincroni.

## Apprendimento model-free

I metodi **model-free** stimano valori o policy senza apprendere esplicitamente la dinamica dell'ambiente. Nella prediction si valuta una policy fissata; nel controllo si cerca invece una policy ottima mentre vengono raccolte nuove transizioni.

La [model-free prediction](04_model_free_prediction/README.md) confronta Monte Carlo e Temporal-Difference learning e introduce $n$-step return ed eligibility traces. Il capitolo sul [model-free control](05_model_free_control/README.md) sviluppa Generalized Policy Iteration, controllo Monte Carlo, SARSA e Q-learning, chiarendo la distinzione tra apprendimento on-policy e off-policy.

## Approssimazione, Deep RL e policy gradient

Negli spazi grandi o continui non è possibile conservare un valore distinto per ogni stato. L'[approssimazione delle funzioni](06_approssimazione_funzioni/README.md) sostituisce quindi le tabelle con modelli parametrici e discute aggiornamenti incrementali, convergenza, metodi batch e least-squares.

Il [Deep Value-Based Reinforcement Learning](07_deep_value_based_rl/README.md) concentra l'attenzione sui problemi introdotti dalle reti neurali, sul replay buffer e sulle Deep Q-Network. Il percorso sui [policy gradient e actor-critic](08_policy_gradient_actor_critic/README.md) considera invece policy parametrizzate direttamente, REINFORCE, baseline e stime del vantaggio.

## Esplorazione

Le azioni determinano non soltanto la ricompensa, ma anche i dati che l'agente osserverà in seguito. Il compromesso tra **exploration ed exploitation** è quindi parte del problema di apprendimento. Il capitolo sull'[esplorazione](09_esplorazione/README.md) parte dai multi-armed bandit e introduce regret, metodi ottimistici, approcci bayesiani e l'estensione agli MDP.

## Model-based Reinforcement Learning e World Models

Nel **Model-Based Reinforcement Learning** l'agente dispone di un modello della dinamica oppure lo apprende dalle transizioni. Il modello può generare esperienza simulata, supportare il planning o valutare sequenze di azioni candidate. La distinzione tra imparare un modello accurato e imparare una rappresentazione utile al controllo conduce direttamente ai World Models moderni.

L'approfondimento sul [Model-Based Reinforcement Learning](10_model_based_rl/README.md) introduce modelli tabulari, Dyna, sample-based planning e ricerca ad albero. Il capitolo generale sui [World Models](../../world_models/README.md) prosegue questo percorso con dinamiche latenti, imagination, predizione visuale e controllo robotico.

## Relazione con VLA e Imitation Learning

Molti VLA vengono addestrati inizialmente mediante Behavioral Cloning e non tramite interazione RL online. La formulazione del Reinforcement Learning resta tuttavia utile per descrivere policy, traiettorie, reward, valutazione closed loop e possibili fasi di post-training. La differenza fondamentale è la provenienza del segnale di apprendimento: il [Behavioral Cloning](../imitation_learning/README.md) replica azioni dimostrate da un esperto, mentre il RL ottimizza un obiettivo definito attraverso le conseguenze dell'interazione.
