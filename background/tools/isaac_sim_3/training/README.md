# Training delle policy e parallelismo

Far avanzare un ambiente produce una traiettoria; **addestrare una policy** significa usare l'esperienza raccolta per modificarne i parametri. Nel Reinforcement Learning il comportamento viene ottimizzato attraverso un segnale di reward, anziché copiando direttamente le azioni di un esperto.

Isaac Lab fornisce gli ambienti e le integrazioni con le librerie RL. La logica fisica del task e l'algoritmo di ottimizzazione rimangono responsabilità distinte: un training può essere eseguito correttamente dal punto di vista software e ottimizzare un obiettivo mal specificato.

## Rollout e aggiornamento della policy

Una policy parametrizzata da $\phi$, indicata con $\pi_\phi$, riceve l'osservazione $o_t$ e produce l'azione $a_t$. L'ambiente restituisce una reward $r_t$ e l'osservazione successiva. L'algoritmo raccoglie sequenze di queste interazioni, dette **rollout**, e usa i dati per aggiornare $\phi$.

Un obiettivo comune è massimizzare il ritorno atteso scontato:

$$
J(\phi) = \mathbb{E}_{\pi_\phi}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right]
$$

Qui $T$ è l'orizzonte dell'episodio e $\gamma$, compreso tra 0 e 1, è il fattore di sconto. L'aspettativa considera le traiettorie prodotte dalla policy e dall'ambiente. Il peso attribuito agli esiti lontani dipende sia da $\gamma$ sia dalla durata fisica di un passo decisionale.

**PPO**, Proximal Policy Optimization, alterna raccolta di esperienza e aggiornamenti della policy, limitando tramite il proprio obiettivo quanto il nuovo comportamento si discosti da quello che ha generato i dati. Una funzione valore stima i ritorni futuri e contribuisce a valutare la qualità delle azioni. Il [percorso sui policy gradient e actor-critic](../../../reinforcement_learning/08_policy_gradient_actor_critic/README.md) sviluppa questi concetti.

## Un primo esperimento

Gli esempi usano **RSL-RL**, una libreria integrata con Isaac Lab. Dopo avere verificato l'avvio del task, un comando di training per il percorso Newton è:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
    --num_envs 1024 --max_iterations 200 --seed 42 \
    physics=newton_mjwarp --viz none
```

Per il debug conviene iniziare con meno ambienti e una durata breve. `max_iterations` indica iterazioni dell'algoritmo, non episodi né singoli passi fisici: la quantità di esperienza dipende anche dalla lunghezza dei rollout e dal numero di copie.

I checkpoint conservano lo stato necessario al caricamento o alla ripresa secondo il formato della libreria. Log e configurazioni vengono scritti sotto la directory delle run; per il percorso RSL-RL standard si può consultare `logs/rsl_rl/` con TensorBoard:

```bash
uv run python -m tensorboard.main --logdir logs/rsl_rl
```

La curva di reward mostra l'evoluzione rispetto all'obiettivo implementato. Va letta insieme a durata degli episodi, terminazioni e successo del task: una crescita del ritorno non esclude comportamenti indesiderati.

## Playback e compatibilità

Il playback carica un checkpoint e ne esegue la policy:

```bash
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct \
    --checkpoint latest --num_envs 8 --viz newton_gl physics=newton_mjwarp
```

`latest` è una scorciatoia operativa; un risultato riproducibile deve identificare il file effettivamente usato. Osservazioni, action space, preprocessing e normalizzazione devono corrispondere al training. Un checkpoint che si carica senza errori può comunque comportarsi diversamente se il significato dei dati è cambiato.

La [documentazione RL di Isaac Lab](https://docs.nvidia.com/isaaclab/3.0/concepts/reinforcement-learning) descrive le integrazioni e il caricamento delle policy. Per un task personalizzato occorre anche rendere disponibili registrazione e configurazione dell'agente, come spiegato nel [capitolo sui task](../custom_tasks/README.md).

## Ambienti vettorializzati e training distribuito

Su una GPU, un singolo processo può simulare molte copie e inviare alla rete un batch di osservazioni. Questa **vettorializzazione** sfrutta operazioni parallele e riduce il costo relativo del coordinamento rispetto a esecuzioni separate.

Il **training distribuito** aggiunge più processi, generalmente uno per GPU. Ogni processo raccoglie esperienza nei propri ambienti e partecipa alla sincronizzazione degli aggiornamenti. Nel launcher multi-GPU della versione 3.0, `--num_envs` indica il numero per GPU:

```bash
uv run isaaclab train_multigpu --rl_library rsl_rl --task Isaac-Cartpole \
    --num_gpus 2 --num_envs 2048
```

In questo esempio il totale è 4,096 ambienti. La [guida NVIDIA multi-GPU e multi-node](https://docs.nvidia.com/isaaclab/3.0/features/multi-gpu-and-multi-node-training) documenta il launcher, indicato come sperimentale, e i requisiti Linux e NCCL per la comunicazione tra GPU.

Aumentare il parallelismo modifica anche la quantità di dati raccolti per aggiornamento. Il confronto deve quindi specificare se vengono mantenuti costanti numero di transizioni, iterazioni o tempo reale. Un raddoppio delle GPU non garantisce un raddoppio della velocità né lo stesso andamento dell'apprendimento.

## Misure e riproducibilità

Il **throughput** esprime quante transizioni vengono raccolte per secondo reale. Va distinto dal numero di passi fisici, perché una decisione può comprenderne diversi. Con immagini, rendering e trasferimenti di memoria possono diventare il costo dominante.

Per diagnosticare le prestazioni servono anche utilizzo GPU, VRAM e tempi di raccolta e aggiornamento. Una configurazione molto veloce con osservazioni di stato non costituisce un riferimento direttamente trasferibile a una policy visuale.

L'esperimento deve conservare seed, revisione del codice, dipendenze, backend, passo fisico, decimation, numero di ambienti e iperparametri. Più seed permettono di osservare la variabilità del processo di apprendimento; la valutazione finale richiede poi episodi separati, descritti nel [capitolo sull'evaluation](../evaluation/README.md).
