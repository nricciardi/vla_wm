# Interazione con un ambiente Gymnasium

**Gymnasium** definisce un'interfaccia per far interagire un agente con un ambiente. L'agente non deve conoscere ogni dettaglio del motore fisico: riceve un'osservazione, invia un'azione e ottiene il risultato della transizione. Questa separazione permette di cambiare algoritmo mantenendo la stessa definizione del task.

In Isaac Lab l'interfaccia viene usata con **ambienti vettorializzati**, cioè più copie simulate contemporaneamente. I valori restituiti sono quindi spesso tensori PyTorch sulla GPU, con una riga per ogni copia, anziché scalari relativi a un unico mondo.

## Costruzione e inizializzazione

L'esempio di accompagnamento è [`01_gym_rollout.py`](lab/01_gym_rollout.py), eseguibile dopo la [verifica dell'installazione](../local_installation/README.md). La sua struttura centrale è:

```python
cfg, _ = resolve_task_config(
    "Isaac-Cartpole-Direct", "", overrides=("physics=newton_mjwarp",)
)
cfg.scene.num_envs = 16
with launch_simulation(cfg, args):
    env = gym.make("Isaac-Cartpole-Direct", cfg=cfg)
    try:
        obs, info = env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(actions)
    finally:
        env.close()
```

Il frammento mostra il ciclo, ma richiede gli import e il parsing degli argomenti presenti nello script completo. `resolve_task_config` recupera la configurazione; `launch_simulation` inizializza il runtime; `gym.make` costruisce l'ambiente registrato. La configurazione descrive ciò che verrà creato, senza essere essa stessa la simulazione in esecuzione.

La **registrazione** associa un nome a un'implementazione. Importare `isaaclab_tasks` rende disponibili le registrazioni dei task forniti dal framework. La documentazione del [workflow Direct](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/creating-a-direct-workflow-rl-environment) sviluppa la classe che implementa Cartpole.

## Significato di reset e step

`reset()` inizializza un episodio e restituisce l'osservazione iniziale insieme a `info`, un contenitore di informazioni aggiuntive. Il reset può campionare condizioni diverse, per esempio una nuova inclinazione del pendolo.

`step(actions)` applica le azioni per un intervallo di controllo e restituisce cinque elementi. `obs` contiene le nuove osservazioni; `reward` il segnale numerico del task; `terminated` indica una fine prevista dalla logica del problema; `truncated` segnala un'interruzione esterna a quella logica, tipicamente il limite temporale. `info` può contenere diagnostica e metriche specifiche.

**Terminazione e successo non coincidono**. Un pendolo che cade può produrre `terminated=True`, mentre un episodio che raggiunge la durata massima può essere troncato. La definizione di successo va cercata nel task. La distinzione influisce anche sul bootstrap della funzione valore negli algoritmi RL, come chiarisce la [guida Gymnasium su termination e truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Dimensioni e significato delle azioni

Se vengono eseguiti $N$ ambienti con azioni di dimensione $d_a$, un batch di azioni ha tipicamente forma $(N,d_a)$. Qui $N$ è il numero di copie e $d_a$ il numero di componenti del comando per copia. Le osservazioni numeriche possono avere forma $(N,d_o)$, dove $d_o$ è la dimensione osservata.

Nel Cartpole standard il vettore osservato comprende posizione e velocità del carrello e angolo e velocità del pendolo. L'ordine delle componenti deve essere verificato nell'implementazione: la sola forma `(16, 4)` non ne specifica il significato.

Analogamente, un'azione numerica può essere scalata dal task prima di diventare una forza. **Un valore di 0.25 non indica automaticamente 0.25 newton**. Occorre leggere limiti, normalizzazione e conversione verso l'attuatore. Lo script alterna comandi artificiali; non contiene una policy addestrata.

## Osservazioni e stato interno

`obs["policy"]` identifica il gruppo di osservazioni destinato alla policy quando il task adotta questa convenzione. L'accesso `env.unwrapped` permette invece di raggiungere l'ambiente sottostante eventuali wrapper, cioè componenti che trasformano l'interfaccia o i dati.

Per ispezionare l'articolazione del Cartpole si può usare:

```python
cartpole = env.unwrapped.scene["cartpole"]
joint_positions = cartpole.data.joint_pos.torch
print(joint_positions.shape)
```

Leggere lo stato interno è utile per il debug, ma inserirlo nell'input della policy modifica il problema osservativo. Una policy che riceve immagini non deve ottenere involontariamente anche pose esatte non disponibili sul robot reale.

## Reset indipendenti

Negli ambienti RL Isaac Lab il reset delle copie terminate è normalmente gestito dal framework. Una copia può iniziare un nuovo episodio mentre le altre proseguono. Il comportamento preciso dei dati terminali e dell'osservazione restituita va verificato nell'ambiente e nei wrapper utilizzati.

Un ciclo di raccolta deve pertanto mantenere ritorno e durata separati per ogni copia, chiudendoli quando `terminated | truncated` è vero. Richiamare indiscriminatamente `env.reset()` dopo la fine di una sola copia interromperebbe anche gli episodi ancora attivi. Il [capitolo sull'evaluation](../evaluation/README.md) sviluppa questa gestione episodica.
