# Accesso allo stato e diagnosi della simulazione

Quando una simulazione produce un comportamento inatteso, occorre individuare il passaggio in cui nasce la discrepanza: costruzione della scena, comando, avanzamento fisico o osservazione. Le API a basso livello permettono di ispezionare questi passaggi senza attribuire ogni problema all'algoritmo di apprendimento.

Il debug risulta più leggibile con un esperimento piccolo e deterministico: un ambiente, poche geometrie e azioni note. Camere, randomizzazione e training vengono introdotti progressivamente, dopo avere verificato le ipotesi del livello precedente.

## Livelli di accesso

L'ambiente Gymnasium gestisce il ciclo decisionale, comprendendo osservazioni, reward e fine episodio. `InteractiveScene` organizza asset e sensori; `RigidObject` e `Articulation` permettono di leggere e modificare il loro stato fisico.

`SimulationContext` controlla invece l'avanzamento della simulazione. È utile per esperimenti che non richiedono ancora un task RL. La stage USD consente di ispezionare la descrizione della scena, ma non sostituisce necessariamente le interfacce ai buffer fisici aggiornati durante il rollout.

Questi livelli vanno usati in modo coerente. Chiamare manualmente `sim.step()` dentro un loop che usa già `env.step()` può introdurre avanzamenti aggiuntivi, alterando tempi, reward e sincronizzazione dei sensori.

## Tensori e viste sui dati

Per un'articolazione presente nella scena si possono leggere posizioni e velocità dei giunti:

```python
robot = env.unwrapped.scene["cartpole"]
joint_positions = robot.data.joint_pos.torch
joint_velocities = robot.data.joint_vel.torch
```

La forma comprende normalmente ambienti e giunti. Le posizioni dei giunti rotativi sono angoli, mentre quelle dei giunti prismatici sono spostamenti lineari: la stessa struttura tensoriale non implica le stesse unità.

Isaac Lab 3.0 usa **ProxyArray** per le proprietà dati interessate dalla nuova interfaccia e permette di accedere esplicitamente alla vista PyTorch con `.torch`. La [guida di migrazione 3.0](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/migrating-to-3-0) descrive anche le convenzioni dei quaternioni e la selezione degli ambienti nelle scritture.

Una vista può condividere memoria con i dati sottostanti. Per conservare una fotografia dello stato o preparare una modifica separata è opportuno creare una copia con `clone()`. Per analisi offline si può trasferire un campione sulla CPU:

```python
sample = joint_positions[:1].detach().cpu().numpy()
```

Trasferimenti e letture di scalari possono sincronizzare CPU e GPU. Effettuarli continuamente su migliaia di copie può ridurre il throughput; durante il debug è preferibile limitare frequenza e quantità dei dati estratti.

## Riferimenti e orientazioni

Per un corpo rigido, il suffisso `_w` indica generalmente il riferimento globale, o world frame:

```python
cube = scene["cube"]
position_world = cube.data.root_pos_w.torch
orientation_world = cube.data.root_quat_w.torch
linear_velocity_world = cube.data.root_lin_vel_w.torch
```

Nelle convenzioni delle API Isaac Lab 3.0 qui considerate, i quaternioni hanno ordine **XYZW** e l'identità è `(0.0, 0.0, 0.0, 1.0)`. Il confine con librerie esterne o API USD richiede una verifica esplicita: quattro componenti normalizzate possono essere valide numericamente e rappresentare comunque la rotazione sbagliata se l'ordine è diverso.

Nelle scene clonate, una coordinata globale include la traslazione dell'origine dell'ambiente. Per confrontare due copie equivalenti occorre usare un riferimento locale coerente. Molti errori di reset derivano dall'assegnazione della stessa posa globale a oggetti appartenenti a copie differenti.

## Scrittura dello stato e invio dei comandi

Le scritture di stato servono soprattutto per reset e test controllati. Nel caso di un corpo rigido già inizializzato:

```python
pose = cube.data.default_root_pose.torch.clone()
velocity = cube.data.default_root_vel.torch.clone()
pose[:, 2] = 1.2
cube.write_root_pose_to_sim_index(root_pose=pose)
cube.write_root_velocity_to_sim_index(root_velocity=velocity)
cube.reset()
```

La famiglia `_index` seleziona le istanze per indici, mentre le corrispondenti API `_mask` usano maschere; firme e supporto vanno controllati per l'asset e il backend utilizzati. Un reset parziale deve selezionare soltanto gli ambienti interessati.

Un comando di controllo ha uno scopo diverso: imposta target o forze lasciando che il solver calcoli il movimento. Se un robot viene continuamente riposizionato scrivendone lo stato, non si sta verificando la capacità del controller di eseguire quel movimento.

## Ciclo manuale della simulazione

Lo script [`02_tabletop_objects.py`](../scenes_and_objects/lab/02_tabletop_objects.py) costruisce una simulazione senza Gymnasium. Dopo la creazione degli asset e `sim.reset()`, ripete tre operazioni: scrive i dati da applicare con `write_data_to_sim()`, avanza con `sim.step()` e aggiorna gli oggetti con `update(dt)`.

Questo ciclo permette di isolare la fisica prima di definire reward e terminazioni. Saltare l'aggiornamento può lasciare osservazioni non coerenti con l'ultimo passo; aggiungere chiamate di reset nel punto sbagliato può cancellare il movimento che si intende osservare.

## Diagnosi per sintomo

La diagnosi deve partire dall'ipotesi più direttamente osservabile. Se un cubo attraversa il tavolo, conviene controllare prima collider e stato iniziale, poi dimensioni, passo temporale e impostazioni del contatto. Cambiare l'algoritmo RL non risolve una superficie priva di collisioni.

| Sintomo | Verifica iniziale |
| --- | --- |
| `gym.make` non trova il task | Import della registrazione e disponibilità del progetto nell'ambiente Python |
| Oggetto visibile ma senza risposta fisica | Corpo dinamico, massa, collisioni e asset inizializzato |
| Posa errata dopo il reset | Riferimento globale o locale, origine della copia e ordine del quaternion |
| Checkpoint incompatibile | Dimensioni e significato di osservazioni e azioni, wrapper e normalizzazione |
| Immagini nere | Posa della camera, renderer, aggiornamento e risorse disponibili |
| Deformabile instabile | Solver supportato, mesh, materiale e passo temporale |
| GPU poco utilizzata | Sincronizzazioni CPU, caricamento asset, rendering e numero di ambienti |

Una volta corretta la configurazione minima, si reintroducono batch, randomizzazione e sensori uno alla volta. Conservare il comando e le versioni insieme a un caso riproducibile rende possibile distinguere un problema della scena da una differenza tra revisioni del software.
