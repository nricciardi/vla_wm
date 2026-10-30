# Scene, corpi rigidi e proprietà di contatto

Una scena robotica comprende elementi visivi e componenti fisiche. La superficie disegnata sullo schermo serve al rendering; la geometria usata dal motore per rilevare i contatti è invece un **collider**. Le due rappresentazioni possono avere dettagli differenti, purché la loro approssimazione sia adeguata al task.

Un tavolo visibile ma privo di collider non sostiene un oggetto. Analogamente, una mesh a forma di cubo non diventa automaticamente un corpo che cade sotto gravità. Comprendere questa separazione permette di interpretare la costruzione della scena prima di aggiungere un robot.

## Corpi statici, dinamici e articolati

Un **corpo statico** partecipa alle collisioni senza essere mosso dalle forze: il piano del tavolo può essere modellato in questo modo. Un **corpo dinamico** possiede massa e inerzia e viene aggiornato dal solver in risposta a gravità, forze e contatti.

Un corpo **cinematico** segue invece un movimento prescritto, mantenendo una partecipazione ai contatti secondo le regole del backend. Prescriverne la traiettoria non equivale a controllarlo attraverso una forza: questa differenza conta quando si misura la risposta di altri oggetti.

Un robot richiede una struttura **articolata**, nella quale più corpi sono collegati da giunti. Una raccolta di mesh separate non specifica da sola né la cinematica né gli attuatori. Il passaggio dal cubo al manipolatore introduce quindi vincoli e controller, oltre alla geometria.

## Una scena procedurale con tavolo e cubi

Lo script [`02_tavolo_oggetti.py`](../scripts/02_tavolo_oggetti.py) costruisce una scena tramite primitive geometriche, senza modelli USD esterni. Per il percorso Isaac Sim previsto dall'esempio:

```bash
uv run --extra isaacsim python \
    /percorso/vla_wm/background/tools/isaac_sim_3/scripts/02_tavolo_oggetti.py \
    --viz kit --mass 0.5 --friction 0.65
```

Il piano del tavolo è un parallelepipedo largo 1.2 m, profondo 0.8 m e spesso 0.10 m. Il centro si trova a quota 0.75 m, quindi la superficie superiore è a 0.80 m. Tre cubi di lato 0.10 m partono più in alto e vengono lasciati cadere.

Il parametro `--mass` stabilisce la massa di base: i tre cubi hanno rispettivamente massa pari a una, due e tre volte questo valore. Lo script modifica le proprietà dei cubi; il materiale della superficie di appoggio deve essere considerato separatamente quando si studia il contatto.

## Configurazione e creazione degli oggetti

Una configurazione descrive dimensioni, materiale, proprietà fisiche e stato iniziale. Uno **spawner** crea poi gli elementi della scena corrispondenti. Nel frammento seguente, tratto dalla logica dell'esempio, la configurazione del cubo include componenti distinte:

```python
sim_utils.CuboidCfg(
    size=(0.10, 0.10, 0.10),
    rigid_props=UsdPhysicsRigidBodyCfg(),
    mass_props=MassCfg(mass=0.5),
    collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
    physics_material=RigidBodyMaterialBaseCfg(
        static_friction=0.65,
        dynamic_friction=0.5,
        restitution=0.05,
    ),
)
```

`rigid_props` abilita il comportamento di corpo rigido dinamico, `collision_props` definisce la partecipazione alle collisioni e `mass_props` assegna la massa. Il materiale fisico descrive il contatto; un materiale visuale controlla invece l'apparenza nel renderer.

Gli oggetti sono collocati in percorsi come `/World/Table` e `/World/Cube0`. Questi identificano nodi nella gerarchia della scena. Nel normale ciclo GPU gli asset fisici vengono creati **prima dell'inizializzazione della simulazione**, affinché il backend possa costruire strutture e buffer coerenti.

## Massa, attrito e restituzione

La **massa** misura l'inerzia traslazionale; l'inerzia rotazionale dipende anche dalla distribuzione della massa e dalla geometria. Cambiare massa può modificare l'interazione con un robot, ma non produce automaticamente una diversa accelerazione di caduta libera sotto la stessa gravità.

L'**attrito statico** riguarda la resistenza all'inizio dello scorrimento, mentre quello **dinamico** riguarda lo scorrimento già in corso. Il contatto dipende da entrambi i materiali e dalla regola con cui il solver ne combina i coefficienti. Una caduta verticale è quindi un esperimento poco informativo per confrontare due valori di attrito.

La **restituzione** controlla il comportamento di rimbalzo nel modello di contatto. Anche in questo caso il risultato dipende dalla discretizzazione temporale e dalle impostazioni del solver: il coefficiente, da solo, non garantisce una traiettoria ideale.

Per isolare l'effetto dell'attrito conviene applicare la stessa spinta orizzontale a oggetti inizialmente uguali. Per studiare il rimbalzo occorre mantenere fissi altezza, geometria e passo fisico, variando il parametro di interesse.

## Lettura e reset dello stato

A simulazione avviata, un `RigidObject` espone pose e velocità. La proprietà `root_pos_w` indica la posizione nel riferimento globale; l'accesso `.torch` fornisce il tensore utilizzabile con PyTorch:

```python
positions = cube.data.root_pos_w.torch
```

Per riportare un cubo a una posa iniziale è necessario reinizializzare anche la velocità, se non si vuole conservare il moto precedente:

```python
pose = cube.data.default_root_pose.torch.clone()
pose[:, 2] = 1.2
cube.write_root_pose_to_sim_index(root_pose=pose)
cube.write_root_velocity_to_sim_index(
    root_velocity=cube.data.default_root_vel.torch.clone()
)
cube.reset()
```

Il frammento presuppone un oggetto già creato e inizializzato. In una scena con copie traslate, le pose globali devono tenere conto dell'origine di ciascun ambiente. La [guida NVIDIA sui corpi rigidi](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/interacting-with-a-rigid-object) descrive il ciclo di scrittura, avanzamento e aggiornamento.

**Reimpostare lo stato è diverso dall'applicare un comando fisico**. Riscrivere la posa a ogni frame impedirebbe di osservare una normale evoluzione libera. Analogamente, modificare un campo della configurazione dopo l'avvio non implica che massa o attrito cambino nel backend: la randomizzazione richiede le API o gli eventi previsti per quella proprietà.
