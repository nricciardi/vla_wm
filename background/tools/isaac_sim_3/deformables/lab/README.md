# Guida a Isaac Lab: simulare oggetti rigidi e deformabili su un tavolo

> **Livello:** principiante assoluto. Questa guida parte dal presupposto che tu conosca le basi di Python, ma non abbia mai usato Isaac Sim, Isaac Lab, PhysX o un simulatore fisico.
>
> **Obiettivo:** capire **cosa fanno**, **come si collegano** e **come modificare** i tre file `06_deformable_tabletop.py`, `scene_config.py` e `tabletop_scene.json`.

## Indice

1. [Che cosa stiamo costruendo?](#1-che-cosa-stiamo-costruendo)
2. [Le parole da conoscere](#2-le-parole-da-conoscere)
3. [Come eseguire il progetto](#3-come-eseguire-il-progetto)
4. [I tre file e il loro rapporto](#4-i-tre-file-e-il-loro-rapporto)
5. [Il file JSON: descrivere la scena](#5-il-file-json-descrivere-la-scena)
6. [`scene_config.py`: leggere e controllare il JSON](#6-scene_configpy-leggere-e-controllare-il-json)
7. [`06_deformable_tabletop.py`: dal programma alla simulazione](#7-06_deformable_tabletoppy-dal-programma-alla-simulazione)
8. [Oggetti rigidi e deformabili: le differenze nel codice](#8-oggetti-rigidi-e-deformabili-le-differenze-nel-codice)
9. [Che cosa significano i parametri fisici?](#9-che-cosa-significano-i-parametri-fisici)
10. [Posizioni, rotazioni e velocità](#10-posizioni-rotazioni-e-velocità)
11. [Come funziona il ciclo di simulazione](#11-come-funziona-il-ciclo-di-simulazione)
12. [Esperimenti guidati](#12-esperimenti-guidati)
13. [Errori comuni e soluzioni](#13-errori-comuni-e-soluzioni)
14. [Glossario e risorse](#14-glossario-e-risorse)

---

## 1. Che cosa stiamo costruendo?

Immagina un laboratorio virtuale. Al suo interno ci sono:

- un pavimento;
- una luce, per vedere la scena;
- un tavolo fisso;
- un **cubo rigido**, che può cadere e muoversi ma non cambiare forma;
- un **cubo deformabile**, che può schiacciarsi e deformarsi;
- una **sfera deformabile**, che può cambiare forma durante urti e contatti.

Il programma crea questo ambiente, inizializza i corpi e fa avanzare il tempo simulato, aggiornando la fisica a ogni passo.

**Non c'è un robot in questi file.** Il progetto è un piccolo esperimento di fisica che, in futuro, potrebbe diventare la base per una scena con un robot.

La scena prevista dal JSON fornito contiene:

| Oggetto | Tipo | Forma | Massa | Posizione orizzontale iniziale | Colore indicativo |
|---|---|---|---|---|---|
| `RigidBox` | Rigido | Cubo 16 cm di lato | 0,5 kg | Sinistra (`x = -0.35`) | Rosso/arancio |
| `SoftBox` | Deformabile | Cubo 16 cm di lato | 0,5 kg | Centro (`x = 0`) | Azzurro |
| `SoftSphere` | Deformabile | Sfera di raggio 8 cm | 0,5 kg | Destra (`x = 0.35`) | Verde |

Questo è ciò che il **codice configura**: il comportamento visualizzato dipenderà dall'esecuzione effettiva del simulatore. Non è stato eseguito un test fisico in questa guida.

### Un modello mentale utile

Pensa a un videogioco, ma con l'obiettivo di simulare proprietà fisiche invece di costruire un gioco:

- **Il JSON** è la lista degli oggetti e dei loro attributi.
- **`scene_config.py`** è il controllore dei dati: rifiuta descrizioni sbagliate.
- **`06_deformable_tabletop.py`** è il costruttore e il regista dell'esperimento.
- **Isaac Sim** fornisce il mondo 3D e l'infrastruttura di simulazione.
- **Isaac Lab** fornisce gli strumenti Python con cui creiamo e gestiamo la scena.
- **PhysX** è il motore che calcola il movimento, i contatti e le deformazioni.

---

## 2. Le parole da conoscere

### Simulazione fisica

È un programma che approssima l'evoluzione di un mondo nel tempo utilizzando modelli matematici: gravità, massa, collisioni, attrito, elasticità e così via.

Un oggetto non viene semplicemente animato lungo un percorso predefinito. Il motore calcola come dovrebbe muoversi in risposta alle condizioni della scena.

### Isaac Sim, Isaac Lab e PhysX

- **Isaac Sim:** ambiente di simulazione 3D basato su tecnologie NVIDIA e USD.
- **Isaac Lab:** libreria Python che semplifica la costruzione di esperimenti e il controllo di asset simulati.
- **PhysX:** motore fisico scelto da questo script, tramite `PhysxCfg()`.
- **CUDA/GPU:** tecnologia e dispositivo utilizzati qui per eseguire la simulazione fisica; questo esempio richiede una GPU CUDA compatibile.

Isaac Lab non è un sostituto di Isaac Sim e PhysX: il codice li utilizza insieme, con ruoli differenti.

### Scena, prim e USD

La **scena** è l'intero mondo virtuale. Un **prim** è un elemento identificabile di quel mondo, ad esempio il tavolo o un cubo. **USD** è il sistema utilizzato per descrivere la struttura e i dati delle scene 3D.

I prim sono organizzati in percorsi simili a cartelle:

```text
/World
  /Ground
  /Light
  /Table
  /Objects
    /RigidBox
    /SoftBox
    /SoftSphere
```

`/World/Objects/SoftBox` non è un file nel computer: è un **indirizzo dentro la scena**.

### Oggetto rigido e oggetto deformabile

Un **rigid body** (corpo rigido) può traslare e ruotare, ma mantiene la forma. Un **deformable body** può modificare anche la propria geometria.

Per rappresentare un solido deformabile, il simulatore usa una **mesh di simulazione**: una suddivisione del volume in piccoli elementi, tipicamente tetraedri. I nodi di questa mesh possono spostarsi l'uno rispetto all'altro.

### Configurazione e istanza

Nelle API troverai spesso classi con il suffisso `Cfg`:

```python
config = DeformableObjectCfg(...)
body = DeformableObject(cfg=config)
```

`DeformableObjectCfg` è la **descrizione di come creare** l'oggetto. `DeformableObject` è l'**istanza Python** che Isaac Lab usa per interagire con l'oggetto nella simulazione. Configurare un oggetto non equivale ad avanzare il tempo fisico.

---

## 3. Come eseguire il progetto

### 3.1 Struttura dei file

Mantieni i tre file nella stessa cartella:

```text
progetto/
  README.md
  06_deformable_tabletop.py
  scene_config.py
  tabletop_scene.json
```

Lo script principale importa `scene_config` e, se non specifichi `--config`, cerca `tabletop_scene.json` **accanto a `scene_config.py`**.

### 3.2 Primo passo: validare senza Isaac Lab

Per il primo controllo è sufficiente Python. Apri un terminale nella cartella dei file ed esegui:

```bash
python 06_deformable_tabletop.py --validate-only
```

Oppure, specificando il file:

```bash
python 06_deformable_tabletop.py --config tabletop_scene.json --validate-only
```

**Che cosa succede?** Il programma legge il JSON, crea descrizioni Python, verifica nomi, numeri, dimensioni e campi, poi stampa un riepilogo. **Non avvia Isaac Sim e non simula il movimento**.

Un riepilogo coerente con il JSON fornito include:

```text
Valid scene: 3 objects; tabletop dimensions=(1.4, 0.9, 0.1) m
  RigidBox: shape=box, deformable=False, mass=0.5 kg, friction=(0.7, 0.5)
  SoftBox: shape=box, deformable=True, mass=0.5 kg, friction=(0.7, 0.5)
  SoftSphere: shape=sphere, deformable=True, mass=0.5 kg, friction=(0.7, 0.5)
```

Questo è un **output atteso ricostruito dal codice**, non il resoconto di un'esecuzione eseguita per questa guida.

Puoi anche vedere i parametri disponibili con:

```bash
python 06_deformable_tabletop.py --help
```

### 3.3 Secondo passo: eseguire la simulazione completa

Il progetto è scritto per le API a *frammenti* di **Isaac Lab 3.0**, in combinazione con Isaac Sim e PhysX. Per farlo partire devi predisporre un'installazione compatibile, tipicamente con una **GPU NVIDIA supportata e CUDA**; il solo Python di sistema non basta.

Segui prima la [guida ufficiale all'installazione di Isaac Lab 3.0](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation).

Il commento iniziale del file principale propone di eseguire il seguente comando **dalla cartella di checkout di Isaac Lab**, sostituendo `/percorso/assoluto/` con il percorso reale del tuo script:

```bash
uv run --extra isaacsim --extra tetrahedralization python /percorso/assoluto/06_deformable_tabletop.py --viz kit
```

`uv run` avvia il programma nell'ambiente Python gestito da `uv`. `--extra isaacsim` richiede le dipendenze di Isaac Sim; `--extra tetrahedralization` include quelle necessarie per generare le mesh tetraedriche dei solidi deformabili; `--viz kit` chiede il visualizzatore Kit, se disponibile nella tua installazione.

**Importante:** il comando e le API sono specifici della versione prevista dal progetto; una versione diversa di Isaac Lab può richiedere modifiche. La riuscita dipende dalle librerie e dai driver effettivamente installati.

### 3.4 Opzioni utili

| Opzione | Cosa controlla | Valore predefinito |
|---|---|---|
| `--config FILE` | Quale JSON leggere | `tabletop_scene.json` accanto a `scene_config.py` |
| `--validate-only` | Controllo dei dati senza simulazione | Disattivato |
| `--steps N` | Numero massimo di passi fisici | `1200` |
| `--dt SECONDI` | Durata virtuale di un passo | `1/120` s |
| `--log-every N` | Ogni quanti passi stampare i dati | `120` |
| `--device cuda:0` | GPU da usare per la simulazione | `cuda:0` quando non specificata |

Le opzioni aggiunte dal launcher di Isaac Lab, come `--device` e `--viz`, sono disponibili solo quando Isaac Lab è presente e il programma viene avviato normalmente, **non** nel semplice percorso `--validate-only`.

Per esempio, per una simulazione più breve con log più frequenti:

```bash
uv run --extra isaacsim --extra tetrahedralization python /percorso/assoluto/06_deformable_tabletop.py --viz kit --steps 240 --log-every 30
```

---

## 4. I tre file e il loro rapporto

```text
                 tabletop_scene.json
          (tavolo, oggetti, materiali)
                         |
                         v
                   scene_config.py
           (legge, interpreta, valida)
                         |
                 TableSpec + ObjectSpec
                         |
                         v
              06_deformable_tabletop.py
        (crea la scena e avvia PhysX)
                         |
                         v
              simulazione passo per passo
                         |
                         v
             posizioni e log nel terminale
```

Questa separazione è intenzionale: puoi modificare dimensioni, masse, oggetti e materiali nel JSON senza riscrivere il codice del ciclo fisico.

**Ordine consigliato di lettura:** prima `tabletop_scene.json`, poi `scene_config.py`, infine le funzioni `main()`, `run_simulation()` e `create_scene()` di `06_deformable_tabletop.py`. Nel file principale, le funzioni sono definite in alto ma partono realmente da `main()`, in fondo.

---

## 5. Il file JSON: descrivere la scena

JSON è un formato testuale per rappresentare dati. Non contiene istruzioni Python: contiene **valori** che il programma legge.

La struttura principale del file è:

```json
{
  "table": { "size": [1.4, 0.9, 0.1], "center": [0.0, 0.0, 0.75] },
  "objects": [
    { "name": "RigidBox", "shape": "box", "deformable": false },
    { "name": "SoftBox", "shape": "box", "deformable": true },
    { "name": "SoftSphere", "shape": "sphere", "deformable": true }
  ]
}
```

Questo è uno **schema abbreviato** per capire la struttura; il file originale contiene molti altri parametri. I campi non indicati ricevono i valori predefiniti definiti in `scene_config.py`.

### 5.1 Le misure del tavolo

Nel tuo file:

```json
"table": {
  "size": [1.4, 0.9, 0.1],
  "center": [0.0, 0.0, 0.75],
  "static_friction": 0.7,
  "dynamic_friction": 0.5
}
```

- `size`: lunghezza lungo X = **1,4 m**, larghezza lungo Y = **0,9 m**, spessore lungo Z = **0,1 m**.
- `center`: centro geometrico del tavolo in coordinate del mondo = **(0, 0, 0,75 m)**.
- `static_friction` e `dynamic_friction`: parametri dell'attrito nei contatti.

La superficie superiore è a:

```text
z_superficie = center_z + spessore / 2
             = 0.75 + 0.10 / 2
             = 0.80 metri
```

Il tavolo è un oggetto **statico**: il codice abilita le collisioni, ma non gli assegna un corpo rigido dinamico. Per questo non cade quando qualcosa lo colpisce.

### 5.2 Le misure degli oggetti

Il cubo morbido del file originale contiene:

```json
{
  "name": "SoftBox",
  "shape": "box",
  "deformable": true,
  "size": [0.16, 0.16, 0.16],
  "position": [0.0, 0.0, 0.30],
  "mass": 0.5,
  "static_friction": 0.7,
  "dynamic_friction": 0.5,
  "youngs_modulus": 10000.0,
  "poissons_ratio": 0.4,
  "elasticity_damping": 0.005,
  "edge_refinement": 4.0,
  "color": [0.15, 0.55, 0.85]
}
```

Traduzione dei campi principali:

| Campo | In parole semplici |
|---|---|
| `name` | Nome univoco usato anche nel percorso USD |
| `shape` | `box`, `sphere`, `cylinder` oppure `usd` |
| `deformable` | `true` = morbido; `false` = rigido |
| `size` | Dimensioni della scatola, in metri |
| `radius` | Raggio di sfera o cilindro, in metri |
| `height` | Altezza del cilindro, in metri |
| `position` | Spostamento rispetto al **centro della superficie del tavolo** |
| `rotation_xyzw` | Orientamento iniziale (quaternione) |
| `linear_velocity` | Velocità iniziale lungo gli assi globali, in m/s |
| `mass` | Massa totale, in kg |
| `color` | Componenti RGB fra 0 e 1, **solo visuali** |
| `youngs_modulus` | Rigidità elastica, in Pa; per deformabili |
| `poissons_ratio` | Espansione laterale durante la compressione; per deformabili |
| `elasticity_damping` | Smorzamento elastico; per deformabili |
| `edge_refinement` | Parametro di risoluzione della mesh procedurale; per deformabili |
| `usd_path`, `scale` | Risorsa USD esterna e relativa scala, per `shape: "usd"` |

**Le unità sono importanti:** il progetto usa il sistema SI (metri, chilogrammi, secondi, Pascal). `0.16` metri significa **16 centimetri**, non 0,16 centimetri.

### 5.3 `true`, `false` e numeri in JSON

In JSON i valori booleani si scrivono `true` e `false`, con l'iniziale minuscola. In Python corrispondono a `True` e `False`.

Le liste JSON, per esempio `[0.0, 0.0, 0.30]`, diventano liste Python e sono accettate dalle strutture di configurazione.

**Non inserire commenti** (`// ...` o `# ...`) in un normale file JSON: il parser usato dal progetto li rifiuta.

---

## 6. `scene_config.py`: leggere e controllare il JSON

Questo modulo usa soltanto la libreria standard di Python. Per questo può controllare le impostazioni senza importare il simulatore.

### 6.1 Le classi `TableSpec` e `ObjectSpec`

```python
@dataclass
class TableSpec:
    size: tuple[float, float, float] = (1.4, 0.9, 0.10)
    center: tuple[float, float, float] = (0.0, 0.0, 0.75)
```

`@dataclass` è una funzionalità Python per creare classi che servono principalmente a conservare dati. `TableSpec` raccoglie dimensioni e caratteristiche del tavolo; `ObjectSpec` fa lo stesso per un singolo oggetto.

Per esempio, concettualmente:

```python
spec = ObjectSpec(name="SoftBox", mass=0.5)
print(spec.mass)  # 0.5
```

I campi non forniti usano i valori predefiniti della classe. Il JSON può sovrascriverli.

### 6.2 `load_scene()`: trasformare il file in oggetti Python

La funzione:

```python
def load_scene(path: Path) -> tuple[TableSpec, list[ObjectSpec]]:
```

esegue questa sequenza:

1. Apre il file e lo interpreta con `json.loads(...)`.
2. Controlla che il documento contenga le chiavi previste (`table` e `objects`).
3. Costruisce un `TableSpec` e una lista di `ObjectSpec`.
4. Chiama le funzioni di validazione.
5. Verifica che non esistano due oggetti con lo stesso nome.
6. Risolve i percorsi dei file USD esterni quando presenti.
7. Restituisce **il tavolo e la lista degli oggetti**.

Perché fa questi controlli? Per intercettare errori facili da diagnosticare **prima** di avviare la parte costosa della simulazione sulla GPU.

### 6.3 `validate_table()` e `validate_object()`: regole concrete

Il programma rifiuta, tra gli altri, questi errori:

- massa nulla o negativa;
- dimensioni o raggi non positivi;
- coefficienti di attrito negativi;
- attrito dinamico maggiore di quello statico (regola di validazione scelta da **questo progetto**);
- nomi con spazi o barre, oppure nomi duplicati;
- colori con componenti fuori dall'intervallo `[0, 1]`;
- `poissons_ratio` fuori da `(-1, 0.5)` per un deformabile;
- `youngs_modulus` non positivo per un deformabile;
- `edge_refinement` minore di `1` per un deformabile;
- quaternioni la cui somma dei quadrati non sia circa `1`;
- riferimenti a file USD locali inesistenti.

Se la configurazione è errata, ottieni un messaggio di errore invece di un esperimento ambiguo.

### 6.4 `resolve_asset_path()`: caricare una forma creata altrove

Per `shape: "usd"`, l'oggetto non è costruito da una forma elementare: viene preso da un file USD esterno.

```json
{
  "name": "OggettoImportato",
  "shape": "usd",
  "deformable": false,
  "usd_path": "assets/modello.usd",
  "scale": [1.0, 1.0, 1.0]
}
```

Nel codice, un percorso relativo come `assets/modello.usd` viene interpretato **rispetto alla cartella del JSON**, non necessariamente rispetto alla cartella da cui hai avviato il terminale. Gli URI con `://` vengono invece lasciati invariati.

La validazione controlla l'esistenza dei file locali, ma **non garantisce** che qualunque USD importato sia adatto a essere simulato come deformabile.

### 6.5 `parse_arguments()`: leggere le opzioni da terminale

Questa funzione configura opzioni come `--config`, `--validate-only`, `--steps`, `--dt` e `--log-every` mediante `argparse`, che fa parte di Python.

Quando chiedi solo la validazione, evita di importare Isaac Lab. Durante l'avvio normale, prova ad aggiungere anche le opzioni del launcher di Isaac Lab. Questa scelta rende possibile un controllo iniziale leggero.

---

## 7. `06_deformable_tabletop.py`: dal programma alla simulazione

### 7.1 Da dove comincia l'esecuzione?

In fondo al file c'è:

```python
if __name__ == "__main__":
    main()
```

Questo è il punto di ingresso: quando avvii il file come programma, viene chiamata `main()`.

`main()`:

1. legge le opzioni da terminale tramite `parse_arguments()`;
2. carica e valida il JSON con `load_scene()`;
3. stampa il riepilogo;
4. termina se hai chiesto `--validate-only`;
5. altrimenti chiama `run_simulation(...)`.

### 7.2 `run_simulation()`: preparare il motore fisico

Le istruzioni fondamentali sono:

```python
device = args.device or "cuda:0"
config = SimulationCfg(dt=args.dt, device=device, physics=PhysxCfg())

with launch_simulation(config, args):
    sim = SimulationContext(config)
    objects = create_scene(table, object_specs)
    sim.reset()
    set_initial_velocities(objects, object_specs)
    simulate(sim, objects, steps=args.steps, dt=args.dt, log_every=args.log_every)
```

Significano:

- **`device`**: quale GPU CUDA usare (il codice rifiuta un dispositivo non CUDA).
- **`SimulationCfg`**: descrive il passo temporale, il dispositivo e il motore scelto (`PhysxCfg`).
- **`launch_simulation(...)`**: avvia l'ambiente necessario alla simulazione.
- **`SimulationContext(...)`**: offre le operazioni per gestire il mondo fisico, come `reset()` e `step()`.
- **`create_scene(...)`**: costruisce tavolo, luce, pavimento e oggetti.
- **`sim.reset()`**: inizializza la scena e la fisica dopo aver creato gli asset.
- **`set_initial_velocities(...)`**: applica eventuali velocità iniziali.
- **`simulate(...)`**: fa effettivamente avanzare il tempo virtuale.

La funzione imposta inoltre la vista della telecamera con `sim.set_camera_view(...)`, puntandola verso la zona del tavolo.

**Dettaglio di ordine:** prima si creano le geometrie con `create_scene()`, *poi* viene chiamato `sim.reset()` per inizializzare la fisica.

### 7.3 `create_scene()`: costruire gli elementi

```python
def create_scene(table, object_specs):
    create_ground_and_light()
    create_table(table)

    objects = {}
    for spec in object_specs:
        if spec.deformable:
            body = create_deformable_object(spec, table)
        else:
            body = create_rigid_object(spec, table)
        objects[spec.name] = body

    return objects
```

Questa funzione è il cuore dell'**allestimento della scena**.

La riga `objects = {}` crea un dizionario Python. Alla fine, contiene associazioni del tipo:

```text
"RigidBox"    -> istanza RigidObject
"SoftBox"     -> istanza DeformableObject
"SoftSphere"  -> istanza DeformableObject
```

`if spec.deformable` sceglie quale tipo di fisica creare per ciascun oggetto. Non occorre una funzione separata per ogni nome: lo stesso codice funziona per tutti gli oggetti del JSON.

### 7.4 `create_ground_and_light()`: pavimento e illuminazione

```python
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)

light = sim_utils.DomeLightCfg(intensity=2500.0)
light.func("/World/Light", light)
```

`GroundPlaneCfg()` prepara un piano di terra. `DomeLightCfg(...)` prepara una luce ambientale. Le chiamate `.func(...)` usano gli spawner configurati per inserire questi elementi nella scena.

La luce serve a **vedere** gli oggetti; non è una forza fisica.

### 7.5 `create_table()`: un tavolo fisso che può essere toccato

```python
contact_material = UsdPhysicsRigidBodyMaterialCfg(
    static_friction=table.static_friction,
    dynamic_friction=table.dynamic_friction,
    restitution=0.0,
)

tabletop = sim_utils.CuboidCfg(
    size=tuple(table.size),
    collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
    physics_material=contact_material,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(table.color)),
)

tabletop.func("/World/Table", tabletop, translation=tuple(table.center))
```

Le proprietà sono di tre categorie:

- **Geometria**: `size` descrive le dimensioni del tavolo.
- **Fisica dei contatti**: `collision_enabled=True` fa sì che gli altri oggetti possano entrare in collisione con il tavolo; l'attrito e la restituzione regolano il contatto.
- **Aspetto**: `diffuse_color` regola il colore visibile.

`restitution=0.0` imposta a zero il coefficiente di restituzione per il materiale di contatto definito qui; non significa che un oggetto elastico non possa mai deformarsi e poi recuperare la forma.

Il tavolo **non** riceve `UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)`: il codice lo lascia statico, anziché farlo cadere sotto la gravità.

---

## 8. Oggetti rigidi e deformabili: le differenze nel codice

### 8.1 Creare un corpo rigido

La funzione `create_rigid_object(spec, table)` prepara:

```python
physics = {
    "rigid_props": UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
    "mass_props": MassCfg(mass=spec.mass),
    "collision_props": UsdPhysicsCollisionCfg(collision_enabled=True),
    "physics_material": contact_material,
}
```

- `rigid_props`: abilita il comportamento rigido dinamico.
- `mass_props`: assegna la massa.
- `collision_props`: abilita il contatto con altre superfici.
- `physics_material`: assegna attrito e restituzione.

Poi chiama `create_geometry(...)` e costruisce un `RigidObjectCfg` con percorso, forma, posizione e rotazione iniziale. Infine restituisce `RigidObject(cfg=config)`.

Per una geometria USD importata e rigida, aggiunge `PhysxConvexHullCfg()` per la rappresentazione della collisione.

### 8.2 Creare un corpo deformabile

Questa è la funzione che motivava il progetto:

```python
def create_deformable_object(spec: ObjectSpec, table: TableSpec):
    body_properties = [
        OmniPhysicsDeformableBodyCfg(mass=spec.mass),
        PhysxDeformableBodyCfg(solver_position_iteration_count=32),
    ]

    elastic_material = [
        OmniPhysicsDeformableMaterialCfg(
            youngs_modulus=spec.youngs_modulus,
            poissons_ratio=spec.poissons_ratio,
            static_friction=spec.static_friction,
            dynamic_friction=spec.dynamic_friction,
        ),
        PhysxDeformableMaterialCfg(elasticity_damping=spec.elasticity_damping),
    ]

    geometry = create_geometry(
        spec,
        volume_deformable_props=body_properties,
        physics_material=elastic_material,
    )

    config = DeformableObjectCfg(
        prim_path=f"/World/Objects/{spec.name}",
        spawn=geometry,
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=world_position(spec, table),
            rot=tuple(spec.rotation_xyzw),
        ),
    )

    return DeformableObject(cfg=config)
```

**Prima parte: `body_properties`.** Specifica la massa totale con `OmniPhysicsDeformableBodyCfg(mass=...)` e il numero di iterazioni di posizione del solver PhysX con `solver_position_iteration_count=32`. La massa di un deformabile, in questa API, non viene assegnata tramite il normale `MassCfg` dei corpi rigidi.

**Seconda parte: `elastic_material`.** Specifica la rigidità, il comportamento durante la compressione, l'attrito e lo smorzamento delle deformazioni.

**Terza parte: `create_geometry(...)`.** Prepara una geometria associandole proprietà da **solido deformabile volumetrico**, non da semplice corpo rigido.

**Quarta parte: `DeformableObjectCfg(...)`.** Assegna all'oggetto un indirizzo nella scena (`prim_path`), la geometria (`spawn`) e lo stato iniziale (`init_state`).

**Quinta parte: `DeformableObject(...)`.** Restituisce l'istanza Python con cui Isaac Lab gestirà l'oggetto durante la simulazione.

#### Perché sono presenti sia `OmniPhysics...` sia `Physx...`?

Non significa che stai avviando due simulazioni separate. Le due famiglie di configurazione rappresentano gruppi di proprietà differenti:

- `OmniPhysics...`: proprietà fisiche comuni, ad esempio la massa e il modulo elastico.
- `Physx...`: parametri specifici del motore PhysX scelto per eseguire questa simulazione.

Le liste raccolgono questi **frammenti di configurazione** e li passano insieme allo spawner, come previsto dalle API del progetto.

### 8.3 `create_geometry()`: scegliere cubo, sfera, cilindro o USD

```python
if spec.shape == "usd":
    return sim_utils.UsdFileCfg(...)

if spec.shape == "box":
    spawner = sim_utils.MeshCuboidCfg if spec.deformable else sim_utils.CuboidCfg

if spec.shape == "sphere":
    spawner = sim_utils.MeshSphereCfg if spec.deformable else sim_utils.SphereCfg

if spec.shape == "cylinder":
    spawner = sim_utils.MeshCylinderCfg if spec.deformable else sim_utils.CylinderCfg
```

Il codice completo passa anche le dimensioni, la fisica e i colori.

| Forma | Se `deformable = false` | Se `deformable = true` |
|---|---|---|
| `box` | `CuboidCfg` | `MeshCuboidCfg` |
| `sphere` | `SphereCfg` | `MeshSphereCfg` |
| `cylinder` | `CylinderCfg` | `MeshCylinderCfg` |
| `usd` | `UsdFileCfg` | `UsdFileCfg` con proprietà deformabili |

Per gli oggetti deformabili **procedurali** (box/sphere/cylinder), la funzione passa anche `edge_refinement`, utilizzato per preparare la risoluzione della mesh. Per i file USD importati questo parametro non viene aggiunto dalla funzione.

**Concetto chiave:** `shape` sceglie la **forma**; `deformable` sceglie il **comportamento fisico**. Un cubo non è rigido soltanto perché è un cubo.

---

## 9. Che cosa significano i parametri fisici?

La simulazione fisica è approssimata numericamente. Questi parametri influenzano il comportamento previsto, ma il risultato dipende anche dalla geometria, dalla mesh, dai contatti e dal passo temporale.

| Parametro | Significato intuitivo | Se aumenti il valore... |
|---|---|---|
| `mass` | Quanta massa ha l'oggetto | In genere cambia la risposta a forze e urti |
| `youngs_modulus` | Quanto è resistente alla deformazione elastica | Il materiale tende a deformarsi meno a parità di sollecitazione |
| `poissons_ratio` | Quanto si allarga lateralmente mentre viene compresso | Vicino a `0.5` tende maggiormente a conservare il volume |
| `static_friction` | Resistenza all'inizio dello scorrimento | Tende a diventare più difficile iniziare a scivolare |
| `dynamic_friction` | Resistenza durante lo scorrimento | Tende ad aumentare la resistenza allo scivolamento in corso |
| `elasticity_damping` | Smorzamento del movimento deformativo | Le oscillazioni elastiche tendono a spegnersi più velocemente |
| `edge_refinement` | Risoluzione di costruzione della mesh deformabile | Cambiano dimensione/numero degli elementi e costo dei calcoli; non è una semplice scala lineare di qualità |
| `solver_position_iteration_count` | Iterazioni del solver per vincoli di posizione | Può migliorare la convergenza, con maggior lavoro computazionale |

### 9.1 Modulo di Young: cubetto morbido e cubetto duro

Pensa a due cubetti con geometria identica. Se eserciti la stessa forza:

- con `youngs_modulus = 10000` Pa il materiale è, in generale, più cedevole;
- con `youngs_modulus = 100000` Pa tende a essere più rigido.

Nel tuo JSON `SoftBox` ha `10000.0` Pa, mentre `SoftSphere` ha `100000.0` Pa. **Non puoi dedurre soltanto da questi numeri quale dei due si schiaccerà di più all'impatto**, perché hanno forme diverse. Per un confronto pulito, cambia un solo parametro mantenendo tutti gli altri identici.

### 9.2 Coefficiente di Poisson

Se schiacci un blocco dall'alto, spesso si allarga sui lati. Il coefficiente di Poisson descrive questo accoppiamento tra deformazione longitudinale e laterale.

Nel JSON il valore è `0.4`: è un parametro di elasticità, non una percentuale di compressione fissa del 40%.

### 9.3 Attrito e smorzamento sono cose diverse

L'**attrito** agisce sui contatti, per esempio fra cubo e tavolo. Lo **smorzamento elastico** riduce nel tempo il movimento di oscillazione dovuto alle deformazioni interne.

Un oggetto può quindi avere molto attrito con il tavolo ma continuare a oscillare nella propria forma, oppure scivolare facilmente pur avendo oscillazioni molto smorzate.

### 9.4 La mesh deformabile e `edge_refinement`

Un simulatore non calcola la deformazione di ogni punto matematico di un solido continuo. Usa una discretizzazione in elementi, per i solidi volumetrici tipicamente **tetraedri**.

`edge_refinement` è uno dei parametri con cui lo spawner decide la risoluzione per generare questa mesh. Non corrisponde semplicemente a "numero di tetraedri". Cambiarlo può influire su tempi di preparazione, costo computazionale e comportamento numerico.

Per iniziare, mantieni il valore `4.0` del file; cambialo solo dopo avere ottenuto una scena che funziona.

---

## 10. Posizioni, rotazioni e velocità

### 10.1 I tre assi dello spazio

Un punto nello spazio viene rappresentato con tre coordinate:

```text
x = posizione verso destra/sinistra lungo un asse del mondo
y = posizione lungo il secondo asse orizzontale
z = altezza nel mondo
```

Le coordinate sono misurate in metri in questo progetto.

### 10.2 `position` nel JSON NON è una coordinata globale

Il codice dice esplicitamente che `ObjectSpec.position` è riferito al **centro della superficie superiore del tavolo**.

La conversione è eseguita qui:

```python
def world_position(spec, table):
    table_x, table_y, table_z = table.center
    offset_x, offset_y, offset_z = spec.position
    surface_height = table_z + table.size[2] / 2
    return table_x + offset_x, table_y + offset_y, surface_height + offset_z
```

Con il JSON originale:

```text
Centro del tavolo:       (0.00, 0.00, 0.75)
Superficie del tavolo:   z = 0.80

RigidBox.position:       (-0.35, 0.00, 0.30)
RigidBox nel mondo:      (-0.35, 0.00, 1.10)

SoftBox.position:        (0.00, 0.00, 0.30)
SoftBox nel mondo:       (0.00, 0.00, 1.10)

SoftSphere.position:     (0.35, 0.00, 0.30)
SoftSphere nel mondo:    (0.35, 0.00, 1.10)
```

**Attenzione alla differenza tra centro e fondo dell'oggetto.** Per i cubi di lato `0.16` m e la sfera di raggio `0.08` m, la parte più bassa si trova inizialmente circa a `1.10 - 0.08 = 1.02` m: quindi circa **22 cm sopra** la superficie del tavolo (`0.80` m), assumendo geometrie iniziali non deformate e senza rotazioni particolari. Non sono appoggiati al tavolo all'inizio: partono sospesi.

### 10.3 Le rotazioni `rotation_xyzw`

La rotazione è rappresentata come un **quaternione** `(x, y, z, w)`. A differenza dei comuni angoli in gradi, è una rappresentazione adatta ai calcoli di orientamento 3D.

```json
"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
```

È la rotazione identità: nessuna rotazione rispetto agli assi iniziali. La validazione controlla che il quaternione abbia norma circa `1`.

**Non scrivere direttamente** `[0, 0, 0, 90]` per ottenere 90 gradi: quel formato non rappresenta gli angoli di Eulero e non supererebbe la validazione.

### 10.4 Velocità iniziale

Ogni oggetto possiede anche:

```json
"linear_velocity": [0.0, 0.0, 0.0]
```

Sono le componenti di velocità lungo X, Y e Z, in **metri al secondo**, in coordinate globali.

Se imposti `[0.5, 0.0, 0.0]`, stai chiedendo una velocità iniziale lungo X di mezzo metro al secondo, senza una velocità iniziale sugli altri assi. L'oggetto potrà comunque accelerare o cambiare direzione quando agiranno gravità e contatti.

---

## 11. Come funziona il ciclo di simulazione

La parte decisiva del programma è `simulate()`:

```python
with torch.inference_mode():
    for step in range(steps):
        if not sim.is_running():
            break

        for body in objects.values():
            body.write_data_to_sim()

        sim.step()

        for body in objects.values():
            body.update(dt)

        if step % log_every == 0:
            heights = {
                name: round(body.data.root_pos_w.torch[0, 2].item(), 3)
                for name, body in objects.items()
            }
            print(f"step={step} simulated_time={(step + 1) * dt:.3f}s z_world={heights}")
```

Leggiamolo come una sequenza di azioni:

1. **`torch.inference_mode()`**: esegue le operazioni PyTorch senza costruire i dati per il calcolo automatico dei gradienti, che qui non servono.
2. **`for step in range(steps)`**: ripete la procedura per un numero massimo di passi.
3. **`sim.is_running()`**: se il contesto non è più in esecuzione, interrompe il ciclo.
4. **`body.write_data_to_sim()`**: invia al simulatore eventuali comandi o dati in attesa sugli asset.
5. **`sim.step()`**: chiede al motore fisico di avanzare di un passo; qui avviene il calcolo fisico.
6. **`body.update(dt)`**: aggiorna i dati gestiti da Isaac Lab dopo il passo.
7. **`if step % log_every == 0`**: ogni N iterazioni stampa un riepilogo.

### 11.1 Che cosa sono `dt` e `steps`?

`dt` significa *delta time*: quanto tempo **simulato** passa a ogni iterazione.

Il codice prevede, per impostazione predefinita:

```text
dt    = 1 / 120 secondi = circa 0.00833 s
steps = 1200

Tempo virtuale massimo = dt * steps = 10 secondi
```

Questi **10 secondi virtuali** non garantiscono che il programma impieghi 10 secondi reali: l'elaborazione può richiedere più o meno tempo, in base all'hardware e alla complessità della scena.

`log_every=120` significa una stampa ogni 120 passi, cioè circa ogni secondo virtuale con `dt = 1/120`. Il primo log viene stampato al passo `0`, **dopo il primo** aggiornamento fisico.

### 11.2 Che cosa significa il log delle altezze?

Il codice legge:

```python
body.data.root_pos_w.torch[0, 2].item()
```

Puoi scomporlo così:

- `body.data`: stato dell'oggetto mantenuto da Isaac Lab;
- `root_pos_w`: posizione di riferimento nel mondo (`w` significa *world*);
- `.torch`: dati accessibili come tensore PyTorch;
- `[0, 2]`: primo oggetto/istanza (`0`), componente Z (`2`);
- `.item()`: converte quel numero in un normale valore Python.

La posizione di riferimento non è identica per rigid e deformable: per i corpi rigidi è la posizione del corpo; per i deformabili Isaac Lab calcola la posizione di riferimento a partire dai nodi della mesh. Il valore stampato **non** è l'altezza del punto più basso, né una misura diretta della deformazione.

### 11.3 `set_initial_velocities()`: il dettaglio dei nodi

Qui c'è una differenza fondamentale fra i due tipi di corpi.

Per un rigido:

```python
velocity = body.data.default_root_vel.torch.clone()
velocity[:, :3] = velocity.new_tensor(spec.linear_velocity)
body.write_root_velocity_to_sim_index(root_velocity=velocity)
```

Il programma imposta la **velocità lineare del corpo** (le prime tre componenti dello stato di velocità).

Per un deformabile:

```python
state = body.data.default_nodal_state_w.torch.clone()
state[..., 3:6] = state.new_tensor(spec.linear_velocity)
body.write_nodal_state_to_sim_index(nodal_state=state)
```

Il deformabile è descritto dai **nodi della mesh**. In ogni stato nodale le prime tre componenti rappresentano la posizione e le successive tre la velocità. `[..., 3:6]` seleziona le velocità di tutti i nodi, assegnando loro la stessa velocità iniziale.

In entrambi i casi `.clone()` prepara una copia dei dati di partenza; alla fine la funzione chiama `body.reset()`.

Questo spiega *perché il codice non può impostare lo stato di un cubo rigido e quello di una gelatina nello stesso identico modo*.

---

## 12. Esperimenti guidati

Esegui **un cambiamento alla volta**, usa `--validate-only` e poi osserva l'effetto nella simulazione. Prima di modificare, conserva una copia del JSON originale.

### Esperimento A: rendere più morbido `SoftBox`

Nel blocco `SoftBox` di `tabletop_scene.json` cambia:

```json
"youngs_modulus": 10000.0
```

in:

```json
"youngs_modulus": 1000.0
```

**Ipotesi:** a parità delle altre condizioni, il cubo dovrebbe essere più cedevole. Controlla qualitativamente la forma durante l'impatto e verifica che la simulazione resti numericamente stabile.

### Esperimento B: cambiare la massa senza cambiare la forma

Nel blocco `SoftBox` cambia:

```json
"mass": 0.5
```

in:

```json
"mass": 1.0
```

**Ipotesi:** l'oggetto ha ora il doppio della massa totale. Non significa automaticamente che cadrà più velocemente in caduta libera ideale: significa che cambiano inerzia e interazioni meccaniche, soprattutto durante i contatti e le deformazioni.

### Esperimento C: partire più in alto

Modifica:

```json
"position": [0.0, 0.0, 0.60]
```

al posto della precedente `[0.0, 0.0, 0.30]` del `SoftBox`.

**Ipotesi:** il centro dell'oggetto parte 30 cm più in alto rispetto alla posizione precedente, quindi avrà più spazio per accelerare verso il tavolo.

### Esperimento D: dare una velocità orizzontale

Aggiungi nel blocco `RigidBox`:

```json
"linear_velocity": [0.5, 0.0, 0.0]
```

**Ipotesi:** il cubo inizierà con una velocità verso X positivo. Dopo il contatto con il tavolo, lo scivolamento sarà influenzato dall'attrito.

### Esperimento E: aggiungere un cilindro deformabile

All'interno dell'array `objects` aggiungi un oggetto come questo, separandolo dagli altri con una virgola valida per JSON:

```json
{
  "name": "SoftCylinder",
  "shape": "cylinder",
  "deformable": true,
  "radius": 0.05,
  "height": 0.12,
  "position": [0.0, 0.25, 0.30],
  "mass": 0.3,
  "youngs_modulus": 20000.0,
  "poissons_ratio": 0.4,
  "color": [0.8, 0.5, 0.2]
}
```

I parametri omessi, come l'attrito e lo smorzamento, useranno i default di `ObjectSpec`. Non devi modificare `create_scene()` perché la funzione itera automaticamente su tutti gli oggetti.

### Un piccolo metodo sperimentale

Per ogni modifica, annota:

| Domanda | Esempio |
|---|---|
| Che cosa cambio? | `SoftBox.youngs_modulus` |
| Da quale valore a quale? | `10000` -> `1000` Pa |
| Che cosa mi aspetto? | Maggiore deformazione |
| Che cosa osservo? | Forma, stabilità, contatto con il tavolo |
| Che cosa tengo costante? | Geometria, massa, altezza, attrito, `dt` |

In questo modo distingui una **ipotesi fisica** da un effetto che hai effettivamente verificato.

---

## 13. Errori comuni e soluzioni

| Problema | Possibile causa | Primo controllo |
|---|---|---|
| `ModuleNotFoundError` per `isaaclab` | Stai provando ad avviare la simulazione con Python ordinario | Usa `--validate-only`, oppure l'ambiente Isaac Lab compatibile |
| Il programma rifiuta una CPU | `run_simulation()` richiede esplicitamente `cuda...` | Usa una GPU CUDA supportata; la validazione resta possibile senza GPU |
| `Invalid scene configuration` | JSON non valido o campi errati | Esegui `--validate-only` e leggi il messaggio |
| `Duplicate object name` | Due oggetti hanno lo stesso `name` | Assegna nomi univoci |
| `rotation_xyzw must be a unit quaternion` | Hai inserito un vettore che non rappresenta un quaternione unitario | Ripristina `[0, 0, 0, 1]` per nessuna rotazione |
| `Asset not found` | Percorso USD locale inesistente | Controlla il percorso relativo alla cartella del JSON |
| Un oggetto non compare come previsto | Posizione, visualizzatore, geometria o ambiente non corretti | Ricontrolla `position`, `shape`, telecamera e messaggi del simulatore |
| Un deformabile si comporta male | Materiale, mesh, passo temporale o contatti non adatti | Prova una scena semplice, mantieni `edge_refinement=4`, controlla `dt` e i valori fisici |
| La simulazione è lenta | Mesh costose, molti corpi, impostazioni solver, hardware | Riduci il numero di oggetti o la complessità e misura di nuovo |

La validazione del JSON **non garantisce** che un asset USD sia fisicamente simulabile, che la mesh deformabile venga generata correttamente o che il comportamento sia realistico: questi aspetti vanno verificati nell'esecuzione vera.

---

## 14. Glossario e risorse

### Glossario minimo

| Termine | Definizione rapida |
|---|---|
| **Asset** | Oggetto gestito dalla libreria di simulazione |
| **Backend fisico** | Motore che esegue i calcoli (qui PhysX) |
| **Collider** | Geometria impiegata per rilevare e risolvere il contatto |
| **Cfg** | Oggetto Python contenente parametri di configurazione |
| **Deformable body** | Corpo in grado di cambiare forma |
| **Delta time (`dt`)** | Durata virtuale di ogni passo di simulazione |
| **Mesh** | Suddivisione geometrica in elementi e nodi |
| **Prim** | Elemento identificabile della scena USD |
| **Rigid body** | Corpo che mantiene la propria forma |
| **Solver** | Algoritmo numerico per calcolare l'evoluzione fisica |
| **Spawner** | Configurazione/funzione che inserisce un elemento in scena |
| **Tensor** | Struttura numerica multidimensionale, qui gestita da PyTorch |
| **USD** | Sistema per descrivere scene e asset 3D |
| **World frame** | Sistema di coordinate globale della scena |

### Documentazione ufficiale di riferimento

- [Installazione di Isaac Lab 3.0](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation)
- [Concetti di corpi deformabili in Isaac Lab 3.0](https://docs.nvidia.com/isaaclab/3.0/concepts/deformables)
- [Schema delle proprietà fisiche in Isaac Lab 3.0](https://docs.nvidia.com/isaaclab/3.0/concepts/physics-schemas)

Queste risorse spiegano l'API di riferimento; i dettagli operativi di questa guida descrivono **i tre file presenti nel progetto**, non un generico esempio NVIDIA.

### Che cosa leggere dopo

Quando avrai compreso questo esperimento, il percorso naturale è: verificare il funzionamento delle collisioni, osservare i nodi di un deformabile, misurare la deformazione nel tempo, confrontare diversi materiali **a parità di geometria** e infine inserire un robot che applichi forze o afferri un oggetto.

**Riassunto finale:** `tabletop_scene.json` dice **che cosa esiste**; `scene_config.py` verifica **che i dati abbiano senso**; `06_deformable_tabletop.py` decide **come creare la scena e far avanzare PhysX nel tempo**.
