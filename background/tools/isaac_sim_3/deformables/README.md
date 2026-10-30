# Simulazione di oggetti deformabili

Un corpo rigido conserva la propria forma: posizione e orientazione permettono di collocarne la geometria nello spazio. Per un **corpo deformabile**, invece, parti diverse possono muoversi l'una rispetto all'altra. La descrizione dello stato deve quindi rappresentare anche la variazione di forma.

Questa differenza è centrale nei task con gomma, tessuti e cavi. Una presa può dipendere dalla compressione del materiale, una piega dalla distribuzione delle tensioni e l'inserimento di un filo dalla sua flessione. Approssimare tutto con un corpo rigido elimina proprio il fenomeno che il robot dovrebbe controllare.

## Rappresentazione del materiale

Un **solido volumetrico** viene discretizzato in elementi che ne rappresentano l'interno, per esempio tetraedri. Una **superficie deformabile** descrive invece un oggetto sottile, come un tessuto, mediante una mesh superficiale e proprietà che ne modellano la risposta.

Un **cavo** può essere rappresentato da segmenti con vincoli e proprietà di estensione, flessione e torsione. Non è quindi sufficiente attribuire un modulo elastico a un oggetto qualsiasi: la rappresentazione geometrica deve corrispondere al fenomeno.

Fluidi e materiali granulari richiedono ulteriori modelli e solver. Il **Material Point Method (MPM)** usa punti materiali e una griglia di calcolo per aggiornare il continuo; il **Finite Element Method (FEM)** approssima il problema mediante elementi e funzioni locali. Questi termini indicano metodi numerici, non tipi di materiali intercambiabili.

## Proprietà elastiche e discretizzazione

Il **modulo di Young** $E$, misurato in pascal, caratterizza la rigidità elastica. Nel caso ideale di una piccola deformazione uniassiale, tensione meccanica $\sigma$ e deformazione relativa $\varepsilon$ sono legate da:

$$
\sigma = E\varepsilon
$$

La tensione rappresenta la forza per unità di area, mentre la deformazione relativa è adimensionale. Questa relazione illustra il ruolo di $E$, ma non descrive da sola tutti i comportamenti di un materiale morbido sottoposto a grandi deformazioni.

Il **rapporto di Poisson** $\nu$ descrive la contrazione trasversale associata a un allungamento; la **densità** $\rho$ specifica la massa per unità di volume. Smorzamento, attrito e parametri specifici della rappresentazione completano il modello.

Anche la mesh influisce sul risultato. Una discretizzazione più fine può rappresentare dettagli aggiuntivi, ma aumenta memoria e costo e può richiedere un passo temporale o un solver differenti. La stabilità deve essere valutata insieme alla fedeltà: un oggetto che rimane numericamente stabile può essere troppo rigido rispetto al materiale desiderato.

## Supporto dei backend

La [documentazione Isaac Lab 3.0 sui deformabili](https://docs.nvidia.com/isaaclab/3.0/concepts/deformables) distingue solidi volumetrici, superfici e cavi. Nel percorso Newton richiede il solver **VBD**, Vertex Block Descent, per queste funzionalità; il preset MJWarp usato negli esempi Cartpole non è una sostituzione equivalente.

Il supporto differisce anche tra famiglie di asset: nella documentazione consultata i cavi sono associati al percorso Newton VBD, mentre solidi e superfici dispongono anche di percorsi PhysX. Una scena che combina robot rigidi e materiali morbidi richiede inoltre una gestione coerente del loro accoppiamento.

La scelta del backend precede quindi l'integrazione nel task. Non è sufficiente cambiare il nome di una classe mantenendo tutte le altre ipotesi degli esempi rigidi.

## Dalla demo a un esperimento controllato

Il primo passo operativo consiste nel verificare gli esempi ufficiali nella checkout utilizzata:

```bash
uv run --extra tetrahedralization isaaclab example deformables
uv run isaaclab example cables
```

L'extra `tetrahedralization` rende disponibili gli strumenti per costruire una mesh volumetrica quando richiesto. Prima di modificare il materiale occorre controllare quale backend venga selezionato dalla demo e che i parametri siano supportati da quel percorso.

Un confronto utile mantiene fissi mesh, densità, condizioni iniziali e impostazioni numeriche, variando soltanto una proprietà, per esempio il modulo elastico. Lasciare cadere il corpo sullo stesso piano permette di confrontare deformazione e recupero della forma; aggiungere subito un robot renderebbe più difficile isolare la causa delle differenze.

## Scena configurabile con tavolo e oggetti

Lo script [`06_deformable_tabletop.py`](lab/06_deformable_tabletop.py) costruisce un piano d'appoggio statico e una collezione di oggetti rigidi o deformabili. La [configurazione JSON di esempio](lab/tabletop_scene.json) contiene un box rigido, un box morbido e una sfera morbida, con parametri indipendenti. Il confronto tra i due box parte dalla stessa geometria e massa, mentre la sfera mostra una seconda forma e una diversa rigidità.

L'esempio utilizza **Isaac Sim con PhysX**, selezionato esplicitamente nello script, e le API a frammenti documentate per Isaac Lab 3.0. Richiede una GPU CUDA supportata e l'extra di tetraedrizzazione per i solidi procedurali. Dalla directory della checkout Isaac Lab si esegue:

```bash
uv run --extra isaacsim --extra tetrahedralization python \
    /percorso/vla_wm/background/tools/isaac_sim_3/deformables/lab/06_deformable_tabletop.py \
    --viz kit --device cuda:0
```

`--steps` controlla il numero di passi fisici, `--dt` la loro durata in secondi e `--log-every` la frequenza delle letture diagnostiche. Lo script stampa la quota globale degli oggetti, che per i deformabili è un valore medio nodale. Con `--viz none` si esegue senza finestra; per osservare forma e contatti è invece utile il visualizzatore Kit.

### Come viene costruita la scena

La funzione **`create_scene`**, all'inizio dello script, espone la sequenza di costruzione: crea ground e illuminazione, chiama `create_table` e poi crea ogni oggetto con `create_rigid_object` oppure `create_deformable_object`. La scelta della forma è isolata in `create_geometry`, mentre il ciclo temporale si trova in `simulate`. La lettura e la validazione del JSON sono raccolte nel modulo separato [`scene_config.py`](lab/scene_config.py).

Le descrizioni `TableSpec` e `ObjectSpec` possono essere costruite anche direttamente in Python. Per esempio, nel contesto di simulazione già avviato da `run_simulation`, prima di `sim.reset()`, la creazione di un tavolo e di un singolo box morbido può essere espressa così:

```python
table = TableSpec(
    size=(1.4, 0.9, 0.10),
    center=(0.0, 0.0, 0.75),
)

soft_box = ObjectSpec(
    name="SoftBox",
    shape="box",
    deformable=True,
    size=(0.16, 0.16, 0.16),
    position=(0.0, 0.0, 0.30),
    mass=0.5,
    youngs_modulus=10000.0,
)

object_specs = [soft_box]
objects = create_scene(table, object_specs)
```

`create_table` costruisce il collider statico, mentre `create_deformable_object` assegna massa e materiale elastico prima di creare l'asset fisico. `create_scene` restituisce un dizionario che associa ogni nome al relativo oggetto simulato. Lo script completo usa lo stesso percorso di costruzione caricando le descrizioni dal JSON, quindi inizializza la fisica e le velocità prima di eseguire `simulate`.

### Configurazione del tavolo e delle forme

Nel JSON, `table.size` contiene lunghezza, larghezza e spessore in metri; `table.center` ne specifica il centro nel riferimento globale. Il tavolo è un collider statico: non viene mosso dalle forze e non richiede una massa dinamica. Attrito statico e dinamico della superficie sono configurati separatamente da quelli degli oggetti.

Ogni elemento di `objects` ha un `name` univoco, usato nel percorso `/World/Objects/<name>`. I parametri geometrici dipendono dalla forma:

| `shape` | Parametri | Rappresentazione |
| --- | --- | --- |
| `box` | `size`: tre dimensioni in metri | Parallelepipedo |
| `sphere` | `radius` in metri | Sfera |
| `cylinder` | `radius` e `height` in metri | Cilindro con asse locale Z |
| `usd` | `usd_path` e `scale` | Asset USD esterno |

`position` indica la posizione dell'origine dell'oggetto **rispetto al centro del piano superiore** del tavolo. Per le primitive questa origine coincide con il centro geometrico; con un asset esterno dipende dal suo pivot. Una quota relativa di `0.30` lascia quindi cadere un box di lato `0.16` da una posizione senza intersezioni iniziali.

`rotation_xyzw` è un quaternion unitario, mentre `linear_velocity` è una velocità iniziale in metri al secondo nel riferimento globale. Una velocità orizzontale, per esempio `[0.4, 0.0, 0.0]`, permette di osservare lo scorrimento dopo il contatto. Per confrontare l'attrito occorre mantenere uguali le altre condizioni e limitare la velocità in modo che gli oggetti non escano subito dal tavolo.

### Massa, frizioni e deformabilità

`deformable: false` crea un corpo rigido; `deformable: true` crea un **solido volumetrico deformabile**. In entrambi i casi `mass` indica la massa totale desiderata in chilogrammi. Lo script usa `MassCfg` per il corpo rigido e `OmniPhysicsDeformableBodyCfg.mass` per il deformabile: la [documentazione degli schemi fisici](https://docs.nvidia.com/isaaclab/3.0/concepts/physics-schemas) distingue esplicitamente questi due percorsi.

`static_friction` e `dynamic_friction` regolano le proprietà di contatto dell'oggetto; la configurazione richiede coefficienti non negativi e attrito dinamico non superiore allo statico. Anche il materiale del tavolo contribuisce al risultato, perciò non basta modificare un solo coefficiente per dedurre direttamente una distanza di arresto.

Per i deformabili, `youngs_modulus` specifica la rigidità in pascal, `poissons_ratio` il rapporto di Poisson ed `elasticity_damping` lo smorzamento elastico. `edge_refinement` controlla la discretizzazione delle primitive deformabili: aumentarlo richiede una mesh più fine e maggiore lavoro numerico. Questi parametri non vengono usati per i corpi rigidi.

La massa è assegnata indipendentemente dalla forma: modificare le dimensioni conservando `mass` cambia quindi la densità media implicita. Le proprietà vengono applicate **alla creazione della scena**, prima dell'inizializzazione fisica. Per confrontare materiali si modifica il JSON e si riavvia lo script; l'esempio non fornisce un editor di materiali durante il rollout.

### Uso di un asset esterno

Un oggetto può essere sostituito con un asset attraverso una voce come questa:

```json
{
  "name": "SoftAsset",
  "shape": "usd",
  "usd_path": "assets/soft_object.usd",
  "scale": [0.1, 0.1, 0.1],
  "deformable": true,
  "position": [0.0, 0.0, 0.35],
  "mass": 0.4,
  "static_friction": 0.7,
  "dynamic_friction": 0.5,
  "youngs_modulus": 50000.0,
  "poissons_ratio": 0.4
}
```

Il percorso locale viene risolto rispetto al JSON; sono accettati anche URI di asset accessibili dal runtime. L'asset deve rappresentare un **singolo corpo compatibile con la modalità scelta**. Per un solido morbido servono una geometria chiusa tetraedrizzabile o una mesh volumetrica già preparata, senza una struttura articolata o corpi rigidi incompatibili. Il caricamento non converte automaticamente qualsiasi modello 3D in un materiale simulabile. Mesh OBJ o STL devono prima essere convertite in un asset USD adeguato.

Per i rigid body importati, lo script richiede un'approssimazione collisionale a inviluppo convesso; le concavità possono quindi non essere rappresentate. Per gli asset deformabili, la discretizzazione dipende dall'asset e dal percorso di importazione: `edge_refinement` viene inoltrato soltanto alle primitive procedurali. I materiali visuali dell'USD sono conservati; `color` controlla le primitive.

### Validazione della configurazione

Si può copiare il JSON, modificarlo e selezionarlo con `--config /percorso/scena.json`. Prima dell'avvio GPU è disponibile un controllo con la sola libreria standard Python:

```bash
python3 /percorso/vla_wm/background/tools/isaac_sim_3/deformables/lab/06_deformable_tabletop.py \
    --config /percorso/scena.json --validate-only
```

Il controllo rileva campi sconosciuti, nomi duplicati, forme non supportate, valori non finiti, masse o dimensioni non positive, quaternioni non unitari e asset locali mancanti. **Non verifica la qualità della mesh, il contenuto fisico dell'USD o la stabilità numerica**: queste proprietà richiedono il runtime e l'ispezione della simulazione. In questa raccolta l'esempio è verificato offline; l'esecuzione PhysX su GPU resta da validare nell'ambiente Isaac previsto.

## Stato nodale e osservazioni della policy

Lo stato di un deformabile comprende posizioni e velocità dei nodi della mesh di simulazione. Per un asset inizializzato che espone `nodal_pos_w`, la lettura con PyTorch segue la forma:

```python
nodes = soft_object.data.nodal_pos_w.torch
print(nodes.shape)
```

La vista tensoriale delle posizioni organizza istanze, nodi e tre coordinate spaziali. Una posizione media può riassumere la traslazione complessiva, ma non distingue un tessuto disteso da uno ripiegato con lo stesso centro. Non va quindi confusa con una posa rigida completa.

Il task deve scegliere quali informazioni osservare: tutti i nodi, alcuni punti caratteristici, misure di contatto o immagini. Questa scelta cambia dimensione e parziale osservabilità del problema, oltre alla trasferibilità sul robot reale.

## Limiti sperimentali

L'accuratezza dipende dalla corrispondenza tra materiale reale e modello numerico, dalla discretizzazione e dai contatti. Una clip plausibile non dimostra che forze, deformazioni e dissipazione siano quantitativamente corrette.

Per un esperimento di manipolazione occorre validare prima il materiale isolato e poi l'interazione con gripper e superfici. Nei risultati vanno riportati anche solver, passo temporale e mesh: conservare soltanto il valore del modulo di Young non basta a ricostruire il comportamento simulato.
