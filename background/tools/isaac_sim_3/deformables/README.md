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
