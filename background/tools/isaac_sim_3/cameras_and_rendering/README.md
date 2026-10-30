# Camere, rendering e apprendimento visuale

Una policy può ricevere misure numeriche dello stato oppure immagini che deve interpretare. Nel secondo caso il sistema deve apprendere anche a estrarre informazioni utili da apparenza, prospettiva e occlusioni. La scelta delle osservazioni cambia quindi il problema di apprendimento, oltre al costo computazionale.

Il **fotorealismo** riguarda la somiglianza delle immagini sintetiche a quelle reali. Può contribuire alla qualità dei dati, ma non garantisce che il modello usi gli indizi corretti o che la dinamica simulata corrisponda al mondo fisico.

## Fisica, renderer, camera e visualizzatore

Il backend fisico aggiorna posizioni, velocità e contatti. Il **renderer** trasforma geometria, materiali, luci e punto di vista in immagini. La **camera** definisce quali immagini vengono acquisite: posa, proiezione, risoluzione, frequenza e canali richiesti.

Il **visualizzatore** permette invece all'utente di osservare il mondo. La sua camera può inquadrare l'intera scena, mentre la policy riceve soltanto una vista dal polso del robot. **Una finestra con rendering realistico non dimostra che la policy osservi RGB**: occorre verificare il percorso dai buffer del sensore al tensore fornito al modello.

L'esecuzione headless elimina la finestra interattiva. Se l'agente usa immagini, il rendering continua a essere necessario e conserva i propri costi di memoria e calcolo.

## Informazione geometrica e immagine

Una policy basata sullo stato può ricevere direttamente la posizione relativa di un cubo. Una policy visuale deve inferirne proprietà rilevanti dai pixel, eventualmente combinandoli con lo stato del robot $q_t$. Una singola immagine può non distinguere oggetti occlusi o movimenti differenti: memoria temporale e viste multiple possono diventare necessarie.

Per un VLA, l'osservazione $o_t$ può comprendere immagini e propriocezione, mentre l'istruzione $l$ specifica l'obiettivo. La simulazione deve produrre entrambe le componenti in modo coerente con le azioni e con il task. Il [capitolo introduttivo sui VLA](../../../../vla/README.md) descrive il loro ruolo nella policy.

In Isaac Lab, la [documentazione delle camere](https://docs.nvidia.com/isaaclab/3.0/concepts/sensors/camera) distingue configurazione del sensore e scelta del renderer. La versione 3.0 organizza il rendering delle viste clonate in passaggi aggregati e buffer GPU; esempi precedenti basati su classi differenti devono essere confrontati con l'API effettiva.

## Parametri della camera

La **posa estrinseca** descrive posizione e orientazione della camera rispetto a un riferimento. Gli **intrinseci** descrivono la proiezione, inclusi focale e centro dell'immagine. Cambiare questi parametri modifica la relazione tra una posizione nel mondo e il pixel in cui appare.

La risoluzione stabilisce quanti pixel vengono prodotti; la frequenza determina quanto spesso arriva una nuova osservazione. Una policy con controllo a 60 Hz e immagini a 15 Hz non riceve una misura visuale nuova a ogni decisione. Occorre sapere se il task ripete l'ultimo frame e quali ritardi vengano rappresentati.

Il canale RGB descrive l'apparenza. La depth misura una distanza secondo la convenzione scelta; la segmentazione identifica classi o istanze. Non sono informazioni equivalenti: un esperimento che usa depth esatta risolve un problema diverso da uno limitato a una camera RGB reale.

## Avvio di un task visuale

Prima di selezionare un renderer si devono ispezionare task e preset della checkout:

```bash
uv run isaaclab list_envs --show_presets
uv run isaaclab train --task Isaac-Cartpole-Camera --help
```

Un esempio del percorso Newton, se supportato dalla configurazione disponibile, è:

```bash
uv run isaaclab zero_agent --task Isaac-Cartpole-Camera \
    --num_envs 4 physics=newton_mjwarp renderer=newton_renderer \
    presets=rgb --viz newton_gl
```

Per un percorso Isaac Sim compatibile, la selezione diventa:

```bash
uv run --extra isaacsim isaaclab zero_agent --task Isaac-Cartpole-Camera \
    --num_envs 4 physics=isaacsim_physx renderer=isaacsim_rtx \
    presets=rgb --viz kit
```

Questi comandi verificano l'esecuzione della configurazione, senza addestrare un encoder visuale. L'algoritmo scelto per il training deve supportare la struttura delle osservazioni e l'architettura necessaria a elaborarle.

## Verifica del percorso delle immagini

In un ambiente con un sensore registrato come `camera`, si possono ispezionare i canali prodotti:

```python
camera = env.unwrapped.scene["camera"]
print(camera.data.output.keys())
rgb = camera.data.output["rgb"]
print(type(rgb), rgb.shape)
```

Il nome del sensore dipende dal task. Tipo, layout e rappresentazione dei valori devono essere controllati prima di scegliere conversioni e normalizzazione: dimensioni apparentemente corrette non escludono uno scambio tra canali e assi spaziali.

Salvare alcuni frame prima del training permette di rilevare camere rivolte altrove, immagini nere, clipping o ambienti scambiati. Va poi verificato che siano proprio quei frame a entrare nel gruppo di osservazioni della policy.

## Costi e trasferimento al mondo reale

Per un batch di $N$ immagini con altezza $H$, larghezza $W$ e $C$ canali, la sola memoria dei pixel è proporzionale a $NHWC$. Durante l'apprendimento si aggiungono rollout, attivazioni dell'encoder e buffer del renderer. Ridurre il numero di ambienti può quindi essere necessario quando si passa da stato a immagini.

La **domain randomization visuale** varia luci, materiali, sfondi o parametri della camera per ridurre la dipendenza da dettagli accidentali. Le variazioni devono rimanere compatibili con il dominio d'uso e con il significato del task.

Il trasferimento richiede inoltre calibrazione, rumore, latenza, qualità delle azioni e fisica dei contatti. Un confronto utile separa gli effetti: prima verifica il controllo con osservazioni di stato, poi il sistema visuale e infine la robustezza a scene e condizioni non presenti nel training.
