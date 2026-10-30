# Esecuzione su cluster HPC

Un cluster **HPC**, High Performance Computing, mette a disposizione risorse condivise attraverso un sistema di assegnazione dei job. Rispetto a una workstation, cambiano l'accesso alla GPU, la gestione dello storage e la disponibilità di rete e interfaccia grafica.

La separazione principale è tra **login node**, usato per preparazione e sottomissione, e **compute node**, sul quale viene eseguito il lavoro assegnato. Una simulazione che richiede GPU deve essere avviata all'interno dell'allocazione concessa dallo scheduler, secondo le regole del cluster.

## Scheduler e risorse

Uno **scheduler**, come SLURM, riceve richieste di risorse e avvia il job quando queste diventano disponibili. Lo script batch descrive numero di nodi, GPU, CPU, memoria e tempo massimo. Queste richieste non modificano il task: determinano dove e per quanto tempo potrà essere eseguito.

Il template [`train_cartpole.sbatch`](train_cartpole.sbatch) contiene un esempio con una GPU. Partizione, sintassi di richiesta GPU e limiti devono essere adattati al cluster: `--gres=gpu:1` e `--gpus=1` non sono opzioni da scegliere indipendentemente dalla configurazione del sito.

Dalla directory della checkout Isaac Lab, dopo avere impostato il suo percorso:

```bash
export ISAACLAB_ROOT="$HOME/projects/IsaacLab"
cd "$ISAACLAB_ROOT"
mkdir -p logs
sbatch /percorso/vla_wm/background/tools/isaac_sim_3/hpc/train_cartpole.sbatch
squeue -u "$USER"
```

La directory `logs` deve esistere prima della sottomissione, perché SLURM apre il file di output prima di eseguire il corpo dello script. Il template è un punto di partenza da adattare e non una configurazione universale.

## Ambiente software e container

Quando il sito lo permette, il compute node può eseguire direttamente l'ambiente gestito da `uv`. In alternativa si può usare un **container**, che raccoglie librerie e applicazione in un'immagine. Il driver NVIDIA rimane una dipendenza del sistema host: non viene sostituito dalla semplice presenza di CUDA nel container.

Molti cluster usano Apptainer o Singularity per eseguire immagini senza un servizio Docker privilegiato. Il percorso generale comprende costruzione dell'immagine, trasferimento e avvio nell'allocazione. La [guida NVIDIA per HPC](https://docs.nvidia.com/isaaclab/3.0/features/running-on-hpc-clusters) descrive la conversione delle immagini e gli adattamenti per scheduler differenti.

Un controllo minimo, su un'immagine già preparata, può avere questa forma:

```bash
apptainer exec --nv --bind "$SCRATCH:/scratch" image.sif \
    python -c 'import torch; print(torch.cuda.is_available())'
```

`--nv` espone le risorse NVIDIA secondo le funzionalità di Apptainer. Il bind rende accessibile nel container una directory host. Nome dell'immagine, interprete e variabile `SCRATCH` dipendono dall'installazione locale.

## Storage, cache e rete

Lo storage persistente conserva codice, configurazioni e checkpoint da mantenere. Lo **scratch** offre spazio di lavoro, talvolta più veloce ma soggetto a cancellazione. Una directory temporanea locale al job può essere utile per cache e file intermedi, purché i risultati vengano copiati prima della sua rimozione.

Isaac Lab e Isaac Sim possono richiedere dipendenze, asset e cache grafiche al primo avvio. Se i compute node non hanno rete, questi dati devono essere predisposti secondo le procedure del sito. Aumentare il numero di job senza preparare le cache può trasformare il download o il filesystem condiviso nel collo di bottiglia.

Il template fornito colloca la cache uv in uno spazio temporaneo. Questa scelta evita alcuni conflitti, ma può ripetere la preparazione a ogni job. Una cache persistente o un'immagine preparata possono essere preferibili per campagne estese, in accordo con le regole dello storage.

## Diagnostica nell'allocazione

Le verifiche devono essere effettuate sul compute node assegnato, perché l'ambiente del login node può essere differente:

```bash
nvidia-smi
which uv
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
uv run isaaclab zero_agent --task Isaac-Cartpole-Direct \
    --num_envs 8 --viz none physics=newton_mjwarp
```

Questi passaggi distinguono visibilità della GPU, disponibilità dell'ambiente Python e avvio del task. Se il job viene terminato dallo scheduler, occorre leggere anche lo stato finale e i limiti di memoria o walltime: un arresto esterno può non generare un'eccezione Python utile.

**Headless non significa senza rendering**. Un training visuale continua a richiedere i buffer delle camere e uno stack grafico compatibile; un training basato sullo stato può invece evitare questo costo. I video possono essere prodotti in job di valutazione separati per rendere più interpretabili le misure di prestazione.

## Più GPU e più nodi

Un job multi-GPU avvia processi che raccolgono esperienza separatamente e sincronizzano gli aggiornamenti. Su più nodi servono anche comunicazione di rete e parametri di coordinamento coerenti. Lo scheduler deve assegnare le risorse effettivamente usate dal launcher: prenotare due GPU non obbliga il programma a utilizzarle entrambe.

La [guida multi-node NVIDIA](https://docs.nvidia.com/isaaclab/3.0/features/multi-gpu-and-multi-node-training) descrive numero di processi, rank dei nodi e indirizzo di coordinamento. Il [capitolo sul training](../training/README.md) spiega il rapporto tra ambienti per GPU e quantità totale di esperienza.

Per ricostruire un esperimento è utile conservare script batch effettivo, identificativo del job, immagine o ambiente, revisione del codice, seed e configurazioni risolte. Queste informazioni collegano il checkpoint alle risorse e alle condizioni che lo hanno prodotto.
