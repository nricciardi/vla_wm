# Installazione locale e verifica dell'ambiente

Installare un simulatore significa rendere compatibili più livelli: driver GPU, librerie di calcolo, ambiente Python e funzionalità richieste dal task. Un errore in uno di questi livelli può impedire l'avvio prima ancora che venga costruita la scena.

Il primo obiettivo è quindi ottenere **un ambiente minimo verificabile**, basato su osservazioni numeriche e un numero ridotto di copie. Camere e rendering avanzato possono essere introdotti dopo avere accertato che la simulazione di base funziona.

## Scelta dello stack

Il percorso Newton di Isaac Lab 3.0 può funzionare senza installare l'applicazione completa Isaac Sim. È un punto di partenza per gli esperimenti di controllo supportati, ma non equivale a una disponibilità universale di robot, sensori e materiali.

Per il percorso con Isaac Sim occorrono anche le dipendenze dell'applicazione e hardware compatibile con le funzionalità RTX utilizzate. La [pagina ufficiale di installazione](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation) distingue questi requisiti: indica Python 3.12 e, per lo stack CUDA 13 predefinito, driver Linux almeno 580.65.06 e GPU compatibile. Specifiche diverse devono essere controllate sulla combinazione effettivamente installata.

RAM e VRAM, rispettivamente memoria di sistema e memoria della GPU, limitano la dimensione degli esperimenti. Il consumo cresce con il numero di ambienti, la complessità delle scene e le immagini conservate per l'apprendimento. Un requisito minimo di avvio non garantisce spazio sufficiente per un training visuale.

## Verifica del sistema

Su Linux i seguenti comandi permettono di identificare GPU, driver e sistema operativo:

```bash
nvidia-smi
uname -a
cat /etc/os-release
```

`nvidia-smi` verifica che il driver riesca a comunicare con la GPU. Non dimostra ancora che tutte le librerie del progetto siano compatibili. Se il comando fallisce, la diagnosi deve partire dal sistema host: l'installazione di un pacchetto Python non sostituisce il driver.

## Ambiente Python e checkout

La documentazione 3.0 propone **uv**, uno strumento che risolve le dipendenze e avvia i comandi nell'ambiente del progetto. Va installato seguendo le [istruzioni ufficiali di uv](https://docs.astral.sh/uv/getting-started/installation/). Una volta disponibile, si può preparare una checkout di Isaac Lab:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
uv run isaaclab list_envs --show_presets
```

Il clone segue inizialmente il branch predefinito, che può evolvere. Per un esperimento riproducibile occorre selezionare una revisione corrispondente alla documentazione utilizzata e conservarne l'identificativo con `git rev-parse HEAD`, insieme al lockfile delle dipendenze. Non basta ripetere il clone a distanza di mesi.

Il comando `list_envs` mostra i task registrati e i preset supportati. Un **preset** raccoglie scelte di configurazione, come un backend o una modalità di osservazione. Leggere questo elenco prima di eseguire gli esempi permette di verificare che i nomi utilizzati siano presenti nella propria revisione.

## Primo avvio controllato

Per una checkout compatibile con gli esempi, il primo test può usare Cartpole con azioni nulle:

```bash
uv run isaaclab zero_agent --task Isaac-Cartpole-Direct \
    --num_envs 16 --viz newton_gl physics=newton_mjwarp
```

Cartpole comprende un carrello che si muove lungo una guida e un pendolo collegato da un giunto. `zero_agent` invia azioni nulle: serve a verificare l'avvio e l'evoluzione del mondo, non a mantenere necessariamente il pendolo in equilibrio.

`--num_envs` stabilisce quante copie simulare; `--viz` seleziona il visualizzatore; `physics=` seleziona il backend esposto dal task. Su una macchina senza display si può usare `--viz none`. Se l'avvio fallisce, conviene risolvere il problema prima di introdurre un ambiente personalizzato.

## Esecuzione degli script della raccolta

Gli script sono conservati in questa raccolta, mentre i comandi vanno lanciati dalla directory della checkout Isaac Lab. Nei capitoli, `/percorso/vla_wm` indica il percorso assoluto locale del presente repository:

```bash
uv run python /percorso/vla_wm/background/tools/isaac_sim_3/scripts/01_gym_rollout.py \
    --num_envs 16 --steps 300 --viz newton_gl
```

Questa convenzione permette a `uv` di usare le dipendenze di Isaac Lab pur eseguendo un file esterno. L'interprete scelto nell'editor deve appartenere allo stesso ambiente; altrimenti un import può funzionare dal terminale e fallire nell'editor, o viceversa.

## Aggiunta di Isaac Sim

Gli **extra** rendono disponibili integrazioni opzionali. Installare un extra e selezionarlo nel task sono due operazioni distinte:

```bash
uv run --extra isaacsim isaaclab zero_agent --task Isaac-Cartpole-Direct \
    --num_envs 4 --viz kit physics=isaacsim_physx
```

Il primo avvio può richiedere download e compilazione delle cache. Per l'esecuzione non interattiva vanno inoltre rispettate le condizioni della licenza prevista dal software, seguendo la procedura ufficiale.

Un avvio riuscito costituisce il punto di partenza per il [ciclo Gymnasium](../gymnasium/README.md). Non valida ancora gli script della raccolta, il training o la generazione di immagini: ciascuna funzionalità richiede una verifica nel proprio contesto.
