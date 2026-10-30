# Valutazione delle policy e registrazione dei rollout

La **valutazione** misura quanto una policy soddisfa un obiettivo in condizioni definite. Un rollout visivamente convincente può mostrare un comportamento possibile, ma non ne quantifica affidabilità, variabilità o capacità di generalizzazione.

Occorre quindi separare tre operazioni: verificare che un checkpoint sia eseguibile, misurare la prestazione su un insieme di episodi e produrre un video illustrativo. Possono usare la stessa policy, ma rispondono a domande differenti.

## Condizioni del confronto

Una valutazione deve identificare checkpoint, configurazione del task, modalità di inferenza e distribuzione delle condizioni iniziali. Preprocessing, normalizzazione e significato delle azioni devono corrispondere al training; eventuali modifiche costituiscono condizioni sperimentali da dichiarare.

Anche il backend fisico fa parte del protocollo. Valutare subito una policy addestrata con Newton passando a PhysX cambia la dinamica, oltre all'eventuale rendering. Per diagnosticare un problema conviene iniziare dalla configurazione originale e introdurre variazioni separatamente.

Gli episodi di test devono essere distinti da quelli usati per scegliere il checkpoint o gli iperparametri. Per studiare generalizzazione si possono riservare pose, oggetti, proprietà fisiche o illuminazioni non viste, specificando quale fattore venga modificato.

## Ritorno, durata e successo

Il **ritorno episodico** somma le reward ricevute. Per un episodio di durata $T$, con reward $r_t$ al passo $t$, la versione non scontata è:

$$
G = \sum_{t=0}^{T-1} r_t
$$

La durata può essere espressa in passi decisionali oppure in secondi simulati, purché la convenzione sia dichiarata. Il ritorno dipende dalla reward progettata e non costituisce da solo una misura universale di qualità.

Il **success rate** richiede un criterio specifico del task. Indicando con $K$ il numero di episodi valutati e con $c_i$ un indicatore pari a 1 se l'episodio $i$ soddisfa il criterio, la stima è:

$$
\widehat p_{\mathrm{success}} = \frac{1}{K}\sum_{i=1}^{K} c_i
$$

I flag `terminated` e `truncated` indicano perché l'esecuzione si interrompe, ma non sostituiscono $c_i$. Per una presa, successo potrebbe significare mantenere l'oggetto sollevato per un intervallo; per Cartpole, evitare il fallimento fino all'orizzonte stabilito può essere il criterio scelto dal valutatore.

Numero di episodi e variabilità vanno riportati insieme alla percentuale. Ripetere il test con più seed di valutazione esplora condizioni iniziali diverse; ripetere il training con più seed misura anche la variabilità dell'apprendimento. Sono fonti di variazione differenti.

## Metriche con ambienti paralleli

Ogni copia possiede il proprio accumulatore di ritorno e durata. Dopo uno `step` si aggiungono reward e un passo; quando `terminated | truncated` è vero per una copia, si salva il risultato e si azzerano soltanto i suoi accumulatori.

Lo script [`05_eval_metrics.py`](../scripts/05_eval_metrics.py) illustra questo meccanismo:

```bash
uv run python /percorso/vla_wm/background/tools/isaac_sim_3/scripts/05_eval_metrics.py \
    --episodes 100 --num_envs 32
```

Il file `outputs/zero_baseline.json` contiene ritorni, durate e flag di fine episodio. Il controller invia **azioni nulle**: è una baseline, non una policy addestrata. `success_rate` rimane `null` perché lo script non definisce un criterio di successo specifico.

La funzione `policy_actions` è il punto in cui integrare l'inferenza del modello, mantenendo wrapper e normalizzazione. Lo script è dimostrativo: la conversione degli indici sulla CPU e l'estrazione di scalari a ogni fine episodio possono limitare il throughput. Inoltre, quando più copie finiscono insieme, conserva soltanto i primi risultati fino al numero richiesto; un protocollo rigoroso deve definire anche questa selezione e l'assegnazione dei seed.

## Checkpoint e playback

Un primo controllo dell'esecuzione può essere effettuato con:

```bash
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct \
    --checkpoint latest --num_envs 8 --viz newton_gl physics=newton_mjwarp
```

Per il report finale va sostituita la selezione implicita `latest` con l'identità del checkpoint effettivo. Osservare otto copie permette di individuare errori evidenti, ma non produce automaticamente statistiche episodiche o intervalli di incertezza.

## Registrazione video

La [guida video di Isaac Lab 3.0](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/recording-video) descrive i recorder configurati nell'ambiente. Una sorgente può essere un visualizzatore oppure un sensore della scena:

```python
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

cfg.video_recorders = [
    VideoRecorderCfg(
        source="visualizer:kit",
        output_dir="videos/tutorial",
        output_filename_prefix="cartpole_demo",
        video_length=100,
        video_interval=0,
        fps=30,
    )
]
```

La configurazione va applicata prima di creare l'ambiente. Il frammento richiede un visualizzatore Kit compatibile; per registrare un sensore si usa una sorgente come `sensor:camera`, con il nome realmente presente nella scena e il canale abilitato.

Lo script [`04_video_gym.py`](../scripts/04_video_gym.py) mostra la registrazione da Kit nel percorso Isaac Sim:

```bash
uv run --extra isaacsim --extra video python \
    /percorso/vla_wm/background/tools/isaac_sim_3/scripts/04_video_gym.py \
    --viz kit --output videos/tutorial
```

Anche questo esempio invia azioni nulle. Per documentare un checkpoint addestrato sul percorso Newton si può usare il playback con una sorgente di cattura supportata:

```bash
uv run --extra video isaaclab play --rl_library rsl_rl \
    --task Isaac-Cartpole-Direct --checkpoint latest \
    --video --video_length 200 --viz newton_gl physics=newton_mjwarp
```

Il frame rate del file determina la velocità di riproduzione: se la cadenza di cattura non corrisponde al tempo simulato, il video può accelerare o rallentare l'azione. Una clip dovrebbe quindi riportare seed, configurazione, checkpoint e l'eventuale variazione di velocità. La documentazione visuale acquista significato quando è associata alle metriche e include anche condizioni di fallimento rappresentative.
