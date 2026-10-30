# Progettazione di un ambiente di apprendimento

Una scena descrive ciò che esiste e può muoversi. Un **task** specifica invece il problema decisionale: quali osservazioni riceve l'agente, quali azioni può applicare, come viene misurata la prestazione e da quali condizioni parte ogni episodio.

La stessa scena può sostenere problemi differenti. Un manipolatore e un cubo possono essere usati per avvicinamento, pushing o presa; cambiano osservazioni, comandi, reward e criterio di successo. Progettare un ambiente significa rendere queste scelte coerenti tra loro.

## Dalla specifica al codice

Prima dell'implementazione occorre chiarire il significato fisico delle azioni. Un target articolare richiede un controller che lo insegua; un comando cartesiano richiede una trasformazione verso i giunti; una coppia agisce più direttamente sulla dinamica. Cambiare action space modifica il problema anche quando la scena rimane identica.

Le osservazioni devono contenere informazioni sufficienti a decidere, evitando accessi impliciti a variabili che si vogliono escludere. La reward deve incentivare il comportamento desiderato, mentre le terminazioni devono distinguere fallimento, successo e limite temporale. Infine, il reset deve ripristinare stato e memoria dell'episodio senza alterare le copie ancora attive.

Queste scelte costituiscono un **contratto dell'ambiente**. Un reward tensor della forma corretta può essere comunque semanticamente sbagliato, per esempio se premia il contatto con il cubo anche quando questo cade dal tavolo.

## Workflow Direct e manager-based

Nel workflow **Direct**, una classe derivata da `DirectRLEnv` implementa direttamente i passaggi del task. Questo rende visibile il collegamento tra azione, avanzamento fisico e calcolo dei segnali.

Il workflow **manager-based** distribuisce invece le funzioni tra componenti configurabili per azioni, osservazioni, reward, eventi e terminazioni. È utile quando molte varianti condividono gli stessi termini. La scelta riguarda soprattutto organizzazione e riuso del codice; non determina da sola la difficoltà del task.

Nel percorso Direct i metodi principali separano responsabilità precise:

| Metodo | Responsabilità |
| --- | --- |
| `_pre_physics_step(actions)` | Prepara e scala il nuovo batch di azioni |
| `_apply_action()` | Invia i comandi agli attuatori nei passi fisici |
| `_get_observations()` | Costruisce i gruppi di osservazioni |
| `_get_rewards()` | Calcola una reward per ambiente |
| `_get_dones()` | Calcola terminazioni e troncamenti |
| `_reset_idx(env_ids)` | Reinizializza le copie selezionate |

La [guida ufficiale al workflow Direct](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/creating-a-direct-workflow-rl-environment) sviluppa questi metodi su Cartpole. La creazione della scena precede il rollout: non deve essere ripetuta durante la lettura delle osservazioni.

## Esempio di reward personalizzata

Lo script [`03_custom_cartpole.py`](lab/03_custom_cartpole.py) eredita scena e dinamica dal Cartpole NVIDIA e modifica la durata dell'episodio e la reward. L'esempio isola così la specifica dell'obiettivo senza introdurre contemporaneamente un nuovo robot:

```bash
uv run python /percorso/vla_wm/background/tools/isaac_sim_3/custom_tasks/lab/03_custom_cartpole.py \
    --num_envs 8 --viz newton_gl
```

Indicando con $\theta_t$ l'angolo del pendolo rispetto alla verticale di equilibrio, con $x_t$ la posizione del carrello e con $b_t$ l'indicatore di terminazione, la reward implementata è:

$$
r_t = 2\cos(\theta_t)-0.1x_t^2-2b_t
$$

I coefficienti esprimono pesi numerici relativi alle unità usate dal task. Il primo termine favorisce il pendolo verticale, il secondo penalizza lo spostamento del carrello e il terzo penalizza le terminazioni, che nel task di equilibrio rappresentano condizioni di fallimento.

Questa modifica **non addestra una policy**: il loop dello script invia azioni nulle. Permette di osservare il nuovo segnale; per apprendere occorre collegare un algoritmo di training. Inoltre, la scala delle reward influenza l'ottimizzazione e va considerata quando si confrontano configurazioni.

## Registrazione e progetto esterno

`gym.register` associa l'identificativo `Tutorial-Cartpole-Custom-v0` alla classe dell'esempio. La registrazione avviene nel processo che esegue lo script: un nuovo processo non eredita automaticamente quell'associazione.

Per rendere un task disponibile ai comandi di training servono un progetto importabile, la registrazione eseguita all'import e le configurazioni dell'agente previste dall'integrazione scelta. Il [generatore ufficiale di progetti](https://docs.nvidia.com/isaaclab/3.0/developer-tools/build-your-own-project-or-task), avviabile con `uv run isaaclab --new`, prepara questa struttura.

Il generatore evita di dover ricostruire subito tutti i collegamenti infrastrutturali. La scelta di un template Direct con RSL-RL permette di partire da un ambiente noto e modificarne progressivamente osservazioni, reward e asset.

## Dal pendolo alla manipolazione

Per un task di sollevamento occorrono almeno un piano d'appoggio, un robot articolato e un oggetto afferrabile. Le osservazioni possono comprendere lo stato del robot $q_t$ e la posizione relativa tra oggetto e gripper, eventualmente sostituendo le informazioni geometriche con immagini.

Il successo va definito in termini osservabili dal valutatore, per esempio mantenere il cubo sopra una quota per un intervallo stabilito. Premiare soltanto l'altezza istantanea potrebbe incentivare il lancio dell'oggetto; premiare soltanto la vicinanza potrebbe produrre un avvicinamento senza presa.

Il reset deve ripristinare pose, velocità, target degli attuatori ed eventuale memoria del task. La **domain randomization**, cioè il campionamento di proprietà e condizioni iniziali tra episodi, viene introdotta dopo una configurazione deterministica verificabile. In questo modo un comportamento inatteso può essere attribuito a una modifica identificabile.
