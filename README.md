# VLA e World Models

I Vision-Language-Action model e i World Model affrontano due problemi **complementari ma distinti**.

Un **VLA** riceve osservazioni visive, istruzioni linguistiche e, in alcuni casi, informazioni sullo stato del robot, per **produrre direttamente una o più azioni** eseguibili.

L’obiettivo principale consiste quindi nel mappare percezione e linguaggio su una policy di controllo.

Il capitolo dedicato ai [Vision-Language-Action model](vla/README.md) ne ricostruisce i precursori e l'evoluzione verso policy robotiche end-to-end.

In forma semplificata, un VLA risponde alla domanda *"quale azione devo eseguire?"*.

$$
\text{VLA:} \quad \text{vision + instruction (+ state/memory)} \longrightarrow \text{action}
$$

Un **World Model**, invece, cerca di **rappresentare la dinamica dell’ambiente** e di prevedere come lo stato del mondo evolverà in funzione delle azioni eseguite.

$$
\text{World Model:} \quad \text{state + action} \longrightarrow \text{next state}
$$

In forma semplificata, un World Model risponde a *"cosa succederà se eseguo questa azione?"*.

Il capitolo dedicato ai [World Models](world_models/README.md) organizza i concetti fondamentali, le principali famiglie di modelli e il loro impiego nella pianificazione e nel controllo.

Tale previsione può essere utilizzata per pianificare sequenze di azioni, valutare alternative o simulare scenari futuri:

$$
a^* = \arg\max_a \; \text{Reward}(\text{WorldModel}(s, a))
$$

**I due approcci non sono alternativi**: un agente può utilizzare un World Model per simulare o valutare possibili evoluzioni future e un VLA, o una policy equivalente, per trasformare la decisione finale in azioni robotiche.

## Background e prerequisiti

Lo studio di VLA e World Model richiede alcuni strumenti comuni: apprendimento da dimostrazioni, controllo robotico e modelli generativi continui. La sezione dedicata ai [background](background/README.md) organizza questi prerequisiti e collega ogni concetto al ruolo che assume nei sistemi discussi nei capitoli principali.

### Imitation Learning e Behavioral Cloning

L'**Imitation Learning** apprende un comportamento da dimostrazioni prodotte da un esperto. Il **Behavioral Cloning** ne fornisce la formulazione supervisionata più diretta, ma introduce covariate shift, accumulo degli errori e difficoltà nel rappresentare azioni multimodali. Questi aspetti spiegano sia la raccolta dei dataset robotici sia molte scelte delle policy VLA. L'[approfondimento su Imitation Learning e Behavioral Cloning](background/imitation_learning/README.md) sviluppa formulazione probabilistica, loss, distribuzioni delle azioni e valutazione closed loop.

### Controllo robotico

Una policy produce normalmente riferimenti articolari o cartesiani, non segnali applicati direttamente ai motori. Cinematica inversa, generazione delle traiettorie e controller a frequenza più elevata trasformano tali riferimenti in movimento fisico. Comprendere questa gerarchia è necessario per interpretare action space, frequenze e vincoli di sicurezza dei VLA. L'[approfondimento sul controllo robotico](background/robot_control/README.md) tratta configurazioni articolari, pose, Jacobiano, controller e interfacce di comando.

### Diffusion Models, score e Flow Matching

Diffusion e Flow Matching modellano distribuzioni complesse attraverso processi dipendenti dal tempo. Nei sistemi generativi producono immagini o latent; nel robot learning possono rappresentare distribuzioni multimodali di azioni e action chunk continui. Il percorso sui [Diffusion Models e Flow Matching](background/diffusion/README.md) collega DDPM, latent diffusion, score-based SDE, Continuous Normalizing Flows e scelte di training e sampling.
