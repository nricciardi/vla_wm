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


