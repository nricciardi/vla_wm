# Strumenti per la simulazione robotica

Gli strumenti di simulazione collegano i concetti di controllo e apprendimento a esperimenti eseguibili. Una scena specifica robot e oggetti; un motore fisico ne aggiorna lo stato; i sensori producono osservazioni; un ambiente organizza azioni, reset e criteri di valutazione. La distinzione tra questi livelli permette di comprendere quali aspetti di un risultato dipendano dalla policy e quali dalla costruzione dell'esperimento.

## Isaac Lab e Isaac Sim

**Isaac Lab** organizza ambienti e pipeline di robot learning, mentre **Isaac Sim** offre un'applicazione di simulazione con strumenti per scene, sensori e rendering. Il percorso considera come scegliere le componenti necessarie e come verificare progressivamente un esperimento, dalla dinamica di pochi corpi alla valutazione di una policy in ambienti paralleli.

L'[approfondimento su Isaac Lab e Isaac Sim](isaac_sim_3/README.md) parte da stato, osservazione e azione, senza presupporre familiarità con ManiSkill. Sviluppa poi installazione locale, Gymnasium, creazione delle scene, task personalizzati, deformabili, training, immagini, metriche episodiche, HPC e debug. Gli esempi Python accompagnano la trattazione e sono distinti dalle verifiche runtime che richiedono lo stack NVIDIA.
