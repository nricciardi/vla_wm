# Interpretabilità e steering della navigazione vision-language

La **Vision-Language Navigation (VLN)** richiede a un agente di seguire un'istruzione linguistica in un ambiente osservato da una camera. Le policy recenti usano spesso un **Vision-Language Model (VLM)** per combinare testo, immagini correnti e memoria visiva, producendo comandi di navigazione. Il *success rate* misura se l'agente arriva a destinazione, ma da solo non chiarisce **quali informazioni abbiano causato le decisioni** né perché una policy riesca a recuperare da un errore.

## What do VLM-Based Vision-Language Navigation Models Rely on (2026)

Nel paper [*What do VLM-Based Vision-Language Navigation Models Rely on: Interpreting and Steering Policy Behavior*](https://arxiv.org/abs/2609.24576), Makowski e colleghi studiano quattro policy VLN basate su VLM: **CorrectNav, DualVLN, AwareVLN e NaVILA**. L'intuizione centrale è confrontare due domande distinte: 

- Quanto cambia l'azione se si altera una parte dell'input 
- Quanto cambia il successo quando l'agente agisce con quell'informazione degradata

Le perturbazioni interessano immagini, memoria visiva e istruzione; ulteriori interventi sulle attivazioni interne verificano se i concetti appresi possano orientare la policy.

La valutazione principale usa un **sottoinsieme dello split validation unseen di R2R in VLN-CE**, con traiettorie in ambienti interni simulati. 

Per l'analisi visiva vengono annotati manualmente **200 punti decisionali**; per quella linguistica, **100 campioni**. 

Una prova separata trasferisce un **vettore di steering**, ricavato in simulazione, a un **robot mobile reale** osservato con una camera fisheye.

Gli esperimenti mostrano che tutte le policy reagiscono a più modalità, ma **sensibilità e robustezza in closed loop non coincidono**. 

Le mappe basate su perturbazioni localizzano spesso indizi utili meglio di quelle basate sui gradienti, pur assegnando peso anche ad aree giudicate irrilevanti dagli annotatori. 

Inoltre, concetti direzionali conservati dal VLM originale possono influenzare DualVLN e un vettore di attivazione può aumentare la frequenza del ragionamento esplicito di AwareVLN sotto uno spostamento della distribuzione visiva.

### Novelty

- Analizza causalmente, nello stesso protocollo, la dipendenza di quattro VLN basati su VLM da **osservazione, istruzione e memoria visiva**, distinguendo interventi open loop e risultati closed loop.
- Verifica che concetti semantici del VLM sopravvivano al fine-tuning per la navigazione e possano **modificare le azioni** anche quando i rispettivi token non appartengono all'output della policy.
- Trasferisce al robot reale uno **steering delle attivazioni appreso in simulazione** per favorire il ragionamento esplicito, senza aggiornare i parametri del modello.

### Limiti

Le analisi principali riguardano un solo benchmark e sottoinsiemi annotati relativamente piccoli. Le mappe di sensibilità dipendono dal tipo e dall'intensità delle perturbazioni; un'azione invariata non prova che l'informazione sia ignorata, perché altri indizi possono compensarla. Viceversa, un'azione diversa non implica un peggioramento del successo.

Nelle prove reali l'intervento è **attivato da un operatore dopo il riconoscimento di un fallimento**. Il miglioramento riportato, dal 37.2% al 74.4% di successo nel protocollo sequenziale, non dimostra quindi che il sistema riconosca autonomamente quando intervenire né che lo stesso vettore generalizzi a robot e ambienti arbitrari.

L'[approfondimento sul paper](vln_interpreting_steering/README.md) sviluppa le metriche di attribuzione, gli interventi sulle attivazioni, il protocollo sperimentale e l'interpretazione dei risultati.
