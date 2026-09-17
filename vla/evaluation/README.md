# Benchmark

## Benchmark basati su simulazione

### LIBERO

**LIBERO** nasce per misurare il trasferimento nel *lifelong robot learning* ed è oggi molto usato anche per confrontare VLA nella manipolazione. Le quattro suite più comuni contengono task complementari:

- **LIBERO-Spatial** varia le relazioni spaziali richieste mantenendo familiari oggetti e comportamenti;

- **LIBERO-Object** verifica il trasferimento verso combinazioni e identità di oggetti differenti;

- **LIBERO-Goal** cambia il goal linguistico associato alla scena;

- **LIBERO-Long**, spesso indicato nel protocollo VLA come **LIBERO-10**, raccoglie attività più lunghe che richiedono più passaggi coordinati.

La metrica normalmente riportata è il **success rate** medio sui task, calcolato su più stati iniziali. Octo fine-tuned e OpenVLA vengono confrontati sulle quattro suite nel lavoro di OpenVLA, rendendo LIBERO uno dei punti di riferimento più diffusi per la generalizzazione dopo adattamento. Il benchmark resta però interamente simulato e può premiare dettagli del controller o del rendering; non misura direttamente il trasferimento su hardware.

### CALVIN

**CALVIN** valuta policy language-conditioned in un ambiente tabletop continuo. Il tratto distintivo è la valutazione **long-horizon**: il modello riceve una sequenza di istruzioni e deve completare più subtask consecutivi senza che l'ambiente venga riportato allo stato iniziale dopo ogni skill. La suite mette quindi in evidenza errori di grounding, esecuzioni parziali e compounding error.

Il protocollo CALVIN ABC→D addestra o adatta la policy sugli ambienti A, B e C e ne misura la generalizzazione nell'ambiente D. Oltre al successo per singolo subtask, viene riportata la lunghezza media della catena completata e la percentuale di sequenze risolte fino a ciascuna profondità. Questa metrica è più informativa del solo successo one-step, ma rimane legata a un insieme finito di primitive e scene.

### SimplerEnv

**SimplerEnv**, associato al benchmark **SIMPLER**, valuta in simulazione policy originariamente addestrate con dati real-world. Ricostruisce setup correlati al **Google Robot** impiegato nella linea RT e al **WidowX** di BridgeData V2, quindi esegue le policy senza richiedere necessariamente un nuovo training esclusivamente simulato.

Il benchmark propone due strategie complementari. *Visual matching* cerca configurazioni della simulazione visivamente vicine al dominio reale; *variant aggregation* valuta molte varianti e aggrega i risultati per ridurre la dipendenza da una singola ricostruzione. Lo scopo è ottenere ranking correlati alle valutazioni fisiche a un costo inferiore, non sostituire universalmente i test reali. Correlazione e validità dipendono infatti da robot, task e policy inclusi nell'analisi.

### RoboCasa

**RoboCasa** è un framework e benchmark per attività domestiche, con particolare attenzione alle cucine. Varia layout, asset, posizioni degli oggetti e istruzioni e comprende skill atomiche e **task composizionali** che richiedono di concatenare più operazioni semanticamente coerenti.

La ricchezza delle scene lo rende utile per valutare generalizzazione visiva e ragionamento operativo in un dominio household. I risultati dipendono però dalla release e dallo split: numero di task, dimostrazioni e protocolli sono evoluti nel tempo e devono essere riportati esplicitamente. Inoltre, la diversità procedurale delle cucine virtuali non elimina il sim-to-real gap.

### RLBench

**RLBench** offre un ampio catalogo di task di manipolazione in CoppeliaSim, con descrizioni linguistiche, condizioni di successo e variazioni procedurali. Le osservazioni possono includere viste RGB-D multiple, maschere e stato propriocettivo; le dimostrazioni sono prodotte tramite motion planning.

La varietà dei task lo rende adatto a valutazioni multi-task, few-shot e language-conditioned. Le dimostrazioni pianificate sono però più regolari delle teleoperazioni umane e non tutti i lavori usano lo stesso sottoinsieme, la stessa modalità osservativa o lo stesso action space. Un confronto richiede pertanto di dichiarare task e protocollo, non soltanto il nome RLBench.

### ManiSkill

**ManiSkill** comprende ambienti e benchmark di manipolazione GPU-accelerated con enfasi su contatti, geometrie e variazioni degli asset. Supporta osservazioni state-based e visuali, robot differenti e task che vanno dal controllo di primitive alla manipolazione più articolata.

L'elevato parallelismo rende praticabili molte prove e permette di valutare robustezza a inizializzazioni diverse. Al tempo stesso, “ManiSkill” identifica una famiglia in evoluzione: versione, task set, modalità sensoriale, controller e budget di training devono essere specificati per rendere il risultato interpretabile.
