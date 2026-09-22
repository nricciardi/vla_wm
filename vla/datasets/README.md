# Dataset per VLA

## Dataset real-world generalisti

I dataset real-world raccolgono l'interazione effettiva tra robot e ambiente. Rispetto alla simulazione conservano fenomeni difficili da modellare, come attrito, deformazioni, occlusioni ed errori di calibrazione, ma sono più costosi da acquisire e spesso riflettono le scelte hardware di un singolo laboratorio.

La loro generalità va quindi valutata lungo almeno tre assi: **diversità dei task**, **diversità degli ambienti** e **diversità degli embodiment**.

### Open X-Embodiment

**Open X-Embodiment (OXE)** è un'iniziativa collaborativa di aggregazione e standardizzazione, non una singola campagna di raccolta.

Il progetto parte da un problema strutturale del robot learning: molti laboratori possiedono dataset utili, ma ciascuno adotta robot, sensori, controller, frequenze di acquisizione e formati differenti. Presi separatamente, questi corpus sono spesso troppo piccoli per sostenere il pre-training di una policy generalista; concatenarli senza conversione, tuttavia, non produce un insieme semanticamente coerente.

La release presentata con RT-X riunisce **oltre un milione di traiettorie reali**, raccolte da 21 istituzioni con **22 embodiment robotici**, e comprende 527 skill distribuite su circa 160.000 task descritti in modi differenti. La scala non deriva quindi dalla ripetizione dello stesso setup, ma dall'unione di manipolatori fissi e mobili, sistemi a uno o due bracci, camere con punti di vista differenti e ambienti che includono laboratori, cucine e spazi domestici.

![Examples](figures/open_x_example.webp)

![Dataset composition](figures/open_x_dataset.webp)

#### Formato RLDS

Il primo livello di unificazione è il formato **RLDS**, che rappresenta **ogni episodio come una sequenza di step**.

Uno step può contenere l'osservazione $o_t$, l'azione $a_t$, una descrizione del task e metadati come l'indicazione di fine episodio.

Questa struttura comune rende possibile caricare e mescolare le sorgenti attraverso la stessa pipeline, ma **non implica che tutti i campi siano presenti o abbiano identica semantica**.

Alcuni dataset offrono più camere, depth o propriocezione; altri soltanto una vista RGB. Anche qualità, risoluzione e durata delle traiettorie rimangono eterogenee.

Per un VLA, l'istruzione linguistica $l$ funziona come interfaccia tra comportamenti differenti. Le sorgenti di OXE non possiedono però tutte annotazioni linguistiche equivalenti: alcune sono state raccolte direttamente con task description, altre richiedono etichette aggiunte o trasformate durante la conversione. Espressioni linguistiche simili possono inoltre riferirsi a condizioni iniziali, oggetti o criteri di successo diversi. La standardizzazione rende quindi le istruzioni utilizzabili nello stesso training mixture, ma non elimina rumore, granularità disomogenea o ambiguità.

#### Spazio delle azioni condiviso

La difficoltà principale riguarda lo **spazio delle azioni**. Quando i dati lo consentono, i comandi originali vengono ricondotti a una rappresentazione centrata sull'end-effector:

$$
a_t^{ee} = (\Delta x_t, \Delta y_t, \Delta z_t,
\Delta \phi_t, \Delta \theta_t, \Delta \psi_t, g_t),
$$

dove $\Delta x_t$, $\Delta y_t$ e $\Delta z_t$ descrivono la traslazione, $\Delta \phi_t$, $\Delta \theta_t$ e $\Delta \psi_t$ la rotazione secondo roll, pitch e yaw, mentre $g_t$ rappresenta il comando del gripper. Può essere aggiunto un segnale discreto di terminazione.

Questa rappresentazione offre un vocabolario operativo condiviso, ma resta un'**approssimazione**. A seconda del dataset, **un valore può indicare un delta di posa, una velocità o un target elaborato da un controller locale**; cambiano inoltre frame di riferimento, scale, frequenze di controllo e convenzioni sulle rotazioni.

La stessa azione numerica può quindi produrre movimenti fisici diversi su robot differenti. Le componenti non riconducibili all'interfaccia comune possono essere ignorate o richiedere trasformazioni specifiche, con una possibile perdita di informazione.

#### Composizione del mixture

Anche la composizione del mixture richiede una scelta esplicita. Se le traiettorie fossero campionate in proporzione alla dimensione originale dei dataset, le sorgenti più grandi dominerebbero l'ottimizzazione. Se ogni dataset avesse lo stesso peso, poche traiettorie di una sorgente piccola verrebbero riutilizzate molto più spesso. Il bilanciamento determina quindi quali embodiment e skill influenzano maggiormente il modello e costituisce parte della definizione del dataset di training, non un dettaglio puramente implementativo.

È infine necessario distinguere Open X-Embodiment completo dal particolare mixture impiegato in un esperimento. RT-1-X e RT-2-X sono stati **addestrati su un sottoinsieme compatibile** con le rispettive architetture, non su ogni traiettoria oggi disponibile sotto il nome OXE.

Analogamente, lavori successivi come Octo e OpenVLA selezionano, filtrano e pesano subset differenti. Dire che due modelli usano Open X-Embodiment non garantisce quindi che abbiano osservato gli stessi dati.


### DROID

**DROID**, acronimo di *Distributed Robot Interaction Dataset*, contiene circa **76.000 traiettorie e 350 ore di interazione**, raccolte da 50 operatori in 564 scene distribuite geograficamente.

Tutti i siti usano una piattaforma Franka standardizzata, tre viste **RGB, depth, stato del robot e azioni**; questa scelta riduce l'eterogeneità hardware per aumentare quella di case, oggetti e comportamenti.

Il dataset è particolarmente utile per studiare la generalizzazione a scene reali non curate. Resta tuttavia legato a un singolo tipo di manipolatore e la raccolta distribuita introduce variazioni di calibrazione e qualità.

### BridgeData V2

**BridgeData V2** raccoglie decine di migliaia di traiettorie di manipolazione eseguite con un WidowX 250 in numerosi ambienti, includendo cucine reali. Le dimostrazioni associano immagini, comandi del robot e annotazioni linguistiche e supportano sia policy condizionate dal linguaggio sia policy condizionate da una goal image.

Il valore del corpus risiede nella varietà di oggetti, scene e comportamenti ottenuta con un embodiment coerente. Proprio questa coerenza semplifica il learning, ma limita il trasferimento diretto ad altre morfologie.

### RH20T

**RH20T** privilegia la ricchezza multimodale: comprende oltre **110.000 sequenze contact-rich** acquisite con più piattaforme robotiche, scene, camere e sensori. Oltre a immagini e stato del robot può includere depth, forza/coppia, audio e segnali tattili, rendendo il dataset adatto a studiare skill in cui il solo feedback visivo è insufficiente.

Questa abbondanza sensoriale comporta però disponibilità non uniforme delle modalità e una pipeline più complessa. Inoltre, una sequenza dimostrativa non coincide necessariamente con una traiettoria linguistica già pronta per il training VLA: annotazioni, sincronizzazione e selezione dei canali dipendono dall'uso previsto.

### RoboSet

**RoboSet** è il corpus introdotto con RoboAgent e contiene circa **7.500 traiettorie** bilanciate su 12 skill e 38 task in scene di cucina. Le dimostrazioni sono raccolte tramite teleoperazione e vengono affiancate da tecniche di *semantic augmentation* per aumentare la varietà visiva e linguistica senza richiedere una nuova esecuzione fisica per ogni variante.

La scala è inferiore a quella dei grandi mixture, ma la distribuzione intenzionalmente bilanciata rende RoboSet utile per analizzare l'efficienza dei dati. Le augmentations aumentano la copertura semantica, non quella delle dinamiche fisiche osservate.

### AgiBot World e Colosseo

**AgiBot World** è il dataset del progetto **Colosseo**, una piattaforma che integra raccolta, verifica, benchmark e modelli. La release descritta nel lavoro originario supera **un milione di traiettorie**, copre 217 task e usa 100 robot in cinque domini applicativi, con particolare attenzione alla manipolazione bimanuale, alle operazioni long-horizon e all'uso di strumenti.

Scala e pipeline standardizzata riducono la frammentazione tipica dei corpus accademici. Rimangono però forti dipendenze dall'hardware, dalle procedure di teleoperazione e dalla distribuzione curata degli scenari; inoltre, “generalista” non significa che ogni forma di robot o locomozione sia rappresentata.

### Dataset proprietari di RT-1 e RT-2

Il corpus di **RT-1** comprende oltre **130.000 episodi**, raccolti nell'arco di 17 mesi con 13 Everyday Robots mobile manipulators in ambienti di tipo office kitchen. Copre più di 700 task espressi linguisticamente e registra immagini, azioni di braccio e pinza, movimento della base e terminazione. **RT-2** riusa sostanzialmente questa esperienza robotica e la combina, mediante co-fine-tuning, con i dati vision-language dei backbone PaLI-X o PaLM-E.

Questi dati mostrano quanto una raccolta coerente e mirata possa sostenere una policy generalista su un embodiment. Poiché il corpus robotico completo non è pubblico, tuttavia, composizione, failure cases e riproducibilità restano meno verificabili rispetto ai dataset aperti.

### Dataset di Physical Intelligence per $\pi_0$

Il training mixture di **$\pi_0$** combina pre-training vision-language su larga scala, dataset robotici open source e dati proprietari raccolti da Physical Intelligence su **otto configurazioni robotiche**. La porzione interna include task destrosi e bimanuali, come piegare indumenti, sparecchiare e assemblare oggetti, ed è usata sia per il pre-training generalista sia per il fine-tuning su task complessi.

La varietà di robot e skill è una parte sostanziale del risultato, ma dimensioni, distribuzione completa e traiettorie non sono pubblicate in forma tale da consentire una replica indipendente. Occorre quindi distinguere le proprietà documentate del mixture dalle inferenze basate sulle prestazioni del modello.

Nel complesso, questi corpus mostrano un trade-off ricorrente: **standardizzare molto facilita il training congiunto ma riduce la varietà hardware; aumentare embodiment e sensori amplia la copertura ma rende più difficile attribuire alle azioni un significato comune**.

## Dataset simulati e pipeline di generazione

Un **dataset simulato** contiene osservazioni, stati e azioni prodotti eseguendo un robot in un ambiente virtuale. La simulazione rende possibile raccogliere dati in parallelo, controllare la distribuzione degli stati iniziali e ottenere annotazioni che nel mondo reale sarebbero costose o impossibili da misurare. Non va però confusa con un benchmark: il dataset serve ad addestrare o pre-addestrare la policy, mentre il benchmark specifica task, split, reset, metriche e protocollo di valutazione.

NVIDIA Isaac Lab, Isaac Sim e robosuite sono innanzitutto **infrastrutture di simulazione**, non dataset con una composizione immutabile. Diventano sorgenti di dataset quando una pipeline definisce robot, scene, policy di raccolta, randomizzazioni e formato di esportazione. Questa distinzione è importante perché due corpus generati con lo stesso simulatore possono avere distribuzioni e finalità completamente diverse.

### NVIDIA Isaac Sim e Replicator

**NVIDIA Isaac Sim** è il livello di simulazione basato su OpenUSD, rendering RTX e motore fisico PhysX. Consente di modellare robot, sensori e scene e di produrre annotazioni sincronizzate con la simulazione. **Replicator** aggiunge primitive per la *synthetic data generation*: variazioni di illuminazione, materiali, pose, camere e asset possono essere campionate programmaticamente, mentre gli annotator esportano RGB, depth, segmentazioni, bounding box, pose e altri ground truth.

Questa pipeline è particolarmente utile per dati percettivi e per la **domain randomization**, con cui si cerca di evitare che il modello dipenda da un solo aspetto grafico della scena. Un corpus di immagini annotate non costituisce però ancora un dataset VLA: per il controllo servono anche una sequenza di azioni $a_t$, lo stato del robot $q_t$ e, quando previsto, l'istruzione $l$. La [documentazione di Isaac Sim Replicator](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/index_tools.html) presenta i workflow di generazione.

### NVIDIA Isaac Lab

**Isaac Lab** è il framework di robot learning costruito sopra Isaac Sim. Aggiunge ambienti vettorializzati, modelli di attuatori e sensori, domain randomization e integrazioni con algoritmi di reinforcement learning e imitation learning. L'esecuzione parallela su GPU permette di raccogliere grandi quantità di transizioni o rollout, ma il contenuto del dataset dipende sempre dal task e dalla policy usata per esplorarlo.

Isaac Lab è quindi una base per creare dataset di locomozione, manipolazione e controllo whole-body più che un singolo corpus da scaricare. La presenza dello stato completo $s_t$ facilita reward, filtraggio e annotazione automatica; una policy visuale destinata al mondo reale dovrà tuttavia essere addestrata sulle osservazioni $o_t$ effettivamente disponibili e affrontare il **sim-to-real gap**. La [documentazione ufficiale di Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/) descrive simulazione, sensori e randomizzazione.

### Isaac Lab Mimic

**Isaac Lab Mimic** è la pipeline più direttamente orientata alla generazione di traiettorie per imitation learning. Parte da un piccolo insieme di dimostrazioni, le segmenta in subtask object-centric e trasforma i segmenti per nuove configurazioni spaziali. Le esecuzioni riuscite vengono esportate, per esempio in HDF5, come nuove dimostrazioni state-based o visuomotor.

Il metodo aumenta la copertura delle pose senza richiedere una nuova teleoperazione per ogni episodio. Non genera però skill arbitrarie: dipende dalla segmentazione, dagli oggetti di riferimento e dalle primitive presenti nei dati sorgente. La [guida Isaac Lab Mimic](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html) documenta raccolta, annotazione, generazione e replay.

### MimicGen

**MimicGen** introduce il paradigma su cui si basa anche Isaac Lab Mimic. A partire da poche dimostrazioni umane, scompone il comportamento in segmenti object-centric e li ricompone rispetto a nuovi stati degli oggetti. Le traiettorie candidate vengono eseguite nel simulatore e conservate quando soddisfano le condizioni di successo.

MimicGen separa il costo di definire una strategia dal costo di produrne molte variazioni. La qualità del corpus dipende tuttavia dalla correttezza delle condizioni di successo e dalla copertura delle dimostrazioni iniziali; una grande quantità di ricombinazioni non equivale automaticamente a diversità strategica. Il [progetto ufficiale MimicGen](https://mimicgen.github.io/) distribuisce codice, ambienti e dataset generati.

### robosuite e MuJoCo

**robosuite** è un framework di manipolazione costruito su MuJoCo e costituisce la base di numerose raccolte sintetiche. Fornisce robot, controller, task e osservazioni modulari; dimostrazioni teleoperate o generate da policy possono essere registrate e poi utilizzate da librerie come robomimic e MimicGen. Anche LIBERO e RoboCasa ne riusano l'infrastruttura, ma aggiungono protocolli e distribuzioni di task propri.

Il vantaggio è un ecosistema relativamente semplice da estendere e riprodurre. I dati rimangono però sensibili alla versione del simulatore, ai parametri fisici, al controller e alla frequenza di campionamento. Il [progetto robosuite](https://robosuite.ai/) descrive ambienti e interfacce di raccolta.

### Genesis

**Genesis** è una piattaforma di simulazione general-purpose orientata all'esecuzione parallela e alla generazione automatizzata di mondi e dati. Può essere usata per produrre rollout robotici, immagini, moto e interazioni con materiali non rigidi, quindi rappresenta una possibile alternativa recente alle pipeline basate su Isaac o MuJoCo.

La promessa di una generazione più automatica non elimina la necessità di specificare distribuzione dei task, controllori, filtri di qualità e formato delle azioni. Trattandosi di un ecosistema più giovane, disponibilità degli asset e riproducibilità dei corpus vanno valutate per ogni release. Il [progetto Genesis](https://genesis-world.readthedocs.io/) documenta simulatori e workflow disponibili.

La provenienza sintetica dovrebbe essere descritta insieme alla **procedura di generazione**: simulatore e versione, robot, controller, sorgente delle dimostrazioni, randomizzazioni, modalità osservate, criterio di successo e rapporto tra episodi riusciti e falliti. Senza queste informazioni, la dimensione del dataset dice poco sulla sua effettiva utilità per un VLA.

## Dataset più specifici

I dataset specialistici non mirano necessariamente a massimizzare il numero di task. Isolano invece un problema per il quale i corpus generalisti offrono poca supervisione: coordinazione tra due braccia, controllo whole-body o apprendimento da attività umane. Il loro uso richiede di distinguere tra **azioni robotiche eseguibili**, **moto umano da retargetizzare** e **video che fornisce soltanto supervisione percettiva o semantica**.

### ALOHA e dataset ACT per la manipolazione bimanuale

**ALOHA** è una piattaforma hardware open source per teleoperazione bimanuale. Due bracci leader, mossi dall'operatore, comandano due bracci follower; camere e posizioni articolari registrano dimostrazioni sincronizzate di task fini e contact-rich. Il lavoro originario raccoglie piccoli dataset task-specifici e introduce **Action Chunking with Transformers (ACT)**, che predice blocchi temporali di azioni per ridurre l'accumulo dell'errore e modellare la variabilità delle dimostrazioni.

Non esiste quindi un unico “ALOHA dataset” universale: il nome identifica una famiglia di raccolte compatibili, dai sei task tabletop originari ai successivi dataset di Mobile ALOHA e ALOHA 2. Sono preziose per coordinazione bimanuale e controllo ad alta precisione, ma spesso hanno poche dimostrazioni per task e un action space specifico dei bracci. Il [progetto ALOHA/ACT](https://tonyzhaozh.github.io/aloha/) e [Mobile ALOHA](https://mobile-aloha.github.io/) distribuiscono dati e strumenti.

### Dataset per umanoidi

Un dataset per umanoidi deve rappresentare più della manipolazione. Locomozione, equilibrio, contatti, postura e coordinazione whole-body rendono l'azione $a_t$ più ampia e fortemente dipendente dalla morfologia. Le fonti disponibili formano un ecosistema frammentato: teleoperazione e motion capture producono traiettorie fisiche; dataset di moto umano forniscono pose da retargetizzare; raccolte recenti su umanoidi reali associano questi segnali a immagini e linguaggio.

Questi dati non possono essere trattati come normali traiettorie di un braccio fisso. Il **retargeting** deve rispettare limiti articolari, stabilità e contatti, mentre la stessa posa umana può non essere dinamicamente realizzabile dal robot. È quindi opportuno documentare per ogni corpus morfologia, modalità di acquisizione, presenza di locomozione e livello di controllo. Lavori come [Humanoid-VLA](https://arxiv.org/abs/2502.14795) mostrano l'uso congiunto di dati language-motion e traiettorie robotiche, ma il settore non dispone ancora di uno standard equivalente a Open X-Embodiment.

### Human video datasets

I **video umani** offrono una scala e una diversità semantica molto superiori ai dati robotici. Collezioni di action recognition, video dimostrativi sul web e registrazioni procedurali mostrano oggetti, affordance e ordine temporale delle attività. Possono pre-addestrare rappresentazioni visive, modelli video e world model o fornire goal e pseudo-annotazioni linguistiche.

Un video non contiene però direttamente l'azione robotica $a_t$: spesso mancano profondità, forze, posa 3D delle mani e stato dell'oggetto, e la cinematica umana differisce da quella del robot. L'uso per il controllo richiede quindi representation learning, pose estimation, inverse dynamics o annotazioni robot-aligned. Dataset come **Something-Something V2** privilegiano interazioni oggetto-azione; raccolte procedurali ed egocentriche aggiungono invece contesto long-horizon. Il punto fondamentale è che questi dati ampliano la conoscenza di “che cosa accade”, ma non specificano da soli “quale comando inviare al robot”.

### Egocentric video datasets

I video **egocentrici**, acquisiti da una camera indossata dalla persona, avvicinano osservazione e azione: mani e oggetti manipolati compaiono dal punto di vista dell'agente. **Ego4D** raccoglie migliaia di ore di attività quotidiane con narrazioni e annotazioni; **EPIC-KITCHENS-100** concentra 100 ore su attività non scripted in cucine reali; **Ego-Exo4D** sincronizza la vista egocentrica con più viste esterne e aggiunge pose 3D, audio, gaze e descrizioni temporali; **HoloAssist** registra anche l'interazione verbale tra esecutore e istruttore.

Queste proprietà favoriscono grounding linguistico, riconoscimento delle affordance e apprendimento di struttura procedurale. Persistono tuttavia l'assenza di comandi robotici e un embodiment gap: mani umane, camera sulla testa e controllo dello sguardo non coincidono con gripper e camere di un robot. Le risorse ufficiali di [Ego4D](https://ego4d-data.org/docs/), [EPIC-KITCHENS](https://epic-kitchens.github.io/), [Ego-Exo4D](https://docs.ego-exo4d-data.org/) e [HoloAssist](https://holoassist.github.io/) permettono di verificare modalità e licenze.

La scelta tra queste sorgenti dipende quindi dal livello di supervisione richiesto. **ALOHA** fornisce azioni direttamente eseguibili su una piattaforma bimanuale; i dataset umanoidi richiedono spesso retargeting; i video umani ed egocentrici sono soprattutto dati di pre-training percettivo, semantico e dinamico. Combinarli è promettente, purché il passaggio tra questi livelli non venga presentato come una semplice concatenazione di dataset.

## Curation, deduplicazione e split semantici

La qualità di un dataset non dipende soltanto dalla raccolta. È necessario anche decidere **quali esempi siano duplicati, quale unità debba rimanere indivisibile e quali differenze debbano separare training, validation e test**. Nei video e nelle traiettorie robotiche, uno split casuale per frame può collocare osservazioni consecutive in partizioni diverse e produrre semantic leakage: il test contiene allora immagini quasi identiche a quelle usate per il training.

### BubbleFence

**BubbleFence** affronta questo problema rappresentando le immagini con un vision foundation model e costruendo lo split direttamente nello spazio degli embedding. La configurazione predefinita usa CLIP, elimina quasi duplicati mediante similarità coseno e colloca anchor tramite sequenze Quasi-Monte Carlo. Attorno a ogni anchor definisce una bubble con raggio adattato alla Local Intrinsic Dimensionality: i punti esterni rimangono nel training set, mentre due shell concentriche formano validation e test.

La pipeline è progettata per dati incrementali. Anchor, embedding e assegnazioni persistono tra round di ingestione; i nuovi esempi riusano le bubble esistenti e ne creano altre soltanto quando serve a mantenere la quota di evaluation. L'articolo dimostra il comportamento su **Zenseact Open Dataset Drives** e su frame Minecraft derivati da **Video PreTraining**, non su robot o dataset VLA. Non presenta inoltre un confronto downstream che dimostri prestazioni di generalizzazione stimate meglio rispetto alle baseline.

#### Novelty

Il contributo consiste nel combinare **semantic fencing, placement QMC, densità locale, controllo closed-loop dello split e persistenza streaming**. In questo modo la partizione segue regioni di similarità visuale e può crescere senza ricalcolare interamente i batch precedenti.

#### Limiti

La semantica dello split dipende dall'encoder e non comprende automaticamente istruzione, azione, stato robotico o appartenenza allo stesso episodio. Per dati VLA è quindi necessario applicare vincoli di gruppo a livello di rollout e integrare il criterio visuale con task, ambiente, embodiment e sessione. L'[approfondimento su BubbleFence](bubblefence/README.md) ricostruisce formalizzazione, placement degli anchor, raggi adattivi, ingestione incrementale, risultati dimostrativi e limiti per il robot learning.
