# Prerequisiti

Questa sezione raccoglie i concetti necessari per discutere Vision-Language-Action model e World Model senza trattarli come componenti isolate. L'obiettivo non è fornire un'enciclopedia completa di ciascun argomento, ma costruire un lessico comune e chiarire le ipotesi che ricorrono nei capitoli successivi.

## Imitation Learning e Behavioral Cloning

L'Imitation Learning studia come apprendere un comportamento a partire da dimostrazioni. Il Behavioral Cloning ne costituisce la formulazione più diretta: una policy viene addestrata a predire le azioni dell'esperto mediante supervised learning. È semplice e permette di sfruttare grandi dataset offline, ma il modello viene addestrato sugli stati visitati dall'esperto e deve poi agire sugli stati prodotti dalle proprie decisioni; questa differenza genera covariate shift e accumulo degli errori. L'[approfondimento su Imitation Learning e Behavioral Cloning](imitation_learning/README.md) introduce la formulazione probabilistica, le loss per azioni discrete e continue, la multimodalità e i criteri di valutazione closed loop.

## Reinforcement Learning

## Diffusion Models

I diffusion model apprendono una distribuzione complessa trasformando progressivamente rumore in dati strutturati. In training il modello impara un target locale lungo un processo di perturbazione noto; in inference un sampler usa ripetutamente tale previsione per generare un campione. Nel robot learning la stessa idea consente di rappresentare distribuzioni multimodali di traiettorie o action chunk, al costo di una procedura iterativa e quindi di maggiore latenza.

L'[approfondimento sui Diffusion Models](diffusion/README.md) organizza il percorso dall'intuizione e dai fondamenti probabilistici fino a DDPM, latent diffusion, score-based SDE e aspetti implementativi.

## Flow Matching

Il Flow Matching addestra direttamente il campo di velocità di una Continuous Normalizing Flow. Invece di ricavare la dinamica generativa da un processo di denoising, costruisce un probability path tra distribuzione base e dati e regredisce la velocità che lo genera. Il conditional flow matching rende il training simulation-free, mentre in inferenza il campo appreso viene integrato mediante un solver ODE.

Il relativo [capitolo su Continuous Normalizing Flows e Flow Matching](diffusion/05_flow_matching/README.md) chiarisce la relazione con score, probability flow ODE, rectified flow e generazione di action chunk continui.

## Controllo Robotico

Una policy robotica non controlla necessariamente i motori in modo diretto. Nella maggior parte dei sistemi produce riferimenti cartesiani o articolari che vengono trasformati in movimento da cinematica inversa, generatori di traiettoria e controller eseguiti a frequenza più elevata. Questa gerarchia spiega perché action space apparentemente simili possano avere significati fisici differenti e perché un VLA in closed loop non sostituisca il servo controller del robot.

L'[approfondimento sul Controllo Robotico](robot_control/README.md) introduce configurazioni articolari e pose cartesiane, cinematica diretta e inversa, Jacobiano, controllo in posizione, velocità e coppia, azioni assolute e relative, generazione delle traiettorie, controller PD/PID, frequenze operative e vincoli di sicurezza.
