# Diffusion Models e Flow Matching

I **diffusion model** costruiscono una distribuzione generativa trasformando una distribuzione semplice, in genere gaussiana, nella distribuzione dei dati. L'idea può essere letta in tre modi complementari: come inversione di un processo di corruzione, come stima dello **score** delle distribuzioni rumorose oppure come apprendimento di un **campo vettoriale dipendente dal tempo**. Queste prospettive conducono rispettivamente ai DDPM, ai modelli score-based formulati tramite equazioni differenziali stocastiche e al Flow Matching.

La distinzione è particolarmente utile nel robot learning. Un'immagine generativa è un singolo campione ad alta dimensionalità; una policy diffusion o flow-matching genera invece un'azione o un'intera sequenza $A_t=[a_t,\ldots,a_{t+H-1}]$ condizionata sull'osservazione $o_t$ e, nei VLA, sull'istruzione linguistica $l$. La capacità di rappresentare più soluzioni compatibili con lo stesso contesto rende questi modelli adatti a distribuzioni di azioni multimodali, ma la generazione iterativa introduce latenza e scelte numeriche che devono essere comprese esplicitamente.

## Intuizione del processo generativo

Il punto di partenza è la separazione tra **training** e **sampling**. Durante il training un campione pulito viene perturbato a un livello di rumore casuale e il modello apprende un target locale; durante il sampling si parte dal prior e si applicano più aggiornamenti fino a ottenere un campione strutturato. Il capitolo sull'[intuizione dei diffusion model](01_introduzione/README.md) chiarisce questa asimmetria, il ruolo del condizionamento e la differenza tra diffusione nei pixel, nei latent e nello spazio delle azioni.

## Fondamenti probabilistici

La diffusione forward definisce una famiglia $p_t(x)$ che collega i dati a una distribuzione semplice. La formula chiusa per $x_t$, il rapporto segnale-rumore e lo score $\nabla_x\log p_t(x)$ spiegano perché la predizione del rumore fornisca l'informazione necessaria per invertire il processo. I [fondamenti matematici](02_fondamenti_matematici/README.md) introducono questi oggetti e fissano la notazione usata nei capitoli successivi.

## Denoising Diffusion Probabilistic Models

I **DDPM** formalizzano forward e reverse process come catene di Markov. Il modello inverso viene addestrato attraverso una evidence lower bound, dalla quale deriva la nota loss di noise prediction. L'[approfondimento sui DDPM](03_ddpm/README.md) sviluppa posteriori gaussiane, parametrizzazioni $\epsilon$, $x_0$ e $v$, sampling ancestrale, DDIM e classifier-free guidance.

## VAE, Latent Diffusion e modelli moderni

La diffusione nei pixel è costosa perché ogni valutazione opera alla risoluzione finale. I **Latent Diffusion Models** separano compressione percettiva e generazione: un autoencoder porta il dato in uno spazio latente più piccolo, il modello generativo opera in tale spazio e il decoder ricostruisce l'output. Il capitolo su [VAE e latent diffusion](03_latent_diffusion/README.md) collega il principio variazionale a Stable Diffusion e alle architetture moderne basate su Diffusion Transformer, modelli multimodali e rectified flow.

## Score matching e formulazione continua

Lo score descrive la direzione locale di crescita della densità. Con il denoising score matching può essere appreso senza conoscere esplicitamente la densità dei dati. Le **stochastic differential equations** estendono il tempo discreto a un intervallo continuo e mostrano che DDPM e precedenti modelli score-based sono discretizzazioni di uno stesso quadro. L'[approfondimento sui modelli score-based](04_score_based_models/README.md) tratta forward SDE, reverse-time SDE, probability flow ODE e sampler predictor-corrector.

## Continuous Normalizing Flows e Flow Matching

Un **Continuous Normalizing Flow** trasporta campioni attraverso l'ODE definita da un campo di velocità. Il Flow Matching evita di simulare l'ODE durante il training: regredisce campi condizionali associati a probability path scelti in anticipo. Il capitolo su [CNF e Flow Matching](05_flow_matching/README.md) spiega continuità, conditional flow matching, cammini diffusivi e optimal-transport path, oltre al passaggio concettuale dallo score alla velocità.

## Training ed engineering

La formulazione matematica non determina da sola un sistema efficace. **Noise schedule, distribuzione dei timestep, loss weighting, normalizzazione, Exponential Moving Average, precisione numerica e sampler** modificano stabilità, qualità e costo. Il capitolo su [training e aspetti implementativi](06_training_and_engineering/README.md) raccoglie le scelte che devono essere specificate per rendere un esperimento interpretabile e riproducibile.

## Relazioni tra le formulazioni

Le famiglie descritte non sono compartimenti indipendenti. Un DDPM può essere interpretato come score model discreto; lo stesso processo ammette una reverse-time SDE e una probability flow ODE; un probability path diffusivo può infine essere usato come path di Flow Matching. Cambiano il target appreso e la dinamica usata in generazione, ma rimane invariato il problema di fondo:

$$
p_{\mathrm{base}}
\longrightarrow
p_{\mathrm{data}}.
$$

La distinzione operativa più utile è quindi tra **oggetto predetto** — rumore, dato pulito, score o velocità — e **procedura di sampling** — catena stocastica, SDE oppure ODE. Architettura neurale, parametrizzazione e integratore sono scelte separabili, anche se nella pratica vengono spesso presentate sotto un'unica etichetta.
